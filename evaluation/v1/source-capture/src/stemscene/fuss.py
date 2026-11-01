"""Optional CPU FUSS comparator: four unlabelled estimates, 16 kHz mono.

Tensor contract adapted from Google Research's Apache-2.0 inference.py,
Copyright 2020 Google LLC, google-research/sound-separation.
Changed: bounded inputs, explicit CPU/session lifetime, rate conversion and
output validation. Checkpoints have separate record-level CC BY 4.0 terms:
https://doi.org/10.5281/zenodo.4012661 . No implicit network access.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

FUSS_DRY_BYTES = 102324048
FUSS_DRY_MD5 = "aca2b32125659a420bd5c307e2505e43"
FUSS_DRY_SHA256 = "49f49e9a9ce3bf78771418c342b031c770e0f0604cbf580eb474593af0589529"
FUSS_DRY_URL = "https://zenodo.org/api/records/4012661/files/FUSS_baseline_dry_model.tar.gz/content"


class FussSeparator:
    """CPU-only comparator returning float64 (4, samples, 1) arrays.

    Input is resampled to 16 kHz mono; stereo is rejected. Polyphase FIR
    conversion compensates its filter delay. Back-converted estimates are
    trimmed to input length, with no gain normalization. This is offline
    full-clip inference. Output indices carry no semantic identities.
    """

    sample_rate = 16000
    source_count = 4

    def __init__(self, model_dir: str | Path, *, max_seconds: float = 30.0,
                 cpu_threads: int = 2) -> None:
        if not np.isfinite(max_seconds) or not 0 < max_seconds <= 120:
            raise ValueError("max_seconds must be finite and in (0, 120]")
        if not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= 16:
            raise ValueError("cpu_threads must be an integer in [1, 16]")
        folder = Path(model_dir).resolve()
        prefix = "baseline_dry_model" if (folder / "baseline_dry_model.index").is_file() else "baseline_model"
        meta = folder / ("baseline_dry_inference.meta" if prefix == "baseline_dry_model" else "baseline_inference.meta")
        for name in (meta.name, f"{prefix}.index", f"{prefix}.data-00000-of-00001"):
            if not (folder / name).is_file():
                raise FileNotFoundError(f"Missing FUSS model artifact: {name}")
        import tensorflow.compat.v1 as tf
        self.max_seconds = max_seconds
        self.graph = tf.Graph()
        config = tf.ConfigProto(device_count={"GPU": 0},
                                intra_op_parallelism_threads=cpu_threads,
                                inter_op_parallelism_threads=1)
        self.session = tf.Session(graph=self.graph, config=config)
        try:
            with self.graph.as_default(), tf.device("/CPU:0"):
                saver = tf.train.import_meta_graph(str(meta), clear_devices=True)
                saver.restore(self.session, str(folder / prefix))
            self.input_tensor = self.graph.get_tensor_by_name("input_audio/receiver_audio:0")
            self.output_tensor = self.graph.get_tensor_by_name("denoised_waveforms:0")
        except BaseException:
            self.session.close()
            raise
        self.provenance: dict[str, Any] = {
            "backend": "fuss-dry-2020" if prefix == "baseline_dry_model" else "fuss-reverb-2020", "source_count": 4,
            "native_sample_rate": 16000, "native_channels": 1,
            "device": "cpu", "normalization": "none",
            "model_record": "https://doi.org/10.5281/zenodo.4012661",
            "model_record_license": "CC-BY-4.0", "code_license": "Apache-2.0",
            "semantic_assignment": "none; outputs are unlabelled",
            "resampling": "scipy.signal.resample_poly; compensated zero-phase FIR",
            "context": "offline full clip", "max_seconds": max_seconds,
        }

    def separate(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return four estimates aligned to the caller's declared sample rate."""
        if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
            raise ValueError("sample_rate must be an integer in [8000, 192000]")
        x = np.asarray(audio)
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError("FUSS accepts (samples, 1) mono audio only")
        if not 0 < x.shape[0] <= int(self.max_seconds * sample_rate):
            raise ValueError("audio must be nonempty and within the clip limit")
        if not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
            raise ValueError("audio must be a finite floating-point array")
        if np.max(np.abs(x)) > 16:
            raise ValueError("audio absolute amplitude exceeds the bound of 16")
        n = x.shape[0]
        divisor = gcd(sample_rate, self.sample_rate)
        mono = x[:, 0].astype(np.float64)
        if sample_rate != self.sample_rate:
            mono = resample_poly(mono, self.sample_rate // divisor, sample_rate // divisor)
        native_length = mono.shape[0]
        padded = np.pad(mono, (0, max(0, 2048 - native_length)))
        pred = self.session.run(self.output_tensor, feed_dict={
            self.input_tensor: padded.astype(np.float32)[None, None, :]
        })[0]
        if pred.ndim != 2 or pred.shape[0] != 4 or pred.shape[1] < native_length:
            raise RuntimeError(f"Unexpected FUSS output shape: {pred.shape}")
        pred = pred[:, :native_length].astype(np.float64)
        if sample_rate != self.sample_rate:
            pred = resample_poly(pred, sample_rate // divisor, self.sample_rate // divisor, axis=1)
        if pred.shape[1] < n or not np.isfinite(pred).all():
            raise RuntimeError("FUSS produced an invalid or too-short output")
        return pred[:, :n, None]

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "FussSeparator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
