"""Explicitly local AudioSep inference, with no implicit downloads.

Text branch inference follows AudioSep's vendored CLAP implementation. The
official AudioSep checkpoint already holds its frozen CLAP text weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import hashlib
import json
import math
import time

import numpy as np
from scipy.signal import resample_poly

AUDIOSEP_SHA256 = "f8cda01bfd0ebd141eef45d41db7a3ada23a56568465840d3cff04b8010ce82c"
MODEL_RATE = 32000


class InferenceCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendLimits:
    max_seconds: float = 600.0
    max_queries: int = 12
    chunk_seconds: float = 5.0
    overlap_seconds: float = 2.0
    timeout_seconds: float = 3600.0
    threads: int = 4

    def __post_init__(self) -> None:
        if (not math.isfinite(self.max_seconds) or not 0 < self.max_seconds <= 600
                or type(self.max_queries) is not int or not 1 <= self.max_queries <= 12
                or not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 3600
                or type(self.threads) is not int or not 1 <= self.threads <= 10
                or not math.isfinite(self.chunk_seconds) or not .1 <= self.chunk_seconds <= 10
                or not math.isfinite(self.overlap_seconds) or not 0 <= self.overlap_seconds < self.chunk_seconds):
            raise ValueError("backend limits exceed the supported resource bounds")


def resample_aligned(samples: np.ndarray, old_rate: int, new_rate: int,
                     length: int | None = None) -> np.ndarray:
    """Zero-phase polyphase FIR, compensated group delay; right trim/pad only.

    This changes bandwidth. Export retains the original decoded signal, so
    back-resampling a proposal never replaces the original recording.
    """
    if old_rate <= 0 or new_rate <= 0:
        raise ValueError("sample rates must be positive")
    a = np.asarray(samples)
    factor = math.gcd(old_rate, new_rate)
    y = a.copy() if old_rate == new_rate else resample_poly(
        a, new_rate // factor, old_rate // factor, axis=0,
        window=("kaiser", 5.0), padtype="constant")
    if length is not None:
        if len(y) < length:
            y = np.pad(y, [(0, length-len(y))]+[(0, 0)]*(y.ndim-1))
        y = y[:length]
    return np.asarray(y, dtype=np.float64)


class AudioSepBackend:
    """Mono CPU backend. Stereo inference is rejected until separately tested.

    Construction loads only explicitly supplied files and verifies checkpoint
    bytes before restricted tensor deserialization. No credentials are read.
    """

    def __init__(self, checkpoint: str | Path, tokenizer_directory: str | Path,
                 *, limits: BackendLimits = BackendLimits(), device: str = "cpu"):
        import torch
        from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
        from ._vendor.audiosep.resunet import ResUNet30
        if device not in {"cpu", "mps"}:
            raise ValueError("supported devices: cpu, mps; MPS is experimental")
        if not 1 <= limits.threads <= 10:
            raise ValueError("threads must be between 1 and 10")
        if not (0 <= limits.overlap_seconds < limits.chunk_seconds <= 10):
            raise ValueError("invalid bounded chunk configuration")
        checkpoint, tokenizer_directory = Path(checkpoint), Path(tokenizer_directory)
        with checkpoint.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != AUDIOSEP_SHA256:
            raise ValueError("checkpoint SHA256 does not match the permitted pinned model")
        torch.set_num_threads(limits.threads)
        self.limits, self.device, self.checkpoint_sha256 = limits, device, digest
        self._torch = torch
        # Legacy convolution buffers require ordinary construction; the text
        # encoder below supports metadata-device construction without RNG work.
        self.separator = ResUNet30(1, 1, 512).eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
        self.separator.load_state_dict({k[len("ss_model."):]: v for k, v in state.items()
                                        if k.startswith("ss_model.")}, strict=True, assign=True)
        config = RobertaConfig(**json.loads((tokenizer_directory / "config.json").read_text()))
        with torch.device("meta"):
            self.text = RobertaModel(config).eval()
        self.text.embeddings.register_buffer("position_ids", torch.arange(config.max_position_embeddings)[None], persistent=False)
        self.text.embeddings.register_buffer("token_type_ids", torch.zeros((1, config.max_position_embeddings), dtype=torch.long), persistent=False)
        prefix = "query_encoder.model.text_branch."
        text_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        # Older transformers persisted this deterministic buffer; newer versions
        # reconstruct it. Verify equality before dropping this one legacy key.
        positions = text_state.pop("embeddings.position_ids", None)
        if positions is not None and not torch.equal(positions, self.text.embeddings.position_ids):
            raise ValueError("checkpoint position IDs differ from tokenizer configuration")
        self.text.load_state_dict(text_state, strict=True, assign=True)
        with torch.device("meta"):
            self.projection = torch.nn.Sequential(torch.nn.Linear(768, 512), torch.nn.ReLU(),
                                                   torch.nn.Linear(512, 512)).eval()
        prefix = "query_encoder.model.text_projection."
        self.projection.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                         if k.startswith(prefix)}, strict=True, assign=True)
        self.tokenizer = RobertaTokenizer(vocab_file=str(tokenizer_directory / "vocab.json"),
                                          merges_file=str(tokenizer_directory / "merges.txt"))
        self.separator.to(device)
        # Text on CPU avoids unsupported MPS encoder operations and is cached.
        self._embeddings: dict[str, np.ndarray] = {}

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        torch = self._torch
        if not queries or len(queries) > self.limits.max_queries:
            raise ValueError("query count exceeds bounds or is empty")
        clean = [" ".join(q.lower().split()) for q in queries]
        if any(not q or len(q) > 512 for q in clean):
            raise ValueError("queries must contain 1 to 512 characters")
        missing = list(dict.fromkeys(q for q in clean if q not in self._embeddings))
        if missing:
            tokens = self.tokenizer(missing, padding="max_length", truncation=True,
                                    max_length=512, return_tensors="pt")
            with torch.inference_mode():
                pooled = self.text(input_ids=tokens["input_ids"],
                                   attention_mask=tokens["attention_mask"]).pooler_output
                embeds = torch.nn.functional.normalize(self.projection(pooled), dim=-1).numpy()
            self._embeddings.update(zip(missing, embeds))
        return np.stack([self._embeddings[q] for q in clean])

    def separate(self, samples: np.ndarray, sample_rate: int, queries: Sequence[str],
                 *, cancelled: Callable[[], bool] | None = None,
                 progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        """Return K aligned `(samples,1)` proposals, offline context only.

        Independent queries see the unchanged original mixture. A raised cancel
        or timeout leaves persistence to the caller; no partial project mutation.
        Hann overlap-add at 32 kHz; no loudness normalization or delay fitting.
        Cancellation is checked between bounded inference chunks.
        """
        torch = self._torch
        view = np.asarray(samples)
        if view.ndim != 2 or view.shape[1] != 1 or view.dtype.kind not in "fiu":
            raise ValueError("AudioSep inference currently supports mono input only")
        if (type(sample_rate) is not int or not 8000 <= sample_rate <= 192000
                or not 1 <= len(view) <= sample_rate*self.limits.max_seconds
                or view.size*8 > 512*1024*1024
                or view.size*8*len(queries) > 2*1024*1024*1024):
            raise ValueError("input sample rate or duration exceeds inference bounds")
        if cancelled is not None and cancelled():
            raise InferenceCancelled("inference cancelled before encoding")
        x = np.asarray(view, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("input must be finite")
        embeddings = self.embed_queries(queries)
        model_input = resample_aligned(x[:, 0], sample_rate, MODEL_RATE)
        chunk = round(self.limits.chunk_seconds*MODEL_RATE)
        hop = chunk-round(self.limits.overlap_seconds*MODEL_RATE)
        starts = list(range(0, max(1, len(model_input)-chunk+hop), hop))
        total = len(starts)*len(queries)
        result = np.empty((len(queries), len(x), 1), dtype=np.float64)
        began = time.monotonic()
        count = 0
        for qi, embed in enumerate(embeddings):
            output = np.zeros(len(model_input), dtype=np.float64)
            weights = np.zeros(len(model_input), dtype=np.float64)
            for start in starts:
                if (cancelled is not None and cancelled()):
                    raise InferenceCancelled("inference cancelled before project mutation")
                if time.monotonic()-began > self.limits.timeout_seconds:
                    raise TimeoutError("inference time budget exceeded")
                segment = model_input[start:start+chunk]
                length = len(segment)
                # Fixed windows bound memory and make edge policy reproducible.
                padded = np.pad(segment, (0, chunk-length)).astype(np.float32)
                with torch.inference_mode():
                    y = self.separator({"mixture": torch.from_numpy(padded)[None, None].to(self.device),
                                        "condition": torch.from_numpy(embed.copy())[None].to(self.device)})["waveform"]
                y = y[0, 0, :length].cpu().numpy().astype(np.float64)
                if len(y) != length or not np.isfinite(y).all():
                    raise RuntimeError("backend returned malformed or unaligned output")
                window = np.maximum(np.hanning(chunk), 1e-4)[:length]
                output[start:start+length] += y*window
                weights[start:start+length] += window
                count += 1
                if progress:
                    progress(count, total)
            result[qi, :, 0] = resample_aligned(output/weights, MODEL_RATE, sample_rate, len(x))
        return result
