"""Bounded lossless media admission. No resampling, downmixing or normalization."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf

Samples = npt.NDArray[np.float64]


class AudioContractError(ValueError):
    """Audio or project data violates the declared audio contract."""


@dataclass(frozen=True)
class AudioLimits:
    max_source_bytes: int = 256 * 1024 * 1024
    max_duration_seconds: float = 600.0
    max_channels: int = 2
    max_sample_rate: int = 192000
    max_decoded_bytes: int = 512 * 1024 * 1024
    max_layers: int = 12
    max_project_array_bytes: int = 2 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise AudioContractError("Resource limits must be positive and finite")


def sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    # Nonblocking admission prevents a crafted project asset from hanging on a
    # FIFO; fstat validates the actual descriptor rather than only its pathname.
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise AudioContractError("Asset must be a readable regular file") from exc
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise AudioContractError("Asset must be a regular file")
        if max_bytes is not None and os.fstat(stream.fileno()).st_size > max_bytes:
            raise AudioContractError("Asset exceeds its byte limit")
        count = 0
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(chunk)
            if max_bytes is not None and count > max_bytes:
                raise AudioContractError("Asset changed or exceeds its byte limit")
            digest.update(chunk)
    return digest.hexdigest()


def validate_samples(samples: npt.ArrayLike, *, shape: tuple[int, int] | None = None,
                     max_bytes: int = 512 * 1024 * 1024) -> Samples:
    # Inspect a view before allocating a float64 copy. A broadcast view may be
    # tiny in storage yet declare arbitrarily many output samples.
    view = np.asarray(samples)
    if view.ndim != 2 or min(view.shape) < 1:
        raise AudioContractError("Audio must have shape (samples, channels)")
    if shape is not None and view.shape != shape:
        raise AudioContractError(f"Aligned audio requires shape {shape}, received {view.shape}")
    if view.size * 8 > max_bytes:
        raise AudioContractError("Audio exceeds the decoded memory limit")
    result = np.array(view, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(result).all():
        raise AudioContractError("Audio contains NaN or infinity")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class AudioData:
    samples: Samples
    sample_rate: int
    source_sha256: str
    source_format: str
    source_subtype: str
    source_bytes: int

    @property
    def frames(self) -> int:
        return self.samples.shape[0]

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    def metadata(self) -> dict[str, Any]:
        return {"frames": self.frames, "channels": self.channels,
                "sample_rate": self.sample_rate, "sha256": self.source_sha256,
                "format": self.source_format, "subtype": self.source_subtype,
                "bytes": self.source_bytes}

    @classmethod
    def read(cls, path: str | Path, limits: AudioLimits | None = None) -> "AudioData":
        limits = limits or AudioLimits()
        source = Path(path).resolve()
        if not source.is_file():
            raise AudioContractError("Input must be a regular WAV or FLAC file")
        initial_stat = source.stat()
        size = initial_stat.st_size
        if size <= 0 or size > limits.max_source_bytes:
            raise AudioContractError("Input exceeds the source byte limit or is empty")
        try:
            with sf.SoundFile(source) as stream:
                if stream.format not in {"WAV", "WAVEX", "RF64", "FLAC"}:
                    raise AudioContractError("Only WAV and FLAC containers are admitted")
                if not 0 < stream.channels <= limits.max_channels:
                    raise AudioContractError("Input exceeds the channel limit")
                if not 0 < stream.samplerate <= limits.max_sample_rate:
                    raise AudioContractError("Input exceeds the sample rate limit")
                if stream.frames <= 0 or stream.frames / stream.samplerate > limits.max_duration_seconds:
                    raise AudioContractError("Input exceeds the duration limit or has no frames")
                if stream.frames * stream.channels * 8 > limits.max_decoded_bytes:
                    raise AudioContractError("Input exceeds the decoded memory limit")
                count = stream.frames
                samples = validate_samples(stream.read(count, dtype="float64", always_2d=True),
                                           max_bytes=limits.max_decoded_bytes)
                if len(samples) != count:
                    raise AudioContractError("Container frame count does not match decoded audio")
                result = cls(samples, stream.samplerate, sha256_file(source),
                             stream.format, stream.subtype, size)
        except (RuntimeError, OSError) as exc:
            raise AudioContractError(f"Cannot decode WAV/FLAC input: {exc}") from exc
        final_stat = source.stat()
        if (final_stat.st_size, final_stat.st_mtime_ns, final_stat.st_ino) != (
                initial_stat.st_size, initial_stat.st_mtime_ns, initial_stat.st_ino):
            raise AudioContractError("Input changed while decoding")
        return result
