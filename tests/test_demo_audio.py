"""The public event previews are exact slices of the attributed recorded demo."""
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf


ASSETS = Path(__file__).resolve().parents[1] / "docs" / "demo" / "assets"


def test_recorded_event_previews_are_exact_unmodified_slices():
    hashes = {
        "original": "e5a6fff63aa8db5829f5284a1488af178377a84e2bce9455fdd4018f97176249",
        "edited": "642da40c7f7781d27daabed1dab3ab96401fd1c7cc2460eb7bf13bab05fe5831",
    }
    for label, digest in hashes.items():
        preview_path = ASSETS / f"event-{label}.wav"
        assert hashlib.sha256(preview_path.read_bytes()).hexdigest() == digest
        source, rate = sf.read(ASSETS / f"{label}-0-8s.wav", dtype="float32", always_2d=True)
        preview, preview_rate = sf.read(preview_path, dtype="float32", always_2d=True)
        assert rate == preview_rate == 48000
        assert len(preview) == 2 * rate
        assert np.array_equal(preview, source[5 * rate : 7 * rate])
