"""Extract the recorded 5–7 second event from the attributed 0–8 second WAVs."""
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent / "assets"
for source, target in (("original-0-8s.wav", "event-original.wav"),
                       ("edited-0-8s.wav", "event-edited.wav")):
    samples, rate = sf.read(HERE / source, dtype="float32", always_2d=True)
    assert rate == 48000 and len(samples) == rate * 8
    path = HERE / target
    if path.exists():
        raise FileExistsError(f"Refusing to replace {path}")
    sf.write(path, samples[5 * rate:7 * rate], rate, format="WAV", subtype="FLOAT")
