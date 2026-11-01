# Launch draft

Draft only. This text has not been posted or sent.

I built STEMSCENE to turn a recording into persistent estimated sound layers,
edit selected events, and export aligned WAVs with a mix another application
can reproduce. It runs locally through a browser mixer, CLI and Python API.

The experimental beta combines AudioSep proposals with a small trained joint
refiner. In a frozen 20-scene controlled study, it reduced mean multi-layer edit
error by 7.25% versus the strongest comparable baseline. There are real limits:
29 of 60 individual edits worsened, loudness accuracy declined, and five of six
absent requests still extracted audio. The repository includes the protocol,
all comparisons and failures, a real-recording demo and independent replay.

It is intended for reviewed sound-design experiments with the original retained.
Feedback on the editing workflow, reproducible exports and specific failure cases
would be useful.
