# Captured-session walkthrough

The composition is 23 seconds, 1920×1080, 30 fps. It uses one actual session
capture, not fabricated clicks or a recorded live inference. Processing and waits
are explicitly omitted. The capture is cropped to the recording, three layers,
waveform and selected-event controls. External titles describe the heard audio.

| Movie interval | Picture | Actual audio source |
| --- | --- | --- |
| [0,3) s | Saved session; opening title | Silence |
| [3,11) s | Same capture; Original label | Preserved original source [0,8) s |
| [11,19) s | Same capture; Edited label | Actual rendered source [0,8) s |
| [19,23) s | Export evidence card | Silence |

Source is RealDESED 001738.wav, 750528 mono samples at 48 kHz. The single event edit
uses the estimated doorbell layer from 5 s to 7 s, gain 0.5, with 20 ms internal fades.
The same original excerpt appears in both comparison passes; there is no music,
level normalization, speech generation or selective re-recording. AAC movie
compression is explicitly distinct from the provided FLOAT32 WAV comparisons and
DOUBLE WAV export verification.

The screenshot displays revision 9. Revision 11 restored its event exactly after
removal/undo. Immutable snapshot checks compared preserved revision 8 layer state
plus the recorded revision 9 event list against revision 11: render difference 0.0.
No app interaction was invented by the video composition.

Text, crop and shot timing remain editable in render_demo.py and composition.json.
Captions are captions.vtt. The capture pixels remain raster footage; this is not
an editable reconstruction of the application's UI. No subjective listening
approval follows from frame inspection or media decode checks.
