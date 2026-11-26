# Recorded doorbell event preview

`assets/event-original.wav` and `assets/event-edited.wav` are exact samples 240,000–335,999 of the existing `original-0-8s.wav` and `edited-0-8s.wav` respectively, both 48 kHz mono float WAV. The source recording is RealDESED test recording 001738, audio CC0; dataset metadata and event annotations are CC BY 4.0. Full attribution and original hashes are in [demo provenance](assets/demo-provenance.json) and [credits](assets/CREDITS.txt).

`extract_event.py` copies the five-to-seven-second slice without normalization or effects and refuses to replace existing files. The edited audio is the previously recorded 50% estimated doorbell-layer edit with 20 ms inside-span fades. It is not a clean isolated physical source or a listening-quality claim. The synthetic editable project linked on the page is a separate example.

- Original 2-second WAV SHA-256: `e5a6fff63aa8db5829f5284a1488af178377a84e2bce9455fdd4018f97176249`
- Edited 2-second WAV SHA-256: `642da40c7f7781d27daabed1dab3ab96401fd1c7cc2460eb7bf13bab05fe5831`
