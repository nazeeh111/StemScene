# Local browser mixer

The browser is a client of the same installed STEMSCENE engine used by the CLI. It has no remote fonts, trackers, model service, generated audio fixtures or hosted dependencies.

## Visual direction

A restrained studio workspace uses warm graphite surfaces, pale waveform marks and a muted yellow selection. The recording and timeline occupy the central column, persistent sound estimates the left column, and selected-event controls the right. Color associates a layer with its controls; names and stable identifiers remain readable without color. System fonts keep the interface self-contained. Source preservation and estimated-class uncertainty appear where they affect editing decisions.

The empty state explains the import-to-edit task. The populated view shows actual waveform peak bins from the engine. Its waveform drawing scales the visual amplitude to the clip peak for inspection; it does not normalize or alter playback. No waveform is invented while audio is absent.

## Interaction contract

- Import WAV/FLAC bytes through `/api/import`. The browser does not decode or modify source samples before sending them to the loopback engine.
- Comma/newline descriptions request actual local computation. With a blank request field, estimating again uses existing layer descriptions so saved anchors can be applied.
- Select a time span by dragging the original waveform or entering start/end seconds. The numeric fields are the keyboard equivalent of the drag control. A level change applies only to the chosen estimate and span, with explicit edge fades.
- Saved event selection restores its span and level. Updating replaces that event; New event deliberately adds another operation. Separate overlapping events multiply. Removing an event is undoable.
- Listen to original, edited mix or a layer through authorized WAV fetches and browser blob playback. Listening to a layer is an estimated isolation, not a quality guarantee.
- A lock protects an estimate during added requests; it does not disable gain edits. Anchors explicitly refer to present/absent spans in the mixed recording. Similar descriptions and physical-source uncertainty remain engine decisions.
- Long inference has visible progress and cancellation. Controls that could mutate the same project are disabled while an estimation job runs. Errors persist until dismissal or another action, rather than disappearing on a timer.
- Export requests DOUBLE WAV precision and an explicit peak policy. The local engine checks its artifacts, then the browser downloads the aligned ZIP bundle. The UI reports a requested browser download, not a verified filesystem save.

Every API fetch, including reads, audio and ZIP downloads, carries the per-session `X-StemScene-Token` injected into the HTML meta tag. Audio/download tokens are not put in URLs. Dynamic text is assigned through `textContent`; project and query names are never interpreted as HTML. The local service owns authorization, path validation, limits and atomic persistence.

## Accessibility and layout

Forms use labels and fieldsets. Buttons expose selected/locked states, anchors have named remove controls, errors use an alert, and general status uses a polite live region. Native audio controls remain available when autoplay is blocked. Focus rings are visible, time selection has numeric controls, and layout supports narrow screens without a fixed desktop canvas. Motion is limited to brief hover-state changes and is removed under reduced-motion preferences.

## Verification state

Static checks cover JavaScript parsing, HTML identifier consistency, a single session-token placeholder and absence of external asset URLs. Actual browser verification imported a new CC0 recording, ran three local estimates, selected an event, preserved a locked estimate while adding a request, undid changes and downloaded/recombined the export outside the app. Saved-event replacement/removal/reload/undo were subsequently exercised. A current 390-pixel layout had no horizontal overflow. Captured states and actual audio appear in [the demo](demo/index.html); [verification](verification.md) distinguishes these checks from a complete accessibility audit or source-quality judgment.
