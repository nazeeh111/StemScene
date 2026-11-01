# Engine and reproducible edit path

The browser and CLI call the same installed `SceneEngine`. Processing uses ordinary authorized recordings and explicit local model files. No clean references, presence labels, remote model service or visual sensors enter production inference.

```mermaid
flowchart LR
    Browser["Loopback browser mixer"] --> API["Installed SceneEngine"]
    CLI["Typed Python API / CLI"] --> API
    Input["Preserved original bytes + SHA256"] --> Decode["Bounded WAV / FLAC decode"]
    Decode --> Original["Immutable decoded original"]
    Original --> Proposals["AudioSep proposals: unchanged mixture per query"]
    Anchors["User anchors in mixed-recording time"] --> Joint
    Queries["Canonical query set + stable class IDs"] --> Proposals
    Queries --> Joint["Trained shared joint mask / residual refiner"]
    Proposals --> Joint
    Joint --> Layers["Persistent estimated stems"]
    Locks["Locked stem bytes"] --> Layers
    Original --> Residual["Explicit residual = original - sum of stems"]
    Layers --> Residual
    Layers --> Edit["Selected-span gains + inside-span fades"]
    Original --> Edit
    Edit --> Mix["Original + sum of selected component changes"]
    Mix --> Export["Aligned WAVs + envelopes + manifest"]
    Layers --> Export
    Residual --> Export
    Export --> Verify["Independent WAV recombination + replay"]
    API --> Proposals
```

The editable source is the Mermaid block above. `architecture.mmd` contains the same graph for diagram tools.

Activity and overlap diagnostics help reveal uncertain associations. A stable query-class ID does not prove that one physical source was tracked. Similar or indistinguishable requests may require grouping or a useful anchor. The explicit residual preserves unresolved energy without forcing every requested description to own part of the mixture.

Training and evaluation are separate utilities. They can open clean source references for loss and scoring; the installed inference API cannot accept them. Frozen recipe and source hashes connect those records to the reported experiment. Reconstruction accuracy and separation accuracy are reported separately.
