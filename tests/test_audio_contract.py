from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stemscene.audio import AudioContractError, AudioData, AudioLimits
from stemscene.export import export_project, replay_export
from stemscene.project import Anchor, LockedLayerError, Project


def original(tmp_path: Path, channels: int = 1) -> tuple[Path, np.ndarray, int]:
    sr = 8000
    t = np.arange(sr * 2) / sr
    x = np.column_stack([0.15 * np.sin(2 * np.pi * (220 + 73 * c) * t) for c in range(channels)])
    path = tmp_path / 'recording.wav'
    sf.write(path, x, sr, subtype='DOUBLE')
    return path, x, sr


def test_preserves_source_bytes_rate_gain_and_stereo(tmp_path: Path) -> None:
    source, x, sr = original(tmp_path, 2)
    project = Project.create(source, tmp_path / 'session')
    assert np.array_equal(project.original.samples, x)
    assert project.sample_rate == sr
    assert project.original.channels == 2
    assert project.original.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (project.root / 'original.wav').read_bytes() == source.read_bytes()
    assert not project.original.samples.flags.writeable
    with pytest.raises(FileExistsError):
        Project.create(source, tmp_path / 'session')


def test_edits_touch_selected_samples_only_and_undo_persists(tmp_path: Path) -> None:
    source, x, sr = original(tmp_path, 2)
    project = Project.create(source, tmp_path / 'session')
    layer_id = project.add_estimates({'tone': x * .75})['tone']
    project.edit(layer_id, .5, 1.25, gain=.2, fade_seconds=.01)
    result = project.render()
    assert np.array_equal(result[:sr // 2], x[:sr // 2])
    assert np.array_equal(result[5 * sr // 4:], x[5 * sr // 4:])
    assert np.max(np.abs(result[sr:sr+100])) < np.max(np.abs(x[sr:sr+100]))
    envelope = project.envelope(layer_id)
    assert envelope[sr//2] == 1
    assert envelope[5*sr//4 - 1] == 1
    restored = Project.load(project.root)
    assert np.array_equal(restored.render(), result)
    assert restored.undo()
    assert np.array_equal(restored.render(), x)
    assert np.array_equal(Project.load(project.root).render(), x)


def test_independent_export_recombination_and_unity_identity(tmp_path: Path) -> None:
    source, x, sr = original(tmp_path, 2)
    project = Project.create(source, tmp_path / 'session')
    ids = project.add_estimates({'low tone': x * .6, 'other': x * .1})
    project.edit(ids['low tone'], .25, .75, 0)
    manifest_path = export_project(project, tmp_path / 'export')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['sample_rate'] == sr
    assert manifest['channels'] == 2
    assert (manifest_path.parent / manifest['source']['path']).read_bytes() == source.read_bytes()
    # Independent consumer only uses soundfile/numpy, not the project engine.
    out = manifest_path.parent
    baseline, _ = sf.read(out / 'original-decoded.wav', always_2d=True)
    residual, _ = sf.read(out / 'residual.wav', always_2d=True)
    combined, unity = baseline.copy(), residual.copy()
    for row in manifest['layers']:
        stem, stem_sr = sf.read(out / row['stem'], always_2d=True)
        gain, gain_sr = sf.read(out / row['envelope'], always_2d=True)
        assert stem_sr == gain_sr == sr
        assert stem.shape == x.shape
        assert gain.shape == (len(x), 1)
        combined += (gain - 1) * stem
        unity += stem
        assert json.loads((out / row['envelope_recipe']).read_text())['frames'] == len(x)
    assert np.max(np.abs(unity - x)) < 1e-15
    assert np.array_equal(combined, project.render())
    assert np.array_equal(replay_export(out), project.render())
    with pytest.raises(FileExistsError):
        export_project(project, out)


def test_locked_layer_rejects_transaction_and_survives_addition(tmp_path: Path) -> None:
    source, x, _ = original(tmp_path)
    project = Project.create(source, tmp_path / 'session')
    identifier = project.add_estimates({'tone': x * .6})['tone']
    project.set_locked(identifier)
    revision = project.revision
    with pytest.raises(LockedLayerError):
        project.add_estimates({'tone': x * .7, 'new layer': x * .05})
    assert project.revision == revision
    assert len(project.layers) == 1
    before = project.layers[0].samples.copy()
    project.add_estimates({'new layer': x * .05})
    loaded = Project.load(project.root)
    assert loaded.layers[0].id == identifier
    assert loaded.layers[0].locked
    assert np.array_equal(loaded.layers[0].samples, before)


def test_duplicate_queries_group_and_unlocked_update_keeps_identity(tmp_path: Path) -> None:
    source, x, _ = original(tmp_path)
    project = Project.create(source, tmp_path / 'session')
    identifiers = project.add_estimates({' DOG ': x, 'dog': x})
    assert len(project.layers) == 1
    identifier = project.layers[0].id
    assert identifiers['dog'] == identifier
    assert project.add_estimates({'DoG': x * .5})['DoG'] == identifier
    assert np.array_equal(project.layers[0].samples, x * .5)
    with pytest.raises(AudioContractError, match='disagree'):
        project.add_estimates({'dog': x, ' DOG ': x * 0})


def test_anchors_are_in_recording_not_reference_data(tmp_path: Path) -> None:
    source, x, _ = original(tmp_path)
    project = Project.create(source, tmp_path / 'session')
    identifier = project.add_estimates({'tone': x}, anchors={'tone': [Anchor(.2, .6), Anchor(1, 1.5, False)]})['tone']
    assert Project.load(project.root).layers[0].anchors == (Anchor(.2, .6), Anchor(1, 1.5, False))
    with pytest.raises(AudioContractError):
        project.set_anchors(identifier, [Anchor(-.1, .2)])
    with pytest.raises(AudioContractError):
        project.set_anchors(identifier, [Anchor(1, 3)])


def test_stale_writer_cannot_discard_another_process_edit(tmp_path: Path) -> None:
    source, x, _ = original(tmp_path)
    first = Project.create(source, tmp_path / 'session')
    identifier = first.add_estimates({'tone': x})['tone']
    second = Project.load(first.root)
    first.edit(identifier, .1, .4, .25)
    with pytest.raises(AudioContractError, match='another process'):
        second.edit(identifier, .5, .9, 0)
    assert np.array_equal(Project.load(first.root).render(), first.render())


def test_malformed_oversize_nonfinite_and_unaligned_media_rejected(tmp_path: Path) -> None:
    source, x, _ = original(tmp_path)
    broken = tmp_path / 'broken.wav'
    broken.write_bytes(b'not a wav')
    with pytest.raises(AudioContractError):
        AudioData.read(broken)
    with pytest.raises(AudioContractError, match='duration'):
        AudioData.read(source, AudioLimits(max_duration_seconds=.1))
    with pytest.raises(AudioContractError, match='source byte'):
        AudioData.read(source, AudioLimits(max_source_bytes=10))
    project = Project.create(source, tmp_path / 'session')
    with pytest.raises(AudioContractError, match='shape'):
        project.add_estimates({'bad': x[:-1]})
    invalid = x.copy()
    invalid[0] = np.nan
    with pytest.raises(AudioContractError, match='NaN'):
        project.add_estimates({'bad': invalid})
    with pytest.raises(AudioContractError):
        Project.create(source, tmp_path / 'small', AudioLimits(max_project_array_bytes=1))


def test_tampered_project_paths_assets_and_source_rejected(tmp_path: Path) -> None:
    source, x, _ = original(tmp_path)
    project = Project.create(source, tmp_path / 'session')
    project.add_estimates({'tone': x})
    path = project.root / 'project.json'
    manifest = json.loads(path.read_text())
    saved = path.read_text()
    manifest['layers'][0]['path'] = '../../recording.wav'
    path.write_text(json.dumps(manifest))
    with pytest.raises(AudioContractError, match='leaves'):
        Project.load(project.root)
    path.write_text(saved)
    layer = project.root / json.loads(saved)['layers'][0]['path']
    layer.write_bytes(b'corrupt')
    with pytest.raises(AudioContractError, match='checksum'):
        Project.load(project.root)


def test_clipping_policy_is_explicit_and_float_export_declares_quantization(tmp_path: Path) -> None:
    source, x, _ = original(tmp_path)
    project = Project.create(source, tmp_path / 'session')
    identifier = project.add_estimates({'tone': x * 10})['tone']
    project.edit(identifier, 0, 2, 4, fade_seconds=0)
    with pytest.raises(AudioContractError, match='exceeds unity'):
        export_project(project, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()
    manifest_path = export_project(project, tmp_path / 'hard', clipping='hard', subtype='FLOAT')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['clipping'] == 'hard'
    assert 'float32' in manifest['precision_note']
    mix, _ = sf.read(manifest_path.parent / 'mix.wav', always_2d=True)
    assert np.max(np.abs(mix)) <= 1
    assert np.max(np.abs(replay_export(manifest_path.parent) - mix)) < 1e-6


def test_next_manifest_is_bounded_before_replacing_loadable_revision(tmp_path: Path) -> None:
    source = tmp_path / 'short.wav'
    sf.write(source, np.full((8, 1), .1), 8000, subtype='DOUBLE')
    project = Project.create(source, tmp_path / 'session')
    project.add_estimates({f'layer {i}': np.zeros((8, 1)) for i in range(12)},
                          metadata={'payload': 'x' * 950_000})
    project.set_locked(project.layers[0].id)
    before = (project.root / 'project.json').read_bytes()
    revision = project.revision
    # The next history snapshot would make a 34 MB document. It must never
    # replace the still-loadable 22 MB revision on disk.
    with pytest.raises(AudioContractError, match='Next JSON document exceeds'):
        project.set_locked(project.layers[1].id)
    assert (project.root / 'project.json').read_bytes() == before
    assert project.revision == revision
    assert Project.load(project.root).revision == revision


def test_export_rejects_actual_double_unity_cancellation_error(tmp_path: Path) -> None:
    source = tmp_path / 'short.wav'
    sf.write(source, np.full((8, 1), .1), 8000, subtype='DOUBLE')
    project = Project.create(source, tmp_path / 'session')
    project.add_estimates({'impossible cancellation': np.full((8, 1), 1e16)})
    with pytest.raises(AudioContractError, match='unity_max_abs_error'):
        export_project(project, tmp_path / 'bad-export', subtype='DOUBLE')
    assert not (tmp_path / 'bad-export').exists()
    assert np.array_equal(Project.load(project.root).original.samples, np.full((8, 1), .1))
    assert not list(tmp_path.glob('.bad-export.*'))


def test_float_export_rejects_gain_quantization_that_breaks_replay(tmp_path: Path) -> None:
    source = tmp_path / 'short.wav'
    sf.write(source, np.full((8, 1), .1), 8000, subtype='DOUBLE')
    project = Project.create(source, tmp_path / 'session')
    identifier = project.add_estimates({'large estimate': np.full((8, 1), 1e8)})['large estimate']
    project.edit(identifier, 0, .001, 1 + 1e-9, fade_seconds=0)
    with pytest.raises(AudioContractError, match='edit_replay_max_abs_error.*Try DOUBLE'):
        export_project(project, tmp_path / 'bad-export', subtype='FLOAT')
    assert not (tmp_path / 'bad-export').exists()
    # A smaller but still precision-sensitive estimate can satisfy DOUBLE.
    project.add_estimates({'large estimate': np.full((8, 1), 1000.)})
    path = export_project(project, tmp_path / 'double-export', subtype='DOUBLE')
    verification = json.loads(path.read_text())['verification']
    assert verification['absolute_tolerance'] == 1e-12
    assert verification['unity_max_abs_error'] <= 1e-12
    assert verification['edit_replay_max_abs_error'] <= 1e-12


def test_fifo_project_asset_is_rejected_without_opening_pipe(tmp_path: Path) -> None:
    import os
    source, x, _ = original(tmp_path)
    project = Project.create(source, tmp_path / 'session')
    project.add_estimates({'tone': x})
    pipe = project.root / 'asset.fifo'
    os.mkfifo(pipe)
    manifest_path = project.root / 'project.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['layers'][0]['path'] = pipe.name
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AudioContractError, match='regular file'):
        Project.load(project.root)


def test_incorrect_broadcast_shape_rejected_before_large_copy(tmp_path: Path) -> None:
    import tracemalloc
    source = tmp_path / 'short.wav'
    sf.write(source, np.full((8, 1), .1), 8000, subtype='DOUBLE')
    project = Project.create(source, tmp_path / 'session')
    values = np.broadcast_to(np.array(0.), (1_000_000, 1))
    tracemalloc.start()
    try:
        with pytest.raises(AudioContractError, match='shape'):
            project.add_estimates({'oversized': values})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000


def test_hard_clipping_cannot_change_unedited_over_unity_original(tmp_path):
    source = tmp_path/'floating.wav'
    samples = np.full((8000,1),1.2)
    sf.write(source,samples,8000,subtype='DOUBLE')
    project = Project.create(source,tmp_path/'clip-project')
    layer = project.add_estimates({'event':np.zeros_like(samples)})['event']
    project.edit(layer,.1,.2,.5)
    with pytest.raises(AudioContractError,match='unedited original samples'):
        export_project(project,tmp_path/'hard-export',clipping='hard')
    assert not (tmp_path/'hard-export').exists()
    export_project(project,tmp_path/'preserved-export',clipping='preserve')
    np.testing.assert_array_equal(replay_export(tmp_path/'preserved-export'),samples)


def test_saved_event_updates_without_stacking_and_remove_undo_restores_it(tmp_path):
    source, samples, sr = original(tmp_path)
    project = Project.create(source,tmp_path/'event-project')
    layer = project.add_estimates({'tone':samples})['tone']
    event = project.edit(layer,.2,.5,.25,0)
    first = project.render().copy()
    project.edit(layer,.7,.9,.5,0,edit_id=event)
    assert len(project.edits)==1 and project.edits[0]['id']==event
    np.testing.assert_array_equal(project.render()[round(.2*sr):round(.5*sr)],samples[round(.2*sr):round(.5*sr)])
    np.testing.assert_allclose(project.render()[round(.7*sr):round(.9*sr)],samples[round(.7*sr):round(.9*sr)]*.5,atol=1e-16)
    assert project.undo()
    np.testing.assert_array_equal(project.render(),first)
    project.remove_edit(event)
    np.testing.assert_array_equal(Project.load(project.root).render(),samples)
    assert project.undo()
    np.testing.assert_array_equal(project.render(),first)
