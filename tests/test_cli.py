"""Structural installed-entry tests. Generated audio is a correctness fixture only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

from stemscene.project import Project


def invoke(*arguments: object) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / 'src')
    environment['PYTHONPATH'] = source_root + os.pathsep + environment.get('PYTHONPATH', '')
    return subprocess.run([sys.executable, '-m', 'stemscene', *(str(arg) for arg in arguments)],
                          text=True, capture_output=True, env=environment, timeout=30)


def make_source(tmp_path: Path) -> tuple[Path, np.ndarray]:
    samples = (.1 * np.sin(2 * np.pi * 220 * np.arange(800) / 8000))[:, None]
    source = tmp_path / 'input.wav'
    sf.write(source, samples, 8000, subtype='DOUBLE')
    return source, samples


def test_cli_init_edit_export_replay_and_undo(tmp_path: Path) -> None:
    source, samples = make_source(tmp_path)
    root = tmp_path / 'project'
    result = invoke('init', source, root)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['source']['sample_rate'] == 8000
    project = Project.load(root)
    # Deterministic test component, not a substituted model inference or demo.
    identifier = project.add_estimates({'fixture tone': samples * .75})['fixture tone']
    result = invoke('edit', root, identifier, '--start', '.02', '--end', '.06', '--gain', '.2', '--fade', '.001')
    assert result.returncode == 0, result.stderr
    changed = Project.load(root).render()
    assert np.array_equal(changed[:160], samples[:160])
    assert np.array_equal(changed[480:], samples[480:])
    rendered = tmp_path / 'render.wav'
    assert invoke('render', root, rendered).returncode == 0
    decoded, rate = sf.read(rendered, always_2d=True)
    assert rate == 8000 and np.array_equal(decoded, changed)
    exported = tmp_path / 'export'
    result = invoke('export', root, exported)
    assert result.returncode == 0, result.stderr
    assert json.loads((exported / 'manifest.json').read_text())['verification']['absolute_tolerance'] == 1e-12
    replayed = tmp_path / 'replayed.wav'
    result = invoke('replay', exported, replayed)
    assert result.returncode == 0, result.stderr
    replay, rate = sf.read(replayed, always_2d=True)
    assert rate == 8000 and np.array_equal(replay, changed)
    assert invoke('undo', root).returncode == 0
    assert np.array_equal(Project.load(root).render(), samples)


def test_cli_locks_anchors_and_inspect(tmp_path: Path) -> None:
    source, samples = make_source(tmp_path)
    root = tmp_path / 'project'
    project = Project.create(source, root)
    identifier = project.add_estimates({'fixture tone': samples})['fixture tone']
    assert invoke('lock', root, identifier).returncode == 0
    assert Project.load(root).layers[0].locked
    assert invoke('unlock', root, identifier).returncode == 0
    result = invoke('anchors', root, identifier, '--present', '.01', '.03', '--absent', '.06', '.09')
    assert result.returncode == 0, result.stderr
    result = invoke('inspect', root)
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload['layers'][0]['id'] == identifier
    assert payload['layers'][0]['locked'] is False
    assert [item['present'] for item in payload['layers'][0]['anchors']] == [True, False]


def test_cli_invalid_requests_and_existing_output_preserve_files(tmp_path: Path) -> None:
    source, _ = make_source(tmp_path)
    root = tmp_path / 'project'
    project = Project.create(source, root)
    source_bytes = source.read_bytes()
    result = invoke('render', root, source)
    assert result.returncode == 2 and 'already exists' in result.stderr
    assert source.read_bytes() == source_bytes
    result = invoke('edit', root, 'unknown', '--start', '0', '--end', '.1', '--gain', '0')
    assert result.returncode == 2 and 'Unknown layer' in result.stderr
    assert Project.load(root).revision == project.revision
    result = invoke('edit', root)
    assert result.returncode == 2 and 'required' in result.stderr
    result = invoke('extract', root, '--query', 'rain', '--checkpoint', tmp_path / 'missing.ckpt',
                    '--tokenizer-dir', tmp_path / 'missing-tokenizer', '--refiner', tmp_path / 'missing-refiner.pt')
    assert result.returncode == 2 and 'Trained refiner artifact does not exist' in result.stderr
    assert 'Traceback' not in result.stderr


def test_model_setup_refuses_existing_directory_before_network(tmp_path: Path) -> None:
    destination = tmp_path / 'models'
    destination.mkdir()
    preserved = destination / 'keep.txt'
    preserved.write_text('retain')
    result = invoke('models', 'download', destination)
    assert result.returncode == 2 and 'already exists' in result.stderr
    assert preserved.read_text() == 'retain'
    assert not list(tmp_path.glob('.models.partial-*'))


def test_help_and_version_do_not_require_model_artifacts() -> None:
    result = invoke('--version')
    assert result.returncode == 0 and 'stemscene 0.1.0b1' in result.stdout
    result = invoke('--help')
    assert result.returncode == 0 and 'extract' in result.stdout and 'replay' in result.stdout
    result = invoke('models', 'download', '--help')
    assert result.returncode == 0 and '1.27 GB' in result.stdout


def test_model_download_integrity_failure_retains_partial_and_never_publishes(tmp_path: Path, monkeypatch) -> None:
    import hashlib
    import io
    from stemscene import models

    class PublicResponse(io.BytesIO):
        headers: dict[str, str] = {}

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == 'https://example.invalid/public-model'
            assert not any(key.lower() == 'authorization' for key in request.headers)
            return PublicResponse(b'wrong')

    expected = b'right'
    monkeypatch.setattr(models, 'ASSETS', (models.PublicAsset('model.bin', 'https://example.invalid/public-model',
                                                           len(expected), hashlib.sha256(expected).hexdigest()),))
    monkeypatch.setattr(models, 'build_opener', lambda *handlers: Opener())
    destination = tmp_path / 'models'
    import pytest
    with pytest.raises(ValueError, match='partial model files retained'):
        models.download_models(destination)
    assert not destination.exists()
    partials = list(tmp_path.glob('.models.partial-*'))
    assert len(partials) == 1
    assert (partials[0] / 'model.bin.partial').read_bytes() == b'wrong'
    status = json.loads((partials[0] / 'download-manifest.json').read_text())
    assert status['status'] == 'failed'
    assert status['assets'][0]['received_bytes'] == 5
