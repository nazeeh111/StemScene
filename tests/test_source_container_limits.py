"""Container byte admission is independent of decoded waveform/estimate size."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stemscene import Project,export_project,replay_export,reopen_export
from stemscene.audio import AudioContractError,AudioLimits,sha256_file


def large_wav(path,junk_bytes):
    sf.write(path,np.zeros((100,1)),8000,subtype='DOUBLE')
    with path.open('r+b') as stream:
        stream.seek(0,2)
        stream.write(b'JUNK'+junk_bytes.to_bytes(4,'little'))
        stream.seek(junk_bytes-1,1);stream.write(b'\0')
        size=stream.tell();stream.seek(4);stream.write((size-8).to_bytes(4,'little'))
    return path


def test_large_admitted_source_container_exports_replays_reopens_and_continues(tmp_path):
    source=large_wav(tmp_path/'source.wav',33*1024*1024)
    project=Project.create(source,tmp_path/'project')
    layer=project.add_estimates({'supplied fixture':np.full((100,1),.01)})['supplied fixture']
    event=project.edit(layer,.001,.005,.5,fade_seconds=0)
    folder=tmp_path/'export';export_project(project,folder)
    assert source.stat().st_size>32*1024*1024
    assert np.array_equal(replay_export(folder),project.render())
    recipient=reopen_export(folder,tmp_path/'recipient')
    assert recipient.original.source_sha256==sha256_file(source)
    assert recipient.layers[0].id==layer and recipient.edits[0]['id']==event
    assert np.array_equal(recipient.render(),project.render())
    assert recipient.undo() and np.array_equal(recipient.render(),recipient.original.samples)
    recipient.edit(layer,.002,.006,.2,fade_seconds=0)
    export_project(recipient,tmp_path/'continued')


def test_over_admission_source_container_still_refused(tmp_path):
    source=large_wav(tmp_path/'oversource.wav',AudioLimits().max_source_bytes)
    assert source.stat().st_size>AudioLimits().max_source_bytes
    with pytest.raises(AudioContractError,match='byte|large'):
        Project.create(source,tmp_path/'refused')
    assert not (tmp_path/'refused').exists()


@pytest.mark.parametrize('role',['current','history','stem'])
def test_source_cap_does_not_relax_estimate_or_history_limits(tmp_path,role):
    source=tmp_path/'small.wav';sf.write(source,np.zeros((100,1)),8000,subtype='DOUBLE')
    project=Project.create(source,tmp_path/'project')
    project.add_estimates({'supplied fixture':np.full((100,1),.01)})
    project.add_estimates({'supplied fixture':np.full((100,1),.02)})
    folder=tmp_path/'export';export_project(project,folder)
    document=json.loads((folder/'manifest.json').read_text())
    portable=json.loads((folder/'project/project.json').read_text())
    if role=='current':relative='project/'+portable['layers'][0]['path']
    elif role=='history':relative='project/'+next(state['layers'][0]['path'] for state in portable['history'] if state['layers'])
    else:relative=document['layers'][0]['stem']
    with (folder/relative).open('r+b') as stream:stream.truncate(33*1024*1024)
    document['files'][relative]=sha256_file(folder/relative)
    (folder/'manifest.json').write_text(json.dumps(document))
    for operation in (lambda:replay_export(folder),lambda:reopen_export(folder,tmp_path/'bad')):
        with pytest.raises(AudioContractError,match='byte limit'):operation()
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('role',['root','portable'])
def test_preserved_source_references_keep_their_own_hard_container_limit(tmp_path,role):
    source=tmp_path/'small.wav';sf.write(source,np.zeros((100,1)),8000,subtype='DOUBLE')
    project=Project.create(source,tmp_path/'project');folder=tmp_path/'export';export_project(project,folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    relative=manifest['source']['path'] if role=='root' else 'project/'+project._manifest['source']['path']
    with (folder/relative).open('r+b') as stream:stream.truncate(AudioLimits().max_source_bytes+1)
    for operation in (lambda:replay_export(folder),lambda:reopen_export(folder,tmp_path/'bad')):
        with pytest.raises(AudioContractError,match='byte limit'):operation()
    assert not (tmp_path/'bad').exists()


def test_container_budget_cannot_be_reassigned_to_a_decoded_role(tmp_path):
    source=tmp_path/'small.wav';sf.write(source,np.zeros((100,1)),8000,subtype='DOUBLE')
    project=Project.create(source,tmp_path/'project');project.add_estimates({'fixture':np.full((100,1),.01)})
    folder=tmp_path/'export';export_project(project,folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    manifest['layers'][0]['stem']=manifest['source']['path']
    (folder/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(AudioContractError,match='roles must be distinct'):replay_export(folder)
