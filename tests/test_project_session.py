"""Synthetic persistence, transport and ownership contracts; no separator model."""
from copy import deepcopy
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
from threading import Event, Thread
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from stemscene import Project, ProjectSession, export_project, replay_export, reopen_export
from stemscene.audio import AudioContractError
from stemscene.storage import project_lock, publish_directory


@pytest.fixture
def scene(tmp_path):
    t = np.arange(800) /8000
    samples = np.column_stack([.15*np.sin(2*np.pi*220*t),.1*np.sin(2*np.pi*330*t)])
    source = tmp_path/'source.wav'
    sf.write(source,samples,8000,subtype='DOUBLE')
    project = Project.create(source,tmp_path/'project')
    layer = project.add_estimates({'fixture tone':samples*.6})['fixture tone']
    return project,layer,samples


def test_live_writer_exclusion_and_sigkill_recovery(scene):
    project, layer, _ = scene
    script = """import sys,time
import stemscene.project as module
from stemscene import Project
p=Project.load(sys.argv[1])
def pause(*args):
 print('OWNED',flush=True)
 time.sleep(60)
module._write_json=pause
p.edit(sys.argv[2],.01,.04,.3)
"""
    child = subprocess.Popen([sys.executable,'-c',script,str(project.root),layer],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={**os.environ,'PYTHONPATH':str(Path(__file__).parents[1]/'src')})
    selector = selectors.DefaultSelector()
    selector.register(child.stdout,selectors.EVENT_READ)
    try:
        assert selector.select(10)
        assert child.stdout.readline().strip()=='OWNED'
        inode = (project.root/'.write.lock').stat().st_ino
        with pytest.raises(AudioContractError,match='being written'):
            project.edit(layer,.01,.04,.3)
        child.kill(); child.wait(timeout=5)
        assert Project.load(project.root).revision==1
        project.edit(layer,.01,.04,.3)
        assert project.revision==2
        assert (project.root/'.write.lock').stat().st_ino==inode
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)
        selector.close()


def test_foreign_legacy_or_live_pid_never_deletes_lock(scene):
    project, layer, _ = scene
    lock=project.root/'.write.lock'
    for content in (b'', b'foreign-marker', b'999999999999999999999999999', b' 123\n',str(os.getpid()).encode()):
        lock.write_bytes(content)
        inode=lock.stat().st_ino
        with pytest.raises(AudioContractError,match='legacy|Legacy'):
            project.edit(layer,.01,.04,.5)
        assert lock.read_bytes()==content and lock.stat().st_ino==inode


def test_stale_memory_edit_undo_preview_export_refuse(scene,tmp_path):
    project, layer, _ = scene
    session=ProjectSession(project.root)
    project.edit(layer,.01,.04,.3)
    for operation in (lambda:session.change('edit',layer,.02,.05,.5),lambda:session.change('undo'),lambda:session.render(),lambda:session.export(tmp_path/'stale-export')):
        with pytest.raises(AudioContractError,match='changed'):
            operation()
    session.reload()
    with pytest.raises(AudioContractError,match='stale'):
        session.change('undo',expected_revision=1)
    assert session.change('undo',expected_revision=2)
    assert session.revision==3


def test_exact_double_and_float_export_reopen_preserve_history_and_continue(scene,tmp_path):
    project, layer, original = scene
    session=ProjectSession(project.root)
    event=session.change('edit',layer,.02,.06,.2,.001)
    edited=session.render()
    assert np.array_equal(edited[:160],original[:160]) and np.array_equal(edited[480:],original[480:])
    for subtype in ('DOUBLE','FLOAT'):
        folder=tmp_path/subtype
        session.export(folder,subtype=subtype)
        assert np.max(np.abs(replay_export(folder)-edited)) <=(1e-12 if subtype=='DOUBLE' else 1e-6)
        reopened=reopen_export(folder,tmp_path/(subtype+'-project'))
        assert reopened.layers[0].id==layer and reopened.edits[0]['id']==event
        assert np.array_equal(reopened.layers[0].samples,Project.load(project.root).layers[0].samples)
        assert reopened.history==Project.load(project.root).history
        assert np.array_equal(reopened.render(),edited)
        assert reopened.original.source_sha256==project.original.source_sha256
        with pytest.raises(FileExistsError):reopen_export(folder,reopened.root)
        recipient=ProjectSession(reopened.root)
        assert recipient.change('undo')
        assert np.array_equal(recipient.render(),original)
        new=recipient.change('edit',layer,.01,.04,.5)
        assert new!=event
        recipient.export(tmp_path/(subtype+'-continued'))


def test_failed_manifest_asset_and_export_preserve_last_valid_state(scene,tmp_path,monkeypatch):
    project, layer, samples = scene
    before=(project.root/'project.json').read_bytes()
    import stemscene.project as project_module
    with patch.object(project_module.os,'replace',side_effect=OSError('injected manifest failure')):
        with pytest.raises(OSError):project.edit(layer,.01,.04,.3)
    assert (project.root/'project.json').read_bytes()==before
    assert not list(project.root.glob('.project.json.*.tmp'))
    original_save=project_module.np.save
    def partial(stream,*args,**kwargs):
        stream.write(b'partial array')
        raise OSError('injected partial asset')
    with patch.object(project_module.np,'save',partial):
        with pytest.raises(OSError):project.add_estimates({'new':samples*.2})
    assert (project.root/'project.json').read_bytes()==before
    assert len(Project.load(project.root).layers)==1
    project.add_estimates({'new':samples*.2})
    import stemscene.export as export_module
    write=export_module.sf.write
    count=0
    def fail_wav(*args,**kwargs):
        nonlocal count
        write(*args,**kwargs);count+=1
        if count==2:raise OSError('injected partial export')
    target=tmp_path/'failed-export'
    with patch.object(export_module.sf,'write',fail_wav):
        with pytest.raises(OSError):export_project(project,target)
    assert not target.exists() and not list(tmp_path.glob('.failed-export.*'))
    export_project(project,target)


def test_empty_concurrent_destination_not_replaced(scene,tmp_path):
    project,_,_=scene
    import stemscene.export as module
    verify=module._verify_written_audio
    target=tmp_path/'collision'
    def collide(*args):
        result=verify(*args);target.mkdir();return result
    with patch.object(module,'_verify_written_audio',collide):
        with pytest.raises(FileExistsError):export_project(project,target)
    assert target.is_dir() and list(target.iterdir())==[]
    staged=tmp_path/'staged';staged.mkdir();(staged/'valid').write_text('complete')
    with pytest.raises(FileExistsError):publish_directory(staged,target)
    assert (staged/'valid').read_text()=='complete'


def test_cancellation_and_stale_proposals_cannot_commit(scene):
    project,layer,_=scene
    session=ProjectSession(project.root)
    class Proposal:
        def commit(self,loaded):
            loaded.edit(layer,.01,.04,.1)
    class Engine:
        def propose(self,*args,**kwargs):return Proposal()
    token=session.begin_extraction()
    session.cancel(token)
    with pytest.raises(RuntimeError,match='Cancelled'):
        session.compute(token,Engine(),['fixture'])
    assert session.revision==1
    token=session.begin_extraction()
    project.edit(layer,.01,.04,.2)
    with pytest.raises(AudioContractError,match='changed'):
        session.compute(token,Engine(),['fixture'])
    assert Project.load(project.root).revision==2
    session.reload()
    token=session.begin_extraction()
    session.compute(token,Engine(),['fixture'])
    assert session.revision==3


def test_reopen_rejects_tampered_payload_or_playback_only_export(scene,tmp_path):
    project,_,_=scene
    folder=tmp_path/'export';export_project(project,folder)
    document=json.loads((folder/'manifest.json').read_text())
    portable=folder/'project/project.json'
    original=portable.read_bytes();portable.write_bytes(original+b' ')
    with pytest.raises(AudioContractError,match='checksum|bound'):
        reopen_export(folder,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    portable.write_bytes(original)
    document.pop('editable_project')
    (folder/'manifest.json').write_text(json.dumps(document))
    with pytest.raises(AudioContractError,match='playback-only'):
        reopen_export(folder,tmp_path/'legacy')


def test_killed_initialized_lock_publication_is_recoverable(scene):
    project,layer,_=scene
    lock=project.root/'.write.lock'
    # Identified fixture lock only; this test starts from a never-opened project.
    folder=project.root.parent/'fresh'
    fresh=Project.create(project.root/'original.wav',folder)
    script="""import sys,time,os
import stemscene.storage as storage
from stemscene import Project
link=os.link
def pause(source,target):
 link(source,target)
 print('PUBLISHED',flush=True)
 time.sleep(60)
storage.os.link=pause
Project.load(sys.argv[1]).save()
"""
    child=subprocess.Popen([sys.executable,'-c',script,str(folder)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={**os.environ,'PYTHONPATH':str(Path(__file__).parents[1]/'src')})
    selector=selectors.DefaultSelector();selector.register(child.stdout,selectors.EVENT_READ)
    try:
        assert selector.select(10)
        assert child.stdout.readline().strip()=='PUBLISHED'
        child.kill();child.wait(timeout=5)
        assert (folder/'.write.lock').stat().st_nlink==2
        Project.load(folder).save()
        assert Project.load(folder).revision==0
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)
        selector.close()


def test_shared_snapshot_readers_coexist_and_exclude_writers(scene):
    project,_,_=scene
    with project_lock(project.root,shared=True):
        with project_lock(project.root,shared=True):
            with pytest.raises(AudioContractError,match="being written"):
                with project_lock(project.root):
                    raise AssertionError("writer entered a held read snapshot")


def test_portable_identity_does_not_conflate_boolean_and_numeric_metadata(scene,tmp_path):
    project,layer,_=scene
    project.edit(layer,.01,.04,1.0)
    folder=tmp_path/'identity';export_project(project,folder)
    manifest=folder/'manifest.json';original=json.loads(manifest.read_text())
    for key in ('event','version','revision'):
        changed=deepcopy(original)
        if key=='event':changed['edit_history'][0]['gain']=True
        elif key=='version':changed['editable_project']['version']=True
        else:changed['revision']=True
        manifest.write_text(json.dumps(changed))
        with pytest.raises(AudioContractError):reopen_export(folder,tmp_path/key)
        assert not (tmp_path/key).exists()
