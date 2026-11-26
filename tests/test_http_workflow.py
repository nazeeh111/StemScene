"""Real loopback HTTP with deterministic supplied estimates, never model inference."""
from concurrent.futures import ThreadPoolExecutor
import http.client
import io
import json
from threading import Thread
import time
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf

from stemscene import Project
from stemscene.server import BrowserSession, make_server


def test_real_http_edit_undo_export_reopen_concurrency_and_failed_zip(tmp_path):
    session=BrowserSession(tmp_path/'workspace')
    source=tmp_path/'original.wav';t=np.arange(800)/8000
    values=(.2*np.sin(2*np.pi*220*t))[:,None]
    sf.write(source,values,8000,subtype='DOUBLE')
    identifier='a'*32
    project=Project.create(source,session.workspace/'projects'/identifier)
    layer=project.add_estimates({'synthetic supplied tone':values*.6})['synthetic supplied tone']
    server=make_server(session,0)
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    def request(method,path,payload=None,mime='application/json'):
        connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=10)
        body=json.dumps(payload).encode() if payload is not None and mime=='application/json' else payload
        host=f'127.0.0.1:{server.server_port}'
        headers={'Host':host,'X-StemScene-Token':session.token,'Origin':'http://'+host}
        if body is not None:headers['Content-Type']=mime
        connection.request(method,path,body,headers)
        response=connection.getresponse();content=response.read();status=response.status;kind=response.getheader('Content-Type','');connection.close()
        return status,json.loads(content) if 'json' in kind else content
    try:
        assert request('GET','/')[0]==200
        assert request('GET','/api/config')[1]['model_ready'] is False
        state=request('GET','/api/state?id='+identifier)[1]
        code,state=request('POST','/api/edit',{'id':identifier,'revision':state['revision'],'layer_id':layer,'start':.02,'end':.06,'gain':.2})
        assert code==200 and state['revision']==2
        # Selected previews are exact sample slices, with no timing leakage.
        for kind in ('original', 'mix'):
            full_code, full = request('GET', '/api/audio?id='+identifier+'&kind='+kind)
            clip_code, clip = request('GET', '/api/audio?id='+identifier+'&kind='+kind+'&start_sample=160&end_sample=480')
            assert full_code == clip_code == 200
            full_values, full_rate = sf.read(io.BytesIO(full), always_2d=True)
            clip_values, clip_rate = sf.read(io.BytesIO(clip), always_2d=True)
            assert full_rate == clip_rate == 8000
            assert np.array_equal(clip_values, full_values[160:480])
        for suffix in ('&start_sample=1', '&start_sample=0&end_sample=0',
                       '&start_sample=4&end_sample=900', '&start_sample=-1&end_sample=2',
                       '&start_sample=&end_sample=',
                       '&start_sample=0&start_sample=1&end_sample=2'):
            assert request('GET', '/api/audio?id='+identifier+'&kind=mix'+suffix)[0] == 400
        event=state['edits'][0]['id']
        for path in ('/api/undo','/api/export'):
            assert request('POST',path,{'id':identifier,'revision':1})[0]==400
        assert request('POST','/api/undo',{'id':identifier,'revision':2})[1]['revision']==3
        code,state=request('POST','/api/edit',{'id':identifier,'revision':3,'layer_id':layer,'start':.01,'end':.05,'gain':.5})
        assert code==200
        event=state['edits'][0]['id']
        code,export=request('POST','/api/export',{'id':identifier,'revision':4})
        assert code==200
        code,archive=request('GET','/api/download?id='+export['export_id']);assert code==200
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            assert 'project/project.json' in bundle.namelist()
        code,reopened=request('POST','/api/reopen',archive,'application/zip');assert code==200
        recipient=reopened['id'];assert recipient!=identifier
        assert reopened['layers'][0]['id']==layer and reopened['edits'][0]['id']==event
        assert Project.load(session.workspace/'projects'/recipient).history==Project.load(project.root).history
        assert np.array_equal(Project.load(session.workspace/'projects'/recipient).layers[0].samples,project.layers[0].samples)
        assert request('POST','/api/undo',{'id':recipient,'revision':4})[1]['revision']==5
        assert np.array_equal(Project.load(session.workspace/'projects'/recipient).render(),values)
        assert request('GET','/api/audio?id='+recipient+'&kind=mix')[0]==200
        import stemscene.server as module
        write=module.zipfile.ZipFile.write;count=0
        def fail(*args,**kwargs):
            nonlocal count
            write(*args,**kwargs);count+=1
            if count==2:raise OSError('injected ZIP failure')
        with patch.object(module.zipfile.ZipFile,'write',fail):
            assert request('POST','/api/export',{'id':identifier,'revision':4})[0]==400
        assert not list((session.workspace/'exports').glob('.*.tmp'))
        assert len(list((session.workspace/'exports').glob('*.zip')))==1
        assert request('GET','/api/download?id='+export['export_id'])[1]==archive
        assert request('POST','/api/export',{'id':identifier,'revision':4})[0]==200
        def competing():return request('POST','/api/edit',{'id':identifier,'revision':4,'layer_id':layer,'start':.02,'end':.03,'gain':.1})[0]
        with ThreadPoolExecutor(2) as workers:
            outcomes=list(workers.map(lambda _:competing(),range(2)))
        assert sorted(outcomes)==[200,400]
        # A model setup failure releases session admission; no weights are read.
        checkpoint=tmp_path/'dummy-checkpoint';checkpoint.write_bytes(b'not a model')
        refiner=tmp_path/'dummy-refiner';refiner.write_bytes(b'not a model')
        session.checkpoint=checkpoint;session.refiner_path=refiner;session.tokenizer_directory=tmp_path
        import builtins
        original_import=builtins.__import__
        def missing_model(name,*args,**kwargs):
            if name in ('engine','stemscene.engine'):
                raise ImportError('injected absent optional runtime')
            return original_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=missing_model):
            code,job=request('POST','/api/extract',{'id':identifier,'revision':5,'queries':['synthetic tone']});assert code==200
            for _ in range(100):
                status=request('GET','/api/job?id='+job['job_id'])[1]
                if status['status']!='running':break
                time.sleep(.01)
        assert status['status']=='error' and 'optional local model runtime' in status['error']
        assert request('POST','/api/undo',{'id':identifier,'revision':5})[0]==200
        before=set((session.workspace/'projects').iterdir())
        bad=io.BytesIO()
        with zipfile.ZipFile(bad,'w') as bundle:bundle.writestr('../escape','invalid')
        assert request('POST','/api/reopen',bad.getvalue(),'application/zip')[0]==400
        assert set((session.workspace/'projects').iterdir())==before
        assert request('POST','/api/reopen',b'PK\x03\x04broken','application/zip')[0]==400
        assert set((session.workspace/'projects').iterdir())==before
    finally:
        server.shutdown();thread.join(5);server.server_close()
