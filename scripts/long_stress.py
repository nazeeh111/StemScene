"""One fixed development stress case. Preparation, inference, scoring are separate processes."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, resource, shutil, sys, threading, time
import numpy as np
import soundfile as sf
import torch
from threadpoolctl import threadpool_limits
from stemscene.backend import AudioSepBackend, BackendLimits, InferenceCancelled, resample_aligned
from stemscene.joint import JointRefiner, RefinerAnchor
from stemscene.engine import SceneEngine
from stemscene.project import Project, Anchor
from stemscene.export import export_project, replay_export

# This research runner requires the installed STEMSCENE package and
# threadpoolctl==3.7.0. It never downloads models or recordings implicitly.
REPO=Path(__file__).resolve().parents[1]
ROOT: Path
SOURCE_CACHE: Path
CHECKPOINT: Path
TOKENIZER: Path
REFINER: Path
RATE=32000; DURATION=48; QUERIES=['clock tick','rain','sneezing']
FROZEN_V1_SHA256='ddbc872006da68440ad3f72435c3b89ecacf422f2f7265d68581f1dee879a9a9'
FROZEN_MANIFEST_SHA256='bd19a599b3165a0648e1ed9fef17e085e18fd4878f090e25e24945454c3d0728'
threadpool_limits(limits=1);torch.set_num_threads(4)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def state(stage,detail):
    (ROOT/'TASK_STATE.md').write_text('# Long stress V1\n\n'+stage+'\n\n'+detail+'\n\nAuthority: one development long stress case, selection sources only; no model/evaluator edits or held-out data access.\n')
def log(value):print(json.dumps(value,allow_nan=False),flush=True)
def peak():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)

def prepare():
    ROOT.mkdir(exist_ok=False)
    for d in ['input','evaluator-only','original-sources']:(ROOT/d).mkdir()
    manifest_path=REPO/'data/esc10-split.json'
    if sha(manifest_path)!=FROZEN_MANIFEST_SHA256:raise ValueError('This fixed case requires the original frozen data manifest')
    # Only selection entries are retained or used. No held-out audio is listed.
    manifest=json.loads(manifest_path.read_text())
    used={f for r in manifest['recipes'] if r['split']=='selection' for f in r['evaluator_only']['source_files']}
    selected=[]; sources={}
    for category in ['clock_tick','rain','sneezing']:
        entry=next(x for x in manifest['splits']['selection'] if x['category']==category and x['filename'] in used and ((SOURCE_CACHE/'audio')/x['filename']).is_file())
        path=(SOURCE_CACHE/'audio')/entry['filename']
        if sha(path)!=entry['sha256']:raise ValueError('source hash mismatch')
        shutil.copyfile(path,ROOT/'original-sources'/path.name)
        a,sr=sf.read(path,dtype='float64',always_2d=True)
        if a.shape!=(220500,1) or sr!=44100:raise ValueError('source contract')
        sources[category.replace('_',' ')]=resample_aligned(a,sr,RATE)[:,0]
        selected.append(entry)
    if sha(SOURCE_CACHE/'LICENSE')!=manifest['license_sha256']:raise ValueError('license hash mismatch')
    shutil.copyfile(SOURCE_CACHE/'LICENSE',ROOT/'original-sources/ESC10-LICENSE')
    truth=np.zeros((3,DURATION*RATE),dtype=np.float64);events=[]
    def place(q,segment,start,gain,kind):
        y=segment.copy();fade=min(round(.01*RATE),len(y)//2)
        y[:fade]*=np.linspace(0,1,fade);y[-fade:]*=np.linspace(1,0,fade)
        lo=round(start*RATE);hi=lo+len(y)
        if lo<0 or hi>truth.shape[1]:raise ValueError('event placement')
        truth[QUERIES.index(q),lo:hi]+=gain*y
        events.append({'query':q,'start_seconds':start,'end_seconds':hi/RATE,'gain_before_shared_peak_bound':gain,'kind':kind})
    rain=sources['rain']
    for lo,hi in [(0.,21.),(27.,45.5)]:
        for start in np.arange(lo,hi,4.9):
            a=rain[:min(len(rain),round((hi-start)*RATE))]
            place('rain',a,float(start),.40,'background repetition, 100ms overlaps')
    # Excerpt centers are determined for stimulus construction only, before any
    # inference. These clean-source locations never become inference anchors.
    for q,width,centers,gain in [
        ('clock tick',.30,[1.5,3.,6.,12.,18.,20.,28.5,36.,38.,43.5],.55),
        ('sneezing',1.0,[7.5,17.99,19.99,31.5,35.99,37.99,43.5],.55)]:
        a=sources[q];count=round(width*RATE);center=int(np.argmax(np.abs(a)))
        first=max(0,min(len(a)-count,center-count//2));excerpt=a[first:first+count]
        for center_time in centers:
            place(q,excerpt,center_time-width/2,gain,'transient near chunk join or paired interior control')
        next(x for x in selected if x['category'].replace('_',' ')==q)['excerpt_samples_32000']=[first,first+count]
    mix=truth.sum(axis=0);factor=min(1.,.8/max(float(np.max(np.abs(mix))),1e-12));truth*=factor;mix=truth.sum(axis=0)
    sf.write(ROOT/'input/mixture.wav',mix,RATE,subtype='DOUBLE')
    (ROOT/'input/request.json').write_text(json.dumps({'queries':QUERIES,'anchors':[]},indent=2)+'\n')
    np.savez_compressed(ROOT/'evaluator-only/references.npz',references=truth)
    for q,y in zip(QUERIES,truth):sf.write(ROOT/'evaluator-only'/f'{q.replace(" ","-")}.wav',y,RATE,subtype='DOUBLE')
    write('stimulus-provenance.json',{'version':'one-fixed-development-case-v1','duration_seconds':DURATION,'sample_rate':RATE,'channels':1,'queries':QUERIES,'sources':selected,'source_split':'selection only','source_manifest_sha256':sha(manifest_path),'shared_peak_bound':factor,'resampling':'scipy zero-phase polyphase FIR, Kaiser5, right trim/pad; 44.1k to32k once','events':events,'all_class_silence_seconds':[[21.5,26.5],[46.,48.]],'audiosep_starts_seconds':list(range(0,46,3)),'audiosep_overlap_ends_seconds':list(range(5,48,3)),'refiner_starts_seconds':[18.,36.],'refiner_overlap_ends_seconds':[20.,38.],'ground_truth_limit':'Controlled edited repetitions of genuine class-labeled recordings, not pure physical-source or natural multitrack truth.','anchors':'No source-derived anchors. Any later negative anchor must be derived from mixed-signal silence alone.'})
    state('Prepared; inference pending.','48-second mono stimulus with 3 classes, overlap, repeated genuine transients, backend/refiner joins, and mixed silence. Original bytes, hashes and attribution retained.')
    log({'prepared':str(ROOT),'mixture_sha256':sha(ROOT/'input/mixture.wav'),'reference_sum_exact':bool(np.array_equal(mix,truth.sum(axis=0)))})

def infer():
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable in this execution context; no CPU substitution')
    if (ROOT/'project').exists():raise FileExistsError('Preserve existing project; do not rerun over valid inference')
    started=time.perf_counter();telemetry={'mps_allocated_peak_observed':0,'mps_driver_peak_observed':0};stop=threading.Event()
    def monitor():
        while not stop.wait(.05):
            try:
                telemetry['mps_allocated_peak_observed']=max(telemetry['mps_allocated_peak_observed'],torch.mps.current_allocated_memory())
                telemetry['mps_driver_peak_observed']=max(telemetry['mps_driver_peak_observed'],torch.mps.driver_allocated_memory())
            except Exception:pass
    t=threading.Thread(target=monitor,daemon=True);t.start()
    loaded=time.perf_counter();backend=AudioSepBackend(CHECKPOINT,TOKENIZER,limits=BackendLimits(threads=4),device='mps');refiner=JointRefiner.load(REFINER);engine=SceneEngine(backend,refiner);load_seconds=time.perf_counter()-loaded
    request=json.loads((ROOT/'input/request.json').read_text());queries=request['queries']
    project=Project.create(ROOT/'input/mixture.wav',ROOT/'project')
    before=time.perf_counter();ids=engine.extract(project,queries,progress=lambda done,total:log({'proposal_queries_done':done,'total':total}));extract_seconds=time.perf_counter()-before
    samples=np.stack([next(l.samples for l in project.layers if l.query==q) for q in queries])
    proposals=np.stack([engine._proposal(project,q,None) for q in queries]);embeddings=backend.embed_queries(queries)
    np.savez_compressed(ROOT/'inference-only.npz',proposals=proposals,trained=samples,embeddings=embeddings)
    write('long-diagnostics.json',project.layers[0].metadata)
    project.edit(ids['sneezing'],17.,21.,.25,.02)
    before=time.perf_counter();export_project(project,ROOT/'export',subtype='DOUBLE',clipping='preserve');export_seconds=time.perf_counter()-before
    replay=replay_export(ROOT/'export');replay_error=float(np.max(np.abs(replay-project.render())))
    export_manifest=json.loads((ROOT/'export/manifest.json').read_text())
    wrote={'installed_module':sys.modules['stemscene'].__file__,'backend_device':backend.device,'threads':4,'checkpoint_sha256':sha(CHECKPOINT),'refiner_sha256':sha(REFINER),'source_sha256':sha(ROOT/'input/mixture.wav'),'model_load_seconds':load_seconds,'three_query_extract_seconds':extract_seconds,'extract_realtime_factor':extract_seconds/DURATION,'export_seconds':export_seconds,'export_independent_replay_max_error':replay_error,'export_verification':export_manifest['verification'],'primary_total_seconds':time.perf_counter()-started,'peak_process_rss_bytes':peak(),'memory_scope':'Cumulative process high-water RSS incl imported runtimes, both models, project arrays; MPS samples every50ms are observed lower bounds, not a true allocator high-water peak.','mps_sampled_memory':dict(telemetry),'source_hashes':{str(p.relative_to(REPO)):sha(p) for p in [REPO/'src/stemscene/engine.py',REPO/'src/stemscene/backend.py',REPO/'src/stemscene/joint.py',REPO/'src/stemscene/project.py']}}
    write('inference-report.json',wrote);state('Primary actual inference/export succeeded; operational smokes and scoring pending.',json.dumps(wrote,indent=2))
    log({'primary_inference_complete':True,**wrote})
    # All following operations see only the mixed recording and existing actual
    # proposals, never evaluator source references or the stimulus event labels.
    smokes={};base=Project.load(ROOT/'project');base.undo()
    def new_project(name):
        p=Project.create(ROOT/'input/mixture.wav',ROOT/name)
        shutil.copytree(ROOT/'project/proposal-cache',p.root/'proposal-cache')
        return p
    seq=new_project('sequential-request-project');engine.extract(seq,['rain','clock tick'])
    shutil.copytree(seq.root,ROOT/'locked-add-project');locked=Project.load(ROOT/'locked-add-project')
    rain_id=next(l.id for l in locked.layers if l.query=='rain');locked.set_locked(rain_id)
    rain_before=next(l.samples.copy() for l in locked.layers if l.query=='rain')
    engine.extract(seq,['sneezing']);engine.extract(locked,['sneezing'])
    smokes['simultaneous_vs_sequential_request_batches']={'maximum_sample_difference':max(float(np.max(np.abs(next(l.samples for l in seq.layers if l.query==q)-samples[i]))) for i,q in enumerate(queries)),'route':'first2classes then add third, unchanged original proposals, full canonical-set refinement; not sequential residual extraction'}
    smokes['locked_add']={'locked_rain_bytes_unchanged':bool(np.array_equal(rain_before,next(l.samples for l in locked.layers if l.query=='rain'))),'layer_count':len(locked.layers),'locked_refinement_versus_primary_rms':float(np.sqrt(np.mean((rain_before-samples[queries.index('rain')])**2)))}
    reverse=np.arange(len(queries))[::-1];x=base.original.samples
    reordered=refiner.refine_long(x,RATE,proposals[reverse],embeddings[reverse]).samples[np.argsort(reverse)]
    smokes['reverse_query_order_max_error']=float(np.max(np.abs(reordered-samples)))
    # Negative anchor grounded solely in zero mixed audio, with no source label.
    silence=x[round(22*RATE):round(25*RATE)]
    if np.max(np.abs(silence))!=0:raise ValueError('Mixed-only silence anchor not supported')
    anchored=new_project('mixed-silence-anchor-project')
    engine.extract(anchored,queries)
    rid=next(l.id for l in anchored.layers if l.query=='rain');anchored.set_anchors(rid,[Anchor(22.,25.,False)]);engine.extract(anchored,['rain'])
    anchored_rain=next(l.samples for l in anchored.layers if l.query=='rain')
    smokes['mixed_only_negative_anchor']={'mixed_interval_max_amplitude':float(np.max(np.abs(silence))),'anchor':[22.,25.,False],'estimated_interval_interior_max_amplitude':float(np.max(np.abs(anchored_rain[round(22.02*RATE):round(24.98*RATE)]))),'limit':'Only a deterministic silence-grounded absent anchor; no positive-source isolation or anchor-quality claim.'}
    revision=base.revision;manifest_hash=sha(base.root/'project.json');calls=0
    def cancelled():
        nonlocal calls
        calls+=1;return calls>=33
    try:engine.extract(base,queries,cancelled=cancelled);smokes['cancel']={'raised':False,'calls':calls}
    except InferenceCancelled as exc:smokes['cancel']={'raised':True,'message':str(exc),'calls':calls,'project_revision_unchanged':base.revision==revision,'manifest_bytes_unchanged':sha(base.root/'project.json')==manifest_hash}
    edit_id=base.edit(next(l.id for l in base.layers if l.query=='sneezing'),17.,21.,.25,.02);rendered=base.render();outside=np.ones(len(x),bool);outside[17*RATE:21*RATE]=False
    smokes['edit_locality']={'outside_edit_max_delta':float(np.max(np.abs(rendered[outside]-x[outside])))};base.undo();smokes['undo_original_max_error']=float(np.max(np.abs(base.render()-x)))
    smokes['source_bytes_preserved']=sha(base.root/'original.wav')==sha(ROOT/'input/mixture.wav')
    stop.set();t.join();write('operational-smokes.json',smokes);write('performance-final.json',{'elapsed_seconds_including_smokes':time.perf_counter()-started,'peak_process_rss_bytes':peak(),'mps_sampled_memory':telemetry});state('Actual primary inference, aligned export and operational smokes complete; evaluator scoring pending.',json.dumps(smokes,indent=2));log({'operational_smokes':smokes})

def evaluate():
    spec=importlib.util.spec_from_file_location('evaluation_read_only',REPO/'scripts/evaluate_refiner.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    x,sr=sf.read(ROOT/'input/mixture.wav',dtype='float64');a=np.load(ROOT/'inference-only.npz',allow_pickle=False)
    with np.load(ROOT/'evaluator-only/references.npz',allow_pickle=False) as f:truth=f['references']
    p=a['proposals'][:,:,0];trained=a['trained'][:,:,0];energy=float(np.mean(x*x));provenance=json.loads((ROOT/'stimulus-provenance.json').read_text())
    methods={'independent':p,'fixed_0.60_gain':p*.60,'trained_v1':trained};summary={}
    edit_cases=[{'name':'refiner_join_18','start_seconds':17.,'end_seconds':21.},
                {'name':'refiner_join_36','start_seconds':35.,'end_seconds':39.},
                {'name':'interior_transient','start_seconds':6.5,'end_seconds':8.5},
                {'name':'whole_recording','start_seconds':0.,'end_seconds':48.}]
    backend_boundaries=sorted(set(provenance['audiosep_starts_seconds'][1:]+provenance['audiosep_overlap_ends_seconds']))
    refiner_boundaries=[18.,20.,36.,38.]
    # Same-excerpt controls preserve the transient's waveform phase relative to
    # the measurement center. They remain observational because the background
    # and surrounding model context differ. Pairing uses references only, before
    # scoring any estimated output, not a search for favorable model errors.
    all_boundaries=sorted(set(backend_boundaries+refiner_boundaries))
    peaks=[]
    for event in provenance['events']:
        if event['query']=='rain':continue
        qi=QUERIES.index(event['query']);lo=round(event['start_seconds']*RATE);hi=round(event['end_seconds']*RATE)
        peaks.append({'query':event['query'],'peak_seconds':(lo+int(np.argmax(np.abs(truth[qi,lo:hi]))))/RATE})
    matched=[]
    for boundary in all_boundaries:
        nearby=[e for e in peaks if abs(e['peak_seconds']-boundary)<=.25]
        if not nearby:continue
        e=min(nearby,key=lambda z:abs(z['peak_seconds']-boundary))
        controls=[c for c in peaks if c['query']==e['query'] and all(abs(c['peak_seconds']-b)>.35 for b in all_boundaries)]
        if not controls:continue
        c=min(controls,key=lambda z:abs(z['peak_seconds']-boundary))
        matched.append({'boundary_seconds':boundary,'transient_query':e['query'],'actual_transient_peak_seconds':e['peak_seconds'],'matched_interior_center_seconds':c['peak_seconds']+boundary-e['peak_seconds'],'comparison':'same excerpt and gain, equal measurement width and transient-relative phase; different background/context'})
    def window_error(y,seconds,width=.05):
        result=[]
        for center in seconds:
            lo=max(0,round((center-width)*RATE));hi=min(len(x),round((center+width)*RATE))
            err=y[:,lo:hi]-truth[:,lo:hi];truth_d=np.diff(truth[:,lo:hi],axis=-1)
            result.append({'center_seconds':center,'error_nmse':float(np.mean(err*err)/energy),'derivative_error_over_global_mixture_derivative_power':float(np.mean(np.diff(err,axis=-1)**2)/max(np.mean(np.diff(x)**2),1e-12)),'reference_power_over_mixture':float(np.mean(truth[:,lo:hi]**2)/energy)})
        return result
    for name,y in methods.items():
        per_query=[];edit_errors=[]
        for qi,q in enumerate(QUERIES):
            edit={'start_seconds':17.,'end_seconds':21.,'gain':.25,'fade_seconds':.02};g=ev.edit_envelope(len(x),RATE,edit)-1
            per_query.append({'query':q,'direct_waveform_sdr_db':ev.sdr(truth[qi],y[qi]),'si_sdr_db':ev.si_sdr(truth[qi],y[qi]),'gain_projection':float(np.dot(y[qi],truth[qi])/np.dot(truth[qi],truth[qi])),'edit_nmse_17_21':float(np.mean((g*(y[qi]-truth[qi]))**2)/energy)})
            edit_errors.append(per_query[-1]['edit_nmse_17_21'])
        activity=ev.power_envelopes(y,RATE);true_activity=ev.power_envelopes(truth,RATE)
        windows=[]
        for lo in range(0,len(x),2*RATE):
            hi=min(len(x),lo+2*RATE);r=truth[:,lo:hi];pred=y[:,lo:hi];active=np.mean(r*r,axis=-1)>.01*energy
            if not active.any():continue
            gram=r[active]@r[active].T;condition=float(np.linalg.cond(gram))
            if condition>1e6:windows.append({'start_seconds':lo/RATE,'ill_conditioned':condition});continue
            coefficients=(pred@r[active].T)@np.linalg.pinv(gram)
            active_ids=np.flatnonzero(active)
            windows.append({'start_seconds':lo/RATE,'active_reference_queries':[QUERIES[i] for i in active_ids],'condition_number':condition,'attribution_coefficients_estimate_rows_reference_columns':coefficients.tolist(),'dominant_estimate_query_for_each_reference':[QUERIES[int(np.argmax(np.abs(coefficients[:,j])))] for j in range(len(active_ids))],'class_attribution_mismatches':sum(int(np.argmax(np.abs(coefficients[:,j])))!=int(qi) for j,qi in enumerate(active_ids))})
        silence_metrics=[]
        for lo,hi in provenance['all_class_silence_seconds']:
            seg=y[:,round(lo*RATE):round(hi*RATE)];silence_metrics.append({'seconds':[lo,hi],'per_query_energy_over_global_mixture':np.mean(seg*seg,axis=1).tolist(),'maximum_amplitude':float(np.max(np.abs(seg)))})
        interiors=[1.5+3*i for i in range(15)]
        matched_errors=[]
        for pair in matched:
            b=window_error(y,[pair['boundary_seconds']])[0];c=window_error(y,[pair['matched_interior_center_seconds']])[0]
            matched_errors.append({**pair,'boundary_metrics':b,'interior_metrics':c,'boundary_minus_interior_nmse':b['error_nmse']-c['error_nmse']})
        summary[name]={'per_query':per_query,'mean_all_query_edit_nmse_17_21':float(np.mean(edit_errors)),'waveform_nmse':float(np.mean((y-truth)**2)/energy),'activity_power_nmse':float(np.mean((activity-true_activity)**2)/energy**2),'activity_change_nmse':float(np.mean((np.diff(activity,axis=-1)-np.diff(true_activity,axis=-1))**2)/energy**2),'backend_boundary_windows':window_error(y,backend_boundaries),'refiner_boundary_windows':window_error(y,refiner_boundaries),'matched_interior_windows':window_error(y,interiors),'paired_transient_boundary_controls':matched_errors,'silence':silence_metrics,'class_attribution_windows':windows,'class_attribution_mismatches':sum(v.get('class_attribution_mismatches',0) for v in windows),'unity_resynthesis_max_error':float(np.max(np.abs((x-y.sum(axis=0))+y.sum(axis=0)-x)))}
        detailed_edits=[]
        for case in edit_cases:
            g=ev.edit_envelope(len(x),RATE,{**case,'gain':.25,'fade_seconds':.02})-1
            selected=g!=0;err=g[None]*(y-truth)
            detailed_edits.append({**case,'gain':.25,'fade_seconds':.02,
                'separate_all_query_mean_edit_nmse_full_record':float(np.mean(err**2)/energy),
                'separate_all_query_mean_edit_nmse_edited_samples_only':float(np.mean(err[:,selected]**2)/energy),
                'per_query_full_record_edit_nmse':np.mean(err**2,axis=1).tolist(),
                'simultaneous_all_query_edit_nmse_full_record':float(np.mean(np.sum(err,axis=0)**2)/energy)})
            detailed_edits[-1]['per_query_full_record_edit_nmse']=(np.mean(err**2,axis=1)/energy).tolist()
        summary[name]['edit_cases']=detailed_edits
        for key in ['backend_boundary_windows','refiner_boundary_windows','matched_interior_windows']:summary[name][key+'_mean_nmse']=float(np.mean([v['error_nmse'] for v in summary[name][key]]))
    write('source-reference-report.json',{'scope':'Single fixed48-second selection-source development stress case, not confirmation; no tuning','sample_rate':RATE,'frames':len(x),'channels':1,'methods':summary,'limits':['Class-labeled recordings can include background sound; not pure physical-source truth.','Repetition of three source files, no independent-scene confidence interval.','Boundary windows have deliberately different event content; compare per-boundary matched transient examples, do not infer causal join artifacts from pooled ratios.','Class attribution via evaluator-only least-squares regression is diagnostic; it does not establish physical identity.','No natural multitrack or subjective listening-quality claim.','Source-reference report is computed only after actual inference/export completed.']})
    state('Complete. Actual long inference, export, operational smokes and source-reference scoring retained.','See inference-report.json, operational-smokes.json, long-diagnostics.json, source-reference-report.json, and report.md. One fixed development case; no confirmation inputs or model tuning.')
    log({'source_reference_scoring_complete':True,'mean_edits':{n:v['mean_all_query_edit_nmse_17_21'] for n,v in summary.items()}})

def extras():
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no CPU substitution')
    load_start=time.perf_counter();backend=AudioSepBackend(CHECKPOINT,TOKENIZER,limits=BackendLimits(threads=4),device='mps');refiner=JointRefiner.load(REFINER);engine=SceneEngine(backend,refiner);load_seconds=time.perf_counter()-load_start
    x=np.zeros((5*RATE,1),dtype=np.float64);path=ROOT/'input/silence-five-seconds.wav';sf.write(path,x,RATE,subtype='DOUBLE')
    original_backend_separate=backend.separate;calls=[]
    def observed(samples,rate,queries,**kwargs):
        start=time.perf_counter();result=original_backend_separate(samples,rate,queries,**kwargs);calls.append({'queries':list(queries),'frames':len(samples),'seconds':time.perf_counter()-start});return result
    backend.separate=observed
    warmup=Project.create(path,ROOT/'silence-duplicate-warmup-project')
    requested=['rain',' RAIN ','rain','clock tick','sneezing'];ids=engine.extract(warmup,requested)
    duplicate={'requested':requested,'actual_layers':[l.query for l in warmup.layers],'rain_ids_equal':ids['rain']==ids[' RAIN '],'distinct_layer_count':len(warmup.layers),'backend_calls':list(calls)}
    before_calls=len(calls);project=Project.create(path,ROOT/'silence-warm-three-query-project');started=time.perf_counter();engine.extract(project,QUERIES);elapsed=time.perf_counter()-started
    proposals=np.stack([engine._proposal(project,q,None) for q in QUERIES]);samples=np.stack([next(l.samples for l in project.layers if l.query==q) for q in QUERIES]);residual=project.residual();original_hash=sha(project.root/'original.wav')
    silence={'warm_three_query_seconds':elapsed,'actual_backend_calls':calls[before_calls:],'source_max_amplitude':float(np.max(np.abs(x))),'proposal_max_amplitude':float(np.max(np.abs(proposals))),'proposal_per_query_mean_energy':np.mean(proposals**2,axis=(1,2)).tolist(),'refined_max_amplitude':float(np.max(np.abs(samples))),'refined_per_query_mean_energy':np.mean(samples**2,axis=(1,2)).tolist(),'residual_max_amplitude':float(np.max(np.abs(residual))),'residual_mean_energy':float(np.mean(residual**2)),'render_original_max_delta':float(np.max(np.abs(project.render()-x))),'no_edit_records':not project.edits,'original_bytes_unchanged':original_hash==sha(path),'refiner_sha256':sha(REFINER)}
    prompts=['rain','rainfall','sound of rain','clock tick','clock ticking','sneezing','sneeze','a person sneezing']
    embeddings=backend.embed_queries(prompts);similarity=embeddings@embeddings.T
    pairs=[{'queries':[prompts[i],prompts[j]],'cosine':float(similarity[i,j]),'above_uncertainty_threshold':bool(similarity[i,j]>=refiner.config.ambiguity_cosine)} for i in range(len(prompts)) for j in range(i+1,len(prompts))]
    report={'scope':'Actual 5s all-zero AudioSep MPS + frozen V1, operational numerical smoke; not absence-detection accuracy','new_process_model_load_seconds':load_seconds,'duplicate_grouping_warmup':duplicate,'warm_three_query_silence':silence,'embedding_overlap':{'prompts':prompts,'threshold':refiner.config.ambiguity_cosine,'pairs':pairs,'exposed_uncertainty_pairs':[v for v in pairs if v['above_uncertainty_threshold']],'limit':'Embedding overlap is prompt similarity, not correct separation, physical-source identity, or a calibrated semantic detector. No threshold was changed.'},'peak_process_rss_bytes':peak(),'mps_memory_current_bytes':torch.mps.current_allocated_memory(),'mps_driver_current_bytes':torch.mps.driver_allocated_memory()}
    write('silence-duplicate-smokes.json',report);log(report)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','infer','evaluate','extras'])
    parser.add_argument('--workspace',type=Path,required=True,help='Fresh output directory for prepare; same directory for later stages')
    parser.add_argument('--source-cache',type=Path,help='Pinned ESC-10 cache root containing audio/ and LICENSE; prepare only')
    parser.add_argument('--models',type=Path,help='Model root containing audiosep_base_4M_steps.ckpt and roberta/')
    parser.add_argument('--checkpoint',type=Path,help='Explicit AudioSep checkpoint, overriding --models')
    parser.add_argument('--tokenizer',type=Path,help='Explicit local tokenizer directory, overriding --models')
    parser.add_argument('--refiner',type=Path,help='Explicit frozen V1 selected.pt; required for infer/extras')
    args=parser.parse_args()
    ROOT=args.workspace.expanduser().resolve()
    SOURCE_CACHE=args.source_cache.expanduser().resolve() if args.source_cache else None
    if args.stage=='prepare' and SOURCE_CACHE is None:parser.error('prepare requires --source-cache')
    model_root=args.models.expanduser().resolve() if args.models else None
    CHECKPOINT=args.checkpoint.expanduser().resolve() if args.checkpoint else (model_root/'audiosep_base_4M_steps.ckpt' if model_root else None)
    TOKENIZER=args.tokenizer.expanduser().resolve() if args.tokenizer else (model_root/'roberta' if model_root else None)
    REFINER=args.refiner.expanduser().resolve() if args.refiner else None
    if args.stage in {'infer','extras'} and any(v is None for v in [CHECKPOINT,TOKENIZER,REFINER]):
        parser.error('infer/extras require --refiner and either --models or both --checkpoint and --tokenizer')
    if args.stage in {'infer','extras'} and sha(REFINER)!=FROZEN_V1_SHA256:
        parser.error('This fixed case requires the frozen V1 refiner SHA-256 recorded in evaluation/long-v1/hashes.json')
    if args.stage=='prepare':ROOT.parent.mkdir(parents=True,exist_ok=True)
    globals()[args.stage]()
