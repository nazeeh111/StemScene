#!/usr/bin/env python3
"""Render an honest captured-session A/B excerpt. No inference is simulated."""
from pathlib import Path
import argparse, hashlib, json, re, shutil, subprocess
import numpy as np
import soundfile as sf

DURATION=23
SOURCE_SHA='7d85ef8b3c89bbd042d1bb340d0e029c50f3d31abaff7f15c7402ebc2dbc0f50'
CREDIT='Audio: RealDESED test recording 001738.wav, CC0. Event annotations and dataset metadata: Florian Schmid et al., RealDESED (2026), DOI 10.5281/zenodo.20056072, CC BY 4.0.'

def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--export',type=Path,required=True);p.add_argument('--screenshot',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--font',default='Arial')
 a=p.parse_args()
 if not re.fullmatch(r'[A-Za-z0-9 _-]{1,80}',a.font):raise ValueError('Use a simple installed font family name')
 ffmpeg=shutil.which('ffmpeg')
 if not ffmpeg:raise RuntimeError('FFmpeg is required')
 source=a.export.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 doc=json.loads((source/'manifest.json').read_text())
 for name,digest in doc['files'].items():
  path=(source/name).resolve()
  if not path.is_relative_to(source) or sha(path)!=digest:raise ValueError('Export integrity check failed')
 if doc['source']['sha256']!=SOURCE_SHA:raise ValueError('This composition expects the documented CC0 recording')
 x,sr=sf.read(source/'original-decoded.wav',dtype='float64',always_2d=True)
 y,mix_sr=sf.read(source/'mix.wav',dtype='float64',always_2d=True)
 assert x.shape==y.shape==(750528,1) and sr==mix_sr==48000
 residual,_=sf.read(source/'residual.wav',dtype='float64',always_2d=True)
 unity=residual.copy();replay=x.copy()
 for layer in doc['layers']:
  stem,ssr=sf.read(source/layer['stem'],dtype='float64',always_2d=True)
  gain,gsr=sf.read(source/layer['envelope'],dtype='float64',always_2d=True)
  assert ssr==gsr==sr and stem.shape==gain.shape==x.shape
  unity+=stem;replay+=(gain-1)*stem
 unity_error=float(np.max(np.abs(unity-x)));replay_error=float(np.max(np.abs(replay-y)))
 assert unity_error<1e-12 and replay_error<1e-12
 outside=np.ones(len(x),bool);outside[5*sr:7*sr]=False
 assert np.array_equal(x[outside],y[outside])
 shutil.copyfile(a.screenshot,out/'ui-session.jpg')
 shutil.copyfile(source/doc['source']['path'],out/'source-001738.wav')
 n=8*sr
 sf.write(out/'original-0-8s.wav',x[:n],sr,subtype='FLOAT')
 sf.write(out/'edited-0-8s.wav',y[:n],sr,subtype='FLOAT')
 timeline=np.concatenate([np.zeros((3*sr,1)),x[:n],y[:n],np.zeros((4*sr,1))]).astype(np.float32)
 sf.write(out/'comparison-timeline.wav',timeline,sr,subtype='FLOAT')
 titles=out/'titles';titles.mkdir()
 filters=["crop=1280:560:0:76", "scale=1600:700:flags=lanczos:in_range=full:out_range=tv:out_color_matrix=bt709", "pad=1920:1080:160:180:color=0x151717",'setsar=1']
 text_sources=[]
 def text(name,words,xpos,ypos,size,color='f0f0ed',enable='lt(t,19)'):
  filename=f'titles/{name}.txt';(out/filename).write_text(words)
  filters.append(f"drawtext=font='{a.font}':textfile='{filename}':x={xpos}:y={ypos}:fontsize={size}:fontcolor=0x{color}:enable='{enable}'")
  text_sources.append({'name':name,'text':words,'x':xpos,'y':ypos,'size':size,'enable':enable})
 text('brand','STEMSCENE',160,42,32,'e7d38e')
 text('intro','Edit an estimated event. Keep the original.',160,101,49,enable='lt(t,3)')
 text('original','Original recording | source 0-8 s',160,101,49,enable='gte(t,3)*lt(t,11)')
 text('edited','Edited mix | same source 0-8 s',160,101,49,enable='gte(t,11)*lt(t,19)')
 text('intro-note','Actual saved session. Source and output keep their gain.',160,937,30,enable='lt(t,3)')
 text('original-note','Natural domestic scene. Three estimated sound layers.',160,937,30,enable='gte(t,3)*lt(t,11)')
 text('edit-note','Doorbell estimate x0.50, 5-7 s. 20 ms edge fades.',160,937,30,enable='gte(t,11)*lt(t,19)')
 text('capture-note','Session capture with audio excerpts. Inference and waiting omitted. No normalization.',160,1015,23,'b3b8b3')
 filters.append("drawbox=x=0:y=0:w=iw:h=ih:color=0x151717:t=fill:enable='gte(t,19)'")
 text('end-brand','STEMSCENE',160,140,32,'e7d38e','gte(t,19)')
 text('end-title','Export an aligned, reproducible mix.',160,245,58,enable='gte(t,19)')
 text('end-files','3 estimated stems + residual + gain envelopes + edit manifest',160,370,34,enable='gte(t,19)')
 text('end-replay',f'Independent edited replay: {replay_error:.1g} max amplitude error',160,450,30,enable='gte(t,19)')
 text('end-unity',f'Unity reconstruction: {unity_error:.3g} max amplitude error',160,507,30,enable='gte(t,19)')
 text('end-locality','Outside 5-7 s: decoded samples remain identical.',160,590,30,enable='gte(t,19)')
 text('credit-a','Audio: RealDESED 001738.wav, CC0.',160,830,24,'b3b8b3','gte(t,19)')
 text('credit-b','Annotations/metadata: Florian Schmid et al. | DOI 10.5281/zenodo.20056072 | CC BY 4.0',160,877,23,'b3b8b3','gte(t,19)')
 text('end-limit','Estimated layers are not clean ground truth. No subjective-quality rating.',160,988,24,'b3b8b3','gte(t,19)')
 (out/'composition.json').write_text(json.dumps({'duration_seconds':23,'fps':30,'size':[1920,1080],'capture_crop':[0,76,1280,560],'shots':[{'start':0,'end':3,'audio':'silence','label':'actual session capture'},{'start':3,'end':11,'audio':'original source0-8s'},{'start':11,'end':19,'audio':'edited same source0-8s'},{'start':19,'end':23,'audio':'silence','label':'executed export checks'}],'text':text_sources,'font_family':a.font},indent=2)+'\n')
 filter_text=','.join(filters)
 (out/'video-filter.txt').write_text(filter_text+'\n')
 commands=[ [ffmpeg,'-hide_banner','-nostdin','-n','-threads','4','-filter_threads','2','-loop','1','-framerate','30','-i','ui-session.jpg','-i','comparison-timeline.wav','-vf',filter_text,'-t','23','-r','30','-c:v','libx264','-threads','4','-preset','medium','-crf','20','-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-c:a','pcm_f32le','-ar','48000','-ac','1','master.mov'],
 [ffmpeg,'-hide_banner','-nostdin','-n','-i','master.mov','-map','0','-c:v','copy','-c:a','aac','-b:a','320k','-ar','48000','-ac','1','-t','23','-movflags','+faststart','stemscene-walkthrough.mp4'] ]
 for index,command in enumerate(commands):
  with (out/f'render-{index+1}.log').open('w') as log:subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
 vtt='''WEBVTT

00:00.000 --> 00:03.000
Actual saved session. Inference and waiting are omitted.

00:03.000 --> 00:11.000
[Original recording, source 0-8 seconds: doorbell, door and keys.]

00:11.000 --> 00:19.000
[Same scene with the estimated doorbell layer at 50% from source 5-7 seconds.]

00:19.000 --> 00:23.000
Aligned export independently replayed. Samples outside the edit remain identical.
'''
 (out/'captions.vtt').write_text(vtt)
 (out/'CREDITS.txt').write_text(CREDIT+'\nNo clean multitrack references or subjective-quality rating. Session excerpts, not a live inference recording.\n')
 report={'source_recording':'001738.wav','dataset':'RealDESED','dataset_url':'https://zenodo.org/records/20056072','source_sha256':SOURCE_SHA,'audio_license':'CC0','annotation_metadata_license':'CC-BY-4.0','credit':CREDIT,'source_frames':len(x),'source_sample_rate':sr,'source_channels':1,'estimated_queries':[l['query'] for l in doc['layers']],'backend_checkpoint_sha256':doc['layers'][0]['metadata']['checkpoint_sha256'],'refinement_method':doc['layers'][0]['metadata']['refinement']['method'],'session_export_revision':doc['revision'],'screenshot_revision':9,'snapshot_revision':11,'session_capture_note':'Screenshot revision 9; its recorded event and preserved layers render identically to revision 11. Processing/waiting omitted.','edit':{'start_seconds':5,'end_seconds':7,'gain':.5,'fade_seconds':.02},'verification':{'unity_max_abs_error':unity_error,'edited_replay_max_abs_error':replay_error,'outside_edit_sample_exact':True},'delivery_audio':'Original gain; no normalization/music. WAV excerpts are FLOAT32; master audio is PCM float32; MP4 AAC at 320 kbps is lossy.','limits':['Estimated layer modification, not proven clean-source attenuation','Natural recording has no clean-reference stems','No subjective listening-panel judgment'],'files':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ['.wav','.jpg','.mp4','.vtt']}}
 (out/'demo-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'output':str(out),'source_frames':len(x),'unity_error':unity_error,'replay_error':replay_error,'mp4_sha256':sha(out/'stemscene-walkthrough.mp4')}),flush=True)
if __name__=='__main__':main()
