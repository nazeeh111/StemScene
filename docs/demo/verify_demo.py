#!/usr/bin/env python3
"""Technical decode/timing/PCM checks; not subjective listening approval."""
from pathlib import Path
from fractions import Fraction
import argparse, hashlib, json, shutil, subprocess
import numpy as np
import soundfile as sf
from scipy.signal import correlate

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('render',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 r=a.render.resolve();o=a.output.resolve();o.mkdir(parents=True,exist_ok=False)
 ffmpeg,probe=shutil.which('ffmpeg'),shutil.which('ffprobe')
 if not ffmpeg or not probe:raise RuntimeError('FFmpeg/ffprobe required')
 def run(command):return subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=180)
 media={}
 for name in ['master.mov','stemscene-walkthrough.mp4']:
  path=r/name
  info=json.loads(run([probe,'-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)]).stdout)
  v=next(s for s in info['streams'] if s['codec_type']=='video');audio=next(s for s in info['streams'] if s['codec_type']=='audio')
  assert (v['width'],v['height'])==(1920,1080) and Fraction(v['avg_frame_rate'])==30 and int(v['nb_read_frames'])==690
  assert abs(float(v['duration'])-23)<1e-6 and int(audio['sample_rate'])==48000 and audio['channels']==1
  decode=run([ffmpeg,'-hide_banner','-nostdin','-v','error','-threads','4','-i',str(path),'-f','null','-'])
  assert not decode.stderr,decode.stderr.decode()
  media[name]={'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'bytes':path.stat().st_size,'video_frames':int(v['nb_read_frames']),'video_duration':float(v['duration']),'audio_duration':float(audio['duration']),'video_codec':v['codec_name'],'audio_codec':audio['codec_name'],'full_decode_errors':0,'width':v['width'],'height':v['height'],'fps':v['avg_frame_rate'],'audio_sample_rate':48000,'audio_channels':1}
 expected,sr=sf.read(r/'comparison-timeline.wav',dtype='float32',always_2d=False)
 assert len(expected)==23*48000
 def decoded_audio(path):return np.frombuffer(run([ffmpeg,'-hide_banner','-nostdin','-v','error','-i',str(path),'-vn','-ac','1','-ar','48000','-c:a','pcm_f32le','-f','f32le','pipe:1']).stdout,dtype='<f4').copy()
 master=decoded_audio(r/'master.mov');assert len(master)==len(expected) and np.array_equal(master,expected)
 delivery=decoded_audio(r/'stemscene-walkthrough.mp4');assert len(delivery)>=len(expected)
 presented=delivery[:len(expected)]
 c=correlate(presented,expected,mode='full',method='fft');center=len(expected)-1;limit=2048
 lag=int(np.argmax(c[center-limit:center+limit+1])-limit);assert lag==0
 def video_hash(path):return hashlib.sha256(run([ffmpeg,'-hide_banner','-nostdin','-v','error','-i',str(path),'-map','0:v:0','-c:v','copy','-f','h264','pipe:1']).stdout).hexdigest()
 same_video=video_hash(r/'master.mov')==video_hash(r/'stemscene-walkthrough.mp4');assert same_video
 for timepoint in [1,5,13,17,21]:
  run([ffmpeg,'-hide_banner','-nostdin','-v','error','-n','-ss',str(timepoint),'-i',str(r/'stemscene-walkthrough.mp4'),'-frames:v','1',str(o/f'frame-{timepoint:02d}.png')])
 report={'media':media,'expected_audio_samples':len(expected),'master_decoded_audio_samples':len(master),'master_audio_max_absolute_error':float(np.max(np.abs(master-expected))),'aac_decoded_audio_samples_including_codec_padding':len(delivery),'aac_samples_beyond_presented_duration':len(delivery)-len(expected),'aac_presented_audio_alignment_lag_samples':lag,'aac_presented_error_rms':float(np.sqrt(np.mean((presented-expected)**2))),'aac_presented_peak':float(np.max(np.abs(presented))),'master_peak':float(np.max(np.abs(expected))),'mp4_video_stream_matches_master':same_video,'representative_frame_times':[1,5,13,17,21],'review_limits':['Full technical decode and timing checks; representative pixels inspected separately','No subjective normal-speed audiovisual listening judgment supplied by these checks','AAC is lossy; exact WAV/export checks remain separate']}
 (o/'media-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
