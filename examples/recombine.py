#!/usr/bin/env python3
"""Independently verify/replay a STEMSCENE folder or ZIP. No engine/model imports."""
from __future__ import annotations
import argparse, hashlib, io, json, os, stat, zipfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import soundfile as sf


@contextmanager
def bundle(path: Path):
    archive = zipfile.ZipFile(path) if path.is_file() else None
    if archive:
        members = archive.infolist()
        if len(members)>128 or sum(x.file_size for x in members)>4*1024**3:
            raise ValueError('Bundle exceeds file/expanded-byte bounds')
        if len({x.filename for x in members}) != len(members):
            raise ValueError('Duplicate archive paths')
    def read(name):
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Invalid bundle path')
        if archive:
            if archive.getinfo(name).file_size>512*1024**2: raise ValueError('Asset exceeds bounds')
            return archive.read(name)
        target=(path/relative).resolve()
        if not target.is_relative_to(path.resolve()): raise ValueError('Asset leaves bundle')
        descriptor=os.open(target,os.O_RDONLY|os.O_NONBLOCK|getattr(os,'O_NOFOLLOW',0))
        with os.fdopen(descriptor,'rb') as f:
            info=os.fstat(f.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size>512*1024**2: raise ValueError('Invalid bounded file')
            return f.read(512*1024**2+1)
    try: yield read
    finally:
        if archive: archive.close()


def verify(path: Path):
    with bundle(path) as read:
        manifest_bytes=read('manifest.json')
        if len(manifest_bytes)>32*1024**2: raise ValueError('Manifest exceeds bounds')
        m=json.loads(manifest_bytes)
        if m['schema']!='stemscene-export/1' or not 0<=len(m['layers'])<=12: raise ValueError('Unsupported export')
        shape=(m['frames'],m['channels'])
        if any(type(x)is not int or x<1 for x in shape) or shape[1]>2 or np.prod(shape)*8>512*1024**2:
            raise ValueError('Declared waveform exceeds bounds')
        for name,digest in m['files'].items():
            if hashlib.sha256(read(name)).hexdigest()!=digest: raise ValueError('Checksum mismatch: '+name)
        def wav(name,channels=shape[1]):
            with sf.SoundFile(io.BytesIO(read(name))) as f:
                if (len(f),f.channels,f.samplerate)!=(shape[0],channels,m['sample_rate']):
                    raise ValueError('Unaligned WAV: '+name)
                a=f.read(dtype='float64',always_2d=True)
            if not np.isfinite(a).all(): raise ValueError('Nonfinite WAV')
            return a
        original=wav('original-decoded.wav'); mixed=wav('mix.wav')
        unity=wav('residual.wav'); replay=original.copy(); recipe_error=0.0
        source=read(m['source']['path'])
        if hashlib.sha256(source).hexdigest()!=m['source']['sha256']: raise ValueError('Original source changed')
        with sf.SoundFile(io.BytesIO(source)) as f:
            if (len(f),f.channels,f.samplerate)!=(shape[0],shape[1],m['sample_rate']):
                raise ValueError('Original source header does not match export')
            decoded=f.read(dtype='float64',always_2d=True)
        source_decode_error=float(np.max(np.abs(decoded-original)))
        for layer in m['layers']:
            stem=wav(layer['stem']); envelope=wav(layer['envelope'],1)[:,0]
            recipe=json.loads(read(layer['envelope_recipe'])); gains=np.ones(shape[0])
            for operation in recipe['operations']:
                start,end,fade=(operation[k] for k in ('start_sample','end_sample','fade_samples'))
                value=operation['gain']; local=np.full(end-start,value,dtype='float64')
                if fade:
                    ramp=np.linspace(1.0,value,fade+1)[:-1];local[:fade]=ramp;local[-fade:]=ramp[::-1]
                gains[start:end]*=local
            recipe_error=max(recipe_error,float(np.max(np.abs(gains-envelope))))
            unity+=stem
            replay+=(envelope-1)[:,None]*stem
        if m['clipping']=='hard': replay=np.clip(replay,-1,1)
        errors={'unity_max_abs_error':float(np.max(np.abs(unity-original))),
                'edited_mix_max_abs_error':float(np.max(np.abs(replay-mixed))),
                'envelope_recipe_max_abs_error':recipe_error,'original_decode_max_abs_error':source_decode_error}
        tolerance=1e-12 if m['subtype']=='DOUBLE' else 1e-6
        if max(errors.values())>tolerance: raise ValueError('Round-trip tolerance exceeded: '+str(errors))
        return replay,m['sample_rate'],{'engine_imported':False,'models_required':False,'stems':len(m['layers']),
               'frames':shape[0],'channels':shape[1],'sample_rate':m['sample_rate'],'tolerance':tolerance,
               'source_sha256':m['source']['sha256'],**errors}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--report',type=Path);a=p.parse_args()
    audio,rate,report=verify(a.bundle)
    if a.output:
        with a.output.open('xb') as stream: sf.write(stream,audio,rate,format='WAV',subtype='DOUBLE')
    if a.report:
        with a.report.open('x') as stream: json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
