"""Authorized 14 Oct: dev events and recording-length metadata only; no signal read.

MNE read_raw_fif(preload=False) is used only for sfreq and n_times. No get_data,
filter, find_events, epochs, gates or HDF5 reads occur here. BAD masks are not
applied to synthetic controls (DESIGN census upper-bound convention).
"""
import argparse,csv,json,hashlib
from pathlib import Path
from fractions import Fraction as F
import mne
from experiments.mlst2026.meg_z_20261013.data_manifest import run_stem
from .windows import gap_geometry
DEV=tuple(f'sub-{i:02}' for i in range(1,5))
def export(raw_root,derived):
    root=Path(raw_root); out={'authorization':'dev event and run-length metadata only, 2026-10-14',
        'clock':'exact rational, 1100/12 Hz, physical onset = (TSV sample - 37)/1100 + 0.0345 s',
        'BAD_annotations':'not applied; synthetic controls use events geometry', 'subjects':{}}
    for sub in DEV:
        runs=[]
        for run in range(1,7):
            stem=root/run_stem(sub,run); event=Path(str(stem)+'_events.tsv')
            rows=list(csv.DictReader(event.open(),delimiter='\t'))
            ons=[];resp=[]
            for row in rows:
                try: code=int(float(row['value']))
                except ValueError: continue
                if code in (5,6,7,13,14,15,17,18,19): ons.append(F(int(row['sample'])-37,1100)+F(69,2000))
                elif code in (256,4096): resp.append(F(row['onset']))
            raw=mne.io.read_raw_fif(str(stem)+'_meg.fif',preload=False,verbose='error')
            assert not raw.preload
            assert raw.info['sfreq']==1100
            n=(int(raw.n_times)+11)//12
            geom=gap_geometry(n,sorted(ons),resp)
            runs.append({'run':run,'G':len(ons),'n_raw':int(raw.n_times),'n_decimated':n,
                'events_sha256':hashlib.sha256(event.read_bytes()).hexdigest(),'gaps':geom})
            raw.close()
        summary=json.loads((Path(derived)/f'{sub}_summary.json').read_text())
        out['subjects'][sub]={'runs':runs,'sealed_channels':summary['channels'],
            'summary_sha256':hashlib.sha256((Path(derived)/f'{sub}_summary.json').read_bytes()).hexdigest()}
    return out
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--raw',required=True);ap.add_argument('--derived',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    Path(a.out).write_text(json.dumps(export(a.raw,a.derived),indent=1)+'\n')
