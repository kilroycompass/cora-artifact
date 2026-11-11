"""Verify and reproduce the delivered CORA experiment from measured scores."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results/cora'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def verify():
 count=0
 for manifest in ['score_manifest.json','tifs_score_manifest.json']:
  for r in json.loads((ROOT/'provenance'/manifest).read_text()):
   p=ROOT/r['path'];assert p.is_file() and digest(p)==r['sha256'],str(p);count+=1
 print('Verified',count,'measured score inputs',flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--recompute',action='store_true');ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
 verify()
 if args.verify_only:return
 if args.recompute:
  # Preserve previous generated outputs for comparison instead of deleting them.
  from datetime import datetime,timezone
  backup=ROOT/'build'/('recompute-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
  for base,names in [(OUT,['grid','traces','tuning','sensitivity','uncertainty']),
                     (ROOT/'experiments/results/tifs',['replays'])]:
   for name in names:
    p=(base/name).resolve();assert p.parent==base.resolve() and p.is_relative_to(ROOT.resolve())
    if p.exists():
     target=(backup/base.name/name).resolve();assert target.is_relative_to((ROOT/'build').resolve())
     target.parent.mkdir(parents=True,exist_ok=True);p.rename(target)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
 for file,args2 in [('test_protocol.py',['-v']),('test_acquisition_certificate.py',['-v']),
                    ('test_tifs_controls.py',['-v']),('replay.py',[]),('analysis.py',[]),
                    ('report.py',[]),('tifs_extension.py',['--stage','audit']),
                    ('tifs_extension.py',['--stage','query']),('tifs_report.py',[])]:
  print('Running',file,flush=True)
  subprocess.run([sys.executable,'-B',str(ROOT/'experiments'/file)]+args2,cwd=ROOT,check=True,env=env)
if __name__=='__main__':main()
