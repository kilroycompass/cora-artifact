"""Acquire original 5G-NIDD session CSV/Argus files with publisher SHA-256 checks.

Public Fairdata download authorization yields an ephemeral delivery URL, which
is never saved. Unavailable original archives remain explicit exclusions.
"""
from pathlib import Path
import hashlib,json,os,ssl,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/real/5gnidd'
IDENTIFIER='9d13ef28-2ca7-44b0-9950-225359afac65'


def context():
    return ssl.create_default_context(cafile=os.environ.get('CORA_CA_BUNDLE'))


def get_json(url,ctx):
    with urllib.request.urlopen(url,context=ctx,timeout=40) as response:return json.load(response)


def fetch():
    RAW.mkdir(parents=True,exist_ok=True);ctx=context()
    metadata=get_json(f'https://metax.fairdata.fi/v3/datasets/{IDENTIFIER}/files',ctx)
    package=get_json(f'https://etsin.fairdata.fi/api/v3/download/requests?cr_id={IDENTIFIER}',ctx)
    selected=[f'BS{s}_each_attack_{kind}.zip' for s in (1,2) for kind in ('csv','argus')];records=[]
    for filename in selected:
        entry=next(x for x in metadata['results'] if x['filename']==filename)
        dest=RAW/filename;expected=entry['checksum'].split(':',1)[1]
        if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest()!=expected:
            ready=next((x for x in package.get('partial',[]) if x.get('scope')==['/'+filename] and x.get('status')=='SUCCESS'),None)
            if ready is None:
                records.append(dict(file=filename,supported=False,reason='No available verified single-file package; abbreviated CSV clocks are not substituted'))
                continue
            req=urllib.request.Request('https://etsin.fairdata.fi/api/v3/download/authorize',
                data=json.dumps(dict(cr_id=IDENTIFIER,package=ready['package'])).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,context=ctx,timeout=40) as r:url=json.load(r)['url']
            outer_path=RAW/(filename+'.delivery.zip')
            with urllib.request.urlopen(url,context=ctx,timeout=60) as src,outer_path.open('wb') as out:
                while chunk:=src.read(1024*1024):out.write(chunk)
            assert 'sha256:'+hashlib.sha256(outer_path.read_bytes()).hexdigest()==ready['checksum']
            with zipfile.ZipFile(outer_path) as outer:
                member=next(k for k in outer.namelist() if Path(k).name==filename);dest.write_bytes(outer.read(member))
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==expected
        station=int(filename[2]);kind='argus' if 'argus' in filename else 'csv'
        folder=RAW/(f'argus_bs{station}' if kind=='argus' else f'bs{station}');folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(dest) as inner:
            for name in inner.namelist():
                if name.endswith('.'+kind):
                    target=folder/Path(name).name
                    with inner.open(name) as src,target.open('wb') as out:
                        while chunk:=src.read(4*1024*1024):out.write(chunk)
        records.append(dict(file=filename,supported=True,sha256=expected,bytes=entry['size']))
        print('Verified 5G-NIDD',filename,flush=True)
    (RAW/'original_session_manifest.json').write_text(json.dumps(dict(doi='10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361',
        publisher=f'https://etsin.fairdata.fi/dataset/{IDENTIFIER}',license='CC BY 4.0',files=records),indent=2))


if __name__=='__main__':fetch()
