"""Download publisher-hosted NetFlow v3 archives and verify BagIt payloads.

No user credentials or dataset-request forms are needed for these open objects.
Signed delivery links are acquired at runtime and never written to disk.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "real" / "netflow-v3"
IDS = {
    "NF-UNSW-NB15-v3": "abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f",
    "NF-ToN-IoT-v3": "343e2e8c-6e6e-4a0c-813d-a46acea1b7f4",
    "NF-BoT-IoT-v3": "db0187cc-1f78-4215-8e34-303364498866",
}
API = "https://api.rdm.uq.edu.au/datasets/"


def get_json(url):
    with urllib.request.urlopen(url, timeout=40) as response:
        return json.load(response)


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name):
    OUT.mkdir(parents=True, exist_ok=True)
    uuid = IDS[name]
    details = get_json(API + uuid + "/file-details")
    delivery = get_json(API + uuid + "/download")
    assert details["accessType"] == "OPEN" and delivery["canDownload"]
    url = delivery["url"]
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=40) as response:
        assert response.status == 206
        size = int(response.headers["Content-Range"].split("/")[-1])
        etag = response.headers["ETag"].strip('"')
        response.read()
    archive = OUT / (name + ".zip")
    parts = OUT / (name + ".parts")
    parts.mkdir(exist_ok=True)
    part_size = 4 * 1024 * 1024

    def part(start):
        end = min(size, start + part_size) - 1
        path = parts / str(start)
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=40) as response:
                    assert response.status == 206
                    assert response.headers["Content-Range"] == f"bytes {start}-{end}/{size}"
                    payload = response.read()
                assert len(payload) == end - start + 1
                path.write_bytes(payload)
                return path
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(5, attempt + 1))

    if not archive.exists() or archive.stat().st_size != size:
        starts = list(range(0, size, part_size))
        print(f"Downloading {name}: {size:,} bytes", flush=True)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(part, start) for start in starts]
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                if i % 15 == 0 or i == len(starts):
                    print(f"{name}: {i}/{len(starts)} ranges complete", flush=True)
        with archive.open("wb") as target:
            for start in starts:
                with (parts / str(start)).open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
    assert archive.stat().st_size == size
    destination = OUT / name
    destination.mkdir(exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            relative = PurePosixPath(info.filename)
            assert ".." not in relative.parts and not relative.is_absolute()
            # Publisher archives contain one top-level packaging directory.
            target = destination / relative.name
            assert target.resolve().is_relative_to(destination.resolve())
            if not target.exists() or target.stat().st_size != info.file_size:
                with z.open(info) as source, target.open("wb") as output:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        output.write(block)
            extracted.append(target.name)
    # The publisher includes BagIt manifests. Verify all listed payloads,
    # including the complete CSV, rather than trusting a filename or local fixture.
    checks = []
    for manifest in destination.glob("manifest-*.txt"):
        algorithm = manifest.stem.split("-", 1)[1]
        for line in manifest.read_text(encoding="utf-8-sig").splitlines():
            expected, relative = line.split(maxsplit=1)
            path = destination / PurePosixPath(relative.lstrip("*")).name
            actual = digest(path, algorithm)
            assert expected.lower() == actual.lower(), f"Publisher checksum mismatch: {path}"
            checks.append(dict(file=path.name, algorithm=algorithm, expected=expected, actual=actual))
    assert checks, "No publisher payload checksums found"
    record = dict(dataset=name, publisher_page="https://staff.itee.uq.edu.au/marius/NIDS_datasets/",
                  delivery_page=f"https://rdm.uq.edu.au/files/{uuid}",
                  retrieved_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  archive_bytes=size, archive_etag=etag, archive_sha256=digest(archive),
                  license=details["datasetLicense"], publisher_details=details,
                  extracted_files=extracted, publisher_payload_verification=checks,
                  benchmark_status="acquired and checksum verified; no fitted experiment implied")
    (destination / "source_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Verified official {name}; {len(checks)} publisher payload checksums", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="*", choices=list(IDS), default=list(IDS))
    args = p.parse_args()
    for name in args.datasets:
        fetch(name)
