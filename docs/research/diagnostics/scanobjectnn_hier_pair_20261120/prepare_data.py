"""Validate an author release and extract only the approved main_split files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from support import DATA_FILES, inspect_data, now, sha_file, write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--verify-existing',action='store_true')
    args=p.parse_args()
    archive=args.root/'h5_files.zip'
    partial=args.root/'h5_files.zip.partial'
    if not archive.exists():
        if partial.stat().st_size!=315941783 or not zipfile.is_zipfile(partial):
            raise RuntimeError('Author archive is incomplete or has the wrong size')
        partial.rename(archive)
    destination=args.root/'main_split'
    if destination.exists():
        if not args.verify_existing:
            raise FileExistsError('Use explicit verification for existing data; never overwrite it')
    else:
        destination.mkdir(exist_ok=False)
        with zipfile.ZipFile(archive) as package:
            for partition,(name,expected,rows) in DATA_FILES.items():
                candidates=[item for item in package.infolist() if Path(item.filename).name==name and '/main_split/' in '/'+item.filename]
                if len(candidates)!=1:
                    raise RuntimeError('Missing or ambiguous main_split member: '+name)
                target=destination/name
                with package.open(candidates[0]) as source,target.open('xb') as output:
                    while True:
                        chunk=source.read(8*1024*1024)
                        if not chunk:break
                        output.write(chunk)
                if sha_file(target)!=expected:
                    raise RuntimeError('Author archive disagrees with independent HDF checksum: '+name)
    arrays,identity=inspect_data(destination)
    del arrays
    record=dict(verified=True,variant='PB_T50_RS',split='main_split',background=True,
        source='https://github.com/ma-xu/pointMLP-pytorch/releases/download/dataset/h5_files.zip',
        source_owner='PointMLP authors public dataset release',archive_sha256=sha_file(archive),
        file_hash_crosscheck='cminst/ScanObjectNN public HDF checksums',files=identity,verified_at=now())
    write(args.root/'dataset_identity.json',record)
    print(json.dumps(record))


if __name__=='__main__':main()
