"""Fetch the pinned upstream model and verify its published provenance entry."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'third_party/josephattalla/e320817d0f87eceb4093c871de6d29d1adca0006'


def main():
    manifest = json.loads((FOLDER/'provenance.json').read_text(encoding='utf-8'))
    item = next(x for x in manifest['files'] if x['file'] == 'bball_model.pt')
    target = FOLDER/item['file']
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Existing model differs; preserve it and use a fresh source folder')
        print('Pinned model already verified');return
    temporary = target.with_suffix('.download')
    try:
        with urlopen(item['url'], timeout=60) as response, temporary.open('xb') as file:
            while block := response.read(1024*1024):file.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Downloaded model checksum mismatch')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print('Pinned model downloaded and verified')


if __name__ == '__main__':main()
