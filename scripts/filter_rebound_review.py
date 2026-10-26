"""Apply a committed supplemental rebound filter to an immutable V1 triage run."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from basketball_poc.media import read_json, write_json, sha256
from basketball_poc.rebound_review import filter_review

ROOT = Path(__file__).resolve().parents[1]


def verify(mapping):
    for path, digest in mapping.items():
        if sha256(path) != digest:
            raise ValueError(f'Evidence changed: {path}')


def load_base(folder):
    return {**{key: read_json(folder/f'{key}-events.json')['events']
               for key in ('supported', 'uncertain', 'excluded')},
            'decisions': read_json(folder/'decisions.json')}


def run(folder, out, profile_path):
    began = time.perf_counter()
    folder, out = Path(folder).resolve(strict=True), Path(out).resolve()
    profile_path = Path(profile_path).resolve(strict=True)
    methods = [Path(__file__).resolve(), ROOT/'src/basketball_poc/rebound_review.py',
               ROOT/'src/basketball_poc/media.py', profile_path]
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--',
                                    *map(str, methods)], cwd=ROOT, text=True).strip()
    if dirty:
        raise ValueError('Commit the rebound method/profile before running')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    summary = read_json(folder/'results.json')
    if summary['status'] != 'completed' or not summary['supported_bytes_unchanged']:
        raise ValueError('Completed frozen triage required')
    analysis = Path(summary['analysis']).resolve(strict=True)
    inputs = read_json(folder/'frozen.json')['inputs_sha256']
    outputs = read_json(folder/'predictions-before-evaluation.json')['files_sha256']
    verify(inputs); verify(outputs)
    base = load_base(folder)
    original = {e['id']: e for key in ('supported', 'uncertain')
                for e in read_json(analysis/f'{key}-events.json')['events']}
    partition = [e for key in ('supported', 'uncertain', 'excluded') for e in base[key]]
    if len(partition) != len(original) or {e['id']: e for e in partition} != original:
        raise ValueError('Source candidate partition mismatch')
    if (folder/'supported-events.json').read_bytes() != (analysis/'supported-events.json').read_bytes():
        raise ValueError('Supported bytes changed')
    paths = [folder/'frozen.json', folder/'predictions-before-evaluation.json',
             analysis/'context.json', *methods]
    frozen = {**inputs, **outputs, **{str(p): sha256(p) for p in paths}}
    profile = read_json(profile_path)
    result = filter_review(base, read_json(analysis/'context.json')['associations'], profile)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'frozen.json', {'inputs_sha256': frozen, 'git_commit': commit,
        'profile': profile, 'base_triage': str(folder), 'labels_used': False,
        'execution': 'post-analysis rebound routing; no new detector/net inference'})
    (out/'supported-events.json').write_bytes((folder/'supported-events.json').read_bytes())
    for key in ('uncertain', 'excluded'):
        write_json(out/f'{key}-events.json', {**read_json(folder/f'{key}-events.json'), 'events': result[key]})
    write_json(out/'decisions.json', result['decisions'])
    write_json(out/'new-exclusions.json', result['new_exclusions'])
    verify(frozen)
    summary = {'status': 'completed', 'analysis': str(analysis), 'base_triage': str(folder),
        'git_commit': commit, 'supported_count': len(result['supported']),
        'uncertain_before': len(base['uncertain']), 'uncertain_after': len(result['uncertain']),
        'excluded_count': len(result['excluded']), 'new_exclusions': len(result['new_exclusions']),
        'supported_bytes_unchanged': True, 'labels_used': False, 'fresh_inference': False,
        'processing_seconds': time.perf_counter()-began}
    write_json(out/'results.json', summary)
    write_json(out/'predictions-before-evaluation.json', {'files_sha256': {
        str(p): sha256(p) for p in out.iterdir() if p.is_file()}, 'labels_used': False})
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('triage', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=ROOT/'configs/rebound-review-v1.json')
    args = parser.parse_args()
    run(args.triage, args.out, args.profile)
