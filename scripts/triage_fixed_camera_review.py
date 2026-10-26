"""Reduce a completed analysis review queue, preserving all original evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from basketball_poc.media import read_json, write_json, sha256
from basketball_poc.review_triage import triage

ROOT = Path(__file__).resolve().parents[1]


def stamp(seconds):
    n = round(seconds*100)
    return f'{n//6000}:{n%6000/100:05.2f}'


def clip_index(document):
    result = {}
    for clip in document['clips']:
        identifier = clip.get('id') or clip.get('event_id')
        if not identifier or identifier in result:
            raise ValueError('Missing or duplicate review clip event identity')
        result[identifier] = clip
    return result


def run(analysis, out, profile_path):
    began = time.perf_counter()
    analysis = Path(analysis).resolve(strict=True)
    out = Path(out).resolve()
    # Accept either the shake run root or the completed analysis folder.
    review_path = analysis/'review.json'
    if (analysis/'result.json').exists():
        analysis = Path(read_json(analysis/'result.json')['pipeline'])
    elif (analysis.parent/'review.json').exists():
        review_path = analysis.parent/'review.json'
    paths = [analysis/name for name in ('supported-events.json', 'uncertain-events.json',
        'review-events.json', 'net-records.json', 'context.json', 'results.json', 'candidates/manifest.json')]
    if review_path.exists():
        paths.append(review_path)
    method_paths = [Path(__file__), ROOT/'src/basketball_poc/review_triage.py',
                    ROOT/'src/basketball_poc/media.py', Path(profile_path).resolve()]
    frozen = {str(p): sha256(p) for p in paths+method_paths}
    if read_json(analysis/'results.json')['status'] != 'completed':
        raise ValueError('Completed analysis required')
    supported = read_json(analysis/'supported-events.json')
    uncertain = read_json(analysis/'uncertain-events.json')
    raw = read_json(analysis/'review-events.json')['events']
    ids = [e['id'] for e in supported['events']+uncertain['events']]
    if len(raw) != len(ids) or {e['id'] for e in raw} != set(ids):
        raise ValueError('Raw candidate partition mismatch')
    profile = read_json(profile_path)
    result = triage(supported['events'], uncertain['events'],
                    read_json(analysis/'net-records.json')['records'],
                    read_json(analysis/'context.json')['associations'], profile)
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--',
        *[str(p) for p in method_paths]], cwd=ROOT, text=True).strip()
    if dirty:
        raise ValueError('Commit the triage method/profile before running')
    write_json(out/'frozen.json', {'inputs_sha256': frozen, 'git_commit': commit,
        'profile': profile, 'labels_used': False, 'execution': 'post-analysis routing; no new detector/net inference'})
    # Exact bytes preserve existing supported-envelope metadata as well as events.
    (out/'supported-events.json').write_bytes((analysis/'supported-events.json').read_bytes())
    for key in ('uncertain', 'excluded'):
        write_json(out/f'{key}-events.json', {**uncertain, 'events': result[key]})
    write_json(out/'decisions.json', result['decisions'])
    clips = clip_index(read_json(review_path)) if review_path.exists() else {}
    decision = {d['id']: d for d in result['decisions']}
    reason_text = {'supported_passthrough': '自动计入，保持原样',
        'projection_before_rim_reach': '未观察到球到达篮筐高度，仅有外推',
        'observed_lateral_net_exit': '经过篮网下缘高度时，球整体位于篮筐外侧',
        'no_sufficient_exclusion_evidence': '证据不足，继续复核'}
    text = ['# 待确认筛选结果', '', f'算法提交：`{commit[:7]}`；自动计入保持不变。', '',
        '排除项保留在诊断记录中，不等于已确认未进。未使用人工标签参与筛选。', '']
    for key, title in [('supported', '自动计入'), ('uncertain', '仍需确认'), ('excluded', '移出主要复核列表')]:
        text += [f'## {title}', '', '| 原编号 | 时间 | 原画面短片 | 原因 |', '| --- | --- | --- | --- |']
        for event in result[key]:
            clip = clips.get(event['id'])
            number = Path(clip['path']).stem if clip else event['id']
            link = f'[播放]({Path(clip["path"]).as_posix()})' if clip else '见原分析记录'
            text.append(f'| {number} | {stamp(event["time_seconds"])} | {link} | {reason_text[decision[event["id"]]["reason"]]} |')
        if not result[key]:
            text.append('| — | 无 | — | — |')
        text.append('')
    (out/'report.md').write_text('\n'.join(text), encoding='utf-8')
    for p, digest in frozen.items():
        if sha256(p) != digest:
            raise ValueError(f'Input changed: {p}')
    summary = {'status': 'completed', 'analysis': str(analysis), 'git_commit': commit,
        'supported_count': len(result['supported']), 'uncertain_before': len(uncertain['events']),
        'uncertain_after': len(result['uncertain']), 'excluded_count': len(result['excluded']),
        'supported_bytes_unchanged': True, 'labels_used': False, 'fresh_inference': False,
        'processing_seconds': time.perf_counter()-began}
    write_json(out/'results.json', summary)
    write_json(out/'predictions-before-evaluation.json', {'files_sha256': {
        str(p): sha256(p) for p in out.iterdir() if p.is_file()}, 'labels_used': False})
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--profile', type=Path, default=ROOT/'configs/review-triage-v1.json')
    args = p.parse_args()
    run(args.analysis, args.out, args.profile)
