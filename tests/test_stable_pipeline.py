from pathlib import Path

import pytest

from basketball_poc import stable_pipeline as pipeline
from basketball_poc.media import read_json, write_json, sha256


def frozen_analysis(tmp_path, source):
    folder = tmp_path/'saved-analysis'
    active = folder/'original'
    digest = sha256(source)
    result = {'status': 'completed', 'mode': 'original_passthrough', 'pipeline': str(active),
              'original_video': str(source), 'original_source_sha256': digest,
              'original_seconds_offset': 0, 'fresh_ball_inference_frames': 10}
    names = ['results.json', 'context.json', 'review-events.json', 'net-records.json',
             'supported-events.json', 'uncertain-events.json', 'candidates/manifest.json']
    for name in names: write_json(active/name, {})
    (active/'candidates/detections.jsonl').write_text('{}\n', encoding='utf-8')
    write_json(folder/'result.json', result)
    write_json(folder/'frozen.json', {'inputs_sha256': {str(source): digest}})
    write_json(folder/'predictions-before-review.json', {'files_sha256': {
        str(p): sha256(p) for p in folder.rglob('*') if p.is_file()}})
    write_json(folder/'verification.json', {'status': 'passed', 'recognition_status': 'completed',
        'frame_and_time_mapping_verified': True, 'candidate_context_net_replay_equal': True,
        'predictions_unchanged': True, 'method_identities_unchanged': True})
    return folder


@pytest.fixture
def source(tmp_path):
    path = tmp_path/'video with 空格.mp4'
    path.write_bytes(b'fixture-only, not recognition evidence')
    return path


def mock_preflight(monkeypatch, source):
    monkeypatch.setattr(pipeline, 'preflight', lambda _: {
        'source_sha256': sha256(source), 'inputs_sha256': {str(source): sha256(source)},
        'git_commit': 'test', 'detector_baseline_commit': 'test', 'rebound_commit': 'test'})


def test_existing_output_is_never_modified(tmp_path, source):
    out = tmp_path/'existing'
    out.mkdir(); (out/'workflow.json').write_bytes(b'previous')
    with pytest.raises(FileExistsError): pipeline.run(source, out)
    assert (out/'workflow.json').read_bytes() == b'previous'


@pytest.mark.parametrize('kwargs', [{'inference_device':'auto'}, {'inference_device':'cuda:0'},
    {'inference_device':'cuda:0','processing_mode':'streaming','verification_mode':'standard','analysis_run':'saved'}])
def test_gpu_requires_explicit_fresh_streaming(tmp_path, source, kwargs):
    out = tmp_path/'invalid'
    with pytest.raises(ValueError):
        pipeline.run(source, out, **kwargs)
    assert not out.exists()


def test_preflight_failure_is_recorded(tmp_path, source, monkeypatch):
    def fail(_): raise ValueError('model mismatch')
    monkeypatch.setattr(pipeline, 'preflight', fail)
    out = tmp_path/'run'
    with pytest.raises(ValueError, match='model mismatch'): pipeline.run(source, out)
    state = read_json(out/'workflow.json')
    assert state['status'] == 'failed' and state['stage'] == 'preflight'
    assert state['review'] is None and state['counts'] is None
    assert state['stages'][0]['status'] == 'failed'
    assert 'model mismatch' in (out/'report.md').read_text(encoding='utf-8')


def test_unsupported_does_not_become_zero_goal_success(tmp_path, source, monkeypatch):
    mock_preflight(monkeypatch, source)
    calls = []
    def command(script, args, out, state):
        calls.append(script)
        write_json(out/'analysis/result.json', {'status': 'unsupported', 'reason': 'No stable hoop'})
    monkeypatch.setattr(pipeline, 'command_stage', command)
    out = tmp_path/'run'
    state = pipeline.run(source, out)
    assert state['status'] == 'unsupported' and state['counts'] is None
    assert calls == ['benchmark_fixed_camera_shake.py']
    assert not (out/'review').exists()
    assert read_json(out/'workflow.json')['stages'][-1]['status'] == 'completed'


@pytest.mark.parametrize('interrupted', [False, True])
def test_stage_failure_or_interrupt_preserves_prior_evidence(tmp_path, source, monkeypatch, interrupted):
    mock_preflight(monkeypatch, source)
    def command(*_):
        if interrupted: raise KeyboardInterrupt()
        raise RuntimeError('detector failed')
    monkeypatch.setattr(pipeline, 'command_stage', command)
    out = tmp_path/'run'
    with pytest.raises(KeyboardInterrupt if interrupted else RuntimeError): pipeline.run(source, out)
    state = read_json(out/'workflow.json')
    assert state['status'] == ('interrupted' if interrupted else 'failed')
    assert state['stage'] == 'analysis' and state['review'] is None
    assert (out/'frozen.json').exists() and not (out/'completion.json').exists()


@pytest.mark.parametrize('damage', ['source', 'context', 'missing_inventory', 'verification', 'pipeline', 'offset'])
def test_reuse_rejects_wrong_or_changed_evidence(tmp_path, source, damage):
    folder = frozen_analysis(tmp_path, source)
    digest = sha256(source)
    if damage == 'source': digest = 'wrong-source'
    elif damage == 'context': (folder/'original/context.json').write_text('changed', encoding='utf-8')
    elif damage == 'missing_inventory':
        (folder/'predictions-before-review.json').write_text('{"files_sha256": {}}', encoding='utf-8')
    elif damage == 'verification':
        (folder/'verification.json').write_text('{"status": "failed"}', encoding='utf-8')
    else:
        result = read_json(folder/'result.json')
        if damage == 'pipeline': result['pipeline'] = str(tmp_path)
        else: result['original_seconds_offset'] = 3
        import json
        (folder/'result.json').write_text(json.dumps(result), encoding='utf-8')
    with pytest.raises(ValueError): pipeline.analysis_identity(folder, digest)


def test_reuse_success_skips_fresh_inference_and_keeps_pending(tmp_path, source, monkeypatch):
    folder = frozen_analysis(tmp_path, source)
    mock_preflight(monkeypatch, source)
    calls = []
    def command(script, args, out, state):
        calls.append(script)
        if script == 'filter_rebound_review.py':
            write_json(out/'filtered/predictions-before-evaluation.json', {'files_sha256': {}})
    def prepare(_, out):
        for name in ('review-manifest.json', 'original-events.json'): write_json(out/name, {})
        (out/'index.html').write_text('test review', encoding='utf-8')
        return {'pack': str(out), 'counts': {'supported': 5, 'uncertain': 3, 'excluded': 3}}
    monkeypatch.setattr(pipeline, 'command_stage', command)
    monkeypatch.setattr(pipeline.stable_review, 'prepare', prepare)
    monkeypatch.setattr(pipeline.stable_review, 'verify_review', lambda _: None)
    out = tmp_path/'run'
    result = pipeline.run(source, out, folder)
    assert result['status'] == 'ready_for_review' and result['counts']['uncertain'] == 3
    assert result['execution_mode'] == 'verified_analysis_reuse'
    assert result['fresh_ball_inference_frames'] == 0 and result['user_acceptance'] == 'pending'
    assert calls == ['triage_fixed_camera_review.py', 'filter_rebound_review.py']
    assert (out/'completion.json').exists()
    assert not (out/'review/exports').exists()


def test_command_arguments_are_not_shell_interpolated(tmp_path, monkeypatch):
    out = tmp_path; (out/'logs').mkdir()
    state = {'stage': 'analysis', 'stages': [{}]}
    recorded = {}
    def fake_run(args, **kwargs):
        recorded.update(args=args, kwargs=kwargs)
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(pipeline.subprocess, 'run', fake_run)
    filename = 'path with spaces/视频;$(literal).mp4'
    pipeline.command_stage('benchmark_fixed_camera_shake.py', [filename, '--out', 'out'], out, state)
    assert recorded['args'][3] == filename
    assert not recorded['kwargs'].get('shell', False)
    assert read_json(out/'workflow.json')['stages'][0]['log'].endswith('analysis.log')
