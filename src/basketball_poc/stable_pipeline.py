"""One local entry around the accepted, independently frozen processing stages."""
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .media import inspect_video, read_json, sha256, write_json
from . import stable_review

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT/'configs/stable-video-workflow-v1.json'
ENTRY = ROOT/'scripts/run_stable_workflow.py'


def now():
    return datetime.now(timezone.utc).isoformat()


def save_status(out, state):
    temporary = out/'workflow.json.tmp'
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                         encoding='utf-8', newline='\n')
    temporary.replace(out/'workflow.json')


@contextmanager
def stage(out, state, name, label):
    began = time.perf_counter()
    row = {'stage': name, 'label': label, 'status': 'running', 'started_at_utc': now()}
    state['stage'] = name
    state['stages'].append(row)
    save_status(out, state)
    print(label, flush=True)
    try:
        yield
    except BaseException:
        row['status'] = 'failed'
        raise
    else:
        row['status'] = 'completed'
    finally:
        row['elapsed_seconds'] = time.perf_counter()-began
        row['finished_at_utc'] = now()
        save_status(out, state)


def preflight(video):
    for executable in ('ffmpeg', 'ffprobe'):
        if shutil.which(executable) is None:
            raise ValueError(f'Missing media dependency: {executable}')
    baseline, baseline_path = stable_review.frozen_method()
    profile = read_json(PROFILE)
    if profile['detector_baseline_commit'] != baseline['commit']:
        raise ValueError('Workflow detector baseline mismatch')
    pinned = {str(ROOT/p): h for p, h in profile['additional_files_sha256'].items()}
    stable_review.verify_hashes(pinned)
    methods = [ENTRY, Path(__file__).resolve(), PROFILE,
        ROOT/'src/basketball_poc/fast_stabilization.py',
        ROOT/'configs/video-processing-performance-v1.json',
        ROOT/'scripts/analyze_video_efficient.py', ROOT/'scripts/verify_video_standard.py',
        ROOT/'scripts/analyze_video_streaming.py', ROOT/'src/basketball_poc/stream_frames.py',
        ROOT/'src/basketball_poc/inference_runtime.py',
        ROOT/'src/basketball_poc/frame_execution.py',
        ROOT/'src/basketball_poc/resource_policy.py', ROOT/'src/basketball_poc/batch_execution.py',
        ROOT/'scripts/verify_stream_analysis.py']
    if (ROOT/'release-manifest.json').exists():
        from .release_integrity import verify_release
        revision = verify_release(ROOT)['source_commit']
    else:
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', *map(str, methods)],
                                        cwd=ROOT, text=True).strip()
        if dirty:
            raise ValueError('Commit the workflow entry/profile before processing')
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    info = inspect_video(video)
    if abs(info['fps']-info['nominal_fps']) > .01:
        raise ValueError('Variable frame rate is unsupported; prepare a constant-frame-rate copy first')
    inputs = {**{str(ROOT/p): h for p, h in baseline['files_sha256'].items()}, **pinned,
              **{str(p): sha256(p) for p in [video, baseline_path, *methods]}}
    return {'source': info, 'source_sha256': inputs[str(video)], 'inputs_sha256': inputs,
            'git_commit': revision,
            'detector_baseline_commit': baseline['commit'], 'rebound_commit': profile['rebound_commit'],
            'runtime_versions': {n: version(n) for n in ('torch', 'ultralytics', 'opencv-python', 'numpy', 'dill')},
            'labels_used': False, 'manual_coordinates_used': False}


def command_stage(script, arguments, out, state):
    log_path = out/'logs'/f'{state["stage"]}.log'
    state['stages'][-1]['log'] = str(log_path)
    save_status(out, state)
    environment = os.environ.copy()
    environment['PYTHONIOENCODING'] = 'utf-8'
    entry = [str(ROOT/'scripts'/script)]
    if (ROOT/'release-manifest.json').exists() and script in ('triage_fixed_camera_review.py', 'filter_rebound_review.py'):
        entry = [str(ROOT/'scripts/run_portable_postprocess.py'), script]
    with log_path.open('x', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-u', *entry, *map(str, arguments)],
                                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{script} failed (exit {result.returncode}); log: {log_path}')


def analysis_identity(folder, source_hash):
    """Validate the frozen shake-wrapper format, including original passthrough."""
    folder = Path(folder).resolve(strict=True)
    frozen = read_json(folder/'frozen.json')['inputs_sha256']
    predictions = read_json(folder/'predictions-before-review.json')['files_sha256']
    result = read_json(folder/'result.json')
    pipeline = Path(result['pipeline']).resolve(strict=True)
    if (result['status'] != 'completed' or result['original_source_sha256'] != source_hash
            or result['original_seconds_offset'] != 0
            or result['mode'] not in ('original_passthrough', 'stabilized_fallback')
            or pipeline != folder/('corrected' if result['mode'] == 'stabilized_fallback' else 'original')):
        raise ValueError('Completed analysis of the exact original video required')
    required = [folder/'result.json', pipeline/'results.json', pipeline/'context.json',
                pipeline/'review-events.json', pipeline/'net-records.json',
                pipeline/'supported-events.json', pipeline/'uncertain-events.json',
                pipeline/'candidates/manifest.json', pipeline/'candidates/detections.jsonl']
    if any(str(path) not in predictions for path in required):
        raise ValueError('Incomplete frozen analysis inventory')
    stable_review.verify_hashes(frozen)
    stable_review.verify_hashes(predictions)
    original = Path(result['original_video']).resolve(strict=True)
    if frozen.get(str(original)) != source_hash or sha256(original) != source_hash:
        raise ValueError('Original source changed')
    verification = read_json(folder/'verification.json')
    validate_verification(verification)
    documents = [folder/'frozen.json', folder/'predictions-before-review.json', folder/'verification.json']
    return result, {**frozen, **predictions, **{str(p): sha256(p) for p in documents}}


def validate_verification(verification):
    common = ('frame_and_time_mapping_verified', 'predictions_unchanged', 'method_identities_unchanged')
    if (verification.get('status') != 'passed' or verification.get('recognition_status') != 'completed'
            or not all(verification.get(k) is True for k in common)):
        raise ValueError('Verified source mapping and evidence required')
    if verification.get('verification_mode') == 'standard-v1':
        if (not all(verification.get(k) is True for k in ('candidate_inventory_verified', 'event_net_partition_verified'))
                or verification.get('full_replay_performed') is not False
                or verification.get('candidate_context_net_replay_equal') is not False):
            raise ValueError('Invalid standard verification scope')
    elif verification.get('candidate_context_net_replay_equal') is not True:
        raise ValueError('Verified analysis replay required')


def report(out, state):
    text = ['# 视频处理结果', '', f'状态：{state["status"]}', f'素材：{state["video"]}',
            f'本轮执行方式：{state["execution_mode"]}', '',
            '## 处理阶段', '', '| 阶段 | 状态 | 耗时（秒） |', '| --- | --- | ---: |']
    for row in state['stages']:
        text.append(f'| {row["label"]} | {row["status"]} | {row.get("elapsed_seconds", 0):.2f} |')
    if state['status'] == 'ready_for_review':
        text += ['', f'自动计入：{state["counts"]["supported"]}；待确认：{state["counts"]["uncertain"]}。',
                 f'复核视频包：{state["review"]}', '',
                 '人工选择仅影响导出，不会改写自动识别成绩。筛除记录保留于诊断文件，不提供页面恢复入口。',
                 '系统仍可能遗漏列表外的进球；本次处理完成不代表已经完成真人验收。']
    else:
        text += ['', f'原因：{state.get("error") or state.get("reason")}',
                 '未生成可用复核结果；不能据此判断视频没有进球。']
    (out/'report.md').write_text('\n'.join(text)+'\n', encoding='utf-8', newline='\n')


def run(video, output, analysis_run=None, processing_mode='reference', verification_mode='full', inference_device='cpu',
        opencv_threads=1, net_mode='full', execution_mode='sequential', batch_size=0):
    if processing_mode not in ('reference', 'efficient', 'streaming') or verification_mode not in ('full', 'standard'):
        raise ValueError('Invalid processing/verification mode')
    if processing_mode == 'streaming' and verification_mode != 'standard':
        raise ValueError('Streaming requires standard verification; independent full replay uses its own verifier')
    if inference_device not in ('auto', 'cpu', 'cuda:0') or (inference_device != 'cpu' and (processing_mode != 'streaming' or analysis_run)):
        raise ValueError('Explicit GPU inference requires a fresh streaming analysis')
    if execution_mode not in ('sequential','batched') or (execution_mode == 'batched' and (processing_mode != 'streaming' or analysis_run)):
        raise ValueError('Batch execution requires fresh streaming analysis')
    if batch_size not in (0,1,2,4,8,16,32):
        raise ValueError('Invalid batch size')
    if (opencv_threads not in (0,1,2,4,8) or net_mode not in ('full','windows')
            or ((opencv_threads != 1 or net_mode != 'full') and (processing_mode != 'streaming' or analysis_run))):
        raise ValueError('Frame execution options require fresh streaming analysis')
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'logs').mkdir()
    began = time.perf_counter()
    state = {'version': 'stable-video-workflow-v1', 'status': 'running', 'stage': 'preflight',
             'video': str(Path(video).resolve()), 'created_at_utc': now(), 'stages': [],
             'execution_mode': 'verified_analysis_reuse' if analysis_run else 'fresh_analysis',
             'fresh_ball_inference_frames': 0, 'review': None, 'counts': None,
             'processing_mode': processing_mode, 'verification_mode': verification_mode,
             'inference_device': inference_device,
             'opencv_threads': opencv_threads, 'net_mode': net_mode,
             'execution_mode_options': execution_mode, 'batch_size': batch_size,
             'labels_used': False, 'user_acceptance': 'pending'}
    save_status(out, state)
    try:
        with stage(out, state, 'preflight', '1/5 检查视频、模型和版本'):
            source = Path(video).resolve(strict=True)
            frozen = preflight(source)
            state.update({k: frozen[k] for k in ('source_sha256', 'git_commit', 'detector_baseline_commit', 'rebound_commit')})
            write_json(out/'frozen.json', frozen)
        if analysis_run:
            analysis = Path(analysis_run).resolve(strict=True)
            with stage(out, state, 'reuse', '2/5 核对已有分析结果'):
                result, evidence = analysis_identity(analysis, frozen['source_sha256'])
        else:
            analysis = out/'analysis'
            with stage(out, state, 'analysis', '2/5 检测篮球、篮筐及篮网，必要时补偿轻微晃动'):
                script = {'efficient': 'analyze_video_efficient.py', 'streaming': 'analyze_video_streaming.py'}.get(processing_mode, 'benchmark_fixed_camera_shake.py')
                arguments = [source, '--out', analysis]
                if processing_mode == 'streaming':
                    arguments += ['--inference-device', inference_device,
                                  '--opencv-threads', str(opencv_threads), '--net-mode', net_mode,
                                  '--execution-mode', execution_mode, '--batch-size', str(batch_size)]
                command_stage(script, arguments, out, state)
                result = read_json(analysis/'result.json')
                if result['status'] == 'unsupported':
                    stable_review.verify_hashes(frozen['inputs_sha256'])
                    state.update(status='unsupported', reason=result['reason'], analysis=str(analysis))
                    return state
                if result['status'] != 'completed':
                    raise ValueError('Analysis did not complete')
                state['fresh_ball_inference_frames'] = result['fresh_ball_inference_frames']
            with stage(out, state, 'verification', '3/5 核验原视频映射与分析结果'):
                script = 'verify_video_standard.py' if verification_mode == 'standard' else 'verify_camera_shake.py'
                command_stage(script, [analysis], out, state)
                result, evidence = analysis_identity(analysis, frozen['source_sha256'])
        state.update(analysis=str(analysis), analysis_mode=result['mode'], pipeline=result['pipeline'])
        write_json(out/'analysis-evidence.json', {'files_sha256': evidence,
            'execution_mode': state['execution_mode'], 'fresh_ball_inference_frames': state['fresh_ball_inference_frames']})
        with stage(out, state, 'triage', '4/5 筛选待确认候选'):
            command_stage('triage_fixed_camera_review.py', [analysis, '--out', out/'triage'], out, state)
        with stage(out, state, 'rebound', '4/5 检查断轨后的反弹'):
            command_stage('filter_rebound_review.py', [out/'triage', '--out', out/'filtered'], out, state)
        with stage(out, state, 'review', '5/5 准备进球预览与导出页面'):
            prepared = stable_review.prepare(out/'filtered', out/'review')
            stable_review.verify_review(out/'review')
            stable_review.verify_hashes(evidence)
            stable_review.verify_hashes(frozen['inputs_sha256'])
        state.update(status='ready_for_review', counts=prepared['counts'], review=prepared['pack'])
        write_json(out/'completion.json', {'files_sha256': {
            str(p): sha256(p) for p in [out/'frozen.json', out/'analysis-evidence.json',
                out/'filtered/predictions-before-evaluation.json', out/'review/review-manifest.json',
                out/'review/index.html', out/'review/original-events.json']},
            'labels_used': False, 'user_acceptance': 'pending'})
        return state
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=str(error) or type(error).__name__, review=None, counts=None)
        raise
    finally:
        state['elapsed_seconds'] = time.perf_counter()-began
        state['finished_at_utc'] = now()
        save_status(out, state)
        report(out, state)
