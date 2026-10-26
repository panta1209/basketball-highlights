"""Loopback upload-first UI around the unchanged stable analysis/review pipeline."""
from datetime import datetime, timezone
from functools import partial
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import threading
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

from . import stable_review
from .media import read_json, sha256, write_json
from .processing_progress import observe
from .workflow_server import ExportJobs, WorkflowReviewHandler

ROOT = Path(__file__).resolve().parents[2]
MAX_UPLOAD = 8 * 1024**3
EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'}
TERMINAL = {'ready_for_review', 'unsupported', 'failed', 'interrupted'}
STAGES = {'preflight': '检查视频', 'analysis': '分析投篮轨迹与篮网',
          'verification': '检查分析结果', 'triage': '筛选待确认片段',
          'rebound': '检查反弹轨迹', 'review': '准备预览页面'}


def atomic_json(path, document):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                         encoding='utf-8', newline='\n')
    temporary.replace(path)


def upload_name(value):
    if (not value or len(value) > 180 or value in ('.', '..')
            or PureWindowsPath(value).name != value or any(c in value for c in '/\\<>:"|?*')
            or any(ord(c) < 32 for c in value) or value[-1] in '. '
            or value.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                *{f'COM{i}' for i in range(1, 10)}, *{f'LPT{i}' for i in range(1, 10)}}):
        raise ValueError('文件名无效，请使用普通视频文件名。')
    if Path(value).suffix.lower() not in EXTENSIONS:
        raise ValueError('请选择 MP4、MOV、MKV、AVI、WEBM 或 M4V 视频。')
    return value


def process_video(video, output, log_path, *, python_executable=None, inference_device='auto', opencv_threads=0,
                  net_mode='windows', execution_mode='batched'):
    from .analysis_launcher import select_python
    runtime, selection = select_python(ROOT, python_executable, inference_device)
    environment = os.environ.copy()
    environment['PYTHONIOENCODING'] = 'utf-8'
    with log_path.open('x', encoding='utf-8') as log:
        log.write(json.dumps({'analysis_runtime_selection':selection},ensure_ascii=False)+'\n')
        log.flush()
        result = subprocess.run([str(runtime), '-u', str(ROOT/'scripts/run_stable_workflow.py'),
            str(video), '--out', str(output), '--processing-mode', 'streaming',
            '--verification-mode', 'standard', '--inference-device', inference_device,
            '--opencv-threads', str(opencv_threads), '--net-mode', net_mode,
            '--execution-mode', execution_mode], cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT)
    if result.returncode not in (0, 2):
        raise RuntimeError(f'Pipeline exit {result.returncode}; see {log_path}')


class UploadJobs:
    def __init__(self, root, processor=None, actor='user', examples=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.processor = processor or process_video
        self.actor = actor
        self.examples = {name: Path(path).resolve(strict=True) for name, path in (examples or {}).items()}
        self.lock = threading.RLock()
        self.work_lock = threading.Lock()
        self.active = None
        self.exports = {}
        self.jobs = {}
        for folder in self.root.iterdir():
            if not folder.is_dir() or not re.fullmatch('[0-9a-f]{32}', folder.name):
                continue
            try:
                job = read_json(folder/'job.json')
                if job['status'] not in TERMINAL:
                    state_path = folder/'output/workflow.json'
                    state = read_json(state_path) if state_path.exists() else {}
                    if state.get('status') in TERMINAL:
                        job['status'] = state['status']
                    else:
                        job.update(status='interrupted', failure_code='service_restarted')
                    atomic_json(folder/'job.json', job)
                self.jobs[folder.name] = job
            except (OSError, ValueError, KeyError):
                continue

    def reserve(self, name, size):
        name = upload_name(name)
        if type(size) is not int or not 0 < size <= MAX_UPLOAD:
            raise ValueError('视频不能为空，且单个文件不能超过 8 GB。')
        with self.lock:
            if self.active:
                raise RuntimeError('已有视频正在上传或分析，请完成后再上传下一个。')
            if not self.work_lock.acquire(blocking=False):
                raise RuntimeError('已有视频分析、人物匹配或导出任务，请稍候。')
            identifier = uuid.uuid4().hex
            folder = self.root/identifier
            try:
                (folder/'input').mkdir(parents=True, exist_ok=False)
                job = {'id': identifier, 'filename': name, 'size_bytes': size, 'status': 'uploading',
                       'created_at_utc': datetime.now(timezone.utc).isoformat(), 'received_bytes': 0}
                atomic_json(folder/'job.json', job)
                self.jobs[identifier] = job
                self.active = identifier
            except Exception:
                self.work_lock.release()
                raise
            return identifier, folder/'input'/(name+'.partial')

    def upload_failed(self, identifier, error):
        with self.lock:
            job = self.jobs[identifier]
            job.update(status='failed', failure_code='upload_failed', diagnostic_error=str(error))
            atomic_json(self.root/identifier/'job.json', job)
            if self.active == identifier:
                self.active = None
                self.work_lock.release()

    def finish_upload(self, identifier, partial_path):
        with self.lock:
            job = self.jobs[identifier]
            if partial_path.stat().st_size != job['size_bytes']:
                raise ValueError('视频上传不完整，请重新选择文件。')
            target = partial_path.with_name(job['filename'])
            partial_path.rename(target)
            digest = sha256(target)
            write_json(self.root/identifier/'upload.json', {**job, 'status': 'uploaded',
                'received_bytes': target.stat().st_size, 'source_sha256': digest,
                'destination': 'loopback local disk', 'external_upload': False})
            job.update(status='queued', received_bytes=target.stat().st_size, source_sha256=digest)
            atomic_json(self.root/identifier/'job.json', job)
        threading.Thread(target=self._process, args=(identifier, target), daemon=True).start()

    def _process(self, identifier, video):
        folder = self.root/identifier
        try:
            with self.lock:
                self.jobs[identifier]['status'] = 'running'
                atomic_json(folder/'job.json', self.jobs[identifier])
            self.processor(video, folder/'output', folder/'processing.log')
            state = read_json(folder/'output/workflow.json')
            if state['status'] not in TERMINAL:
                raise ValueError('Pipeline did not reach a terminal state')
            with self.lock:
                self.jobs[identifier]['status'] = state['status']
        except Exception as error:
            with self.lock:
                self.jobs[identifier].update(status='failed', failure_code='processing_failed', diagnostic_error=str(error))
        finally:
            with self.lock:
                atomic_json(folder/'job.json', self.jobs[identifier])
                if self.active == identifier:
                    self.active = None
                    self.work_lock.release()

    def summary(self, identifier):
        with self.lock:
            job = dict(self.jobs[identifier])
        state = {}
        path = self.root/identifier/'output/workflow.json'
        try:
            if path.exists():
                state = read_json(path)
        except (OSError, ValueError):
            pass  # Atomic replacement may be briefly locked by a Windows writer.
        result = {k: job[k] for k in ('id', 'filename', 'size_bytes', 'status', 'created_at_utc')}
        result.update(observe(self.root/identifier/'output', state, job))
        result['stage'] = STAGES.get(state.get('stage'), '等待开始分析')
        result['counts'] = state.get('counts') if job['status'] == 'ready_for_review' else None
        if job['status'] == 'ready_for_review':
            result['review_url'] = f'/jobs/{identifier}/review/'
        elif job['status'] == 'unsupported':
            result['message'] = '暂时无法稳定分析这段视频。请确认机位基本固定，篮筐和篮网清晰可见。'
        elif job['status'] in ('failed', 'interrupted'):
            code = job.get('failure_code')
            if code == 'upload_failed':
                result['message'] = '视频上传未完成，请重新选择文件并上传。'
            elif code == 'service_restarted' or job['status'] == 'interrupted':
                result['message'] = '上次处理已中断，请重新上传开始新的任务。'
            elif state.get('stage') == 'preflight':
                result['message'] = '无法处理此视频。请确认文件可以正常播放，并使用常见视频格式。'
            else:
                result['message'] = '视频处理未完成，可以重新上传重试。原始记录已保留。'
        return result

    def listing(self):
        with self.lock:
            identifiers = sorted(self.jobs, key=lambda i: self.jobs[i]['created_at_utc'], reverse=True)
            active = self.active
        return {'jobs': [self.summary(i) for i in identifiers[:20]], 'active_id': active}

    def review(self, kind, identifier):
        key = (kind, identifier)
        with self.lock:
            if key in self.exports:
                return self.exports[key]
            if kind == 'examples':
                root = self.examples[identifier]
            else:
                if self.jobs[identifier]['status'] != 'ready_for_review':
                    raise ValueError('此视频尚未处理完成。')
                root = self.root/identifier/'output/review'
            exports = ExportJobs(root, self.actor, backend=stable_review, work_lock=self.work_lock)
            self.exports[key] = exports
            return exports


class PortalHandler(WorkflowReviewHandler):
    def __init__(self, *args, portal, **kwargs):
        self.portal = portal
        self.review_prefix = None
        super().__init__(*args, jobs=None, directory=str(portal.root), **kwargs)

    def json_response(self, status, value):
        if self.review_prefix and isinstance(value, dict):
            value = dict(value)
            for key in ('montage_url', 'selection_url'):
                if value.get(key):
                    value[key] = self.review_prefix+value[key].lstrip('/')
        return super().json_response(status, value)

    def page(self, text):
        body = text.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def mount_review(self):
        match = re.fullmatch(r'/(jobs|examples)/([a-zA-Z0-9-]+)/review/(.*)', urlsplit(self.path).path)
        if not match:
            return False
        kind, identifier, relative = match.groups()
        if kind == 'jobs' and not re.fullmatch('[0-9a-f]{32}', identifier):
            self.send_error(404); return True
        try:
            self.jobs = self.portal.review(kind, identifier)
        except (KeyError, ValueError, OSError):
            self.send_error(404); return True
        self.directory = str(self.jobs.root)
        self.review_prefix = f'/{kind}/{identifier}/review/'
        relative = unquote(relative)
        if '..' in Path(relative).parts or '\\' in relative or relative.startswith('/'):
            self.send_error(404); return True
        if self.command == 'POST':
            if relative not in ('api/export', 'api/players/start', 'api/players/save', 'api/players/export'):
                self.send_error(404); return True
            self.path = '/'+relative
            super().do_POST()
        elif relative == 'preview-media':
            self.path = '/preview-media'
            if self.command == 'HEAD':
                super().do_HEAD()
            else:
                super().do_GET()
        elif relative in ('', 'index.html'):
            text = stable_review.render_review(self.jobs.manifest, self.jobs.digest)
            text = text.replace("fetch('/api/status'", "fetch('api/status'")
            text = text.replace("fetch('/api/export'", "fetch('api/export'")
            label = '示例结果 · ' if kind == 'examples' else ''
            text = text.replace('<header>', '<header><p><a href="/">← 上传其他视频</a></p>')
            text = text.replace('<h1>', f'<h1>{label}', 1)
            self.page(text)
        elif relative in ('api/status', 'api/players/status'):
            self.path = '/'+relative; super().do_GET()
        elif (re.fullmatch(r'clips/[^/]+\.mp4', relative)
                or re.fullmatch(r'exports/[0-9a-f]{32}/(?:selection\.json|result/clips/[^/]+\.mp4)', relative)
                or re.fullmatch(r'players/[0-9a-f]{32}/images/e[0-9]{4}-t[0-9]{4}-[0-2]\.jpg', relative)):
            self.path = '/'+relative; super().do_GET()
        else:
            self.send_error(404)
        return True

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/':
            text = Path(__file__).with_name('upload_portal.html').read_text(encoding='utf-8')
            self.page(text.replace('/*EXAMPLE_ENABLED*/false', 'true' if 'indoor' in self.portal.examples else 'false'))
        elif path == '/api/jobs':
            self.json_response(200, self.portal.listing())
        elif re.fullmatch(r'/api/jobs/[0-9a-f]{32}', path):
            try:
                self.json_response(200, self.portal.summary(path.rsplit('/', 1)[-1]))
            except KeyError:
                self.json_response(404, {'error': '找不到这个任务。'})
        elif not self.mount_review():
            self.send_error(404)

    def do_HEAD(self):
        if urlsplit(self.path).path.endswith('/review/preview-media') and self.mount_review():
            return
        self.send_error(405)

    def do_POST(self):
        allowed_hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        host, origin = self.headers.get('Host'), self.headers.get('Origin')
        if host not in allowed_hosts or (origin and origin != 'http://'+host):
            return self.json_response(403, {'error': '请从本机页面上传或操作。'})
        if self.mount_review():
            return
        if urlsplit(self.path).path != '/api/uploads':
            return self.json_response(404, {'error': '找不到此操作。'})
        identifier = None
        partial_path = None
        try:
            if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Type', '').split(';')[0] != 'application/octet-stream':
                raise ValueError('上传格式无效，请从页面重新选择视频。')
            size = int(self.headers.get('Content-Length', '0'))
            if size > MAX_UPLOAD:
                return self.json_response(413, {'error': '单个视频不能超过 8 GB。'})
            query = parse_qs(urlsplit(self.path).query)
            names = query.get('filename', [])
            if len(names) != 1:
                raise ValueError('缺少视频文件名。')
            identifier, partial_path = self.portal.reserve(names[0], size)
            self.connection.settimeout(60)
            remaining = size
            with partial_path.open('xb') as file:
                while remaining:
                    block = self.rfile.read(min(1024*1024, remaining))
                    if not block:
                        raise ValueError('视频上传不完整。')
                    file.write(block)
                    remaining -= len(block)
            self.portal.finish_upload(identifier, partial_path)
        except RuntimeError as error:
            if identifier:
                self.portal.upload_failed(identifier, error)
            return self.json_response(409, {'error': str(error)})
        except (ValueError, OSError) as error:
            if identifier:
                self.portal.upload_failed(identifier, error)
                if partial_path.exists():
                    partial_path.unlink()
            return self.json_response(400, {'error': str(error) if isinstance(error, ValueError) else '上传未完成，请重试。'})
        # A browser can close after the full transfer. Do not cancel accepted
        # processing or free its active slot just because the response is lost.
        try:
            self.json_response(202, self.portal.summary(identifier))
        except (BrokenPipeError, ConnectionResetError):
            pass


def make_portal(root, port=8894, processor=None, actor='user', examples=None):
    portal = UploadJobs(root, processor, actor, examples)
    server = ThreadingHTTPServer(('127.0.0.1', port), partial(PortalHandler, portal=portal))
    server.portal = portal
    return server
