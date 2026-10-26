"""Single-pack local review with durable choices and asynchronous media export."""
from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import uuid

from .media import read_json, write_json
from .review import verify_review, selection_events, apply_selection
from .review_server import ReviewHandler
from .source_timeline import serve_media


class ExportJobs:
    def __init__(self, root, actor='user', backend=None, work_lock=None):
        self.root = Path(root).resolve(strict=True)
        self.actor = actor
        self.backend = backend
        self.manifest, self.original, self.digest = (backend.verify_review if backend else verify_review)(self.root)
        if not self.manifest.get('browser_export_enabled'):
            raise ValueError('Browser export is not enabled for this pack')
        self.lock = threading.Lock()
        self.work_lock = work_lock if work_lock is not None else threading.Lock()
        self.state = {'status': 'idle', 'selection_actor': actor}
        exports = self.root/'exports'
        if exports.exists():
            records = sorted((p/'job.json' for p in exports.iterdir()
                if p.is_dir() and re.fullmatch('[0-9a-f]{32}', p.name) and (p/'job.json').is_file()),
                key=lambda p: p.stat().st_mtime_ns, reverse=True)
            for path in records:
                try:
                    saved = read_json(path)
                except (OSError, ValueError):
                    continue
                if (saved.get('status') == 'completed' and saved.get('selection_actor') == actor
                        and isinstance(saved.get('selection'), dict)
                        and saved['selection'].get('review_manifest_sha256') == self.digest):
                    self.state = saved
                    break
        self.players = None
        if backend and hasattr(backend, 'SELECTION_VERSION') and backend.SELECTION_VERSION == 'stable-selection-v2':
            from .player_workflow import PlayerJobs
            self.players = PlayerJobs(self)

    def status(self):
        with self.lock: return dict(self.state)

    def claim_work(self):
        if not self.work_lock.acquire(blocking=False):
            raise RuntimeError('已有视频分析、人物匹配或导出任务，请稍候。')

    def start(self, selection):
        # Validate before creating any job directory or changing current status.
        (self.backend.selection_events if self.backend else selection_events)(self.manifest, self.original, selection, self.digest, self.actor)
        with self.lock:
            if self.state['status'] == 'running': raise RuntimeError('Export already running')
            if self.players and self.players._state()['status'] == 'running':
                raise RuntimeError('人物正在匹配，请完成后再导出。')
            self.claim_work()
            try:
                job_id = uuid.uuid4().hex
                folder = self.root/'exports'/job_id
                folder.mkdir(parents=True, exist_ok=False)
                write_json(folder/'selection.json', selection)
            except Exception:
                self.work_lock.release()
                raise
            self.state = {'status': 'running', 'job_id': job_id, 'selection_actor': self.actor}
            state = dict(self.state)
        threading.Thread(target=self._export, args=(folder,), daemon=True).start()
        return state

    def _export(self, folder):
        try:
            result = (self.backend.apply_selection if self.backend else apply_selection)(self.root, folder/'selection.json', folder/'result', actor=self.actor)
            montage = Path(result['montage']).relative_to(self.root).as_posix() if result['montage'] else None
            state = {'status': 'completed', 'job_id': folder.name, 'selection_actor': self.actor,
                'selected_count': result['selected_count'], 'montage_url': '/'+montage if montage else None,
                'selection': read_json(folder/'selection.json'),
                'selection_url': '/'+(folder/'selection.json').relative_to(self.root).as_posix()}
        except Exception as error:
            state = {'status': 'failed', 'job_id': folder.name, 'selection_actor': self.actor, 'error': str(error)}
        try:
            write_json(folder/'job.json', state)
        finally:
            with self.lock:
                self.state = state
                self.work_lock.release()


class WorkflowReviewHandler(ReviewHandler):
    def __init__(self, *args, jobs, **kwargs):
        self.jobs = jobs
        super().__init__(*args, **kwargs)

    def json_response(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers();self.wfile.write(body)

    def do_GET(self):
        if self.path == '/api/status': return self.json_response(200, self.jobs.status())
        if self.path == '/api/players/status' and self.jobs.players:
            return self.json_response(200, self.jobs.players.status())
        if self.path == '/preview-media' and self.jobs.backend and hasattr(self.jobs.backend, 'preview_resource'):
            return self.preview_media()
        if self.path in ('/', '/index.html') and self.jobs.backend and hasattr(self.jobs.backend, 'render_review'):
            body = self.jobs.backend.render_review(self.jobs.manifest, self.jobs.digest).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body)
            return
        return super().do_GET()

    def preview_media(self, head=False):
        try:
            resource = self.jobs.backend.preview_resource(self.jobs.root, self.jobs.manifest)
            return serve_media(self, *resource, head=head)
        except (OSError, ValueError, KeyError):
            self.send_error(404)

    def do_HEAD(self):
        if self.path == '/preview-media' and self.jobs.backend and hasattr(self.jobs.backend, 'preview_resource'):
            return self.preview_media(head=True)
        return super().do_HEAD()

    def do_POST(self):
        player_routes = {'/api/players/start', '/api/players/save', '/api/players/export'}
        if self.path != '/api/export' and (self.path not in player_routes or not self.jobs.players):
            return self.json_response(404, {'error': 'Unknown endpoint'})
        # JSON-only same-origin endpoint; local media files are never client-specified.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://'+self.headers.get('Host', ''):
            return self.json_response(403, {'error': 'Origin mismatch'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.json_response(415, {'error': 'JSON required'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 1024*1024: raise ValueError('Invalid request size')
            selection = json.loads(self.rfile.read(length))
            if not isinstance(selection, dict): raise ValueError('Selection object required')
            if self.path == '/api/players/start':
                result = self.jobs.players.start(selection)
            elif self.path == '/api/players/save':
                result = self.jobs.players.save(selection)
            elif self.path == '/api/players/export':
                result = self.jobs.players.export(selection)
            else:
                result = self.jobs.start(selection)
            self.json_response(200 if self.path == '/api/players/save' else 202, result)
        except RuntimeError as error: self.json_response(409, {'error': str(error)})
        except (ValueError, KeyError, TypeError) as error: self.json_response(400, {'error': str(error)})
        except OSError as error: self.json_response(500, {'error': str(error)})


def make_server(root, port=8877, actor='user', backend=None):
    jobs = ExportJobs(root, actor, backend)
    return ThreadingHTTPServer(('127.0.0.1', port), partial(WorkflowReviewHandler, jobs=jobs, directory=str(jobs.root)))
