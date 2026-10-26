"""Versioned, local-only player decisions on a fixed goal selection."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import uuid

from .media import export_clips, read_json, sha256, write_json
from .source_timeline import interval

ROOT = Path(__file__).resolve().parents[2]
METHOD = 'confirmed-clip-players-v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def replace_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def analyze(snapshot_path, folder):
    """Use a healthy installed runtime; no runtime installs or remote media."""
    from .analysis_launcher import select_python
    runtime, selection = select_python(ROOT)
    write_json(folder/'runtime.json', selection)
    with (folder/'analysis.log').open('x', encoding='utf-8') as log:
        process = subprocess.run([str(runtime), '-u', str(ROOT/'scripts/analyze_goal_players.py'),
            '--snapshot', str(snapshot_path), '--out', str(folder)], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise ValueError('人物匹配未完成。可重试，或继续导出全部进球。')
    return read_json(folder/'automatic.json')


class PlayerJobs:
    def __init__(self, exports, analyzer=None):
        self.exports = exports
        self.root = exports.root
        self.analyzer = analyzer or analyze
        self.current = None
        pointer = self.root/'players/current.json'
        if pointer.exists():
            identifier = read_json(pointer).get('analysis_id', '')
            if re.fullmatch('[0-9a-f]{32}', identifier):
                self.current = identifier
                state = self._state()
                if state['status'] == 'running':
                    replace_json(self.folder/'status.json', {**state, 'status': 'failed',
                        'error': '服务已重启，请重新匹配投篮人。'})

    @property
    def folder(self):
        return self.root/'players'/self.current

    def _state(self):
        return read_json(self.folder/'status.json') if self.current else {'status': 'idle'}

    def snapshot(self, selection):
        owner = self.exports
        if not isinstance(selection, dict):
            raise ValueError('请提交有效的进球选择。')
        selected = owner.backend.selection_events(owner.manifest, owner.original, selection,
                                                    owner.digest, owner.actor)
        if not selected['events']:
            raise ValueError('请先保留至少一个进球。')
        duration = owner.manifest['source']['duration_seconds']
        events = [{**event, 'segment_start_seconds': interval(event['time_seconds'], duration)[0],
                   'segment_end_seconds': interval(event['time_seconds'], duration)[1]}
                  for event in selected['events']]
        value = {'version': 'confirmed-selection-v1', 'method': METHOD,
            'method_sha256': {name: sha256(ROOT/name) for name in
                ('src/basketball_poc/player_analysis.py', 'src/basketball_poc/player_workflow.py',
                 'scripts/analyze_goal_players.py')},
            'source': owner.manifest['source'], 'source_sha256': owner.manifest['source_sha256'],
            'review_manifest_sha256': owner.digest, 'selection_actor': owner.actor,
            'selection': selection, 'selected_events': {**selected, 'events': events}}
        return {**value, 'snapshot_sha256': fingerprint(value)}

    def status(self):
        with self.exports.lock:
            state = self._state()
            if self.current:
                snapshot = read_json(self.folder/'confirmed-selection.json')
                state.update(analysis_id=self.current, snapshot_sha256=snapshot['snapshot_sha256'],
                             selection=snapshot['selection'])
                progress = self.folder/'progress.json'
                if state['status'] == 'running' and progress.exists():
                    state['progress'] = read_json(progress)
                if state['status'] == 'completed':
                    state['automatic'] = read_json(self.folder/'automatic.json')
                    state['decisions'] = self._decisions()
            return state

    def start(self, selection):
        snapshot = self.snapshot(selection)
        with self.exports.lock:
            if self.exports.state['status'] == 'running' or self._state()['status'] == 'running':
                raise RuntimeError('已有匹配或导出任务，请稍候。')
            if self.current and self._state()['status'] == 'completed':
                old = read_json(self.folder/'confirmed-selection.json')
                if old['snapshot_sha256'] == snapshot['snapshot_sha256']:
                    return {'status': 'completed', 'analysis_id': self.current}
            self.exports.claim_work()
            previous = self.current
            try:
                self.current = uuid.uuid4().hex
                self.folder.mkdir(parents=True, exist_ok=False)
                write_json(self.folder/'confirmed-selection.json', snapshot)
                write_json(self.folder/'status.json', {'status': 'running'})
                replace_json(self.root/'players/current.json', {'analysis_id': self.current})
                folder = self.folder
            except Exception:
                self.current = previous
                self.exports.work_lock.release()
                raise
        threading.Thread(target=self._run, args=(folder,), daemon=True).start()
        return {'status': 'running', 'analysis_id': folder.name}

    def _run(self, folder):
        try:
            result = self.analyzer(folder/'confirmed-selection.json', folder)
            if not (folder/'automatic.json').exists():
                write_json(folder/'automatic.json', result)
            # Suggestions are editable drafts, never human accuracy labels.
            decisions = {'analysis_id': folder.name, 'revision': 0, 'players': result['players'],
                'assignments': result['assignments'], 'skipped': [], 'actor': 'automatic_suggestion',
                'chosen_candidates': {event['id']: event['suggested_candidate_id'] for event in result['events']}}
            write_json(folder/'decisions/000000.json', decisions)
            replace_json(folder/'decision-head.json', {'revision': 0})
            state = {'status': 'completed'}
        except Exception as error:
            state = {'status': 'failed', 'error': str(error)}
        finally:
            with self.exports.lock:
                try:
                    replace_json(folder/'status.json', state)
                finally:
                    self.exports.work_lock.release()

    def _decisions(self):
        revision = read_json(self.folder/'decision-head.json')['revision']
        return read_json(self.folder/f'decisions/{revision:06d}.json')

    def validate_current(self, request):
        if not self.current or request.get('analysis_id') != self.current or self._state()['status'] != 'completed':
            raise ValueError('人物结果已过期，请返回确认进球并重新匹配。')
        snapshot = self.snapshot(request.get('selection', {}))
        saved = read_json(self.folder/'confirmed-selection.json')
        if snapshot['snapshot_sha256'] != saved['snapshot_sha256']:
            raise ValueError('进球选择已变化，请按新选择重新匹配投篮人。')
        return saved

    def save(self, request):
        with self.exports.lock:
            self.validate_current(request)
            previous = self._decisions()
            if type(request.get('revision')) is not int or request['revision'] != previous['revision']:
                raise ValueError('人物选择已有更新，请刷新后重试。')
            players, assignments, skipped = request.get('players'), request.get('assignments'), request.get('skipped')
            if (not isinstance(players, list) or len(players) > 1000 or any(not isinstance(p, dict) for p in players)
                    or not isinstance(assignments, dict) or not isinstance(skipped, list)
                    or any(not isinstance(key, str) for key in skipped)
                    or any(value is not None and not isinstance(value, str) for value in assignments.values())):
                raise ValueError('人物选择格式无效。')
            automatic = read_json(self.folder/'automatic.json')
            events = {event['id'] for event in automatic['events']}
            candidates = {candidate['id']: candidate for event in automatic['events'] for candidate in event['candidates']}
            chosen = request.get('chosen_candidates', {})
            event_candidates = {event['id']: {c['id'] for c in event['candidates']} for event in automatic['events']}
            if (not isinstance(chosen, dict) or set(chosen) != events
                    or any(value is not None and (not isinstance(value, str) or value not in event_candidates[key]) for key, value in chosen.items())):
                raise ValueError('投篮人画面与进球不匹配。')
            validated, identifiers = [], set()
            for player in players:
                identifier, name, representative = player.get('id'), player.get('name'), player.get('representative')
                if (not isinstance(identifier, str) or not re.fullmatch('p-[0-9a-f]{12}', identifier)
                        or identifier in identifiers or not isinstance(name, str) or not 1 <= len(name.strip()) <= 60
                        or any(ord(char) < 32 for char in name)
                        or (representative is not None and (not isinstance(representative, str) or representative not in candidates))):
                    raise ValueError('人物编号、名称或代表图片无效。')
                identifiers.add(identifier)
                validated.append({'id': identifier, 'name': name.strip(), 'representative': representative})
            if (set(assignments) != events or any(value is not None and value not in identifiers for value in assignments.values())
                    or len(set(skipped)) != len(skipped) or not set(skipped) <= events
                    or any(assignments[key] is not None for key in skipped)):
                raise ValueError('人物归属包含无效或重复的进球。')
            revision = previous['revision']+1
            result = {'analysis_id': self.current, 'revision': revision, 'players': validated,
                'assignments': assignments, 'skipped': skipped, 'actor': self.exports.actor,
                'chosen_candidates': chosen}
            write_json(self.folder/f'decisions/{revision:06d}.json', result)
            replace_json(self.folder/'decision-head.json', {'revision': revision})
            return result

    def export(self, request):
        with self.exports.lock:
            snapshot = self.validate_current(request)
            decisions = self._decisions()
            if request.get('revision') != decisions['revision'] or decisions['actor'] == 'automatic_suggestion':
                raise ValueError('请先核对并保存人物归属。')
            assignments = decisions['assignments']
            if any(value is None and key not in decisions['skipped'] for key, value in assignments.items()):
                raise ValueError('请指定剩余进球的人物，或明确选择暂不分配。')
            player = next((p for p in decisions['players'] if p['id'] == request.get('player_id')), None)
            if player is None:
                raise ValueError('请选择当前视频中的人物。')
            events = [event for event in snapshot['selected_events']['events'] if assignments[event['id']] == player['id']]
            if not events:
                raise ValueError('这位人物没有可导出的进球。')
            if self.exports.state['status'] == 'running' or self._state()['status'] == 'running':
                raise RuntimeError('已有匹配或导出任务，请稍候。')
            record = {'version': 'player-export-v1', 'analysis_id': self.current,
                'snapshot_sha256': snapshot['snapshot_sha256'], 'player': player, 'decisions': decisions,
                'selection': snapshot['selection'], 'source_sha256': snapshot['source_sha256'],
                'selected_events': {**snapshot['selected_events'], 'events': events},
                'skipped_count': len(decisions['skipped']), 'automatic_metrics_modified': False}
            self.exports.claim_work()
            try:
                folder = self.root/'exports'/uuid.uuid4().hex
                folder.mkdir(parents=True, exist_ok=False)
                write_json(folder/'selection.json', record)
            except Exception:
                self.exports.work_lock.release()
                raise
            self.exports.state = {'status': 'running', 'job_id': folder.name, 'kind': 'player'}
            state = dict(self.exports.state)
        threading.Thread(target=self._export, args=(folder, record), daemon=True).start()
        return state

    def _export(self, folder, record):
        try:
            manifest, _, digest = self.exports.backend.verify_review(self.root)
            if digest != record['selection']['review_manifest_sha256']:
                raise ValueError('Review identity changed')
            result = export_clips(manifest['source']['path'], record['selected_events'], folder/'result/clips')
            state = {'status': 'completed', 'kind': 'player', 'job_id': folder.name,
                'selection_actor': self.exports.actor, 'selected_count': len(record['selected_events']['events']),
                'selection': record['selection'], 'analysis_id': record['analysis_id'],
                'revision': record['decisions']['revision'], 'player_id': record['player']['id'],
                'player_name': record['player']['name'], 'skipped_count': record['skipped_count'],
                'montage_url': '/'+(folder/'result/clips'/result['montage']).relative_to(self.root).as_posix(),
                'selection_url': '/'+(folder/'selection.json').relative_to(self.root).as_posix()}
        except Exception as error:
            state = {'status': 'failed', 'kind': 'player', 'job_id': folder.name, 'error': str(error)}
        try:
            write_json(folder/'job.json', state)
        finally:
            with self.exports.lock:
                self.exports.state = state
                self.exports.work_lock.release()
