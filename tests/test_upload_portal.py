import json
from pathlib import Path
import socket
import threading
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

from basketball_poc import stable_review
from basketball_poc.media import command, inspect_video, read_json, write_json, sha256
from basketball_poc.upload_portal import UploadJobs, make_portal, MAX_UPLOAD, upload_name


def post(base, name, body, **headers):
    return urlopen(Request(base+'/api/uploads?filename='+quote(name), body,
        headers={'Content-Type': 'application/octet-stream', **headers}), timeout=10)


def json_get(url):
    with urlopen(url, timeout=10) as response:
        return json.load(response)


def wait_job(base, identifier):
    deadline = time.monotonic()+10
    while time.monotonic() < deadline:
        state = json_get(base+'/api/jobs/'+identifier)
        if state['status'] not in ('queued', 'running', 'uploading'):
            return state
        time.sleep(.05)
    raise AssertionError('Job did not finish')


@pytest.fixture
def server(tmp_path):
    servers = []
    def start(processor):
        app = make_portal(tmp_path/'jobs', 0, processor=processor, actor='integration_test')
        thread = threading.Thread(target=app.serve_forever, daemon=True)
        thread.start(); servers.append((app, thread))
        return app, f'http://127.0.0.1:{app.server_port}'
    yield start
    for app, thread in servers:
        app.shutdown(); app.server_close(); thread.join()


@pytest.fixture(scope='module')
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp('upload-media')/'原声 片段.mp4'
    command(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'lavfi', '-i', 'testsrc2=s=160x90:r=10:d=3',
             '-f', 'lavfi', '-i', 'sine=frequency=440:duration=3', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
             '-c:a', 'aac', '-shortest', str(path)])
    return path


def synthetic_process(video, output, log):
    # Synthetic routing/export fixture, never detector or physical-goal evidence.
    analysis, triage = output/'analysis', output/'triage'
    events = {'source_sha256': sha256(video), 'events': [{'id': 'auto', 'time_seconds': .5},
                                                       {'id': 'pending', 'time_seconds': 1.5}]}
    write_json(analysis/'review-events.json', events)
    write_json(analysis/'candidates/manifest.json', {'source': inspect_video(video), 'source_sha256': sha256(video)})
    for folder in (analysis, triage):
        for kind, subset in [('supported', events['events'][:1]), ('uncertain', events['events'][1:]), ('excluded', [])]:
            write_json(folder/f'{kind}-events.json', {**events, 'events': subset})
    write_json(triage/'results.json', {'status': 'completed', 'supported_bytes_unchanged': True, 'analysis': str(analysis)})
    write_json(triage/'frozen.json', {'inputs_sha256': {str(p): sha256(p) for p in analysis.rglob('*.json')}})
    write_json(triage/'predictions-before-evaluation.json', {'files_sha256': {str(p): sha256(p) for p in triage.glob('*.json')}})
    prepared = stable_review.prepare(triage, output/'review')
    write_json(output/'workflow.json', {'status': 'ready_for_review', 'stage': 'review', 'counts': prepared['counts']})


@pytest.mark.parametrize('name', ['../x.mp4', 'C:\\x.mp4', '/x.mp4', 'CON.mp4', 'a.mp4 ',
                                  'x.txt', 'x.html', 'a\x00.mp4', 'a:stream.mp4', ''])
def test_unsafe_names_rejected(name):
    with pytest.raises(ValueError): upload_name(name)


def test_home_is_file_picker_and_private_files_are_not_served(server):
    _, base = server(lambda *_: None)
    with urlopen(base) as response: page = response.read().decode()
    assert 'type="file"' in page and '开始分析' in page
    assert 'const exampleEnabled=false' in page and '/*REVIEW_DATA*/' not in page
    assert json_get(base+'/api/jobs') == {'jobs': [], 'active_id': None}
    for path in ['/job.json', '/input/video.mp4', '/../../README.md', '/api/jobs/unknown']:
        with pytest.raises(HTTPError) as error: urlopen(base+path)
        assert error.value.code == 404


def test_empty_large_and_cross_origin_requests_do_not_start_work(server):
    app, base = server(lambda *_: pytest.fail('Invalid request processed'))
    for data, headers, expected in [(b'', {}, 400), (b'x', {'Content-Length': str(MAX_UPLOAD+1)}, 413),
                                    (b'x', {'Origin': 'https://other.example'}, 403),
                                    (b'x', {'Host': 'other.example'}, 403)]:
        with pytest.raises(HTTPError) as error: post(base, 'video.mp4', data, **headers)
        assert error.value.code == expected
    assert not app.portal.jobs


def test_overlapping_upload_rejected_and_failed_upload_releases_slot(server):
    app, base = server(lambda *_: None)
    first, _ = app.portal.reserve('first.mp4', 100)
    with pytest.raises(HTTPError) as error: post(base, 'next.mp4', b'x')
    assert error.value.code == 409
    app.portal.upload_failed(first, 'fixture failure')
    second, _ = app.portal.reserve('next.mp4', 1)
    assert second != first


def test_interrupted_request_is_not_analyzed(server):
    app, base = server(lambda *_: pytest.fail('Partial upload processed'))
    with socket.create_connection(('127.0.0.1', app.server_port), timeout=5) as connection:
        request = f'POST /api/uploads?filename=clip.mp4 HTTP/1.0\r\nHost: 127.0.0.1:{app.server_port}\r\nContent-Type: application/octet-stream\r\nContent-Length: 100\r\n\r\nx'
        connection.sendall(request.encode()); connection.shutdown(socket.SHUT_WR)
        assert b'400' in connection.recv(1000)
    job = json_get(base+'/api/jobs')['jobs'][0]
    assert job['status'] == 'failed' and 'review_url' not in job
    assert app.portal.active is None


@pytest.mark.parametrize('terminal', ['failed', 'unsupported'])
def test_failed_or_unsupported_analysis_has_no_review_link(server, terminal):
    def process(video, output, log):
        if terminal == 'failed': raise ValueError('fixture invalid media')
        write_json(output/'workflow.json', {'status': 'unsupported', 'counts': None})
    _, base = server(process)
    with post(base, 'video.mp4', b'fixture') as response: job = json.load(response)
    result = wait_job(base, job['id'])
    assert result['status'] == terminal and result['counts'] is None and 'review_url' not in result
    with pytest.raises(HTTPError) as error: urlopen(base+f'/jobs/{job["id"]}/review/')
    assert error.value.code == 404


def test_restart_marks_incomplete_upload_interrupted(tmp_path):
    first = UploadJobs(tmp_path)
    identifier, _ = first.reserve('video.mp4', 100)
    restarted = UploadJobs(tmp_path)
    assert restarted.active is None
    assert restarted.summary(identifier)['status'] == 'interrupted'
    assert 'review_url' not in restarted.summary(identifier)


def test_upload_prefixed_review_ranges_export_and_isolation(server, video):
    app, base = server(synthetic_process)
    body = video.read_bytes()
    with post(base, video.name, body) as response: job = json.load(response)
    ready = wait_job(base, job['id'])
    assert ready['status'] == 'ready_for_review'
    saved = app.portal.root/job['id']/'input'/video.name
    assert sha256(saved) == sha256(video)
    prefix = ready['review_url']
    with urlopen(base+prefix) as response: page = response.read().decode()
    assert "fetch('api/status'" in page and "fetch('api/export'" in page
    assert '上传其他视频' in page
    assert json_get(base+prefix+'api/status')['status'] == 'idle'
    pack = app.portal.root/job['id']/'output/review'
    manifest = read_json(pack/'review-manifest.json')
    assert manifest['version'] == stable_review.VERSION and not (pack/'clips').exists()
    with urlopen(Request(base+prefix+'preview-media', headers={'Range': 'bytes=0-15'})) as response:
        assert response.status == 206 and response.read() == body[:16]
    with urlopen(Request(base+prefix+'preview-media', method='HEAD')) as response:
        assert response.status == 200 and int(response.headers['Content-Length']) == len(body)
    selection = {'version': stable_review.SELECTION_VERSION, 'review_manifest_sha256': sha256(pack/'review-manifest.json'),
                 'choices': {'pending': True}}
    with urlopen(Request(base+prefix+'api/export', json.dumps(selection).encode(),
                         headers={'Content-Type': 'application/json'})) as response: assert response.status == 202
    deadline = time.monotonic()+10
    while time.monotonic() < deadline:
        status = json_get(base+prefix+'api/status')
        if status['status'] != 'running': break
        time.sleep(.1)
    assert status['status'] == 'completed' and status['selected_count'] == 2
    assert status['montage_url'].startswith(prefix) and status['selection_url'].startswith(prefix)
    with urlopen(base+status['montage_url']) as response: assert response.status == 200
    with urlopen(base+status['selection_url']) as response: assert json.load(response) == selection
    for relative in ['../input/'+video.name, 'review-manifest.json', 'exports/', 'api/jobs']:
        with pytest.raises(HTTPError): urlopen(base+prefix+quote(relative, safe='/'))
    with post(base, video.name, body) as response: second = json.load(response)
    second = wait_job(base, second['id'])
    assert second['id'] != job['id'] and second['review_url'] != prefix
    assert json_get(base+second['review_url']+'api/status')['status'] == 'idle'
