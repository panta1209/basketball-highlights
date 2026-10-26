import json
from pathlib import Path
import shutil
import threading
import time
from urllib.request import Request,urlopen
from urllib.error import HTTPError

import pytest

from basketball_poc import stable_review as backend
from basketball_poc.media import command,inspect_video,read_json,write_json,sha256
from basketball_poc.workflow_server import make_server


@pytest.fixture(scope='module')
def pack(tmp_path_factory):
    root=tmp_path_factory.mktemp('stable-review');video=root/'source.mp4'
    command(['ffmpeg','-v','error','-nostdin','-n','-f','lavfi','-i','testsrc2=s=160x90:r=10:d=5',
             '-f','lavfi','-i','sine=frequency=440:duration=5','-c:v','libx264','-pix_fmt','yuv420p',
             '-c:a','aac','-shortest',str(video)])
    analysis=root/'analysis';triage=root/'triage'
    events={'source_sha256':sha256(video),'events':[{'id':f'event-{i}','time_seconds':i+.5} for i in range(4)]}
    write_json(analysis/'review-events.json',events)
    write_json(analysis/'candidates/manifest.json',{'source':inspect_video(video),'source_sha256':sha256(video)})
    for folder in [analysis,triage]:
        for kind,indices in [('supported',[0]),('uncertain',[1,2,3] if folder==analysis else [1,2]),('excluded',[3])]:
            write_json(folder/f'{kind}-events.json',{**events,'events':[events['events'][i] for i in indices]})
    write_json(triage/'results.json',{'status':'completed','supported_bytes_unchanged':True,'analysis':str(analysis)})
    write_json(triage/'frozen.json',{'inputs_sha256':{str(p):sha256(p) for p in analysis.rglob('*.json')}})
    write_json(triage/'predictions-before-evaluation.json',{'files_sha256':{str(p):sha256(p) for p in triage.glob('*.json')}})
    out=root/'pack';backend.prepare(triage,out,preview_mode='clips')
    return out


@pytest.fixture(scope='module')
def timeline(pack, tmp_path_factory):
    target = tmp_path_factory.mktemp('timeline')/'pack'
    backend.prepare(read_json(pack/'review-manifest.json')['input_triage'], target)
    return target


def selection(pack,choices):
    return {'version':backend.SELECTION_VERSION,'review_manifest_sha256':sha256(pack/'review-manifest.json'),'choices':choices}


def test_default_and_manual_provenance(pack):
    m,e,h=backend.verify_review(pack)
    result=backend.selection_events(m,e,selection(pack,{'event-1':True,'event-2':False}),h)
    assert [x['id'] for x in result['events']]==['event-0','event-1']
    assert [x['selection_source'] for x in result['events']]==['automatic_default','user_choice']
    changed=backend.selection_events(m,e,selection(pack,{'event-0':False,'event-1':False,'event-2':True}),h,'integration_test')
    assert [x['id'] for x in changed['events']]==['event-2']
    assert changed['events'][0]['selection_source']=='integration_test'
    assert 'event-3' not in {c['id'] for c in m['candidates']}


@pytest.mark.parametrize('choices',[{}, {'event-1':True}, {'event-1':True,'event-2':False,'unknown':True}, {'event-1':1,'event-2':False}, {'event-1':True,'event-2':False,'event-3':True}])
def test_incomplete_or_invalid_choices_rejected(pack,choices):
    m,e,h=backend.verify_review(pack)
    with pytest.raises(ValueError):backend.selection_events(m,e,selection(pack,choices),h)


def test_wrong_pack_rejected(pack):
    m,e,h=backend.verify_review(pack)
    with pytest.raises(ValueError):backend.selection_events(m,e,selection(pack,{'event-1':True,'event-2':False}),'wrong')


def test_manual_goal_and_custom_ranges_export_without_changing_evidence(timeline, tmp_path):
    m, original, digest = backend.verify_review(timeline)
    before = (timeline/'original-events.json').read_bytes()
    choice = selection(timeline, {'event-1':False, 'event-2':False})
    choice.update(manual_events=[{'id':'manual-test', 'time_seconds':3.8, 'start_seconds':3.2, 'end_seconds':4.6}],
                  segment_overrides={'event-0':{'start_seconds':.2, 'end_seconds':1.1}})
    record = tmp_path/'selection.json'; write_json(record, choice)
    result = backend.apply_selection(timeline, record, tmp_path/'export')
    assert result['selected_count'] == 2
    output = read_json(tmp_path/'export/clips/export.json')
    assert [(c['start_seconds'],c['end_seconds']) for c in output['clips']] == [(.2,1.1),(3.2,4.6)]
    assert inspect_video(result['montage'])['audio_streams'] == 1
    assert abs(inspect_video(result['montage'])['duration_seconds']-2.3) < .25
    command(['ffmpeg','-v','error','-xerror','-i',result['montage'],'-f','null','-'])
    selected = read_json(tmp_path/'export/selected-events.json')
    assert selected['events'][1]['selection_source'] == 'user_added'
    assert selected['review']['automatic_metrics_modified'] is False
    assert (timeline/'original-events.json').read_bytes() == before
    backend.verify_review(timeline)


@pytest.mark.parametrize('change', [
    {'manual_events':[{'id':'manual-x','time_seconds':4,'start_seconds':3,'end_seconds':6}]},
    {'manual_events':[{'id':'../x','time_seconds':4,'start_seconds':3,'end_seconds':5}]},
    {'manual_events':[{'id':'manual-x','time_seconds':float('nan'),'start_seconds':3,'end_seconds':5}]},
    {'manual_events':[{'id':'manual-x','time_seconds':True,'start_seconds':0,'end_seconds':5}]},
    {'manual_events':[{'id':'manual-x','time_seconds':.5,'start_seconds':0,'end_seconds':1}]},
    {'segment_overrides':{'event-0':{'start_seconds':1,'end_seconds':2}}},
    {'segment_overrides':{'event-3':{'start_seconds':0,'end_seconds':5}}},
])
def test_invalid_manual_decisions_rejected_before_export(timeline, change):
    m, original, digest = backend.verify_review(timeline)
    choice = {**selection(timeline, {'event-1':False,'event-2':False}), **change}
    with pytest.raises(ValueError):backend.selection_events(m,original,choice,digest)


@pytest.mark.parametrize('kind',['clip','category','snapshot'])
def test_changed_evidence_rejected(pack,tmp_path,kind):
    copy=tmp_path/'pack';shutil.copytree(pack,copy,ignore=shutil.ignore_patterns('exports'))
    m=read_json(copy/'review-manifest.json')
    if kind=='clip':
        with (copy/m['candidates'][0]['clip']).open('ab') as f:f.write(b'changed')
    elif kind=='snapshot':
        (copy/'original-events.json').write_text('{}')
    else:
        m['candidates'][0]['category']='excluded'
        (copy/'review-manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):backend.verify_review(copy)


def test_http_real_export_and_empty_output(pack):
    server=make_server(pack,0,'integration_test',backend=backend)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def post(choice):
        return urlopen(Request(base+'/api/export',json.dumps(selection(pack,choice)).encode(),headers={'Content-Type':'application/json'}),timeout=15)
    def finish():
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            with urlopen(base+'/api/status') as response:state=json.load(response)
            if state['status']!='running':return state
            time.sleep(.05)
        raise AssertionError('Export timeout')
    try:
        with pytest.raises(HTTPError) as e:post({})
        assert e.value.code==400
        with post({'event-1':True,'event-2':False}) as response:assert response.status==202
        state=finish();assert state['status']=='completed' and state['selected_count']==2
        path=pack/state['montage_url'].lstrip('/')
        assert inspect_video(path)['audio_streams']==1
        command(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'])
        with urlopen(Request(base+state['montage_url'],headers={'Range':'bytes=0-127'})) as r:assert r.status==206 and len(r.read())==128
        with post({f'event-{i}':False for i in range(3)}) as response:assert response.status==202
        empty=finish();assert empty['status']=='completed' and empty['montage_url'] is None
        assert backend.verify_review(pack)[0]['automatic_metrics_modified'] is False
    finally:
        server.shutdown();thread.join();server.server_close()


def test_timeline_has_no_duplicate_media_and_preserves_events(timeline, pack):
    new, events, _ = backend.verify_review(timeline)
    old, old_events, _ = backend.verify_review(pack)
    assert events == old_events
    assert new['version'] == backend.VERSION and new['preview']['kind'] == 'original'
    assert not list(timeline.glob('*.mp4')) and not (timeline/'clips').exists()
    assert not (timeline/'exports').exists()
    for a, b in zip(new['candidates'], old['candidates'], strict=True):
        assert (a['id'],a['time_seconds'],a['category']) == (b['id'],b['time_seconds'],b['category'])
        assert a['segment_start_seconds'] == max(0, a['time_seconds']-4)
        assert a['segment_end_seconds'] == min(5, a['time_seconds']+2)
    assert 'source_segments' in (timeline/'index.html').read_text(encoding='utf-8')


def test_timeline_source_ranges_head_and_export(timeline):
    server = make_server(timeline, 0, 'integration_test', backend=backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    source = Path(read_json(timeline/'review-manifest.json')['source']['path'])
    try:
        with urlopen(Request(base+'/preview-media', method='HEAD')) as response:
            assert response.status == 200 and int(response.headers['Content-Length']) == source.stat().st_size
            assert not response.read() and response.headers['Accept-Ranges'] == 'bytes'
        for header, expected in [('bytes=10-25',source.read_bytes()[10:26]), ('bytes=-16',source.read_bytes()[-16:])]:
            with urlopen(Request(base+'/preview-media', headers={'Range':header})) as response:
                assert response.status == 206 and response.read() == expected
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base+'/preview-media', headers={'Range':'bytes=99999999-'}))
        assert error.value.code == 416
        m, original, digest = backend.verify_review(timeline)
        choice = selection(timeline, {'event-1':True,'event-2':False})
        selected = backend.selection_events(m, original, choice, digest, 'integration_test')
        old = read_json(Path(m['input_triage'])/'supported-events.json')['events']
        assert selected['events'][0]['id'] == old[0]['id']
        export = timeline.parent/'chosen-export'
        record = timeline.parent/'choices.json'; write_json(record, choice)
        result = backend.apply_selection(timeline, record, export, actor='integration_test')
        assert result['selected_count'] == 2 and inspect_video(result['montage'])['audio_streams'] == 1
        command(['ffmpeg','-v','error','-xerror','-i',result['montage'],'-f','null','-'])
    finally:
        server.shutdown(); thread.join(); server.server_close()


def test_timeline_modified_interval_rejected(timeline, tmp_path):
    target = tmp_path/'pack'; shutil.copytree(timeline,target)
    manifest = read_json(target/'review-manifest.json')
    manifest['candidates'][0]['segment_end_seconds'] = 999
    (target/'review-manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError,match='Segment boundaries'):
        backend.verify_review(target)


def test_legacy_dynamic_page_keeps_manifest_and_selection_identity(pack):
    manifest, _, digest = backend.verify_review(pack)
    before = (pack/'review-manifest.json').read_bytes()
    page = backend.render_review(manifest,digest)
    assert 'source_segments' in page and digest in page
    assert (pack/'review-manifest.json').read_bytes() == before
    assert backend.verify_review(pack)[2] == digest
    assert 'const PLAYERS_ENABLED=false;' in page
    assert 'const PLAYERS_ENABLED=true;' in backend.render_review(manifest, digest, include_players=True)
