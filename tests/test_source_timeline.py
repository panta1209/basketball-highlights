from pathlib import Path

import pytest

from basketball_poc.source_timeline import interval, compatibility, prepare_media, resource
from basketball_poc.media import command, inspect_video, sha256


@pytest.mark.parametrize('time', [-1, 3, float('nan'), float('inf'), True])
def test_invalid_event_time_is_rejected(time):
    with pytest.raises(ValueError): interval(time, 3)


def test_intervals_at_source_edges():
    assert interval(0, 20) == (0, 2)
    assert interval(19.5, 20) == (15.5,20)


@pytest.mark.parametrize('mode', ['transcode','remux'])
def test_one_compatible_preview_preserves_time_and_audio(tmp_path, mode):
    source = tmp_path/'source.mkv'
    arguments = ['-c:v','ffv1','-c:a','pcm_s16le'] if mode == 'transcode' else ['-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac']
    command(['ffmpeg','-v','error','-nostdin','-n','-f','lavfi','-i','testsrc2=s=160x90:r=10:d=2',
             '-f','lavfi','-i','sine=frequency=440:duration=2',*arguments,'-shortest',str(source)])
    digest = sha256(source); info = inspect_video(source)
    assert compatibility(source)['direct_mime'] is None
    out = tmp_path/'pack'; out.mkdir()
    preview = prepare_media(info,digest,out)
    assert preview['kind'] == 'compatible' and preview['conversion'] == mode
    assert len(list(out.glob('*.mp4'))) == 1
    result = inspect_video(out/'preview.mp4')
    assert result['audio_streams'] == 1 and abs(result['duration_seconds']-info['duration_seconds']) <= .1
    assert sha256(source) == digest
    assert resource(out, {'preview':preview})[0] == out/'preview.mp4'


def test_preview_resource_rejects_client_like_paths(tmp_path):
    with pytest.raises(ValueError):
        resource(tmp_path, {'preview':{'kind':'compatible','file':'../other.mp4'}})
