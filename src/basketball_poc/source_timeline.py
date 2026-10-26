"""Shared preview media and source-time intervals; export always uses the original."""
import json
import math
from pathlib import Path

from .media import command, inspect_video, sha256
from .review_server import byte_range


def interval(time, duration):
    if (isinstance(time, bool) or not isinstance(time, (int, float))
            or not math.isfinite(time) or not 0 <= time < duration):
        raise ValueError('Event outside original media')
    return max(0., time-4), min(duration, time+2)


def compatibility(source):
    raw = json.loads(command(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(source)]))
    video = next(s for s in raw['streams'] if s['codec_type'] == 'video' and not s.get('disposition', {}).get('attached_pic'))
    audio = next((s for s in raw['streams'] if s['codec_type'] == 'audio'), {})
    mp4 = (video.get('codec_name') == 'h264' and video.get('pix_fmt') in ('yuv420p', 'yuvj420p')
           and audio.get('codec_name') in (None, 'aac', 'mp3'))
    webm = (video.get('codec_name') in ('vp8', 'vp9') and video.get('pix_fmt') == 'yuv420p'
            and audio.get('codec_name') in (None, 'opus', 'vorbis'))
    suffix = Path(source).suffix.lower()
    mime = 'video/mp4' if mp4 and suffix in ('.mp4', '.m4v') else 'video/webm' if webm and suffix == '.webm' else None
    return {'direct_mime': mime, 'mp4_streamcopy': mp4,
            'video_codec': video.get('codec_name'), 'audio_codec': audio.get('codec_name')}


def prepare_media(info, digest, output):
    check = compatibility(info['path'])
    if check['direct_mime']:
        return {'kind': 'original', 'mime_type': check['direct_mime'], 'sha256': digest,
                'generated_media_bytes': 0, 'compatibility': check}
    target = Path(output)/'preview.mp4'
    arguments = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', info['path'],
                 '-map', '0:v:0', '-map', '0:a:0?', '-sn', '-dn']
    if check['mp4_streamcopy']:
        arguments += ['-c', 'copy']
    else:
        arguments += ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-preset', 'fast',
                      '-crf', '20', '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough',
                      '-c:a', 'aac', '-ar', '48000', '-ac', '2']
    command([*arguments, '-movflags', '+faststart', str(target)])
    converted = inspect_video(target)
    if (abs(converted['fps']-info['fps']) > 1e-6
            or abs(converted['duration_seconds']-info['duration_seconds']) > max(.05, 1/info['fps'])
            or bool(converted['audio_streams']) != bool(info['audio_streams'])
            or any(abs(converted[k]-info[k]) > 1 for k in ('width', 'height'))):
        raise ValueError('Compatible preview timebase/audio/dimensions changed')
    return {'kind': 'compatible', 'file': 'preview.mp4', 'mime_type': 'video/mp4',
            'sha256': sha256(target), 'generated_media_bytes': target.stat().st_size,
            'conversion': 'remux' if check['mp4_streamcopy'] else 'transcode', 'compatibility': check}


def resource(root, manifest):
    preview = manifest.get('preview')
    if preview:
        if preview['kind'] == 'original':
            return Path(manifest['source']['path']), preview['mime_type'], manifest['source_sha256']
        if preview['kind'] == 'compatible' and preview.get('file') == 'preview.mp4':
            return Path(root)/'preview.mp4', 'video/mp4', preview['sha256']
        raise ValueError('Invalid shared preview media')
    mime = compatibility(manifest['source']['path'])['direct_mime']
    if not mime:
        raise ValueError('Legacy source needs existing clip previews')
    return Path(manifest['source']['path']), mime, manifest['source_sha256']


def playback_data(manifest):
    if manifest.get('preview') or compatibility(manifest['source']['path'])['direct_mime']:
        return {'mode': 'source_segments', 'url': 'preview-media', 'segments': [
            {'id': c['id'], 'start': interval(c['time_seconds'], manifest['source']['duration_seconds'])[0],
             'end': interval(c['time_seconds'], manifest['source']['duration_seconds'])[1]}
            for c in manifest['candidates']]}
    return {'mode': 'clips'}


def serve_media(handler, path, mime, digest, head=False):
    """Only call with a verified manifest resource, never a browser-supplied path."""
    handler._byte_range = None
    size = path.stat().st_size
    header = handler.headers.get('Range')
    if header:
        try:
            handler._byte_range = byte_range(header, size)
        except ValueError:
            handler.send_response(416)
            handler.send_header('Content-Range', f'bytes */{size}')
            handler.send_header('Content-Length', '0'); handler.end_headers()
            return
    elif handler.headers.get('If-None-Match') == f'"{digest}"':
        handler.send_response(304); handler.end_headers()
        return
    with path.open('rb') as file:
        handler.send_response(206 if header else 200)
        handler.send_header('Content-Type', mime)
        handler.send_header('Accept-Ranges', 'bytes')
        handler.send_header('Cache-Control', 'private, max-age=86400')
        handler.send_header('ETag', f'"{digest}"')
        if header:
            start, end = handler._byte_range
            file.seek(start)
            handler.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            handler.send_header('Content-Length', str(end-start+1))
        else:
            handler.send_header('Content-Length', str(size))
        handler.end_headers()
        if not head:
            handler.copyfile(file, handler.wfile)
