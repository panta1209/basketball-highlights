"""Bundled Windows app service; all operations stay on loopback and local disk."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready-file', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--port', type=int, default=8899)
    args = parser.parse_args()
    data = args.data_dir or Path(os.environ['LOCALAPPDATA'])/'BasketballHighlights'
    data.mkdir(parents=True, exist_ok=True)
    # pythonw has no inherited streams. Explicit logging also preserves failures.
    log = (data/'app.log').open('a', encoding='utf-8', buffering=1)
    sys.stdout = sys.stderr = log
    try:
        from basketball_poc.release_integrity import verify_release
        manifest = verify_release(ROOT)
        tools = ROOT.parent/'tools'
        os.environ['PATH'] = str(tools)+os.pathsep+os.environ.get('PATH', '')
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.environ.pop('PYTHONPATH', None)
        os.environ['YOLO_CONFIG_DIR'] = str(data/'settings')
        os.environ['YOLO_OFFLINE'] = 'true'
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ['YOLO_VERBOSE'] = 'false'
        from basketball_poc.upload_portal import make_portal, process_video
        processor = partial(process_video, python_executable=Path(sys.executable), inference_device='cpu')
        # A stable browser origin preserves local review drafts across restarts.
        with make_portal(data/'jobs', args.port, processor=processor) as server:
            url = f'http://127.0.0.1:{server.server_port}/'
            temporary = args.ready_file.with_suffix('.tmp')
            temporary.write_text(url, encoding='utf-8')
            temporary.replace(args.ready_file)
            print(json.dumps({'version':manifest['app_version'], 'url':url}), flush=True)
            server.serve_forever()
    except Exception as error:
        traceback.print_exc()
        reason = '本机端口已被占用，请关闭已打开的软件后重试。' if getattr(error, 'winerror', None) == 10048 else '请重新解压完整软件包。'
        args.ready_file.with_suffix('.error').write_text('启动失败。'+reason+'详细记录：'+str(data/'app.log'), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
