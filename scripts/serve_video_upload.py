"""Start the local file-picker, processing and review portal."""
import argparse
from functools import partial
from pathlib import Path

from basketball_poc.upload_portal import make_portal, process_video

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'runs/local-video-upload')
    parser.add_argument('--port', type=int, default=8894)
    parser.add_argument('--example', type=Path, help='Optional verified pack behind a separately labeled example link')
    parser.add_argument('--integration-test', action='store_true')
    parser.add_argument('--analysis-python', type=Path, help='Optional isolated analysis runtime')
    parser.add_argument('--inference-device', choices=['auto', 'cpu', 'cuda:0'], default='auto')
    parser.add_argument('--opencv-threads', type=int, choices=[0,1,2,4,8], default=0)
    parser.add_argument('--net-mode', choices=['full','windows'], default='windows')
    parser.add_argument('--execution-mode', choices=['sequential','batched'], default='batched')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    examples = {'indoor': args.example} if args.example else {}
    runtime = args.analysis_python.resolve(strict=True) if args.analysis_python else None
    processor = partial(process_video, python_executable=runtime, inference_device=args.inference_device,
                        opencv_threads=args.opencv_threads, net_mode=args.net_mode, execution_mode=args.execution_mode)
    with make_portal(args.out, args.port, processor=processor,
                     actor='integration_test' if args.integration_test else 'user', examples=examples) as server:
        print(f'选择本地视频：http://127.0.0.1:{server.server_port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
