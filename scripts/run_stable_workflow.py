"""One local video -> stable analysis -> filtering -> review and optional serving."""
import argparse
from pathlib import Path

from basketball_poc.stable_pipeline import run
from basketball_poc import stable_review
from basketball_poc.workflow_server import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path, help='Local fixed-camera or minor-shake video with a visible net')
    parser.add_argument('--out', required=True, type=Path, help='New output directory; existing paths are never overwritten')
    parser.add_argument('--analysis-run', type=Path, help='Explicit reuse of a verified shake-wrapper run for this exact video')
    parser.add_argument('--serve', action='store_true', help='Serve the completed review pack locally')
    parser.add_argument('--port', type=int, default=8891)
    parser.add_argument('--processing-mode', choices=['reference', 'efficient', 'streaming'], default='reference')
    parser.add_argument('--verification-mode', choices=['full', 'standard'], default='full')
    parser.add_argument('--inference-device', choices=['auto', 'cpu', 'cuda:0'], default='cpu')
    parser.add_argument('--opencv-threads', type=int, choices=[0,1,2,4,8], default=1)
    parser.add_argument('--execution-mode', choices=['sequential','batched'], default='sequential')
    parser.add_argument('--batch-size', type=int, choices=[0,1,2,4,8,16,32], default=0)
    parser.add_argument('--net-mode', choices=['full','windows'], default='full')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    try:
        state = run(args.video, args.out, args.analysis_run, args.processing_mode, args.verification_mode,
                    args.inference_device, args.opencv_threads, args.net_mode, args.execution_mode, args.batch_size)
    except KeyboardInterrupt:
        print(f'处理已中断；已保留阶段记录：{args.out.resolve()}', flush=True)
        return 130
    except Exception as error:
        print(f'处理失败：{error}\n记录位置：{args.out.resolve()}', flush=True)
        return 1
    if state['status'] == 'unsupported':
        print(f'当前场景暂不支持：{state["reason"]}\n这不表示视频没有进球。', flush=True)
        return 2
    print(f'处理完成：自动计入 {state["counts"]["supported"]}，待确认 {state["counts"]["uncertain"]}。', flush=True)
    print(f'视频包：{state["review"]}\n阶段记录：{args.out.resolve()/"workflow.json"}', flush=True)
    print('使用方式：在本地页面完成待确认选择后生成集锦。', flush=True)
    if args.serve:
        try:
            with make_server(state['review'], args.port, 'user', backend=stable_review) as server:
                print(f'复核与导出：http://127.0.0.1:{server.server_port}/', flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
        except Exception as error:
            print(f'视频包已完成，但页面启动失败：{error}\n可以换一个端口单独启动已完成的视频包。', flush=True)
            return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
