# 第三方组件与源码

本项目应用源码以 GNU AGPL-3.0 发布。第三方组件保留原作者版权和各自许可，不将其统一改为本项目许可。完整许可证随各 Python 包的 dist-info、cv2 和 Python 运行时一起提供。

## 模型

篮球与篮筐权重来自 josephattalla/Basketball-Shot-Detection，固定提交 e320817d0f87eceb4093c871de6d29d1adca0006。上游仓库声明 MIT；原 LICENSE、README、文件 URL 与 SHA-256 保存在 third_party 来源目录。模型未在本项目重新训练，未发布训练素材。Ultralytics 引擎及随附的 AGPL 组件按 AGPL-3.0 开源条件使用；MIT 来源说明不替代引擎的许可。

## 本地视频导出

Windows 便携包使用从源码构建的 FFmpeg 7.1.1（db69d06eeeab4f46da15030a80d539efb4503ca8）和 x264（373697b467f7cd0af88f1e9e32d4f10540df4687），GPL-2.0-or-later。只启用 x264 这一外部库，禁用网络和其它外部库自动探测。无源码修改。对应原始源码、构建脚本和配置保存在 licenses/sources。GCC/NASM 是构建工具，不随应用运行时分发。

## OpenCV 解码组件

OpenCV 的 Windows FFmpeg DLL 保持独立动态文件，可由用户替换。其 LGPL-2.1 文本随 cv2/LICENSE-3RD-PARTY.txt 提供。已核对 opencv_videoio_ffmpeg4140_64.dll 的 MD5 为 84757ed0f16ddedce99227529d3165e9，与 OpenCV 4.14.0 的 ffmpeg.cmake 一致。对应二进制提交为 bd9418020a5c342be979c56d6e6434261959d3af，完整对应源码和构建脚本来自 opencv/opencv_3rdparty 的 33d5aa904ed04725cedd042394227520e49db453（ffmpeg/4.x_20260715_src）。所需 sources 和 ffmpeg 构建脚本随 licenses/sources/opencv-ffmpeg 提供；上游的其它预编译 DLL 不重复分发。

## Python 与 Python 组件

Python 3.12.14 运行时保留 PSF 许可证（runtime/LICENSE.txt）。纯 Python 源码随运行时分发；AGPL 组件和 certifi 的原始发布源码另附于 licenses/sources。其它本地库的版权说明保留于包内，精确版本见 requirements-windows.lock.txt 和 licenses/python-packages.json。

| 组件 | 版本 | 许可元数据（完整条款见包内） |
|---|---|---|
| anyio | 4.15.1 | MIT |
| certifi | 2026.7.22 | MPL-2.0 |
| charset-normalizer | 3.5.2 | MIT |
| contourpy | 1.4.0 | BSD-3-Clause |
| cycler | 0.12.1 | Copyright (c) 2015, matplotlib project |
| dill | 0.4.1 | BSD-3-Clause |
| filelock | 4.0.12 | MIT |
| fonttools | 4.66.1 | MIT |
| fsspec | 2026.9.0 | BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| idna | 3.20 | BSD-3-Clause |
| jinja2 | 3.1.6 | 见包内 LICENSE |
| kiwisolver | 1.5.1 | BSD-3-Clause |
| markupsafe | 3.0.4 | BSD-3-Clause |
| matplotlib | 3.11.2 | License agreement for matplotlib versions 1.3.0 and later |
| mpmath | 1.3.0 | BSD |
| networkx | 3.7 | BSD-3-Clause |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| nvidia-ml-py | 13.615.71 | BSD |
| opencv-python | 4.14.0.94 | Apache 2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pillow | 12.3.0 | MIT-CMU |
| polars | 2.0.0 | Copyright (c) 2025 Ritchie Vink |
| polars-runtime-32 | 2.0.0 | 见包内 LICENSE |
| psutil | 7.2.2 | BSD-3-Clause |
| pyparsing | 3.3.3 | MIT |
| python-dateutil | 2.9.0.post0 | Dual License |
| pyyaml | 6.0.3 | MIT |
| requests | 2.34.2 | Apache-2.0 |
| setuptools | 84.0.0 | MIT |
| six | 1.17.0 | MIT |
| sympy | 1.14.0 | BSD |
| torch | 2.14.1 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| torchvision | 0.29.1 | BSD |
| typing-extensions | 4.16.0 | PSF-2.0 |
| ultralytics | 8.4.174 | AGPL-3.0 |
| ultralytics-platform | 0.1.82 | AGPL-3.0-only |
| ultralytics-thop | 2.2.2 | AGPL-3.0 |
| urllib3 | 2.8.0 | MIT |

## 来源地址

- Ultralytics: https://github.com/ultralytics/ultralytics
- FFmpeg: https://github.com/FFmpeg/FFmpeg/tree/db69d06eeeab4f46da15030a80d539efb4503ca8
- x264: https://code.videolan.org/videolan/x264 (源码镜像 https://github.com/mirror/x264/tree/373697b467f7cd0af88f1e9e32d4f10540df4687)
- OpenCV FFmpeg 源码: https://github.com/opencv/opencv_3rdparty/tree/33d5aa904ed04725cedd042394227520e49db453
- Python: https://github.com/python/cpython/tree/v3.12.14
- 模型: https://github.com/josephattalla/Basketball-Shot-Detection/tree/e320817d0f87eceb4093c871de6d29d1adca0006

下载源码文件的精确 URL 和 SHA-256 见 licenses/sources/sources.json。模型 SHA-256 见 third_party/josephattalla/e320817d0f87eceb4093c871de6d29d1adca0006/provenance.json。私人视频、标注、浏览器草稿、用户路径和任务日志不包含在公开源码或发行包中。
