"""Verify a portable distribution without requiring Git on the user's PC."""
import json
from pathlib import Path
from .media import sha256


def verify_release(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'release-manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != 'windows-portable-v1' or not manifest.get('source_commit'):
        raise ValueError('发行文件清单无效，请重新解压完整安装包。')
    files = manifest.get('files_sha256')
    if not isinstance(files, dict) or not files:
        raise ValueError('发行文件清单为空。')
    for name, digest in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError(f'发行文件缺失或已改变：{name}。请重新解压。')
    for name, digest in manifest.get('tools_sha256', {}).items():
        path = (root.parent/'tools'/name).resolve()
        if not path.is_relative_to(root.parent/'tools') or sha256(path) != digest:
            raise ValueError('视频处理组件已改变，请重新解压完整软件包。')
    return manifest
