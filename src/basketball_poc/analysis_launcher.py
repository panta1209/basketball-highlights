"""Select a healthy installed analysis runtime; never install GPU drivers or packages."""
import json
from pathlib import Path
import subprocess
import sys


def select_python(root, explicit=None, device='auto'):
    if explicit is not None:
        return Path(explicit).resolve(strict=True), {'mode': 'explicit', 'probes': []}
    if device != 'auto':
        return Path(sys.executable), {'mode': 'current', 'probes': []}
    root = Path(root)
    candidates = [root/'.venv-gpu/Scripts/python.exe', Path(sys.executable), root/'.venv/Scripts/python.exe']
    seen, probes, usable = set(), [], []
    code = ('import json,torch; '
            'print(json.dumps({"cuda_available":torch.cuda.is_available(),"torch":torch.__version__}))')
    for path in candidates:
        if not path.is_file() or path.resolve() in seen:
            continue
        path = path.resolve()
        seen.add(path)
        try:
            process = subprocess.run([str(path), '-c', code], capture_output=True, text=True, timeout=30)
            data = json.loads(process.stdout.strip().splitlines()[-1]) if process.returncode == 0 else {}
            healthy = process.returncode == 0 and 'cuda_available' in data
            probes.append({'python':str(path), 'healthy':healthy, **data})
            if healthy:
                usable.append(path)
                if data['cuda_available']:
                    return path, {'mode':'auto', 'probes':probes}
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as error:
            probes.append({'python':str(path), 'healthy':False, 'error':type(error).__name__})
    if usable:
        cpu_runtime=(root/'.venv/Scripts/python.exe').resolve()
        preferred=cpu_runtime if cpu_runtime in usable else Path(sys.executable).resolve()
        return preferred if preferred in usable else usable[0], {'mode':'auto-cpu', 'probes':probes}
    raise ValueError('No usable installed PyTorch runtime; see installation instructions')
