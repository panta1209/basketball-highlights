"""Build an inspectable Windows x64 folder with its own CPU Python runtime."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def dependencies(roots):
    found, pending = {}, list(roots)
    while pending:
        name = canonicalize_name(pending.pop())
        if name in found:
            continue
        dist = metadata.distribution(name)
        found[name] = dist
        for text in dist.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({'extra':''}):
                pending.append(requirement.name)
    return found


def application_files():
    # Source-only explicit surfaces: no videos, private notes, run logs or Git history.
    paths = set()
    for pattern in ('src/basketball_poc/*.py','src/basketball_poc/*.html','src/basketball_poc/*.js'):
        paths.update(ROOT.glob(pattern))
    baseline = json.loads((ROOT/'configs/stable-workflow-baseline.json').read_text(encoding='utf-8'))
    profile = json.loads((ROOT/'configs/stable-video-workflow-v1.json').read_text(encoding='utf-8'))
    paths.update(ROOT/name for name in baseline['files_sha256'])
    paths.update(ROOT/name for name in profile['additional_files_sha256'])
    paths.update(ROOT/name for name in (
        'configs/stable-workflow-baseline.json','configs/stable-video-workflow-v1.json',
        'configs/video-processing-performance-v1.json','scripts/run_stable_workflow.py',
        'scripts/analyze_video_efficient.py','scripts/analyze_video_streaming.py',
        'scripts/verify_video_standard.py','scripts/verify_stream_analysis.py',
        'scripts/run_windows_app.py','scripts/run_portable_postprocess.py','scripts/serve_video_upload.py',
        'scripts/benchmark_automatic_scale.py','scripts/benchmark_fixed_camera_scale.py',
        'scripts/benchmark_fixed_camera_net.py'))
    return sorted(paths)


def build(output, python_home, ffmpeg_home, version, source_commit=None, license_sources=None):
    if sys.platform != 'win32' or sys.maxsize <= 2**32:
        raise ValueError('Windows x64 build runtime required')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    app, runtime = output/'app', output/'runtime'
    shutil.copytree(python_home, runtime, ignore=shutil.ignore_patterns('__pycache__','site-packages','Scripts','include','libs'))
    site = runtime/'Lib/site-packages'; site.mkdir(parents=True, exist_ok=True)
    packages = dependencies(['ultralytics','opencv-python','dill','psutil'])
    package_rows = []
    for name, dist in sorted(packages.items()):
        copied = 0
        for entry in dist.files or []:
            if entry.suffix == '.pyc' or '__pycache__' in entry.parts or '..' in entry.parts:
                continue
            source = Path(dist.locate_file(entry))
            if not source.is_file():
                continue
            target = site/str(entry)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target); copied += 1
        package_rows.append({'name':name,'version':dist.version,'files':copied,
                             'license':dist.metadata.get('License-Expression') or dist.metadata.get('License') or '',
                             'project_urls':dist.metadata.get_all('Project-URL') or []})
    (runtime/'python312._pth').write_text('Lib\nDLLs\nLib/site-packages\n../app/src\n../app/scripts\nimport site\n',encoding='utf-8')
    for source in application_files():
        target = app/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(source,target)
    tools = output/'tools';tools.mkdir()
    for name in ('ffmpeg.exe','ffprobe.exe'):
        shutil.copy2(ffmpeg_home/'bin'/name,tools/name)
    licenses = output/'licenses';licenses.mkdir()
    for name in ('LICENSE','README.txt'):
        shutil.copy2(ffmpeg_home/name, licenses/('ffmpeg-'+name))
    if license_sources:
        shutil.copytree(license_sources, licenses/'sources')
    if (ROOT/'THIRD_PARTY_NOTICES.md').is_file():
        shutil.copy2(ROOT/'THIRD_PARTY_NOTICES.md',output/'THIRD_PARTY_NOTICES.md')
    if (ROOT/'LICENSE').is_file():shutil.copy2(ROOT/'LICENSE',output/'LICENSE')
    if (ROOT/'docs/windows-v0.1.md').is_file():shutil.copy2(ROOT/'docs/windows-v0.1.md',output/'使用说明.md')
    (licenses/'python-packages.json').write_text(json.dumps(package_rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    csc = Path(__import__('os').environ['WINDIR'])/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    subprocess.run([str(csc),'/nologo','/target:winexe','/platform:x64',
                    '/reference:System.Windows.Forms.dll','/reference:System.Drawing.dll',
                    '/out:'+str(output/'BasketballHighlights.exe'),str(ROOT/'apps/windows/Launcher.cs')],check=True)
    revision = source_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest = {'version':'windows-portable-v1','app_version':version,'source_commit':revision,
                'files_sha256':{p.relative_to(app).as_posix():digest(p) for p in app.rglob('*') if p.is_file()},
                'tools_sha256':{p.name:digest(p) for p in tools.iterdir()}}
    (app/'release-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'packages':len(packages),'application_files':len(manifest['files_sha256'])}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--python-home',type=Path,default=Path(sys.base_prefix))
    parser.add_argument('--ffmpeg-home',type=Path,required=True)
    parser.add_argument('--version',default='0.1.0-preview.1')
    parser.add_argument('--source-commit')
    parser.add_argument('--license-sources',type=Path,help='Corresponding source archives and build scripts to include')
    args=parser.parse_args()
    build(args.out,args.python_home,args.ffmpeg_home,args.version,args.source_commit,args.license_sources)
