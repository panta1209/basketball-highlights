"""Create a clean public source directory without research history or user assets."""
import argparse
from pathlib import Path
import shutil

from build_windows_portable import application_files, ROOT


def prepare(output):
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    files=set(application_files())
    files.update(ROOT/name for name in (
        'LICENSE','THIRD_PARTY_NOTICES.md','requirements-windows.lock.txt','pyproject.toml',
        'docs/windows-v0.1.md','apps/windows/Launcher.cs','scripts/build_windows_portable.py',
        'scripts/build_windows_media.sh','scripts/fetch_release_model.py','scripts/prepare_public_source.py',
        'tests/test_stable_review.py','tests/test_upload_portal.py','tests/test_stable_pipeline.py',
        'tests/test_analysis_launcher.py','tests/test_source_timeline.py','tests/test_release_integrity.py'))
    for source in sorted(files):
        if source.suffix=='.pt':continue
        target=output/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    shutil.copy2(ROOT/'docs/windows-v0.1.md',output/'README.md')
    (output/'.gitignore').write_text('.venv/\n__pycache__/\n.pytest_cache/\n*.egg-info/\ndist/\nruns/\n*.mp4\n*.mov\n*.mkv\n*.pt\n*.download\n',encoding='utf-8')
    # Frozen source hashes include original line endings, including vendored files.
    (output/'.gitattributes').write_text('* -text\n*.sh text eol=lf\n',encoding='utf-8',newline='\n')
    print(f'Prepared {len(files)} source paths in {output}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    prepare(parser.parse_args().out)
