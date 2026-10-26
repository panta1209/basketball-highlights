"""Supply verified release provenance to unchanged research postprocessors."""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from basketball_poc.release_integrity import verify_release

ROOT = Path(__file__).resolve().parents[1]
STAGES = {'triage_fixed_camera_review.py':'review-triage-v1.json',
          'filter_rebound_review.py':'rebound-review-v1.json'}


def run(stage, source, output):
    verify_release(ROOT)
    def provenance(arguments, **kwargs):
        manifest = verify_release(ROOT)
        if arguments == ['git', 'rev-parse', 'HEAD']:
            return manifest['source_commit']+'\n'
        if arguments[:4] == ['git', 'status', '--porcelain', '--']:
            for name in arguments[4:]:
                path = Path(name).resolve()
                if not path.is_relative_to(ROOT) or path.relative_to(ROOT).as_posix() not in manifest['files_sha256']:
                    raise ValueError('Postprocessor outside the verified release')
            return ''
        raise ValueError('Unsupported provenance query')
    spec = importlib.util.spec_from_file_location('portable_postprocessor', ROOT/'scripts'/stage)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only the legacy module's provenance calls are adapted. No Git shim is installed.
    module.subprocess = SimpleNamespace(check_output=provenance)
    module.run(source, output, ROOT/'configs'/STAGES[stage])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.stage, args.source, args.out)
