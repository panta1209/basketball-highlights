import hashlib
import json

import pytest

from basketball_poc.release_integrity import verify_release


def test_portable_checks_fail_before_changed_model_or_tool_runs(tmp_path):
    app=tmp_path/'app';app.mkdir();tools=tmp_path/'tools';tools.mkdir()
    model=app/'model.pt';model.write_bytes(b'pinned weights')
    tool=tools/'ffmpeg.exe';tool.write_bytes(b'pinned media tool')
    manifest={'version':'windows-portable-v1','source_commit':'public-version',
              'files_sha256':{'model.pt':hashlib.sha256(model.read_bytes()).hexdigest()},
              'tools_sha256':{'ffmpeg.exe':hashlib.sha256(tool.read_bytes()).hexdigest()}}
    (app/'release-manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    assert verify_release(app)['source_commit']=='public-version'
    model.write_bytes(b'other weights')
    with pytest.raises(ValueError):verify_release(app)
    model.write_bytes(b'pinned weights');tool.write_bytes(b'other executable')
    with pytest.raises(ValueError):verify_release(app)
