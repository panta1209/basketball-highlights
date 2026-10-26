"""Pinned upstream class contract, separate from generic checkpoint names."""
import ast
import hashlib
from pathlib import Path

from .media import sha256
from .named_relation import PROFILE as BASE_PROFILE

REVISION = 'b72b7875764a86c68e147850c6eca3d8521c0e41'
MODEL_SHA = 'ce7d916271ac87f204ef2829ada25f0565b2ad3017dadbdd27756c1467b3381c'
MODEL_BLOB = '28dbcd94ffc2923fe0608c8c0bc58fc09bfd3dd0'
MODEL_SIZE = 44056729
RUN_NAME = 'yolo26m_5classes_2'
UPSTREAM_NAMES = {0: 'Ball', 1: 'Ball in Basket', 2: 'Player', 3: 'Basket', 4: 'Player Shooting'}
NAMES = {0: 'ball', 1: 'ball-in-basket', 2: 'player', 3: 'basket', 4: 'player-shooting'}
PROFILE = {**BASE_PROFILE, 'version': 'basketball-ai-relation-probe-v1',
           'label_identity': 'pinned_upstream_program_contract_and_checkpoint_training_run',
           'native_postprocess': 'YOLO26 end-to-end; iou does not apply if end2end=True'}
FILES = {
    'BE/live_test.py': ('3b35a55fce783fbc8fbf0d75539e49684f9368c1', 6268),
    'BE/app.py': ('d53f0cc747efa0b6951450532b18b4a069e09e82', 59052),
    'BE/README.md': ('2ee1a5640c60d10291fe724cc31c44171b8c4c05', 3973),
}


def verify_blob(data, digest, size):
    actual = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
    if len(data) != size or actual != digest:
        raise ValueError('Pinned Git blob mismatch')


def class_contract(source):
    """Read literal constants without executing the upstream application."""
    assignments = {n.targets[0].id: n.value for n in ast.parse(source).body
                   if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    names = ast.literal_eval(assignments['CLASSES'])
    if names != UPSTREAM_NAMES:
        raise ValueError('Upstream five-class contract changed')
    # This path is the upstream intended model path, not a byte-equivalence claim.
    fragments = [n.value for n in ast.walk(assignments['MODEL_PATH']) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    if fragments != ['best.pt', 'weights', RUN_NAME, 'basketball_training']:
        raise ValueError('Upstream model training run changed')
    return names


def validate_model_identity(model, provenance):
    embedded = {int(k): v for k, v in model.names.items()}
    if embedded != {i: str(i) for i in range(5)}:
        raise ValueError('Expected original generic checkpoint names')
    args = model.ckpt.get('train_args', {})
    if args.get('name') != RUN_NAME or f'/{RUN_NAME}/weights/' not in args.get('model', ''):
        raise ValueError('Checkpoint belongs to another training run')
    if args.get('data') != '/home/diego/Basketball_AI/BE/basketball-detection-srfkd-1/data.yaml':
        raise ValueError('Checkpoint training dataset identity changed')
    if provenance['sha256'] != MODEL_SHA or provenance['revision'] != REVISION:
        raise ValueError('Checkpoint/provenance mismatch')
    if {int(k): v for k, v in provenance['names'].items()} != NAMES:
        raise ValueError('Declared canonical class contract changed')
    if {int(k): v for k, v in provenance['upstream_names'].items()} != UPSTREAM_NAMES:
        raise ValueError('Upstream names changed')
    for entry in provenance['source_files']:
        path = Path(entry['local_path'])
        if sha256(path) != entry['sha256']:
            raise ValueError('Source contract file changed')
        digest, size = FILES[entry['upstream_path']]
        verify_blob(path.read_bytes(), digest, size)
    if {e['upstream_path'] for e in provenance['source_files']} != set(FILES):
        raise ValueError('Incomplete source contract evidence')
    live = next(e for e in provenance['source_files'] if e['upstream_path'] == 'BE/live_test.py')
    class_contract(Path(live['local_path']).read_text(encoding='utf-8'))
    return NAMES.copy()
