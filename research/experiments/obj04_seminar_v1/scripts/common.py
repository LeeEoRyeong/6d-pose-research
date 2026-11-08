import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/RESEARCH_DATA/HCCEPose')
PBR=Path('/RESEARCH_DATA/HCCEPose_experiments/realistic_final100')
REAL=Path('/RESEARCH_DATA/HCCEPose_obj04_transfer')
SCENE=PBR/'train_pbr/000000'

def write_json(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def load_json(path):return json.loads(Path(path).read_text())

def original_function(path,dotted,namespace):
    tree=ast.parse((SOURCE/path).read_text())
    node=tree
    for name in dotted.split('.'):
        node=next(n for n in node.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
    env=dict(namespace)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE/path),'exec'),env)
    return env[node.name]

def verify_originals():
    manifest=load_json(ROOT/'checks/original_manifest.json')
    changed=[x['path'] for x in manifest if hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()!=x['sha256']]
    if changed:raise RuntimeError('원본 변경: '+str(changed))
    return len(manifest)
