import csv,json,hashlib,zipfile,tempfile,shutil,re
from pathlib import Path,PurePosixPath
from PIL import Image
P=Path(__file__).resolve().parent;K=P/'package';out=P/'BB100_pose_group_review_Windows_offline.zip'
def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
g=read(K/'group_metadata.csv');f=read(K/'frame_review_template.csv');m=read(K/'motion_diagnostics.csv');t=read(K/'group_review_template.csv');assert len(g)==len(t)==36 and len(f)==len(m)==100;assert all(x['frame_status']=='UNREVIEWED' and x['training_status']=='HOLD' for x in f);assert all(x['group_status']=='UNREVIEWED' and x['training_status']=='HOLD' for x in t)
source=(K/'index.html').read_text(encoding='utf8');match=re.search(r'const DATA=(.*?);const groups=DATA\.groups',source,re.S);assert match;payload=json.loads(match.group(1));assert len(payload['groups'])==36 and len(payload['frames'])==len(payload['metrics'])==100;assert '__DATA__' not in source;assert not re.search(r'<script[^>]+src=|<link[^>]+href=|https?://',source,re.I)
paths=[x['group_comparison_image'] for x in payload['groups']]+[x['phase12_frame_image'] for x in payload['frames']];assert len(paths)==len(set(paths))==136
for name in paths:
 x=PurePosixPath(name);assert not x.is_absolute() and '..' not in x.parts and '\\' not in name and (K/name).is_file()
 with Image.open(K/name) as im:im.verify()
# Original frame IDs and source assets must be preserved byte-for-byte.
assert {int(x['frame']) for x in f}=={int(x['frame']) for x in m};assert {int(x['pose_group']) for x in g}=={int(x['pose_group']) for x in t}
V=P.parent/'bb100_visual_review_20261109'
for x in f:assert sha(K/x['phase12_frame_image'])==sha(V/'comparisons'/f"{int(x['frame']):06d}.jpg")
# All original source hashes were captured before any generated outputs.
sourcehash=json.loads((P/'manifest.json').read_text())['source_sha256'];assert all(sha(Path(n))==h for n,h in sourcehash.items())
assets=sorted(x for x in K.rglob('*') if x.is_file());assert not (K/'MANIFEST.csv').exists()
with (K/'MANIFEST.csv').open('x',newline='',encoding='utf8') as file:
 w=csv.DictWriter(file,fieldnames=['relative_path','size_bytes','sha256']);w.writeheader();w.writerows([{'relative_path':x.relative_to(K).as_posix(),'size_bytes':x.stat().st_size,'sha256':sha(x)} for x in assets])
assert not out.exists()
with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
 for path in sorted(x for x in K.rglob('*') if x.is_file()):
  assert not path.is_symlink();z.write(path,path.relative_to(K).as_posix())
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None;names=z.namelist();assert len(names)==len(set(names));assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and '\\' not in n for n in names);assert all(n in names for n in paths);assert 'index.html' in names and 'MANIFEST.csv' in names
 with tempfile.TemporaryDirectory(prefix='bb100_zip_verify_') as tmp:
  z.extractall(tmp);root=Path(tmp);assert all((root/n).is_file() for n in paths);assert sha(root/'index.html')==sha(K/'index.html');
  for n in ['groups/000043.jpg','frames/000043.jpg','frames/000060.jpg']:
   with Image.open(root/n) as im:im.verify()
result={'status':'PASS','group_count':36,'frame_count':100,'relative_image_references':len(paths),'all_images_exist_and_decodable':True,'frame_assets_match_phase12':True,'zip_entries':len(names),'zip_crc':'PASS','zip_extract':'PASS','offline_external_network_references':'none','all_initial_unreviewed_and_hold':True,'source_hashes_unchanged':True,'javascript':'simulated DOM smoke separately; no installed Windows/Chrome browser','zip_path':str(out),'zip_bytes':out.stat().st_size,'zip_sha256':sha(out)};json.dump(result,(P/'배포검증.json').open('x'),ensure_ascii=False,indent=2);print(json.dumps(result,ensure_ascii=False),flush=True)
