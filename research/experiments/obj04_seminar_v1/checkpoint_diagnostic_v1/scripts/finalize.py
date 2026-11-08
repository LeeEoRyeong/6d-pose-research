from pathlib import Path
import csv,json,ast,re,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1]
# Mark unavailable quantities explicitly, rather than presenting blank values as measurements.
for name in ['checkpoint_comparison.csv','epoch_loss_history.csv','correspondence_geometry.csv']:
 p=R/name
 with p.open(newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
 for row in rows:
  if name=='checkpoint_comparison.csv':
   for b in ('front','back'):
    for k in ('add_mean_valid','rotation_deg_mean_valid','translation_mean_valid'):
     key=b+'_'+k
     if not row[key]:row[key]='N/A'
  elif name=='epoch_loss_history.csv':
   for key in ('train_optimization_loss','train_evaluation_loss','validation_loss','train_mask_iou','val_mask_iou','amp_skips_since_last_validation','saved_checkpoint_names','mask_loss_recomputed','front_HCCE_weighted_loss_recomputed','back_HCCE_weighted_loss_recomputed','val_loss_recomputed'):
    if not row.get(key):row[key]='N/A'
  else:
   if not row['PnP_add']:row['PnP_add']='N/A'
   if not row['PnP_failure']:row['PnP_failure']='없음'
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
# Clarify unchanged zero-valued labels at any GT-mask pixel; original loss had no validity gate.
report=R/'reports/Checkpoint_선택_원인분석.md';text=report.read_text();text=text.replace('같은 유효 라벨·가중치를 고정한 게이트 교체가','같은 라벨 배열·가중치를 고정한 게이트 교체가');needle='두 모델 각각의 history 가중치는 기존 예측 Mask로 형성됐다.'
text=text.replace(needle,needle+' 원본 Loss와 동일하게 GT Mask 안의 별도 무효 라벨도 추가 필터링하지 않았으므로 이 반사실은 GT 유효 Mask 기반 선택 지표와 구분한다.');report.write_text(text)
# Maintain report generator compatibility with explicit N/A placeholders.
p=R/'scripts/build_reports.py';s=p.read_text().replace("v is None or v==''", "v is None or v in ('','N/A')").replace("x['validation_loss'] and int(x['epoch'])>=1", "x['validation_loss'] not in ('','N/A') and int(x['epoch'])>=1").replace("and x['PnP_add']]", "and x['PnP_add'] not in ('','N/A')]")
s=s.replace('같은 유효 라벨·가중치를 고정한 게이트 교체가','같은 라벨 배열·가중치를 고정한 게이트 교체가').replace(needle,needle+' 원본 Loss와 동일하게 GT Mask 안의 별도 무효 라벨도 추가 필터링하지 않았으므로 이 반사실은 GT 유효 Mask 기반 선택 지표와 구분한다.');p.write_text(s)
# Redraw the scatter with leader lines to avoid overlapping baseline epoch labels.
with (R/'checkpoint_comparison.csv').open() as f:C=list(csv.DictReader(f))
def font(n):
 try:return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
 except OSError:return ImageFont.load_default()
F=font(16);S=font(13);TITLE=font(23);blue='#216ba5';red='#c34c3d';gray='#777777'
im=Image.new('RGB',(1250,760),'white');d=ImageDraw.Draw(im);d.text((45,20),'Saved checkpoints: Val loss vs Front XYZ error',font=TITLE,fill='black');d.text((45,55),'Val20, GT BBox ROI; each point is an actually saved checkpoint, epoch0 is untrained reference',font=F,fill='black');l,t,r,b=130,120,1190,610;xmin=.65;xmax=2.15;ymax=100
for i in range(6):
 x=l+(r-l)*i/5;y=b-(b-t)*i/5;d.line((l,y,r,y),fill='#ddd');d.line((x,t,x,b),fill='#eee');d.text((l-55,y-8),str(i*20),font=S,fill='black');d.text((x-12,b+15),f'{xmin+(xmax-xmin)*i/5:.2f}',font=S,fill='black')
offsets={('baseline',155):(-58,20),('baseline',150):(-50,-35),('baseline',180):(18,-40),('baseline',195):(22,10)}
for c in C:
 epoch=int(c['epoch']);x=l+(float(c['val_loss_recomputed'])-xmin)/(xmax-xmin)*(r-l);y=b-float(c['front_xyz_l2'])/ymax*(b-t);m=c['model'];color=blue if m=='baseline' else red if m=='rotation' else gray;label=('B' if m=='baseline' else 'R' if m=='rotation' else 'Init')+str(epoch);dx,dy=offsets.get((m,epoch),(8,-16));d.ellipse((x-6,y-6,x+6,y+6),fill=color)
 if (m,epoch) in offsets:d.line((x,y,x+dx+(4 if dx<0 else 0),y+dy+7),fill=color,width=1)
 d.text((x+dx,y+dy),label,font=S,fill=color)
d.text((525,670),'Recomputed Val loss',font=F,fill='black');d.text((130,85),'Front XYZ L2 (native units)',font=F,fill='black');d.text((120,715),'B = Baseline',font=F,fill=blue);d.text((470,715),'R = Rotation',font=F,fill=red);d.text((820,715),'Init = separate epoch0',font=F,fill=gray);im.save(R/'figures/loss_vs_xyz.png')
# Validate requested artifacts, N/A placeholders, actual checkpoint counts and report links.
required=['checkpoint_inventory.csv','epoch_loss_history.csv','checkpoint_comparison.csv','correspondence_geometry.csv','figures/loss_vs_xyz.png','figures/checkpoint_pose_comparison.png','reports/Checkpoint_선택_원인분석.md','reports/세미나_추가진단_요약.md']
assert all((R/x).is_file() for x in required)
for p in (R/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
for p in (R/'reports').glob('*.md'):
 for q in re.findall(r'\]\((/[^)]+)\)',p.read_text()):assert Path(q).is_file(),q
with (R/'checkpoint_comparison.csv').open() as f:c=list(csv.DictReader(f))
assert len(c)==10 and next(x for x in c if x['model']=='rotation' and x['epoch']=='10')['front_add_mean_valid']=='N/A'
with (R/'epoch_loss_history.csv').open() as f:h=list(csv.DictReader(f))
assert len(h)==247 and next(x for x in h if x['model']=='rotation' and x['epoch']=='2')['validation_loss']=='N/A'
manifest=[]
for p in sorted(R.rglob('*')):
 if p.is_file():
  z=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(1048576),b''):z.update(chunk)
  manifest.append({'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':z.hexdigest()})
with (R/'checks/final_artifact_manifest.json').open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
with (R/'checks/final_completion.json').open('x') as f:json.dump({'status':'PASS','actual_checkpoints_evaluated':10,'required_artifacts':len(required),'epoch_rows':len(h),'labels_and_links':'PASS','missing_values':'explicit N/A','source_and_prior_runs':'unchanged per final_integrity.json','retraining':False,'real27_used':False},f,ensure_ascii=False,indent=2)
print(json.dumps({'status':'PASS','checkpoints':len(c),'epoch_rows':len(h),'artifacts':len(manifest),'bytes':sum(x['bytes'] for x in manifest)},ensure_ascii=False))
