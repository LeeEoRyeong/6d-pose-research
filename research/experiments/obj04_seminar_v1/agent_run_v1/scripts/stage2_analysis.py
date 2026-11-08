from pathlib import Path
import json,csv,numpy as np,cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;S=ROOT/'stage1';O=ROOT/'stage2';data=json.loads((S/'pose_results.json').read_text());rows=data['rows'];assert len(rows)==27
# Coarse annotations copied from earlier human-reviewed reprojection report, not inferred masks.
occluded={26,32,61,65,66,68,70,72,73,78,83,91,92,94,96,98,99};clear={38,39,51,56,108,118,119,126};uncertain={66,68,70,72,73};
for r in rows:r['occlusion_prior']='도구 가림 관찰' if r['frame'] in occluded else ('가림 없거나 적음' if r['frame'] in clear else '미분류');r['GT_quality_prior']='판단불가' if r['frame'] in uncertain else '거시 정합'
with (O/'real_frame_analysis.csv').open('x',newline='') as f:
 keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
groups=[]
for label in ['도구 가림 관찰','가림 없거나 적음','미분류']:
 rr=[r for r in rows if r['occlusion_prior']==label];good=[r for r in rr if r['valid_pose']];groups.append({'prior_annotation':label,'n':len(rr),'ADD_pass':sum(r['add_pass'] for r in rr),'mean_ADD_valid_only':float(np.mean([r['add'] for r in good])) if good else None,'mean_inlier_ratio':float(np.mean([r['inlier_ratio_actual'] for r in rr])) if rr else None})
# Compact review image shows original image and existing GT/estimate wireframe in fixed priority frames.
priority=[26,51,56,92,126];sheet=Image.new('RGB',(1200,5*310),'#171717');draw=ImageDraw.Draw(sheet)
for i,frame in enumerate(priority):
 rr=next(r for r in rows if r['frame']==frame);xywh=json.loads(rr['bbox_padded_xywh']);x,y,w,h=xywh;x0=max(0,int(x-w*.25));y0=max(0,int(y-h*.25));x1=min(1280,int(x+w*1.25));y1=min(720,int(y+h*1.25));
 for j,kind in enumerate(['controlled_rgb','overlays']):
  im=Image.open(S/kind/f'{frame:06d}.png').crop((x0,y0,x1,y1)).convert('RGB');im.thumbnail((580,270));sheet.paste(im,(j*600+(600-im.width)//2,i*310+35));draw.text((j*600+8,i*310+8),f'{frame:06d} '+('RGB' if j==0 else f'GT green / prediction magenta ADD={rr["add"]:.2f}'),fill='white')
sheet.save(O/'priority_examples.png')
base={'val20_ADD_pass':1,'val20_total':20,'val20_ADD_rate':0.05,'val20_mean_ADD':62.24738430230629,'val20_mean_inlier_ratio':0.04008987350488901,'real_ADD_pass':sum(r['add_pass'] for r in rows),'real_total':27,'real_ADD_rate':sum(r['add_pass'] for r in rows)/27,'real_mean_ADD_valid':float(np.mean([r['add'] for r in rows if r['valid_pose']])),'real_mean_inlier_ratio':float(np.mean([r['inlier_ratio_actual'] for r in rows])),'caveat':'PBR ROI=GT visible BBox; real ROI=GT CAD projected amodal BBox; synthetic/real GT quality and domain differ; this is not a controlled causal attribution.'}
summary={'status':'PASS','groups':groups,'comparison':base,'prior_occlusion_annotations_source':'HCCEPose/validation_obj04_transfer_20261103/reprojection_report_ko.md','GT_uncertainty_frames':sorted(uncertain),'overlay_review':'000051,000092를 직접 확인. 녹색GT와 자홍예측 wireframe의 크기·방향 차이가 보임; 물체 자체/도구 가림·투영깊이에 의해 가시 형상의 해석 제한','confirmed':'ADD 실패27/27, 낮은inlier 비율, 유효PnP반환27/27','hypotheses':'PBR-실사 외관·조명·가림 차이, ROI 유형 차이, 학습 데이터·좌표 일반화 및 GT정밀도가 각각 기여할 수 있으나 이번 데이터로 인과 분리 불가'}
with (O/'analysis_summary.json').open('x') as f:json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(summary,ensure_ascii=False,indent=2))
