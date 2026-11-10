from pathlib import Path
import csv,json,hashlib,zipfile,shutil
import cv2,numpy as np
P=Path('/home/oylee/projects/HCCEPose_experiments');O=Path(__file__).resolve().parent;old=P/'bb100_visible_mask_phase16a1_20261110';flow=P/'bb100_liteflownet_phase16a2f_20261110';pkg=O/'package'
for d in ('frames','sam','intersection','cad','flow'): (pkg/d).mkdir(parents=True)
with (old/'selected_frames.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
assert len(rows)==10
flowmap={int(r['before_frame']):r for r in csv.DictReader((flow/'pairs.csv').open(encoding='utf-8-sig'))};data=[];manifest=[]
for r in rows:
 i=int(r['frame']);id=f'{i:06d}';d={'frame':i,'occlusion':r['occlusion'],'pose_group':r['pose_group'],'panel':f'frames/{id}.jpg','rgb':f'frames/{id}_RGB.jpg','sam':f'sam/{id}_selected_NOT_GT.png','intersection':f'intersection/{id}_CAD_SAM_NOT_GT.png','cad':f'cad/{id}_AMODAL_NOT_VISIBLE_GT.png','flow':None}
 for k in ('panel','rgb','sam','intersection','cad'):
  assert (old/'package'/d[k]).is_file();shutil.copy2(old/'package'/d[k],pkg/d[k])
 if i in flowmap:
  j=int(flowmap[i]['after_frame']);src=flow/'package/panels'/f'{id}_{j:06d}.jpg';d['flow']=f'flow/{id}_{j:06d}.jpg';shutil.copy2(src,pkg/d['flow'])
 data.append(d);manifest.append({'frame':i,'occlusion':r['occlusion'],'pose_group':r['pose_group'],'robotseg_gripper_mask':'NOT_RUN','A_sam':'EXISTING_CANDIDATE','B_cad_sam':'EXISTING_CANDIDATE','C_sam_minus_robotseg':'NOT_CREATED','D_cad_sam_minus_robotseg':'NOT_CREATED','E_cad_minus_robotseg':'NOT_CREATED','flow_existing':'YES' if d['flow'] else 'NO','manual_reference':'NOT_PROVIDED'})
with (O/'representative_status.csv').open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=manifest[0]);w.writeheader();w.writerows(manifest)
template=(O/'template.html').read_text();(pkg/'index.html').write_text(template.replace('DATA_PLACEHOLDER',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8');shutil.copy2(O/'representative_status.csv',pkg/'representative_status.csv')
(pkg/'사용법.txt').write_text('RobotSeg 모델을 실행하지 못했으며 결과가 없습니다. 이 페이지는 기존 SAM/CAD/Flow 자료를 검수하기 위한 대체 자료입니다. 별도 프로그램에서 보정한 마스크가 있으면 원본 1280x720, PNG 8-bit 단일 채널 0/255로 불러와 수동 참조 파일로 다운로드할 수 있습니다. 파일 형식 검사는 정확도 인증이 아닙니다. 자동 GT 승인은 하지 않습니다.\n',encoding='utf-8')
with zipfile.ZipFile(O/'BB100_robotseg_blocked_review_windows.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(pkg.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(pkg))
with zipfile.ZipFile(O/'BB100_robotseg_blocked_review_windows.zip') as z:
 assert z.testzip() is None;names=set(z.namelist());assert all(all(d[k] in names for k in ('panel','rgb','sam','intersection','cad')) and (not d['flow'] or d['flow'] in names) for d in data)
summary={'frames':10,'partially_occluded':6,'flow_prior_pairs':len(flowmap),'robotseg_inference':'NOT_RUN_CHECKPOINT_UNAVAILABLE','robotseg_masks':0,'new_robotseg_candidates':0,'zip_references_crc':'PASS','windows_chrome':'NOT_TESTED'}
(O/'package_validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
