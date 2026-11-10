from pathlib import Path
import csv,json,hashlib,collections,sys
import numpy as np,cv2
P=Path('/home/oylee/projects/HCCEPose_experiments'); OUT=Path(__file__).resolve().parent
BB=P/'real_bb100_20261108/BB_obj04'; REVIEW=P/'bb100_review_export (2).csv'; AUDIT=P/'bb100_gt_camera_verification_20261109/bb100_frame_audit.csv'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def rows(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,data):
 with p.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def writejson(p,o):
 with p.open('x',encoding='utf-8') as f:json.dump(o,f,indent=2,ensure_ascii=False);f.write('\n')
class InspectionDataset:
 """CPU-only metadata reader, explicitly not an HCCEPose training Dataset."""
 def __init__(self, manifest):self.entries=manifest
 def __len__(self):return len(self.entries)
 def __getitem__(self,i):
  row=self.entries[i];img=cv2.imread(row['image_path']);T=np.load(row['pose_path'],allow_pickle=False)
  if img is None or T.shape!=(4,4):raise ValueError('invalid image or pose')
  return {'frame':int(row['frame']),'image':img,'R':T[:3,:3],'t':T[:3,3],'K':np.array(json.loads(row['K'])).reshape(3,3)}
 def training_sample(self,i):raise RuntimeError('BLOCKED: visible supervision, verified physical split, symmetry-compatible XYZ and trainer are required')
def main():
 review=rows(REVIEW);audit=rows(AUDIT);gt=json.loads((BB/'scene_gt_obj04.json').read_text());cams=json.loads((BB/'scene_camera_obj04.json').read_text())
 assert len(review)==100 and len(set(x['frame'] for x in review))==100
 assert set(x['frame'] for x in review)==set(x['frame'] for x in audit)==set(gt)==set(cams)
 am={x['frame']:x for x in audit};byhash=collections.defaultdict(list);out=[];k0=np.array([984.70009004,0,635.41416344,0,983.75715611,362.61850196,0,0,1]).reshape(3,3)
 for r in review:
  a=am[r['frame']];i=int(r['frame']); im=Path(a['image_labeling_path']); pp=Path(a['pose_path']);assert im.is_file() and pp.is_file()
  image=cv2.imread(str(im));assert image is not None and image.shape[:2]==(720,1280)
  T=np.load(pp,allow_pickle=False);assert T.shape==(4,4) and np.isfinite(T).all() and np.allclose(T[3],[0,0,0,1],atol=1e-8)
  R=T[:3,:3];t=T[:3,3];orth=float(np.max(np.abs(R.T@R-np.eye(3))));det=float(np.linalg.det(R));assert orth<1e-5 and abs(det-1)<1e-5 and t[2]>0
  g=gt[r['frame']];assert len(g)==1 and g[0]['obj_id']==4
  diff=max(float(np.max(np.abs(R-np.asarray(g[0]['cam_R_m2c']).reshape(3,3)))),float(np.max(np.abs(t-g[0]['cam_t_m2c']))));assert diff<1e-8
  K=np.asarray(cams[r['frame']]['cam_K']).reshape(3,3);assert np.allclose(K,k0,atol=1e-8)
  assert r['pose_group']==a['pose_exact_group'] and r['conservative_connected_group']==a['connected_group']
  ih=sha(im);byhash[ih].append(i)
  out.append({'frame':i,'review_status':r['review_status'],'occlusion':r['occlusion'],'source_training_status':r['training_status'],'eligibility':'HOLD','hold_reason':'physical split and valid visible supervision / symmetry-compatible training targets not established','image_path':str(im),'image_sha256':ih,'pose_path':str(pp),'pose_sha256':sha(pp),'pose_group':r['pose_group'],'connected_group':r['conservative_connected_group'],'capture_time':r['capture_time'],'state':r['state'],'K':json.dumps(K.tolist(),separators=(',',':')),'orthogonality_max_error':orth,'rotation_det':det,'pose_json_max_diff':diff,'split':'UNASSIGNED','tentative_split_only':('validation' if r['conservative_connected_group']=='31' else 'train')})
 assert all(x['review_status']=='ACCEPT' for x in out)
 writecsv(OUT/'bb100_candidate_manifest.csv',out)
 split=json.loads((P/'obj04_seminar_v1/runs/baseline_seed42_20261103_214353/trained_only/checks/split.json').read_text()); tr=set(split['train_ids']);va=set(split['val_ids']);assert len(tr)==80 and len(va)==20 and not tr&va
 s=P/'realistic_final100/train_pbr/000000';pgt=json.loads((s/'scene_gt.json').read_text());pcam=json.loads((s/'scene_camera.json').read_text());pbr=[]
 for i in sorted(tr|va):
  images=[p for p in (s/'rgb').glob(f'{i:06d}.*')];assert len(images)==1 and str(i) in pgt and str(i) in pcam
  instances=[(j,g) for j,g in enumerate(pgt[str(i)]) if g['obj_id']==4];assert len(instances)==1
  j,g=instances[0];mask=s/'mask_visib'/f'{i:06d}_{j:06d}.png';assert mask.is_file()
  m=cv2.imread(str(mask),0);im=cv2.imread(str(images[0]));assert m is not None and im is not None and m.shape==im.shape[:2] and set(np.unique(m)).issubset({0,255})
  pbr.append({'id':i,'split':'train' if i in tr else 'val','image':str(images[0]),'mask_visib':str(mask),'K':pcam[str(i)]['cam_K'],'R':g['cam_R_m2c'],'t':g['cam_t_m2c']})
 writejson(OUT/'pbr_existing_split_manifest.json',pbr)
 dup=[v for v in byhash.values() if len(v)>1]; cross=[v for v in dup if len({next(x['connected_group'] for x in out if x['frame']==i) for i in v})>1]
 summary={'review_file':str(REVIEW),'review_sha256':sha(REVIEW),'count':len(out),'review_status':dict(collections.Counter(x['review_status'] for x in out)),'source_training_status':dict(collections.Counter(x['source_training_status'] for x in out)),'occlusion':dict(collections.Counter(x['occlusion'] for x in out)),'connected_groups':dict(collections.Counter(x['connected_group'] for x in out)),'pose_groups':len({x['pose_group'] for x in out}),'exact_rgb_duplicate_groups':dup,'exact_rgb_duplicates_cross_connected_groups':cross,'max_rotation_orthogonality_error':max(x['orthogonality_max_error'] for x in out),'max_rotation_det_error':max(abs(x['rotation_det']-1) for x in out),'max_pose_json_difference':max(x['pose_json_max_diff'] for x in out),'bb_k_matches_user_100':True,'image_size':'1280x720','pbr_training_count':len(tr),'pbr_val_count':len(va),'final_split':'UNASSIGNED','training_readiness':'BLOCKED'}
 writejson(OUT/'audit_summary.json',summary)
 dataset=InspectionDataset(out);assert len(dataset)==100
 for ix in [0,13,50,99]:
  v=dataset[ix];assert v['image'].shape==(720,1280,3) and v['K'].shape==(3,3)
 try:dataset.training_sample(0);raise AssertionError('guard failed')
 except RuntimeError:pass
 assert not cross
 writejson(OUT/'cpu_tests.json',{'review_and_frame_bijection':'PASS','100_image_and_pose_checks':'PASS','100_rotations_and_K':'PASS','pose_json_matches_npy':'PASS','pbr_split_disjoint_and_visible_masks_100':'PASS','inspection_dataset_4_samples':'PASS','training_guard':'PASS','tentative_split_exact_duplicate_check':'PASS','physical_independence':'NOT_VERIFIED','training_dataloader':'NOT_IMPLEMENTED','real_visible_mask':'NOT_AVAILABLE','checkpoint_strict_load':'see checkpoint_check.json'})
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
