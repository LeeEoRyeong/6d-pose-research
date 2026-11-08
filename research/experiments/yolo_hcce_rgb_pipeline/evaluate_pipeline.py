import os,sys,json,csv,math,time,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));from run_inference import Pipeline,e,symmetry
import cv2,numpy as np,torch

def iou(a,b):
 lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);inter=float(np.prod(np.maximum(hi-lo,0)));return inter/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter+1e-12)
def angle(a,b):return float(np.degrees(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1))))
def dumpcsv(name,rows):
 with (P/name).open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def main():
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip(),'GPU occupied';torch.set_num_threads(2);cv2.setNumThreads(1);model=P/'runs/yolo11n_pbr_attempt3/weights/best.pt';assert model.is_file();pipe=Pipeline(model);v,faces,syms,info=symmetry.load();data,_=e.get_data(v);samples=data['real'];rows=[];pose=[];fail=[];over=P/'overlays/real27';over.mkdir(parents=True,exist_ok=False);threshold=.1*info['diameter']
 for s in samples:
  r=pipe.predict(s['image'],s['K'],42+s['id']);gt=e.project(v,s['R'],s['t'],s['K']);g=np.r_[gt.min(0),gt.max(0)];dr={'frame':s['id'],'detected':r['detected'],'confidence':r['confidence'],'detector_bbox_xyxy':json.dumps(r['bbox_xyxy']),'cad_amodal_bbox_xyxy':json.dumps(g.tolist()),'iou_to_cad_amodal':'','center_error_px':'','width_ratio':'','height_ratio':'','tp_iou_0_5':False,'failure':r['failure']}
  if r['detected']:
   b=np.array(r['bbox_xyxy']);dr.update(iou_to_cad_amodal=iou(b,g),center_error_px=float(np.linalg.norm((b[:2]+b[2:]-g[:2]-g[2:])/2)),width_ratio=float((b[2]-b[0])/(g[2]-g[0])),height_ratio=float((b[3]-b[1])/(g[3]-g[1])));dr['tp_iou_0_5']=dr['iou_to_cad_amodal']>=.5
  pr={'frame':s['id'],'detected':r['detected'],'pnp_valid':r['pnp_valid'],'failure':r['failure'],'ADD':'','ADD_S':'','symmetry_ADD':'','ADD_pass':False,'ADD_S_pass':False,'symmetry_ADD_pass':False,'rotation_sym_deg':'','translation_sym_mm':'','rotation_raw_deg':'','translation_raw_mm':'','pred_R':json.dumps(r['R']),'pred_t':json.dumps(r['t'])}
  category=[]
  if not r['detected']:category.append('detector_miss')
  elif not dr['tp_iou_0_5']:category.append('bbox_low_iou_amodal')
  if r['pnp_valid']:
   R=np.array(r['R']);t=np.array(r['t']);a=symmetry.add(v,R,t,s['R'],s['t']);aa=symmetry.add_s(v,R,t,s['R'],s['t']);ss,idx=symmetry.symmetry_add(v,R,t,s['R'],s['t'],syms);Rg,tg=symmetry.compose(s['R'],s['t'],syms[idx]);pr.update(ADD=a,ADD_S=aa,symmetry_ADD=ss,ADD_pass=a<threshold,ADD_S_pass=aa<threshold,symmetry_ADD_pass=ss<threshold,rotation_sym_deg=angle(R,Rg),translation_sym_mm=float(np.linalg.norm(t-tg)),rotation_raw_deg=angle(R,s['R']),translation_raw_mm=float(np.linalg.norm(t-s['t'])))
   if ss>=threshold:category.append('pose_error_symmetry_ADD')
  elif r['detected']:category.append('pnp_failure')
  if s['id'] in (66,68,70,72,73):category.append('GT_pose_previously_uncertain')
  fail.append({'frame':s['id'],'observed_categories':';'.join(category) if category else 'none','unverified_possible_causes':'occlusion; background confusion; calibration error; symmetry ambiguity cannot be assigned from these metrics alone'})
  pipe.K=s['K'];im=pipe.overlay(s['image'],r);cv2.rectangle(im,tuple(g[:2].astype(int)),tuple(g[2:].astype(int)),(0,255,0),1);cv2.putText(im,f"{s['id']} GT CAD green, det yellow, pose magenta",(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA);cv2.imwrite(str(over/f"{s['id']:06d}.png"),im);rows.append(dr);pose.append(pr);print('FRAME',s['id'],r['detected'],r['pnp_valid'],dr['iou_to_cad_amodal'],pr['symmetry_ADD'],flush=True)
 dumpcsv('detector_eval.csv',rows);dumpcsv('pose_eval.csv',pose);dumpcsv('failure_analysis.csv',fail)
 tp=sum(x['tp_iou_0_5'] for x in rows);det=sum(x['detected'] for x in rows);summary={'total':len(rows),'detected':det,'tp_iou_0_5_vs_cad_amodal':tp,'precision_at_iou_0_5':tp/det if det else None,'recall_at_iou_0_5':tp/len(rows),'mean_iou_detected':float(np.mean([x['iou_to_cad_amodal'] for x in rows if x['detected']])) if det else None,'pnp_valid':sum(x['pnp_valid'] for x in pose),'ADD_pass':sum(x['ADD_pass'] for x in pose),'ADD_S_pass':sum(x['ADD_S_pass'] for x in pose),'symmetry_ADD_pass':sum(x['symmetry_ADD_pass'] for x in pose),'threshold':.25,'note':'all images denominator for pose; precision/recall vs CAD amodal projection, one highest confidence prediction per frame; not visible bbox GT'};json.dump(summary,(P/'eval_summary.json').open('x'),indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x));print('SUMMARY',summary,flush=True)
 # Repeat first actual detected frame; warmup measurements discarded. Includes miss if any.
 ids={r['frame'] for r in rows if r['detected']};s=next((x for x in samples if x['id'] in ids),samples[0]);times={};success=0
 for i in range(35):
  z=pipe.predict(s['image'],s['K'],42+s['id']);success+=int(i>=5 and z['pnp_valid'])
  if i>=5:
   for k,val in z['timing_ms'].items():times.setdefault(k,[]).append(val)
 out={'frame':s['id'],'warmup':5,'repetitions':30,'pnp_success':success,'gpu':torch.cuda.get_device_name(0),'input_scope':'already-undistorted in-memory image; includes YOLO predict and NMS, bbox select, crop and H2D, model, decode, PnP; excludes disk IO, camera capture, undistortion, overlay','end_to_end_not_including':'I/O, undistortion, overlay','timings':{k:{'mean_ms':float(np.mean(x)),'median_ms':float(np.median(x)),'p95_ms':float(np.quantile(x,.95)),'n':len(x)} for k,x in times.items()}};json.dump(out,(P/'latency.json').open('x'),indent=2);print('LATENCY',out,flush=True)
if __name__=='__main__':main()
