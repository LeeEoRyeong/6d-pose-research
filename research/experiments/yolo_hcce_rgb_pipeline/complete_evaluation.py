import sys,csv,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));from run_inference import Pipeline,e
import torch,numpy as np,cv2

def main():
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip();d=list(csv.DictReader((P/'detector_eval.csv').open()));ps=list(csv.DictReader((P/'pose_eval.csv').open()));n=len(d);assert n==27 and len(ps)==n;tp=sum(r['tp_iou_0_5']=='True' for r in d);det=sum(r['detected']=='True' for r in d);o={'total':n,'detected':det,'tp_iou_0_5_vs_cad_amodal':tp,'precision_at_iou_0_5':tp/det if det else None,'recall_at_iou_0_5':tp/n,'mean_iou_detected':float(np.mean([float(r['iou_to_cad_amodal']) for r in d if r['detected']=='True'])) if det else None,'pnp_valid':sum(r['pnp_valid']=='True' for r in ps),'ADD_pass':sum(r['ADD_pass']=='True' for r in ps),'ADD_S_pass':sum(r['ADD_S_pass']=='True' for r in ps),'symmetry_ADD_pass':sum(r['symmetry_ADD_pass']=='True' for r in ps),'threshold':.25,'note':'full set denominator, one highest scoring box/frame; precision and recall vs CAD amodal, not visible GT'};json.dump(o,(P/'eval_summary.json').open('x'),indent=2);print('SUMMARY',o,flush=True)
 torch.set_num_threads(2);cv2.setNumThreads(1);pipe=Pipeline(P/'runs/yolo11n_pbr_attempt3/weights/best.pt');v,_,_,_=__import__('symmetry').load();data,_=e.get_data(v);ids={int(x['frame']) for x in d if x['detected']=='True'};s=next((x for x in data['real'] if x['id'] in ids),data['real'][0]);times={};success=0
 for i in range(35):
  z=pipe.predict(s['image'],s['K'],42+s['id']);success+=int(i>=5 and z['pnp_valid'])
  if i>=5:
   for k,val in z['timing_ms'].items():times.setdefault(k,[]).append(val)
 out={'frame':s['id'],'warmup':5,'repetitions':30,'pnp_success':success,'gpu':torch.cuda.get_device_name(0),'input_scope':'already-undistorted in-memory image; includes YOLO predict and NMS, bbox select, crop and H2D, model, decode, PnP; excludes disk IO, camera capture, undistortion, overlay','timings':{k:{'mean_ms':float(np.mean(x)),'median_ms':float(np.median(x)),'p95_ms':float(np.quantile(x,.95)),'n':len(x)} for k,x in times.items()}};json.dump(out,(P/'latency.json').open('x'),indent=2);print('LATENCY',out,flush=True)
if __name__=='__main__':main()
