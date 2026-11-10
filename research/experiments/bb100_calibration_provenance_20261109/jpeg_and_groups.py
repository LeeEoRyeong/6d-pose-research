import csv,json,io,hashlib
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,JpegImagePlugin
P=Path(__file__).resolve().parent;DEV=Path('/home/oylee/projects/HCCEPose_obj04_transfer');OLD=P.parent/'bb100_gt_camera_verification_20261109'
def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def qinfo(a):
 with Image.open(a) as im:return (im.quantization,JpegImagePlugin.get_sampling(im))
def main():
 cv2.setNumThreads(2);rows=read(P/'image_hypotheses.csv');old={int(x['frame']):x for x in read(OLD/'bb100_frame_audit.csv')};k=json.loads((DEV/'camera_params.json').read_text());K=np.array(k['camera_matrix']);D=np.array(k['distortion_coefficients']);mx,my=cv2.initUndistortRectifyMap(K,D,None,K,(1280,720),cv2.CV_32FC1);out=[]
 for r in rows:
  raw=Path(r['raw_path']);lab=Path(r['label_path']);x=cv2.imread(str(raw));y=cv2.imread(str(lab));one=cv2.remap(x,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);ok,buf=cv2.imencode('.jpg',one,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok;encoded=buf.tobytes();im=cv2.imdecode(buf,cv2.IMREAD_COLOR);lq,ls=qinfo(lab);pq,ps=qinfo(io.BytesIO(encoded));diff=np.abs(im.astype('int16')-y.astype('int16'));out.append({'frame':r['frame'],'jpeg_q95_encoded_byte_sha256':hashlib.sha256(encoded).hexdigest(),'label_byte_sha256':hashlib.sha256(lab.read_bytes()).hexdigest(),'jpeg_bytes_exact_match':encoded==lab.read_bytes(),'jpeg_quant_tables_match':lq==pq,'jpeg_sampling_match':ls==ps,'label_sampling':ls,'candidate_sampling':ps,'decoded_component_exact_fraction':float(np.mean(diff==0)),'decoded_max_abs_diff':int(diff.max()),'decoded_mae':float(diff.mean()),'interpretation':'compatible with assumed remap+q95; encoder provenance and parameter uniqueness not proven'})
 with (P/'jpeg_evidence.csv').open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 pairs=[(31,75),(32,42),(43,60),(235,247),(420,428)];(P/'same_pose_pairs').mkdir();pairdata=[]
 for a,b in pairs:
  A=old[a];B=old[b];assert A['pose_exact_group']==B['pose_exact_group'];x=cv2.imread(A['image_labeling_path']);y=cv2.imread(B['image_labeling_path']);xo=cv2.imread(A['overlay_path']);yo=cv2.imread(B['overlay_path']);bounds=np.array(json.loads(A['projection_bbox_xyxy']));cx=(bounds[0]+bounds[2])/2;cy=(bounds[1]+bounds[3])/2;side=max(bounds[2]-bounds[0],bounds[3]-bounds[1])*1.8;x1=max(0,int(cx-side/2));x2=min(1280,int(cx+side/2));y1=max(0,int(cy-side/2));y2=min(720,int(cy+side/2));delta=np.abs(x.astype('int16')-y.astype('int16'));pairdata.append({'frame_a':a,'frame_b':b,'pose_exact_group':A['pose_exact_group'],'state_a':A['state'],'state_b':B['state'],'time_a':A['frame_time'],'time_b':B['frame_time'],'capture_delta_seconds':abs(float(A['frame_mono'])-float(B['frame_mono'])),'whole_image_mae':float(delta.mean()),'projected_neighborhood_mae':float(delta[y1:y2,x1:x2].mean()),'note':'difference includes gripper/background/lighting; same pose does not prove annotation correctness'})
  panel=Image.new('RGB',(1800,580),'black');draw=ImageDraw.Draw(panel)
  for i,(title,arr) in enumerate([(f'{a:03d} label',x),(f'{a:03d} projected; calibration unknown',xo),(f'{b:03d} label',y),(f'{b:03d} projected; calibration unknown',yo)]):
   rgb=cv2.cvtColor(arr,cv2.COLOR_BGR2RGB);panel.paste(Image.fromarray(rgb).resize((450,253)),(450*i,25));panel.paste(Image.fromarray(rgb[y1:y2,x1:x2]).resize((290,290)),(450*i+80,285));draw.text((450*i+5,5),title,fill='white')
  panel.save(P/'same_pose_pairs'/f'{a:06d}_{b:06d}.jpg',quality=92)
 with (P/'same_pose_pair_metrics.csv').open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(pairdata[0]));w.writeheader();w.writerows(pairdata)
 print('JPEG byte exact',sum(x['jpeg_bytes_exact_match'] for x in out),'quant tables',sum(x['jpeg_quant_tables_match'] for x in out),'sampling',sum(x['jpeg_sampling_match'] for x in out),'decoded exact mean',np.mean([x['decoded_component_exact_fraction'] for x in out]),'max component',max(x['decoded_max_abs_diff'] for x in out),flush=True)
if __name__=='__main__':main()
