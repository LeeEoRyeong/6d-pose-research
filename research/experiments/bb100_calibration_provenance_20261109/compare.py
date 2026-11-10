"""CPU, read-only source comparison; JPEG and conditional calibration hypotheses."""
import csv,json,hashlib,collections
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parent;E=P.parent;OLD=E/'bb100_gt_camera_verification_20261109';DEV=Path('/home/oylee/projects/HCCEPose_obj04_transfer');RUN=Path('/home/oylee/projects/qwen_smt/data/BB_rgb_obj04/run_20260813_164421')
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def write(p,r):
 with p.open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def jpeg(im,q):
 ok,buf=cv2.imencode('.jpg',im,[int(cv2.IMWRITE_JPEG_QUALITY),q]);assert ok;return cv2.imdecode(buf,cv2.IMREAD_COLOR)
def mae(a,b):return float(np.mean(np.abs(a.astype(np.int16)-b.astype(np.int16))))
def exif(p):
 with Image.open(p) as im:
  x=im.getexif();return {'datetime':str(x.get(306,'')),'datetime_original':str(x.get(36867,'')),'camera_model':str(x.get(272,'')),'camera_serial':str(x.get(42033,'')),'width':im.width,'height':im.height}
def main():
 cv2.setNumThreads(2);audit=read(OLD/'bb100_frame_audit.csv');assert len(audit)==100;mapping=read(RUN/'frame_opc_mapping.csv');assert len({x['frame_name'] for x in mapping})==len(mapping);mp={int(Path(x['frame_name']).stem):x for x in mapping};cal=json.loads((DEV/'camera_params.json').read_text());K=np.array(cal['camera_matrix'],np.float64);D=np.array(cal['distortion_coefficients'],np.float64);maps=cv2.initUndistortRectifyMap(K,D,None,K,(1280,720),cv2.CV_32FC1);rep=[31,32,43,60,67,75,77,90,99,107,336,359,420,428];by={int(x['frame']):x for x in audit};assert all(by[i]['priority']=='P1' for i in rep)
 files=[OLD/'bb100_frame_audit.csv',OLD/'pose_groups.csv',OLD/'audit_summary.json',DEV/'camera_params.json',RUN/'frame_opc_mapping.csv']+[Path(x['image_raw_path']) for x in audit]+[Path(x['image_labeling_path']) for x in audit]+[Path(by[i]['overlay_path']) for i in rep];before={str(p):sha(p) for p in files};(P/'differences').mkdir();(P/'comparison_panels').mkdir();(P/'p1_raw_projection').mkdir();out=[]
 for j,r in enumerate(audit):
  i=int(r['frame']);pr=Path(r['image_raw_path']);pl=Path(r['image_labeling_path']);x=cv2.imread(str(pr));y=cv2.imread(str(pl));assert x is not None and y is not None and x.shape==y.shape==(720,1280,3);assert pr.stem==pl.stem==f'{i:06d}' and int(Path(mp[i]['frame_name']).stem)==i;one=cv2.remap(x,*maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);two=cv2.remap(one,*maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);q90=jpeg(one,90);q95=jpeg(one,95);q100=jpeg(one,100);none_j95=jpeg(x,95);swap=one[:,:,::-1].copy();er=exif(pr);el=exif(pl);assert er['width']==el['width']==1280 and er['height']==el['height']==720
  ch=np.mean(np.abs(one.astype('int16')-y.astype('int16')),axis=(0,1));record={'frame':i,'raw_path':str(pr),'label_path':str(pl),'raw_sha256':sha(pr),'label_sha256':sha(pl),'raw_bytes':pr.stat().st_size,'label_bytes':pl.stat().st_size,'width':1280,'height':720,'raw_exif_datetime':er['datetime_original'] or er['datetime'],'label_exif_datetime':el['datetime_original'] or el['datetime'],'raw_exif_camera_model':er['camera_model'],'label_exif_camera_model':el['camera_model'],'filename_match':True,'directory_state_match':pr.parent.name==pl.parent.name,'capture_wall_time':mp[i]['frame_read_wall_time'],'image_save_wall_time':mp[i]['image_save_wall_time'],'robot_match_valid':mp[i]['match_valid'],'raw_label_time_relation':'same basename and acquisition log; label derivative time not independently recorded','pose_group':r['pose_exact_group'],'priority':r['priority'],'mae_no_correction':mae(x,y),'mae_assumed_devKD_once':mae(one,y),'mae_assumed_devKD_twice':mae(two,y),'mae_assumed_once_jpeg90':mae(q90,y),'mae_assumed_once_jpeg95':mae(q95,y),'mae_assumed_once_jpeg100':mae(q100,y),'mae_raw_jpeg95':mae(none_j95,y),'mae_assumed_once_channel_swap':mae(swap,y),'once_vs_own_jpeg95_mae':mae(one,q95),'once_residual_B':float(ch[0]),'once_residual_G':float(ch[1]),'once_residual_R':float(ch[2]),'bb_K_D_actual':'UNKNOWN','actual_BB_setting_comparison':'not available','training_status':'HOLD'};out.append(record)
  if i in rep:
   for label,img in [('no_correction',x),('assumed_once',one),('assumed_twice',two)]:
    diff=np.abs(img.astype(np.int16)-y.astype(np.int16));vis=np.clip(diff*4,0,255).astype('uint8');cv2.putText(vis,f'{i:06d} {label} | abs diff x4 vs label',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA);assert cv2.imwrite(str(P/'differences'/f'{i:06d}_{label}.png'),vis)
   # Full image comparison of three hypotheses and label. Text labels explicitly mark assumption.
   panel=Image.new('RGB',(1600,252),'black');draw=ImageDraw.Draw(panel)
   for k,(title,img) in enumerate([('RAW / no correction',x),('dev K,D assumed 1x',one),('dev K,D assumed 2x',two),('label JPEG',y)]):
    panel.paste(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).resize((400,225)),(k*400,27));draw.text((k*400+4,7),title,fill='white')
   panel.save(P/'comparison_panels'/f'{i:06d}.jpg',quality=92)
   # Display raw, label and existing Phase 10 conditional projected image, plus zoom views.
   overlay=cv2.imread(r['overlay_path']);assert overlay is not None;bounds=np.array(json.loads(r['projection_bbox_xyxy']),float);cx=(bounds[0]+bounds[2])/2;cy=(bounds[1]+bounds[3])/2;side=max(bounds[2]-bounds[0],bounds[3]-bounds[1])*1.8;lx=max(0,int(cx-side/2));rx=min(1280,int(cx+side/2));ty=max(0,int(cy-side/2));byy=min(720,int(cy+side/2));panel=Image.new('RGB',(1500,780),'black');draw=ImageDraw.Draw(panel)
   for k,(title,img) in enumerate([('RAW candidate; camera ISP unknown',x),('label JPEG; correction provenance unknown',y),('label + CAD; K unverified, amodal',overlay)]):
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);panel.paste(Image.fromarray(rgb).resize((500,281)),(500*k,30));panel.paste(Image.fromarray(rgb[ty:byy,lx:rx]).resize((450,450)),(500*k+25,320));draw.text((500*k+5,8),title,fill='white')
   panel.save(P/'p1_raw_projection'/f'{i:06d}.jpg',quality=92)
  if (j+1)%20==0:print('COMPARED',j+1,flush=True)
 write(P/'image_hypotheses.csv',out);assert before=={p:sha(Path(p)) for p in before};keys=['mae_no_correction','mae_assumed_devKD_once','mae_assumed_devKD_twice','mae_assumed_once_jpeg90','mae_assumed_once_jpeg95','mae_assumed_once_jpeg100','mae_raw_jpeg95','mae_assumed_once_channel_swap','once_vs_own_jpeg95_mae'];stats={k:{'mean':float(np.mean([r[k] for r in out])),'median':float(np.median([r[k] for r in out])),'min':min(r[k] for r in out),'max':max(r[k] for r in out)} for k in keys};res={'frames':len(out),'representative_P1_ids':rep,'raw_files_log_name_match':sum(x['filename_match'] for x in out),'raw_label_same_state_dir':sum(x['directory_state_match'] for x in out),'robot_match_valid':sum(x['robot_match_valid']=='1' for x in out),'raw_exif_capture_time_count':sum(bool(x['raw_exif_datetime']) for x in out),'label_exif_capture_time_count':sum(bool(x['label_exif_datetime']) for x in out),'raw_exif_camera_model_count':sum(bool(x['raw_exif_camera_model']) for x in out),'label_exif_camera_model_count':sum(bool(x['label_exif_camera_model']) for x in out),'raw_label_exact_hash_equal':sum(x['raw_sha256']==x['label_sha256'] for x in out),'once_better_than_twice_count':sum(x['mae_assumed_devKD_once']<x['mae_assumed_devKD_twice'] for x in out),'jpeg95_better_than_unencoded_once_count':sum(x['mae_assumed_once_jpeg95']<x['mae_assumed_devKD_once'] for x in out),'stats':stats,'bb_specific_calibration_found':False,'source_hashes_unchanged':True,'source_file_count':len(before),'source_sha256':before,'status_note':'calibration hypotheses only; jpeg test does not establish actual encoder or correction provenance'};json.dump(res,(P/'summary.json').open('x'),indent=2);print('SUMMARY',json.dumps({k:v for k,v in res.items() if k not in ('source_sha256','stats')}),'stats',stats,flush=True)
if __name__=='__main__':main()
