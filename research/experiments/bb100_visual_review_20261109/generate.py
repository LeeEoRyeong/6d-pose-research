"""Human review panels and offline HTML; never changes original labels."""
import csv,json,hashlib,shutil,sys,html
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parent;E=P.parent;OLD=E/'bb100_gt_camera_verification_20261109';PREV=E/'bb100_calibration_provenance_20261109';B=E/'real_bb100_20261108/BB_obj04';DEV=Path('/home/oylee/projects/HCCEPose_obj04_transfer');sys.path.insert(0,str(E/'itodd_symmetry_eval_20261108'));import symmetry

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 cv2.setNumThreads(2);prev=read(OLD/'bb100_frame_audit.csv');groups=read(OLD/'pose_groups.csv');gt=json.loads((B/'scene_gt_obj04.json').read_text());cams=json.loads((B/'scene_camera_obj04.json').read_text());cal=json.loads((DEV/'camera_params.json').read_text());K=np.array(cal['camera_matrix'],float);D=np.array(cal['distortion_coefficients'],float);K_user=np.array([[984.70009004,0,635.41416344],[0,983.75715611,362.61850196],[0,0,1]]);D_user=np.array([-.28801442,-.07604108,-.00069984,.00055886,.12258375]);assert np.max(abs(K-K_user))<1e-9 and np.max(abs(D-D_user))<1e-9 and len(prev)==100 and len(groups)==36;v,faces,syms,info=symmetry.load();expected_sha='0933ba753a7aa8d673df6fa180e35807869903661b1bc8d1419ebb0ac890424f';mesh=E/'itodd_official_audit_20261108/official_obj_000004.ply';assert sha(mesh)==expected_sha
 sources=[OLD/'bb100_frame_audit.csv',OLD/'pose_groups.csv',PREV/'summary.json',B/'scene_gt_obj04.json',B/'scene_camera_obj04.json',DEV/'camera_params.json',mesh]+[Path(r['image_labeling_path']) for r in prev]+[Path(r['pose_path']) for r in prev]+[Path(r['overlay_path']) for r in prev];before={str(f):sha(f) for f in sources};out=P/'comparisons';out.mkdir();records=[]
 # local x/y/z arrows of 50 model units = BOP mm; projected onto previously undistorted JPEG with D=0.
 for x in prev:
  i=int(x['frame']);im=cv2.imread(x['image_labeling_path']);over=cv2.imread(x['overlay_path']);assert im is not None and im.shape==(720,1280,3) and over is not None and over.shape==im.shape;pose=np.load(x['pose_path'],allow_pickle=False);g=gt[str(i)][0];R=np.array(g['cam_R_m2c']).reshape(3,3);t=np.array(g['cam_t_m2c']);Km=np.array(cams[str(i)]['cam_K']).reshape(3,3);assert np.max(abs(Km-K_user))<1e-9 and np.max(abs(R-pose[:3,:3]))<1e-8 and np.max(abs(t-pose[:3,3]))<1e-8
  pts=np.array([[0,0,0],[50,0,0],[0,50,0],[0,0,50]],np.float64);cam=pts@R.T+t;assert (cam[:,2]>0).all();q=cam@Km.T;uv=np.rint(q[:,:2]/q[:,2:]).astype(int);origin=tuple(uv[0]);col=[(0,0,255),(0,255,0),(255,100,0)]
  for j,(color,lab) in enumerate(zip(col,'XYZ'),1):cv2.arrowedLine(over,origin,tuple(uv[j]),color,2,cv2.LINE_AA,tipLength=.17);cv2.putText(over,lab,tuple(uv[j]),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2,cv2.LINE_AA)
  bbox=np.array(json.loads(x['projection_bbox_xyxy']),float);pc=v@R.T+t;proj=pc@Km.T;proj=proj[:,:2]/proj[:,2:];calc=np.r_[proj.min(0),proj.max(0)];assert np.max(abs(calc-bbox))<1e-7;cv2.rectangle(over,tuple(np.rint(bbox[:2]).astype(int)),tuple(np.rint(bbox[2:]).astype(int)),(255,255,0),1)
  center=(bbox[:2]+bbox[2:])/2;side=max(bbox[2]-bbox[0],bbox[3]-bbox[1])*1.8;l=max(0,int(center[0]-side/2));rr=min(1280,int(center[0]+side/2));top=max(0,int(center[1]-side/2));bottom=min(720,int(center[1]+side/2));assert rr>l and bottom>top
  panel=Image.new('RGB',(1280,1130),'black');d=ImageDraw.Draw(panel);d.text((8,6),f'FRAME {i:03d} | {x["priority"]} | POSE GROUP {x["pose_exact_group"]} | UNREVIEWED / HOLD',fill='white')
  for j,(title,img) in enumerate([('LABEL IMAGE: no new correction',im),('PHASE 10 AMODAL + XYZ AXES + BBOX',over)]):
   rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);full=Image.fromarray(rgb).resize((640,360));zoom=Image.fromarray(rgb[top:bottom,l:rr]);zoom.thumbnail((540,540),Image.Resampling.LANCZOS);panel.paste(full,(j*640,45));panel.paste(zoom,(j*640+50+(540-zoom.width)//2,570+(540-zoom.height)//2));d.text((j*640+8,27),title,fill='white');d.text((j*640+8,540),'ZOOM: same image-coordinate crop',fill='white')
  d.text((8,420),'Yellow contour=CAD amodal | Cyan box=projected bbox | X red, Y green, Z blue (50 CAD units)',fill='white');d.text((8,445),'Camera same per user; imaging/settings and GT accuracy not independently certified.',fill='white');d.text((8,470),'Occlusion / reflection / boundary remain UNKNOWN until reviewer records them.',fill='white');d.text((8,495),'Frame 043 -> 060: documented pose copy; review both against their own images.',fill='yellow' if i in (43,60) else '#aaaaaa');panel.save(out/f'{i:06d}.jpg',quality=92)
  records.append({'frame':i,'priority':x['priority'],'priority_reason':x['priority_reason'],'pose_group':x['pose_exact_group'],'conservative_connected_group':x['connected_group'],'capture_time':x['frame_time'],'state':x['state'],'documented_pose_copy':'043_to_060' if i in (43,60) else '', 'review_status':'UNREVIEWED','occlusion':'UNKNOWN','reflection':'UNKNOWN','boundary':'UNKNOWN','blur':'UNKNOWN','reviewer':'','reviewed_at':'','notes':'','training_status':'HOLD','comparison_image':f'comparisons/{i:06d}.jpg'})
  if len(records)%20==0:print('PANEL',len(records),flush=True)
 write(P/'review_template.csv',records);write(P/'p1_priority.csv',[r for r in records if r['priority']=='P1']);shutil.copyfile(OLD/'pose_groups.csv',P/'pose_groups.csv');assert len([r for r in records if r['priority']=='P1'])==45 and before=={n:sha(Path(n)) for n in before};j=Path(__file__).with_name('index_template.html').read_text(encoding='utf8');data=json.dumps(records,ensure_ascii=False).replace('<','\\u003c');assert '__FRAME_DATA__' in j; (P/'index.html').write_text(j.replace('__FRAME_DATA__',data),encoding='utf8');
 provenance={'source_count':len(before),'sources_sha256':before,'source_hashes_unchanged':True,'frames':len(records),'pose_groups':len(groups),'p1_count':45,'statuses':{'UNREVIEWED':100},'training_status':'HOLD 100','K_D_user_values_match_dev_camera_json':True,'scene_K_match_all':True,'model_axes':'X red, Y green, Z blue; 50 CAD units; model-to-camera R X+t','distortion':'label JPEG taken as undistorted with K based on user-provided history plus Phase11 pixel evidence; acquisition settings and actual GT accuracy not independently certified','camera_identity':'same Tapo camera per user statement, not independent measurement','overlay_reuse':'Phase10 existing conditional overlay; image and original sources read only'};json.dump(provenance,(P/'manifest.json').open('x'),indent=2,ensure_ascii=False);print('PASS',len(records),'P1',sum(r['priority']=='P1' for r in records),'source',len(before),flush=True)
if __name__=='__main__':main()
