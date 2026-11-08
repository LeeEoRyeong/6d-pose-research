"""Oracle ROI perturbation: same official model, K, front-only PnP and metric."""
import sys,json,csv,itertools,torch,numpy as np,cv2
from pathlib import Path
from PIL import Image,ImageDraw
OLD=Path('/RESEARCH_DATA/HCCEPose_experiments/itodd_symmetry_eval_20261108');sys.path.insert(0,str(OLD));import evaluate as e, symmetry
HERE=Path(__file__).resolve().parent

def prep(img,box,crop):
 im=crop(img,box,256,cv2.INTER_LINEAR);x=torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1))).float()/255
 return (x-torch.tensor([.406,.456,.485])[:,None,None])/torch.tensor([.225,.224,.229])[:,None,None]
def angle(a,b):return float(np.degrees(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1))))
def main():
 assert json.load((HERE/'reproduction.json').open())['status']=='PASS';torch.set_num_threads(2);cv2.setNumThreads(1)
 apps=e.subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip();assert not apps,apps
 v,faces,syms,info=symmetry.load();data,_=e.get_data(v);samples=data['real'];mod=e.explicit_module();local=e.json_read(e.BASE/'data/models/models_info.json')['4'];lo=np.array([local['min_'+a] for a in 'xyz']);size=np.array([local['size_'+a] for a in 'xyz']);a=torch.tensor(lo,dtype=torch.float32,device='cuda');b=torch.tensor(size,dtype=torch.float32,device='cuda')
 with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64'))]):ck=torch.load(e.AUD/'itodd_obj04_1_0000step50000.pt',map_location='cpu',weights_only=True)
 net=mod.HccePose_BF_Net(efficientnet_key=1,input_channels=3,min_xyz=a,size_xyz=b);net.load_state_dict(ck['model_state_dict'],strict=True);net.cuda().eval()
 crop=e.original_function('HccePose/bop_loader.py','crop_square_resize',{'np':np,'cv2':cv2});solve=e.original_function('HccePose/PnP_solver.py','solve_PnP',{'np':np,'cv2':cv2,'itertools':itertools});conditions=[('oracle',0.,0.,1.)]
 for axis in ['x','y']:
  for shift in [-.10,-.05,.05,.10]:conditions.append((f'{axis}{shift:+.0%}',shift if axis=='x' else 0.,shift if axis=='y' else 0.,1.))
 for mult in [.8,.9,1.1,1.2]:conditions.append((f'scale{mult:.1f}',0.,0.,mult))
 rows=[];overlay=HERE/'overlays';overlay.mkdir(exist_ok=False)
 for label,dx,dy,mult in conditions:
  for start in range(0,len(samples),4):
   batch=samples[start:start+4];boxes=[];xs=[]
   for s in batch:
    x,y,w,h=s['box'];cx=x+w/2+dx*w;cy=y+h/2+dy*h;w2=w*mult;h2=h*mult;box=np.array([cx-w2/2,cy-h2/2,w2,h2]);boxes.append(box);xs.append(prep(s['image'],box,crop))
   with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):out=net.inference_batch(torch.stack(xs).cuda(),torch.tensor(np.array(boxes),dtype=torch.float32,device='cuda'))
   for j,s in enumerate(batch):
    mask=out['pred_mask'][j].cpu().numpy()>0;front=out['pred_front_code_obj'][j].float().cpu().numpy();uv=out['coord_2d_image'][j].float().cpu().numpy();r={'condition':label,'frame':s['id'],'dx_fraction':dx,'dy_fraction':dy,'scale':mult,'bbox_xywh':json.dumps(boxes[j].tolist()),'pnp_valid':False,'failure':'','ADD_S':'','symmetry_ADD':'','ADD_S_pass':False,'symmetry_ADD_pass':False,'rotation_raw_deg':'','translation_raw':'','rotation_sym_deg':'','translation_sym':'','symmetry_index':''};p=None
    try:
     if mask.sum()<=4:raise ValueError('mask has <=4 points')
     cv2.setRNGSeed(42+s['id']);res=solve((mask,front,uv,s['K']),pnp_op=2,reprojectionError=1.5,iterationsCount=150)
     if not res['success']:raise ValueError('PnP success=False')
     R=np.asarray(res['rot'],float);t=np.asarray(res['tvecs'],float).reshape(3)
     if not np.isfinite(R).all() or not np.isfinite(t).all() or np.min((v@R.T+t)[:,2])<=0:raise ValueError('invalid pose')
     adds=symmetry.add_s(v,R,t,s['R'],s['t']);val,idx=symmetry.symmetry_add(v,R,t,s['R'],s['t'],syms);Rg,tg=symmetry.compose(s['R'],s['t'],syms[idx]);th=.1*info['diameter'];r.update(pnp_valid=True,ADD_S=adds,symmetry_ADD=val,ADD_S_pass=adds<th,symmetry_ADD_pass=val<th,rotation_raw_deg=angle(R,s['R']),translation_raw=float(np.linalg.norm(t-s['t'])),rotation_sym_deg=angle(R,Rg),translation_sym=float(np.linalg.norm(t-tg)),symmetry_index=idx);p=(R,t)
    except Exception as ex:r['failure']=type(ex).__name__+': '+str(ex)
    rows.append(r)
    if s['id'] in (26,32,72,78,99,126) and label in ('oracle','x+10%','x-10%','scale0.8','scale1.2'):
     image=s['image'].copy();
     for R,t,color in [(s['R'],s['t'],(0,255,0))]+([] if p is None else [(p[0],p[1],(255,0,255))]):
      uvp=e.project(v,R,t,s['K']);polys=[np.clip(uvp[f],-50000,50000).astype(np.int32).reshape(-1,1,2) for f in faces];cv2.polylines(image,polys,True,color,1,cv2.LINE_AA)
     cv2.rectangle(image,tuple(np.array(boxes[j][:2],int)),tuple(np.array(boxes[j][:2]+boxes[j][2:],int)),(0,255,255),2);cv2.putText(image,f'{label} frame {s["id"]} GT green / pred magenta / ROI yellow',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA);e.png(overlay/f'{label}_{s["id"]:06d}.png',image)
  q=[r for r in rows if r['condition']==label];print('ROI',label,'PnP',sum(x['pnp_valid'] for x in q),'sym',sum(x['symmetry_ADD_pass'] for x in q),'ADD-S',sum(x['ADD_S_pass'] for x in q),flush=True)
 assert len(rows)==13*27
 with (HERE/'roi_robustness.csv').open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summaries=[]
 for label,*_ in conditions:
  z=[r for r in rows if r['condition']==label];valid=[r for r in z if r['pnp_valid']];summaries.append({'condition':label,'n':len(z),'pnp_valid':len(valid),'symmetry_pass':sum(r['symmetry_ADD_pass'] for r in z),'adds_pass':sum(r['ADD_S_pass'] for r in z),'sym_rotation_median_deg':float(np.median([r['rotation_sym_deg'] for r in valid])) if valid else None,'sym_translation_median':float(np.median([r['translation_sym'] for r in valid])) if valid else None,'failures':{str(r['frame']):r['failure'] for r in z if not r['pnp_valid']}})
 with (HERE/'roi_summary.json').open('x') as f:json.dump(summaries,f,indent=2);f.write('\n')
 # Static bar chart, labeled counts; no extra plotting dependency.
 W,H=1400,700;im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im);d.text((35,20),'Oracle ROI perturbation: passing frames / 27 (not detection)',fill='black');left=80;bottom=590;d.line((left,70,left,bottom),fill='black',width=2);d.line((left,bottom,W-20,bottom),fill='black',width=2)
 for y in range(0,28,5):Y=bottom-y*17;d.line((left-4,Y,W-20,Y),fill='#dddddd');d.text((40,Y-6),str(y),fill='black')
 for i,row in enumerate(summaries):
  x=left+20+i*98;v1=row['symmetry_pass'];v2=row['adds_pass'];d.rectangle((x,bottom-v1*17,x+30,bottom),fill='#3268ba');d.rectangle((x+32,bottom-v2*17,x+62,bottom),fill='#e6992e');d.text((x,bottom+8),row['condition'],fill='black');d.text((x,bottom-v1*17-17),str(v1),fill='#3268ba');d.text((x+32,bottom-v2*17-17),str(v2),fill='#995500')
 d.text((100,650),'Blue: explicit symmetry ADD | Orange: ADD-S',fill='black');im.save(HERE/'roi_robustness.png')
 print('ROI DONE',flush=True)
if __name__=='__main__':main()
