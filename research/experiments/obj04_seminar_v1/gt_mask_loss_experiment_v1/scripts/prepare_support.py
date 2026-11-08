"""원래 XYZ 생성의 depth-valid support를 재구성하고 저장 라벨과 100장 정확히 대조."""
from pathlib import Path
import sys,json,cv2,numpy as np,time,traceback
R=Path(__file__).resolve().parents[1];B=R.parent;sys.path.insert(0,str(B/'diagnostics_v1/scripts'));import diagnose as D
sys.path.insert(0,str(B/'scripts'));import generate_xyz_v2 as G

def save(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 cv2.setNumThreads(1);samples,train,val=D.S.dataset();assert len(train)==80 and len(val)==20
 gt=D.S.load_json(D.S.SCENE/'scene_gt.json');cam=D.S.load_json(D.S.SCENE/'scene_camera.json');rows=[];start=time.monotonic()
 for frame in train+val:
  g=gt[str(frame)][0];rot=np.array(g['cam_R_m2c']).reshape(3,3);trans=np.array(g['cam_t_m2c']);K=np.array(cam[str(frame)]['cam_K']).reshape(3,3);near,far,valid,_=G.render(D.V,D.F,rot,trans,K,720,1280,.5)
  yy,xx=np.nonzero(valid);assert len(xx)>0
  counts=[];zeroval=[]
  for kind,z in [('front',near),('back',far)]:
   Z=z[valid];xc=np.column_stack([(xx-K[0,2])/K[0,0]*Z,(yy-K[1,2])/K[1,1]*Z,Z]);coord=(xc-trans)@rot;code=np.rint(np.clip((coord-D.LO)/D.SIZE,0,1)*255).astype(np.uint8)
   img=np.zeros((720,1280,3),np.uint8);img[valid]=code;original=cv2.imread(str(B/'labels'/kind/f'{frame:06d}_000000.png'));assert original is not None and np.array_equal(img,original),('stored label differs',frame,kind)
   zeroval.append(int(np.count_nonzero(np.all(code==0,axis=1))))
   # Generator uses common finite near/far surface validity for both outputs; persist separately.
   crop=D.CROP(valid.astype(np.uint8),samples[frame]['bbox'].numpy(),128,cv2.INTER_NEAREST)>0;path=R/'support'/f'{frame:06d}_{kind}.png';assert cv2.imwrite(str(path),crop.astype(np.uint8)*255);counts.append(int(crop.sum()))
  mask=samples[frame]['tensors'][1].numpy()>.5;vf=cv2.imread(str(R/'support'/f'{frame:06d}_front.png'),0)>0;vb=cv2.imread(str(R/'support'/f'{frame:06d}_back.png'),0)>0
  rows.append({'frame':frame,'split':'train' if frame in train else 'val','full_render_valid':int(valid.sum()),'full_valid_allzero_quant_front':zeroval[0],'full_valid_allzero_quant_back':zeroval[1],'front_valid_crop':counts[0],'back_valid_crop':counts[1],'GT_visible_pixels':int(mask.sum()),'front_gate_pixels':int((mask&vf).sum()),'back_gate_pixels':int((mask&vb).sum()),'GT_visible_invalid_front':int((mask&~vf).sum()),'GT_visible_invalid_back':int((mask&~vb).sum()),'front_back_support_difference':int((vf!=vb).sum())})
  print(json.dumps({'frame':frame,'GT':rows[-1]['GT_visible_pixels'],'valid_front':rows[-1]['front_gate_pixels'],'valid_back':rows[-1]['back_gate_pixels']}),flush=True)
 assert len(rows)==100 and all(r['front_gate_pixels']>0 and r['back_gate_pixels']>0 for r in rows)
 out={'status':'PASS','frames':100,'stored_front_back_labels_exact_match':True,'source':'기존 CPU near/far renderer의 finite depth support를 CAD/R/t/K로 재생성; 라벨의 0값 자체로 validity를 추정하지 않음','physical_limit':'이전 원본 VisPy와 직접 bitwise 비교는 미완료','full_valid_allzero_quant_front':sum(r['full_valid_allzero_quant_front'] for r in rows),'full_valid_allzero_quant_back':sum(r['full_valid_allzero_quant_back'] for r in rows),'front_back_support_difference_total':sum(r['front_back_support_difference'] for r in rows),'rows':rows,'seconds':time.monotonic()-start}
 save(R/'checks/support_validation.json',out);print(json.dumps({k:v for k,v in out.items() if k!='rows'},ensure_ascii=False,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:save(R/'checks/support_failure.json',{'status':'FAIL','reason':str(e),'traceback':traceback.format_exc()});raise
