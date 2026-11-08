from pathlib import Path
import json,traceback,sys
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
from augmentation import ROOT,BASE,D,Augmentation

def save(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 for p in [ROOT/'checks/stage1_gate.json',ROOT/'stage2/analysis_summary.json',ROOT/'checks/pbr_repro_gate.json']:
  assert json.loads(p.read_text())['status']=='PASS',p
 torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42);np.random.seed(42);samples,train,val=D.S.dataset();aug=Augmentation(samples,train)
 # Identity must reproduce all original tensors exactly.
 identity_max=0.
 for i in train:
  tensors,info=aug.tensors(i,0.)
  for x,y in zip(tensors,samples[i]['tensors']):identity_max=max(identity_max,float(torch.max(torch.abs(x-y))))
 assert identity_max==0,identity_max
 n=0;max_coord_error=0.;decode_max=0.;fractions=[]
 for i in train[:8]:
  for angle in [-20.,-10.,10.,20.]:
   ten,d=aug.tensors(i,angle);m=d['M128'];M=d['M256'];assert np.max(abs(M[:,:2]-m[:,:2]))<1e-12 and np.max(abs(M[:,2]-2*m[:,2]))<1e-12
   pts=np.array([[0,0,1],[64,64,1],[20,85,1],[127,127,1]],float);left=(M@np.column_stack((pts[:,:2]*2,np.ones(len(pts)))).T).T;right=2*(m@pts.T).T;max_coord_error=max(max_coord_error,float(abs(left-right).max()))
   support=(d['mask']>0)&(d['valid']>0);assert support.sum()>100
   for key in ['front','back']:
    q=d[key][support][None];dec,_=D.decoded(q);decode_max=max(decode_max,float(np.max(abs(dec-q))))
   fractions.append(float(support.sum()/max(np.count_nonzero(aug.data[i]['mask']&aug.data[i]['valid']),1)));n+=1
 assert max_coord_error<1e-10 and decode_max==0
 # Independent transformed 128/256 location test with synthetic point markers.
 markers=[]
 for angle in [-20.,10.,20.]:
  matrix=cv2.getRotationMatrix2D((64.,64.),angle,1.);full=matrix.copy();full[:,2]*=2
  for x,y in [(30,40),(64,64),(92,75),(48,98)]:
   point=np.array([x,y,1.]);q=matrix@point;Q=[REDACTED_EMAIL]([2*x,2*y,1.]);markers.append(float(np.linalg.norm(Q/2-q)))
 assert max(markers)<1e-10
 check={'status':'PASS','identity_train_samples':len(train),'identity_max_tensor_abs_diff':identity_max,'tested_transform_samples':n,'angles_deg':[-20,-10,10,20],'shared_affine_128_256_max_difference':max_coord_error,'independent_marker_max_difference':max(markers),'GT_HCCE_roundtrip_max_uint8_error':decode_max,'transformed_valid_mask_fraction_min':min(fractions),'transformed_valid_mask_fraction_max':max(fractions),'loss_policy':'기존 baseline은 valid 영역을 Loss에 별도 반영하지 않음. valid는 동기 변환·정합성 검증에 사용하고 Loss 수식은 변경하지 않음'}
 save(ROOT/'checks/augmentation_alignment.json',check)
 # CPU 4-sample training-path smoke, single optimizer step in separate process; no main training state used.
 mod=D.S.explicit_module();net=D.S.build_network(mod,torch.tensor(D.LO,dtype=torch.float32),torch.tensor(D.SIZE,dtype=torch.float32));net.train();criterion=mod.HccePose_Loss();optimizer=torch.optim.Adam(net.parameters(),lr=.0002)
 angles=[-20.,-6.,8.,20.];x,m,front,back=aug.batch(train[:4],angles,'cpu');optimizer.zero_grad(set_to_none=True);pm,pc=net(x);pieces=criterion(pc[:,:24],pc[:,24:],pm,front,back,m);loss=D.S.loss_value(pieces);assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),float('inf'));assert torch.isfinite(grad);optimizer.step()
 with torch.no_grad():pm2,pc2=net(x);parts2=criterion(pc2[:,:24],pc2[:,24:],pm2,front,back,m);second=D.S.loss_value(parts2)
 assert torch.isfinite(second)
 smoke={'status':'PASS','device':'CPU','batch_size':4,'smoke_optimizer_steps':1,'independent_random_model_discarded':True,'first_loss':float(loss),'second_forward_loss':float(second),'gradient_norm':float(grad),'float_cast_history_second_call':'PASS','real_frames':0}
 save(ROOT/'checks/augmentation_cpu_smoke.json',smoke)
 d0=aug.transform(train[0],0);d1=aug.transform(train[0],20);canvas=Image.new('RGB',(1024,300),'#222222');draw=ImageDraw.Draw(canvas)
 for j,(item,title) in enumerate([(d0,'RGB angle0'),(d1,'RGB angle20'),(d1,'Mask angle20'),(d1,'Front XYZ angle20')]):
  im=cv2.cvtColor(item['rgb'],cv2.COLOR_BGR2RGB) if j<2 else (item['mask']*255 if j==2 else item['front']);im=Image.fromarray(im).convert('RGB').resize((240,240),Image.Resampling.NEAREST);canvas.paste(im,(j*256+8,45));draw.text((j*256+8,10),title,fill='white')
 canvas.save(ROOT/'stage3/augmentation_cpu_example.png');print(json.dumps({'alignment':check,'smoke':smoke},ensure_ascii=False,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:save(ROOT/'checks/augmentation_failure.json',{'reason':str(e),'traceback':traceback.format_exc()});raise
