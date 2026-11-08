"""PBR crop에 동기 in-plane affine만 추가. 모델 좌표값은 유지하고 HCCE는 재생성."""
from pathlib import Path
import sys,cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;sys.path.insert(0,str(BASE/'diagnostics_v1/scripts'));import diagnose as D
MEAN=torch.tensor([.406,.456,.485])[:,None,None];STD=torch.tensor([.225,.224,.229])[:,None,None]
class Augmentation:
 def __init__(self,samples,train):
  self.samples=samples;self.data={}
  for frame in train:
   s=samples[frame];box=s['bbox'].numpy();ff=cv2.imread(str(BASE/'labels/front'/f'{frame:06d}_000000.png'));bb=cv2.imread(str(BASE/'labels/back'/f'{frame:06d}_000000.png'))
   assert ff is not None and bb is not None
   valid=D.CROP((ff.any(-1)&bb.any(-1)).astype(np.uint8),box,128,cv2.INTER_NEAREST)
   self.data[frame]={'rgb':s['rgb_crop'],'mask':(s['tensors'][1].numpy()>.5).astype(np.uint8),'front':D.CROP(ff,box,128,cv2.INTER_NEAREST),'back':D.CROP(bb,box,128,cv2.INTER_NEAREST),'valid':valid}
 def transform(self,frame,angle):
  source=self.data[frame];m128=cv2.getRotationMatrix2D((64.,64.),float(angle),1.);m256=m128.copy();m256[:,2]*=2
  rgb=cv2.warpAffine(source['rgb'],m256,(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
  out={'rgb':rgb,'angle':float(angle),'M128':m128,'M256':m256}
  for key in ('mask','front','back','valid'):out[key]=cv2.warpAffine(source[key],m128,(128,128),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
  return out
 def tensors(self,frame,angle):
  d=self.transform(frame,angle);image=torch.from_numpy(np.ascontiguousarray(d['rgb'].transpose(2,0,1))).float()/255
  gt=[(image-MEAN)/STD,torch.tensor(d['mask'],dtype=torch.float32)]
  for key in ('front','back'):gt.append(torch.from_numpy(np.ascontiguousarray(D.ENC(None,d[key]).transpose(2,0,1))).float())
  return gt,d
 def batch(self,ids,angles,device):
  rows=[self.tensors(i,a)[0] for i,a in zip(ids,angles)]
  return [torch.stack([v[k] for v in rows]).to(device) for k in range(4)]
