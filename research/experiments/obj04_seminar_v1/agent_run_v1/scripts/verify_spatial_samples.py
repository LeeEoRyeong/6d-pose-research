"""回転ラベルを逆 affine による独立最近傍参照と比較する CPU 追加検査。"""
import json,cv2,numpy as np
from augmentation import ROOT,D,Augmentation

def main():
 samples,train,val=D.S.dataset();aug=Augmentation(samples,train);n=0;mismatch={k:0 for k in ['mask','front','back','valid']};centroid_errors=[]
 yy,xx=np.indices((128,128),dtype=float);dest=np.stack([xx,yy,np.ones_like(xx)],axis=-1)
 for frame in train[:8]:
  for angle in [-20.,-10.,10.,20.]:
   out=aug.transform(frame,angle);inv=cv2.invertAffineTransform(out['M128']);sx=dest@inv[0];sy=dest@inv[1]
   # Exclude near-half ties where OpenCV fixed-point rounding may legitimately differ.
   safe=(np.abs((sx-np.floor(sx))-.5)>.05)&(np.abs((sy-np.floor(sy))-.5)>.05)
   xi=np.floor(sx+.5).astype(int);yi=np.floor(sy+.5).astype(int);inside=(xi>=0)&(xi<128)&(yi>=0)&(yi<128);test=safe&inside;n+=int(test.sum())
   for k in mismatch:
    src=aug.data[frame][k];expected=src[yi[test],xi[test]];got=out[k][test];mismatch[k]+=int(np.count_nonzero(np.any(got!=expected,axis=-1) if got.ndim==2 else got!=expected))
 for angle in [-20.,-10.,10.,20.]:
  M=cv2.getRotationMatrix2D((64,64),angle,1.);M2=M.copy();M2[:,2]*=2
  for x,y in [(30,40),(64,64),(92,75),(48,98)]:
   mask=np.zeros((128,128),np.uint8);color=np.zeros((256,256),np.uint8);cv2.circle(mask,(x,y),3,255,-1);cv2.circle(color,(2*x,2*y),6,255,-1)
   m=cv2.warpAffine(mask,M,(128,128),flags=cv2.INTER_NEAREST);c=cv2.warpAffine(color,M2,(256,256),flags=cv2.INTER_LINEAR)
   y1,x1=np.indices(m.shape);y2,x2=np.indices(c.shape);v1=np.array([(x1*m).sum()/m.sum(),(y1*m).sum()/m.sum()]);v2=np.array([(x2*c).sum()/c.sum(),(y2*c).sum()/c.sum()])/2;centroid_errors.append(float(np.linalg.norm(v1-v2)))
 assert all(v==0 for v in mismatch.values()),mismatch
 assert max(centroid_errors)<.75,centroid_errors
 result={'status':'PASS','independent_reference_pixels_per_channel':n,'nearest_pixel_mismatches':mismatch,'half_pixel_tie_exclusion':0.05,'RGB_bilinear_vs_mask_nearest_marker_count':len(centroid_errors),'marker_centroid_max_difference_in_128_pixel':max(centroid_errors),'reference':'inverse affine computed independently at eligible interior destination pixels'}
 with (ROOT/'checks/augmentation_spatial_independent.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
