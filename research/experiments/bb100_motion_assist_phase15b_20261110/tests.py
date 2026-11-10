import numpy as np,cv2,json
from pathlib import Path
from analyze import fit,transform
rng=np.random.default_rng(42);a=rng.uniform([0,0],[1200,700],(150,2)).astype('float32');b=a+np.array([7.,-4.],np.float32);b[:20]=rng.uniform([0,0],[1200,700],(20,2));M,ins,n,ratio,res,cov=fit(a,b,True);d=transform(M,[500,300])-np.array([500,300]);assert n>=125 and np.linalg.norm(d-[7,-4])<.01
p=rng.uniform([350,200],[550,400],(120,2)).astype('float32');q=p+np.array([10.,-2.],np.float32);q[:15]=rng.uniform([350,200],[550,400],(15,2));A,ins,n,ratio,res,cov=fit(p,q);delta=transform(A,[450,300])-transform(M,[450,300]);assert n>=100 and np.linalg.norm(delta-[3,2])<.01
assert fit(a[:3],b[:3],True)[0] is None and fit(p[:3],q[:3],False)[0] is None
result={'known_background_translation_with_outliers':'PASS','known_object_minus_background_translation':'PASS','insufficient_correspondences':'PASS','scope':'synthetic geometry only; does not verify foreground feature ownership'}
(Path(__file__).resolve().parent/'synthetic_tests.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
