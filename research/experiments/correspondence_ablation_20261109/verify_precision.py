from pathlib import Path
import numpy as np,csv,json
from ablate import choose_unique,choose_grid
P=Path(__file__).resolve().parent;C=P.parent/'pnp_stability_20261109/full/cache';out=[]
for f in sorted(C.glob('*.npz')):
 z=np.load(f);p3=z['points3d'];ids=[choose_unique(p3,i) for i in (3,5,7)];out.append({'cache':f.name,'n3':len(ids[0]),'n5':len(ids[1]),'n7':len(ids[2]),'same_selected_indices':bool(np.array_equal(ids[0],ids[1]) and np.array_equal(ids[1],ids[2]))})
assert len(out)==104
with (P/'full/dedup_precision_indices.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
# Unit checks: selection returns unchanged indices, representative tie goes to first index, grid nearest centre.
pts=np.array([[1,2,3],[1,2,3],[4,5,6]],np.float32);assert np.array_equal(choose_unique(pts,5),[0,2]);p2=np.array([[.1,.1],[.24,.24],[.8,.8]],np.float32);assert np.array_equal(choose_grid(p2,np.array([0,0,1,1]),np.arange(3),2),[1,2]);print('PASS: precision indices',sum(x['same_selected_indices'] for x in out),'of',len(out),'unit checks PASS')
