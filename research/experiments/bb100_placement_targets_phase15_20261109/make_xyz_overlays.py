"""Generate four diagnostic RGB/XYZ overlays; output is never visible GT."""
import csv,cv2,numpy as np
from pathlib import Path
P=Path('/home/oylee/projects/HCCEPose_experiments');O=Path(__file__).resolve().parent
with (P/'bb100_finetuning_preparation_20261109/bb100_candidate_manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
for frame in (43,75,235,247):
 r=next(x for x in rows if int(x['frame'])==frame);rgb=cv2.imread(r['image_path']);g=r['pose_group'];mask=cv2.imread(str(O/f'amodal_masks/pose_{g}.png'),0)>0;images=[rgb]
 for side in ('front','back'):
  xyz=cv2.imread(str(O/f'xyz_raw_{side}/pose_{g}.png'));im=rgb.copy();im[mask]=(rgb[mask]*.4+xyz[mask]*.6).astype('uint8');images.append(im)
 canvas=np.hstack([cv2.resize(im,(640,360)) for im in images])
 for i,title in enumerate(('RGB','FRONT XYZ BLENDED NOT GT','BACK XYZ BLENDED NOT GT')):cv2.putText(canvas,title,(i*640+5,25),0,.6,(255,255,255),2)
 path=O/'previews'/f'{frame:06d}_RGB_XYZ_NOT_GT.jpg'
 if path.exists():raise FileExistsError(path)
 assert cv2.imwrite(str(path),canvas)
