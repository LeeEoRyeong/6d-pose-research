from pathlib import Path
import json,hashlib,cv2,numpy as np,collections
from app import Store,FRAMES,mask_decode
P=Path('/home/oylee/projects/HCCEPose_experiments');O=Path(__file__).resolve().parent;s=Store();snapshot=json.loads((O/'accepted_snapshot.json').read_text());old=Path(snapshot['source']);matched=[]
for fr,z in snapshot['copied_user_accepted_frames'].items():
 d=s.latest(fr);src=old/fr/'versions'/z['revision'];assert d.name==z['revision'];info=s.state(fr);assert info['status']=='ACCEPTED' and info['accepted_explicitly'] and info['mask_sha256']==z['mask_sha256'];
 for p in src.rglob('*'):
  if p.is_file():q=O/'annotations'/fr/'versions'/p.relative_to(old/fr/'versions');assert q.is_file() and hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(q.read_bytes()).digest()
 a=mask_decode((d/'working_mask.png').read_bytes());valid=cv2.imread(str(d/'evaluation_valid_255.png'),0);binary=cv2.imread(str(d/'evaluation_foreground_255.png'),0);assert np.array_equal(valid,(a!=128).astype('uint8')*255) and np.array_equal(binary,(a==255).astype('uint8')*255);matched.append(fr)
for fr in FRAMES:
 for v in s.entries[fr]['candidates'].values():mask_decode(v.read_bytes(),(0,255))
status=collections.Counter((s.state(fr)['status'],s.state(fr)['saved']) for fr in FRAMES);assert status[('ACCEPTED',True)]==10 and status[('DRAFT',False)]==90
out={'frames':len(FRAMES),'candidate_masks':sum(len(x['candidates']) for x in s.entries.values()),'accepted_original_copies_checked':len(matched),'accepted_untouched':'PASS','approved_mask_ignore_reloads':'PASS','draft_unsaved':90,'source_training':'NOT_RUN','windows_browser':'NOT_TESTED'};(O/'final_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
