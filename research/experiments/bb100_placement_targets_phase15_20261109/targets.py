"""CPU diagnostic HCCE adapter. NOT a training Dataset; no validated visible mask.
Uses existing renderer and original crop/encoding functions. Does not canonicalize official symmetry.
"""
from pathlib import Path
import sys
P=Path('/home/oylee/projects/HCCEPose_experiments')
sys.path.insert(0,str(P/'bb_weak_supervision_20261108'))
import pipeline
class DiagnosticTargets:
 def __init__(self,ids):self.ids=tuple(map(int,ids));self.adapter=pipeline.BBAdapter(self.ids)
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  s=self.adapter.load(self.ids[i]); t=self.adapter.tensors(s)
  return {'frame':self.ids[i],'cad_amodal':s['cad'],'front_uncanonicalized':s['front'],'back_uncanonicalized':s['back'],'diagnostic_tensors':t,'status':'NOT_FOR_TRAINING'}
 def training_sample(self,i):
  raise RuntimeError('BLOCKED: verified visible supervision, physical split, symmetry canonicalization and training loop required')
