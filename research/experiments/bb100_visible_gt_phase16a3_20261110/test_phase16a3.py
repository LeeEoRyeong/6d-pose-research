import unittest,tempfile,cv2,numpy as np,json
from pathlib import Path
from app import Store,mask_decode,png
from evaluate import metrics
class TestPhase16A3(unittest.TestCase):
 def test_ignore_excluded(self):
  g=np.array([[255,255,0],[0,128,128]],np.uint8);p=np.array([[255,0,255],[0,255,0]],np.uint8);m=metrics(g,p)
  self.assertEqual([m[k] for k in ('tp','fp','fn','tn','ignore_pixels')],[1,1,1,1,2]);self.assertEqual(m['iou'],1/3);self.assertEqual(m['dice'],.5);self.assertEqual(m['false_positive_rate'],.5);self.assertEqual(m['false_negative_rate'],.5)
 def test_immutability_status_and_reload(self):
  with tempfile.TemporaryDirectory() as tmp:
   st=Store(Path(tmp));self.assertEqual(sum(st.state(f)['saved'] for f in st.entries),0);self.assertEqual(sum('D_SAM_ROBOT' in x['candidates'] for x in st.entries.values()),6)
   fr='000075';a=mask_decode(st.entries[fr]['candidates']['C_CAD_SAM'].read_bytes());a[0,0]=128;raw=png(a)
   with self.assertRaises(ValueError):st.save(fr,raw,'ACCEPTED','C_CAD_SAM',None,False,'test')
   x=st.save(fr,raw,'DRAFT','C_CAD_SAM',None,False,'test');d=st.latest(fr);self.assertEqual(mask_decode((d/'working_mask.png').read_bytes())[0,0],128);self.assertEqual(cv2.imread(str(d/'evaluation_valid_255.png'),0)[0,0],0);self.assertEqual(st.state(fr)['status'],'DRAFT')
   with self.assertRaises(RuntimeError):st.save(fr,raw,'DRAFT','C_CAD_SAM',None,False,'stale')
   y=st.save(fr,raw,'ACCEPTED','C_CAD_SAM',x['revision'],True,'manual test only');self.assertTrue(y['accepted_explicitly']);self.assertFalse(y['binary_is_final_bop_compatible']);self.assertTrue((d/'metadata.json').is_file());self.assertEqual(st.state(fr)['status'],'ACCEPTED')
 def test_invalid(self):
  with self.assertRaises(ValueError):mask_decode(png(np.full((720,1280),77,np.uint8)))
if __name__=='__main__':unittest.main(verbosity=2)
