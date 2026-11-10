from pathlib import Path
import importlib.util,argparse
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'bb100_visible_gt_phase16a4_20261110';spec=importlib.util.spec_from_file_location('phase16a4_base_editor',OLD/'app.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def make_app(output):
 store=base.Store(output)
 candidate=HERE/'F_HYBRID_000063_NOT_GT.png';data=candidate.read_bytes();base.mask_decode(data,(0,255));store.entries['000063']['candidates']['F_HYBRID']=candidate;store.entries['000063']['hashes'][str(candidate)]=base.sha(data)
 # Change static assets only after original Store initializes its A–E paths.
 base.ROOT=HERE
 class H(base.handler(store)):
  def do_GET(self):
   path=base.urlparse(self.path).path
   if path in ('/hybrid/uncertain','/hybrid/comparison'):
    if not self.local():return self.out(403,{'error':'localhost only'})
    source=HERE/('H_CAD_uncertain_review.png' if path.endswith('uncertain') else '000063_B_C_D_F_comparison.png');return self.out(200,source.read_bytes(),'image/png')
   return super().do_GET()
 return H
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=7870);p.add_argument('--output',type=Path,default=OLD/'annotations');a=p.parse_args();server=base.Server(('127.0.0.1',a.port),make_app(a.output));print(f'http://127.0.0.1:{a.port}',flush=True);server.serve_forever()
