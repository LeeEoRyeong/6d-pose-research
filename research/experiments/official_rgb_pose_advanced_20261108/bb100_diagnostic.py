"""Official checkpoint, inference-only, BB100 one-session diagnostic; oracle ROI."""
import sys,json,csv,torch,numpy as np,cv2
from pathlib import Path
OLD=Path('/RESEARCH_DATA/HCCEPose_experiments/itodd_symmetry_eval_20261108');sys.path.insert(0,str(OLD));import evaluate as e,symmetry
HERE=Path(__file__).resolve().parent

def main():
 assert json.load((HERE/'reproduction.json').open())['status']=='PASS';audit=json.load((HERE/'bb100_summary.json').open());assert audit['bb_frames']==100 and audit['raw_hashes_unchanged'] and audit['undistort_existing_max_mae']<1 and audit['all_projection_in_view']
 torch.set_num_threads(2);cv2.setNumThreads(1);apps=e.subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip();assert not apps,apps
 v,faces,syms,info=symmetry.load();mod=e.explicit_module();li=e.json_read(e.BASE/'data/models/models_info.json')['4'];lo=np.array([li['min_'+a] for a in 'xyz']);sz=np.array([li['size_'+a] for a in 'xyz']);minn=torch.tensor(lo,dtype=torch.float32,device='cuda');size=torch.tensor(sz,dtype=torch.float32,device='cuda')
 with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64'))]):ck=torch.load(e.AUD/'itodd_obj04_1_0000step50000.pt',map_location='cpu',weights_only=True)
 net=mod.HccePose_BF_Net(efficientnet_key=1,input_channels=3,min_xyz=minn,size_xyz=size)
 BB=e.EXP/'real_bb100_20261108/BB_obj04';gt=e.json_read(BB/'scene_gt_obj04.json');cam=e.json_read(BB/'scene_camera_obj04.json');crop=e.original_function('HccePose/bop_loader.py','crop_square_resize',{'np':np,'cv2':cv2});pad=e.original_function('HccePose/bop_loader.py','pad_square_fp32',{'np':np});items=[];files=[]
 for i in sorted(map(int,gt)):
  x=gt[str(i)][0];R=np.array(x['cam_R_m2c']).reshape(3,3);t=np.array(x['cam_t_m2c']);K=np.array(cam[str(i)]['cam_K']).reshape(3,3);uv=e.project(v,R,t,K);low=uv.min(0);high=uv.max(0);box=pad(np.r_[low,high-low],1.5);p=HERE/'bb100_undistorted_K_original'/f'{i:06d}.png';im=cv2.imread(str(p));assert im is not None;files.append(p);items.append(e.sample(i,im,K,R,t,box,'bb100',crop))
 assert len(items)==100;cp=e.AUD/'itodd_obj04_1_0000step50000.pt';before={str(p):e.sha(p) for p in files+[cp,BB/'scene_gt_obj04.json',BB/'scene_camera_obj04.json']}
 out=HERE/'bb100_inference';out.mkdir(exist_ok=False);rows,peak=e.run_one('official',net,ck,[('bb100',items)],syms,v,faces,info,lo,sz,out,True);after={p:e.sha(Path(p)) for p in before};assert before==after
 with (HERE/'bb100_pose_diagnostic.csv').open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
 result={'status':'CONDITIONAL_DEVELOPMENT_DIAGNOSTIC_NOT_INDEPENDENT','frames':len(rows),'PnP_valid':sum(r['pnp_valid'] for r in rows),'ADD_pass':sum(r['ADD_pass'] for r in rows),'ADD_S_pass':sum(r['ADD_S_pass'] for r in rows),'symmetry_ADD_pass':sum(r['symmetry_ADD_pass'] for r in rows),'group_source':'two overlapping components in one recorded session','ROI':'GT CAD projection oracle pad1.5','K':'new remap uses dev27 K,D, BB K equal; BB D provenance unverified','pose_quality':'not independently validated','peak':peak,'input_hashes_unchanged':True,'source_file_count':len(before)}
 with (HERE/'bb100_diagnostic_summary.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('BB100 DIAGNOSTIC',json.dumps(result),flush=True)
if __name__=='__main__':main()
