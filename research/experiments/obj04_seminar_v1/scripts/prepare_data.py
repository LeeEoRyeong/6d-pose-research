import collections,hashlib,json,shutil
from common import ROOT,SOURCE,PBR,REAL,SCENE,write_json,load_json,verify_originals

def main():
    n=verify_originals()
    models=ROOT/'data/models';models.mkdir(exist_ok=False)
    for a,b in [(REAL/'obj_000004.ply',models/'obj_000004.ply'),(PBR/'models/models_info.json',models/'models_info.json')]:
        shutil.copyfile(a,b)
        assert hashlib.sha256(a.read_bytes()).digest()==hashlib.sha256(b.read_bytes()).digest()
    split={}
    for key in ('train','val'):
        file=PBR/f'yolo11/train_obj_s/autosplit_{key}.txt'
        split[key]=[int(x.rsplit('_',1)[1].split('.')[0]) for x in file.read_text().splitlines() if x.strip()]
    assert len(split['train'])==80 and len(split['val'])==20
    assert not set(split['train'])&set(split['val'])
    assert not {x//10 for x in split['train']}&{x//10 for x in split['val']}
    gt=load_json(SCENE/'scene_gt.json');cam=load_json(SCENE/'scene_camera.json');info=load_json(SCENE/'scene_gt_info.json')
    assert set(gt)==set(cam)==set(info)=={str(x) for x in range(100)}
    for i in range(100):
        assert len(gt[str(i)])==1 and gt[str(i)][0]['obj_id']==4
        for folder,suffix in [('rgb',f'{i:06d}.jpg'),('depth',f'{i:06d}.png'),('mask',f'{i:06d}_000000.png'),('mask_visib',f'{i:06d}_000000.png')]:
            assert (SCENE/folder/suffix).is_file()
    result={'protected_files':n,'scene':str(SCENE),'cad':str(models/'obj_000004.ply'),'model_info':str(models/'models_info.json'),'split':split,'train_blocks':sorted({x//10 for x in split['train']}),'val_blocks':sorted({x//10 for x in split['val']}),'frame_count':100,'real_training_frames':0,'reuse_audit':str(SOURCE/'validation_obj04_transfer_20261103/pbr_baseline_readiness_ko.md'),'known_gt_info_count_discrepancy_frames':[24,36,77],'gt_info_original_preserved':True,'physical_unit':'PBR provenance에 object_scale_m_per_mm=0.001 명시, 실물 치수 별도 미확인','real_dev_count':27,'real_dev_unique_exact_poses':15}
    write_json(ROOT/'data/dataset.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
