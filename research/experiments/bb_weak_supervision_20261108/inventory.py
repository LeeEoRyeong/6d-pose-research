"""Metadata-only fail-closed eligibility audit. Does not render labels or train."""
import csv,datetime,json,uuid
from pathlib import Path
from pipeline import BBAdapter,BB,load_json

HERE=Path(__file__).resolve().parent

def main():
    gt=load_json(BB/'BB_obj04/scene_gt_obj04.json');ids=sorted(map(int,gt));adapter=BBAdapter(ids)
    assert len(ids)==100 and set(ids)==set(adapter.rgb_paths)
    folder=HERE/'runs'/('inventory_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True,exist_ok=False)
    rows=[]
    for i in ids:
        gid=next(g for g,items in adapter.group_map.items() if i in items)
        near=[j for j in adapter.group_map[gid] if j!=i and abs(j-i)<=adapter.gap]
        ref=min(near,key=lambda j:(abs(j-i),j)) if near else None
        rows.append({'frame':i,'state':adapter.rgb_paths[i].parent.name,'pose_group':gid,'nearest_same_pose_candidate':ref,
                     'weak_supervision_eligible':False,'low_occlusion_selected':False,'decision':'excluded_no_validated_occlusion_estimator',
                     'valid_pixel_fraction':'not_measured','reason':'fail-closed policy; no verified confidence/occlusion model'})
    with (folder/'frame_selection.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    result={'frames':len(rows),'candidate_pose_groups':len(adapter.group_map),'weak_supervision_eligible':0,'low_occlusion_selected':0,
            'not_measured':len(rows),'exclusion_reason':'automatic occlusion estimator unavailable; gates must remain zero','no_rendering_or_training':True}
    with (folder/'summary.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False));print('RESULT',folder)
if __name__=='__main__':main()
