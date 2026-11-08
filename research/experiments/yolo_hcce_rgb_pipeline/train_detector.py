import os,sys,subprocess,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(P/'.ultralytics');os.environ['MPLCONFIGDIR']=str(P/'.mpl')
from ultralytics import YOLO
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip(),'GPU already occupied'
a=json.loads((P/'dataset_manifest.json').read_text());s=json.loads((P/'split_manifest.json').read_text());assert a['pbr']['rejected']==0 and a['pbr']['train']==80 and a['pbr']['val']==20 and not s['image_hash_cross_split_duplicates'] and not s['pbr_to_dev_raw_exact_duplicates']
m=YOLO(str(P/'weights/yolo11n.pt'));assert len(m.names)==80,'not COCO pretrained'
m.train(data=str(P/'dataset.yaml'),epochs=80,patience=15,imgsz=640,batch=8,seed=42,workers=2,device=0,optimizer='auto',pretrained=True,plots=True,amp=True,project=str(P/'runs'),name='yolo11n_pbr_attempt3',exist_ok=False)
