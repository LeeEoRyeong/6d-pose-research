from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
import argparse,base64,binascii,csv,hashlib,json,os,re,threading,uuid
from datetime import datetime,timezone
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;EXP=ROOT.parent;VIS=EXP/'bb100_visible_mask_phase16a1_20261110';ROB=EXP/'bb100_robotseg_phase16a2r_resume_20261110';FRAMES=tuple(f"{int(x['frame']):06d}" for x in json.loads((ROOT/'sam_generation.json').read_text())['frames']);STATUSES=('DRAFT','REVIEWED','ACCEPTED','REWORK')
def sha(data):return hashlib.sha256(data).hexdigest()
def mask_decode(data,values=(0,128,255)):
 if not data.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('PNG 형식 필요')
 a=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED)
 if a is None or a.dtype!=np.uint8 or a.shape[:2]!=(720,1280):raise ValueError('1280×720 8비트 PNG 필요')
 if a.ndim==3:
  if a.shape[2] not in (3,4) or not np.array_equal(a[:,:,0],a[:,:,1]) or not np.array_equal(a[:,:,1],a[:,:,2]) or (a.shape[2]==4 and np.any(a[:,:,3]!=255)):raise ValueError('불투명 회색 마스크 필요')
  a=a[:,:,0]
 if a.ndim!=2 or not np.isin(a,values).all():raise ValueError('0,128,255만 허용')
 return a
def png(a):
 ok,b=cv2.imencode('.png',a);assert ok;return b.tobytes()
class Store:
 def __init__(self,output=ROOT/'annotations'):
  self.output=Path(output); resolved=self.output.resolve()
  for source in (VIS.resolve(),ROB.resolve(),(EXP/'real_bb100_20261108').resolve()):
   if resolved==source or source in resolved.parents:raise ValueError('원본 데이터 디렉터리에 저장할 수 없습니다')
  self.output.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock();self.entries={}
  with (EXP/'bb100_finetuning_preparation_20261109/bb100_candidate_manifest.csv').open(encoding='utf-8-sig') as f:rows={int(x['frame']):x for x in csv.DictReader(f)}
  assert len(rows)==100 and len(FRAMES)==100
  self.robot_diff={int(x['frame']):int(x['B_D_difference']) for x in json.loads((ROOT/'robot_generation.json').read_text())['frames']}
  for fr in FRAMES:
   i=int(fr);r=rows[i]; candidate={}
   for key,k in [('A_CAD','A'),('B_SAM','B'),('C_CAD_SAM','C'),('D_SAM_ROBOT','D'),('E_CAD_SAM_ROBOT','E')]:
    path=ROOT/'candidates'/k/f'{fr}.png'
    if path.is_file():candidate[key]=path
   image=Path(r['image_path']);assert image.is_file();im=cv2.imread(str(image));assert im is not None and im.shape==(720,1280,3)
   for path in candidate.values():mask_decode(path.read_bytes(),(0,255))
   self.entries[fr]={'rgb':image,'occlusion':r['occlusion'],'candidates':candidate,'hashes':{str(path):sha(path.read_bytes()) for path in [image,*candidate.values()]}}
  self.write_status()
 def verify_inputs(self,fr):
  for p,h in self.entries[fr]['hashes'].items():
   if sha(Path(p).read_bytes())!=h:raise ValueError('입력 해시 변경: '+p)
 def latest(self,fr):
  d=self.output/fr/'versions';return max((p for p in d.glob('*') if (p/'metadata.json').is_file()),default=None) if d.exists() else None
 def state(self,fr):
  if fr not in self.entries:raise ValueError('프레임 오류')
  p=self.latest(fr)
  if not p:return {'frame':fr,'status':'DRAFT','revision':None,'saved':False,'candidate':None,'occlusion':self.entries[fr]['occlusion'],'available':list(self.entries[fr]['candidates']),'robot_removed_pixels':self.robot_diff[int(fr)]}
  m=json.loads((p/'metadata.json').read_text());raw=(p/'working_mask.png').read_bytes()
  if sha(raw)!=m['mask_sha256']:raise ValueError('저장본 해시 불일치')
  a=mask_decode(raw)
  key=m.get('candidate');source=self.entries[fr]['candidates'].get(key)
  if source is not None:
   ref=mask_decode(source.read_bytes(),(0,255));m['modified_pixels']=int(np.count_nonzero(a!=ref));m['modified']=bool(m['modified_pixels'])
  else:m['modified_pixels']=None;m['modified']=None
  m['save_path']=str(p/'working_mask.png');m['approved_utc']=m.get('saved_utc') if m['status']=='ACCEPTED' else None
  m['available']=list(self.entries[fr]['candidates']);m['robot_removed_pixels']=self.robot_diff[int(fr)];return m
 def write_status(self):
  path=self.output/'frame_status.csv';tmp=self.output/('status_'+uuid.uuid4().hex+'.tmp')
  with tmp.open('x',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['frame','occlusion','status','saved','revision','candidate','modified','modified_pixels','ignore_pixels','foreground_pixels','approved_utc','save_path']);w.writeheader()
   for fr in FRAMES:
    s=self.state(fr);w.writerow({k:s.get(k,'') for k in w.fieldnames})
  os.replace(tmp,path)
 def save(self,fr,raw,status,candidate,revision,confirmed,note):
  if status not in STATUSES:raise ValueError('유효하지 않은 상태')
  if status=='ACCEPTED' and confirmed is not True:raise ValueError('ACCEPTED는 사용자의 명시적 확인이 필요합니다')
  with self.lock:
   state=self.state(fr)
   if state['revision']!=revision:raise RuntimeError('다른 버전이 저장됨. 다시 불러오세요.')
   self.verify_inputs(fr)
   if candidate not in self.entries[fr]['candidates']:raise ValueError('초기 후보 없음')
   a=mask_decode(raw);fg=a==255;ign=a==128
   if not fg.any():raise ValueError('전경 없는 마스크는 저장할 수 없습니다')
   now=datetime.now(timezone.utc);rev=now.strftime('%Y%m%dT%H%M%S.%fZ')+'_'+uuid.uuid4().hex[:8];d=self.output/fr/'versions'/rev;d.mkdir(parents=True,exist_ok=False)
   encoded=png(a);(d/'working_mask.png').write_bytes(encoded)
   # These two files are a coupled evaluation representation, not an ignore-free BOP GT.
   (d/'evaluation_foreground_255.png').write_bytes(png(fg.astype('uint8')*255));(d/'evaluation_valid_255.png').write_bytes(png((~ign).astype('uint8')*255))
   im=cv2.imread(str(self.entries[fr]['rgb']));im[fg]=(im[fg]*.55+np.array([0,0,255])*.45).astype('uint8');im[ign]=(im[ign]*.55+np.array([0,220,255])*.45).astype('uint8');(d/'overlay.png').write_bytes(png(im))
   initial=mask_decode(self.entries[fr]['candidates'][candidate].read_bytes(),(0,255));modified_pixels=int(np.count_nonzero(a!=initial));box=list(map(int,cv2.boundingRect(fg.astype('uint8'))));info={'frame':fr,'occlusion':self.entries[fr]['occlusion'],'status':status,'saved':True,'revision':rev,'parent_revision':revision,'saved_utc':now.isoformat(),'candidate':candidate,'candidate_sha256':self.entries[fr]['hashes'][str(self.entries[fr]['candidates'][candidate])],'rgb_sha256':self.entries[fr]['hashes'][str(self.entries[fr]['rgb'])],'mask_sha256':sha(encoded),'modified':bool(modified_pixels),'modified_pixels':modified_pixels,'approved_utc':now.isoformat() if status=='ACCEPTED' else None,'save_path':str(d/'working_mask.png'),'foreground_pixels':int(fg.sum()),'ignore_pixels':int(ign.sum()),'background_pixels':int((a==0).sum()),'bbox_visib_xywh':box,'note':str(note)[:4000],'accepted_explicitly':status=='ACCEPTED' and confirmed is True,'is_accepted_reference':status=='ACCEPTED','binary_is_final_bop_compatible':status=='ACCEPTED' and not ign.any(),'warning':'128은 제외; 이진 전경과 유효영역을 함께 사용'}
   with (d/'metadata.json').open('x',encoding='utf-8') as f:json.dump(info,f,ensure_ascii=False,indent=2)
   self.write_status();return info
class Server(ThreadingHTTPServer):pass
def handler(store):
 class H(BaseHTTPRequestHandler):
  def out(self,code,data,typ='application/json; charset=utf-8'):
   if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode()
   self.send_response(code);self.send_header('Content-Type',typ);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
  def local(self):return self.headers.get('Host','').split(':')[0] in ('127.0.0.1','localhost')
  def do_GET(self):
   if not self.local():return self.out(403,{'error':'localhost only'})
   path=urlparse(self.path).path
   try:
    if path=='/':return self.out(200,(ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
    if path=='/app.js':return self.out(200,(ROOT/'app.js').read_bytes(),'text/javascript; charset=utf-8')
    if path=='/api/frames':return self.out(200,[store.state(f) for f in FRAMES])
    m=re.fullmatch(r'/api/(\d{6})/(rgb|state|saved|candidate/([A-Z_]+))',path)
    if not m or m[1] not in FRAMES:return self.out(404,{'error':'not found'})
    fr,kind,key=m.group(1,2,3);store.verify_inputs(fr)
    if kind=='state':return self.out(200,store.state(fr))
    if kind=='rgb':return self.out(200,store.entries[fr]['rgb'].read_bytes(),'image/jpeg')
    if kind=='saved':
     d=store.latest(fr)
     return self.out(200,(d/'working_mask.png').read_bytes(),'image/png') if d else self.out(404,{'error':'저장본 없음'})
    p=store.entries[fr]['candidates'].get(key)
    return self.out(200,p.read_bytes(),'image/png') if p else self.out(404,{'error':'후보 없음'})
   except (ValueError,OSError) as e:return self.out(400,{'error':str(e)})
  def do_POST(self):
   if not self.local():return self.out(403,{'error':'localhost only'})
   origin=self.headers.get('Origin')
   if origin and origin!='http://'+self.headers.get('Host',''):return self.out(403,{'error':'origin mismatch'})
   m=re.fullmatch(r'/api/(\d{6})/save',urlparse(self.path).path)
   if not m or m[1] not in FRAMES:return self.out(404,{'error':'not found'})
   try:
    n=int(self.headers.get('Content-Length','0'))
    if n<=0 or n>10*1024*1024:raise ValueError('요청 크기 오류')
    d=json.loads(self.rfile.read(n));raw=base64.b64decode(d['mask_png'],validate=True)
    return self.out(200,store.save(m[1],raw,d['status'],d['candidate'],d['revision'],d.get('confirmed',False),d.get('note','')))
   except RuntimeError as e:return self.out(409,{'error':str(e)})
   except (ValueError,KeyError,TypeError,OSError,binascii.Error) as e:return self.out(400,{'error':str(e)})
 return H
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=7867);p.add_argument('--output',type=Path,default=ROOT/'annotations');a=p.parse_args();server=Server(('127.0.0.1',a.port),handler(Store(a.output)));print(f'http://127.0.0.1:{a.port}',flush=True);server.serve_forever()
