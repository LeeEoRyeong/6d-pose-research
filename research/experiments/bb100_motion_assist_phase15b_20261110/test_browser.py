import urllib.request,json
base='http://127.0.0.1:45678'
def req(m,p,v=None):
 with urllib.request.urlopen(urllib.request.Request(base+p,data=None if v is None else json.dumps(v).encode(),headers={'Content-Type':'application/json'},method=m),timeout=30) as f:return json.load(f)['value']
s=None
try:
 s=req('POST','/session',{'capabilities':{'alwaysMatch':{'browserName':'firefox','moz:firefoxOptions':{'args':['-headless']}}}})['sessionId'];r='/session/'+s
 req('POST',r+'/url',{'url':'file:///home/oylee/projects/HCCEPose_experiments/bb100_motion_assist_phase15b_20261110/package/index.html'})
 def js(t):return req('POST',r+'/execute/sync',{'script':t,'args':[]})
 def ajs(t):return req('POST',r+'/execute/async',{'script':t,'args':[]})
 page=js('return {cards:document.querySelectorAll("article").length,images:document.querySelectorAll("img").length,first:document.querySelector("img").complete&&document.querySelector("img").naturalWidth>0}')
 assert page['cards']==99 and page['images']==297 and page['first']
 js('let s=document.querySelector("article select");s.value="ACCEPT";s.dispatchEvent(new Event("change"));let n=document.querySelector("textarea");n.value="manual,keep";n.dispatchEvent(new Event("input"));')
 imp=ajs('let done=arguments[arguments.length-1];let text="boundary_id,before_frame,after_frame,status,note\\nB001,31,32,REJECT,from_old\\nB002,32,34,ACCEPT,imported\\n";let f=new File([text],"old.csv",{type:"text/csv"});let dt=new DataTransfer();dt.items.add(f);let x=document.querySelector("#import");x.files=dt.files;x.dispatchEvent(new Event("change"));setTimeout(()=>done({message:document.querySelector("#message").textContent,state:JSON.parse(localStorage.getItem("hcce_15b_manual_review_v1"))}),500)')
 assert imp['state']['B001']=={'status':'ACCEPT','note':'manual,keep'} and imp['state']['B002']['status']=='ACCEPT'
 out=ajs('let done=arguments[arguments.length-1];URL.createObjectURL=(blob)=>{blob.text().then(done);return "blob:captured"};document.querySelector("#export").click();')
 assert 'B001' in out and 'manual,keep' in out and 'B002' in out and 'imported' in out
 req('POST',r+'/refresh',{});persist=js('return {s:document.querySelector("article select").value,n:document.querySelector("textarea").value,p:document.querySelector("#progress").textContent}')
 assert persist['s']=='ACCEPT' and persist['n']=='manual,keep'
 print(json.dumps({'page':page,'import_preserves_existing':True,'import_message':imp['message'],'csv_export_content':'PASS','reload':persist},ensure_ascii=False))
finally:
 if s:
  try:req('DELETE','/session/'+s)
  except Exception:pass
