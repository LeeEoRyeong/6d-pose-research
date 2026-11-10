import urllib.request,json
base='http://127.0.0.1:45677'
def req(m,p,v=None):
 with urllib.request.urlopen(urllib.request.Request(base+p,data=None if v is None else json.dumps(v).encode(),headers={'Content-Type':'application/json'},method=m),timeout=30) as f:return json.load(f)['value']
s=None
try:
 s=req('POST','/session',{'capabilities':{'alwaysMatch':{'browserName':'firefox','moz:firefoxOptions':{'args':['-headless']}}}})['sessionId'];r='/session/'+s
 req('POST',r+'/url',{'url':'file:///home/oylee/projects/HCCEPose_experiments/bb100_visible_mask_phase16a1_20261110/package/index.html'})
 def js(t):return req('POST',r+'/execute/sync',{'script':t,'args':[]})
 def aj(t):return req('POST',r+'/execute/async',{'script':t,'args':[]})
 p=js('return {cards:document.querySelectorAll("article").length,first:document.querySelector("article img").complete&&document.querySelector("article img").naturalWidth>0}')
 assert p['cards']==10 and p['first']
 js('let el=document.querySelector("article");let s=el.querySelectorAll("select")[1];s.value="CORRECT";s.dispatchEvent(new Event("change"));let n=el.querySelector("textarea");n.value="메모, 테스트";n.dispatchEvent(new Event("input"))')
 out=aj('let done=arguments[arguments.length-1];URL.createObjectURL=(b)=>{b.text().then(done);return "blob:captured"};document.querySelector("#export").click()');assert 'CORRECT' in out and '메모, 테스트' in out
 imp=aj('let done=arguments[arguments.length-1];let text="frame,occlusion,pose_group,method,status,note\\n43,NONE,43,B_SAM,ACCEPT,old\\n247,NONE,235,B_SAM,UNCERTAIN,imported\\n";let f=new File([text],"old.csv",{type:"text/csv"}),dt=new DataTransfer();dt.items.add(f);let x=document.querySelector("#import");x.files=dt.files;x.dispatchEvent(new Event("change"));setTimeout(()=>done({msg:document.querySelector("#message").textContent,s:JSON.parse(localStorage.getItem("hcce_16a1_visible_review_v1"))}),500)');assert imp['s']['43']['status']=='CORRECT' and imp['s']['247']['status']=='UNCERTAIN'
 req('POST',r+'/refresh',{});assert js('return document.querySelector("article select:nth-of-type(2)").value')=='CORRECT'
 print(json.dumps({'cards':10,'first_image_loaded':'PASS','export_csv':'PASS','import_preserves_existing':'PASS','reload':'PASS'},ensure_ascii=False))
finally:
 if s:
  try:req('DELETE','/session/'+s)
  except Exception:pass
