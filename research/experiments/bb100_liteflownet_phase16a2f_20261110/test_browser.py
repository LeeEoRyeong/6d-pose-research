import urllib.request,json
B='http://127.0.0.1:45676'
def q(m,p,v=None):
 with urllib.request.urlopen(urllib.request.Request(B+p,data=None if v is None else json.dumps(v).encode(),headers={'Content-Type':'application/json'},method=m),timeout=30) as f:return json.load(f)['value']
s=None
try:
 s=q('POST','/session',{'capabilities':{'alwaysMatch':{'browserName':'firefox','moz:firefoxOptions':{'args':['-headless']}}}})['sessionId'];r='/session/'+s;q('POST',r+'/url',{'url':'file:///home/oylee/projects/HCCEPose_experiments/bb100_liteflownet_phase16a2f_20261110/package/index.html'})
 def js(t):return q('POST',r+'/execute/sync',{'script':t,'args':[]})
 def aj(t):return q('POST',r+'/execute/async',{'script':t,'args':[]})
 a=js('return {cards:document.querySelectorAll("article").length,images:document.querySelectorAll("article img").length,first:document.querySelector("article img").complete && document.querySelector("article img").naturalWidth>0}');assert a['cards']==6 and a['images']==6 and a['first']
 js('let s=document.querySelector("article select");s.value="UNCERTAIN";s.dispatchEvent(new Event("change"));let n=document.querySelector("article textarea");n.value="수동 테스트";n.dispatchEvent(new Event("input"))')
 out=aj('let done=arguments[arguments.length-1];URL.createObjectURL=(b)=>{b.text().then(done);return "blob:captured"};document.querySelector("#export").click()');assert 'UNCERTAIN' in out and '수동 테스트' in out
 q('POST',r+'/refresh',{});assert js('return document.querySelector("article select").value')=='UNCERTAIN'
 print(json.dumps({'cards':6,'images':'PASS','export':'PASS','reload':'PASS'}))
finally:
 if s:
  try:q('DELETE','/session/'+s)
  except Exception:pass
