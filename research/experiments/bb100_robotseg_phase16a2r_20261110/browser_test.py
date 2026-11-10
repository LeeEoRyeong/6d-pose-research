import urllib.request,json,time
B='http://127.0.0.1:45675'
def req(m,p,v=None):
 with urllib.request.urlopen(urllib.request.Request(B+p,data=None if v is None else json.dumps(v).encode(),headers={'Content-Type':'application/json'},method=m),timeout=30) as f:return json.load(f)['value']
s=None
try:
 s=req('POST','/session',{'capabilities':{'alwaysMatch':{'browserName':'firefox','moz:firefoxOptions':{'args':['-headless']}}}})['sessionId'];r='/session/'+s
 req('POST',r+'/url',{'url':'file:///home/oylee/projects/HCCEPose_experiments/bb100_robotseg_phase16a2r_20261110/package/index.html'})
 def js(t):return req('POST',r+'/execute/sync',{'script':t,'args':[]})
 def aj(t):return req('POST',r+'/execute/async',{'script':t,'args':[]})
 page=js('return {cards:document.querySelectorAll("article").length,first:document.querySelector("article img").complete&&document.querySelector("article img").naturalWidth>0,warning:document.querySelector(".warn").textContent}')
 assert page['cards']==10 and page['first'] and '미실행' in page['warning']
 el=req('POST',r+'/element',{'using':'css selector','value':'article input[type=file]'});eid=el['element-6066-11e4-a52e-4f735466cecf'];path='/home/oylee/projects/HCCEPose_experiments/bb100_robotseg_phase16a2r_20261110/package/cad/000043_AMODAL_NOT_VISIBLE_GT.png';req('POST',r+'/element/'+eid+'/value',{'text':path});time.sleep(.5)
 v=js('let p=document.querySelector("article input[type=file]").parentElement;return {message:p.querySelector("span").textContent,enabled:!p.querySelector("button").disabled}');assert v['enabled'] and '내용 정확도 미검증' in v['message']
 js('let s=document.querySelector("article select");s.value="CORRECT";s.dispatchEvent(new Event("change"));let n=document.querySelector("textarea");n.value="test";n.dispatchEvent(new Event("input"))')
 out=aj('let done=arguments[arguments.length-1];URL.createObjectURL=(b)=>{b.text().then(done);return "blob:captured"};document.querySelector("#export").click()');assert 'CORRECT' in out and 'NOT_RUN_NO_CHECKPOINT' in out
 print(json.dumps({'cards':10,'blocked_warning':'PASS','valid_manual_png_format_test':'PASS','csv_export':'PASS'},ensure_ascii=False))
finally:
 if s:
  try:req('DELETE','/session/'+s)
  except Exception:pass
