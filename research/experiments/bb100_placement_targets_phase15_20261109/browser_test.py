import urllib.request,json,time
base='http://127.0.0.1:45679'
def call(method,path,data=None):
 req=urllib.request.Request(base+path,data=None if data is None else json.dumps(data).encode(),method=method,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=30) as f:return json.load(f)['value']
sid=None
try:
 sid=call('POST','/session',{'capabilities':{'alwaysMatch':{'browserName':'firefox','moz:firefoxOptions':{'args':['-headless']}}}})['sessionId'];root='/session/'+sid
 call('POST',root+'/url',{'url':'file:///home/oylee/projects/HCCEPose_experiments/bb100_placement_targets_phase15_20261109/review_package/index.html'})
 def js(code):return call('POST',root+'/execute/sync',{'script':code,'args':[]})
 print('page',js('return {title:document.title, count:document.querySelectorAll("article").length, images:document.querySelectorAll("img").length, first:document.querySelector("img").complete && document.querySelector("img").naturalWidth>0};'))
 print('change',js('let s=document.querySelector("article select");s.value="ACCEPT";s.dispatchEvent(new Event("change"));let n=document.querySelector("article textarea");n.value="test, quote \\\" newline";n.dispatchEvent(new Event("input"));return document.querySelector("#progress").textContent;'))
 print('filter',js('let f=document.querySelector("#filter");f.value="SAME_POSE_CHECK";f.dispatchEvent(new Event("input"));return [...document.querySelectorAll("article")].filter(e=>!e.hidden).length;'))
 call('POST',root+'/refresh',{});print('persist',js('return {status:document.querySelector("article select").value, note:document.querySelector("article textarea").value, progress:document.querySelector("#progress").textContent}'))
 print('csv_parser',js('return parseCSV("a,b\\n\\\"x,y\\\",\\\"z\\\"\\\"k\\\"\\n");'))
finally:
 if sid:
  try:call('DELETE','/session/'+sid)
  except Exception:pass
