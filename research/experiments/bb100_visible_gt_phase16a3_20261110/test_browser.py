import urllib.request,json,time
B='http://127.0.0.1:45672'
def q(m,p,v=None):
 req=urllib.request.Request(B+p,data=None if v is None else json.dumps(v).encode(),headers={'Content-Type':'application/json'},method=m)
 try:
  with urllib.request.urlopen(req,timeout=30) as f:return json.load(f)['value']
 except urllib.error.HTTPError as e:print(e.read().decode());raise
s=None
try:
 s=q('POST','/session',{'capabilities':{'alwaysMatch':{'browserName':'firefox','moz:firefoxOptions':{'args':['-headless']}}}})['sessionId'];r='/session/'+s;q('POST',r+'/url',{'url':'http://127.0.0.1:7868/'})
 def js(t):return q('POST',r+'/execute/sync',{'script':t,'args':[]})
 time.sleep(1);assert js('return document.querySelector("canvas").width')==1280
 v=js('let z=document.querySelector("#zoom");z.value=200;z.dispatchEvent(new Event("input"));let c=document.querySelector("#editor");let r=c.getBoundingClientRect();let p=pos({clientX:r.left+200,clientY:r.top+240});return {x:p.x,y:p.y}');assert abs(v['x']-100)<.01 and abs(v['y']-120)<.01,v
 js('document.querySelector("#ignore").click();document.querySelector(".viewport").scrollTop=0;document.querySelector(".viewport").scrollLeft=0;document.querySelector("#editor").scrollIntoView();')
 time.sleep(.2);xy=js('let r=document.querySelector("#editor").getBoundingClientRect();return {x:Math.round(r.left+200),y:Math.round(r.top+240),h:innerHeight,w:innerWidth}')
 assert 0<=xy['x']<xy['w'] and 0<=xy['y']<xy['h'],xy
 q('POST',r+'/actions',{'actions':[{'type':'pointer','id':'m','parameters':{'pointerType':'mouse'},'actions':[{'type':'pointerMove','duration':0,'x':xy['x'],'y':xy['y'],'origin':'viewport'},{'type':'pointerDown','button':0},{'type':'pointerUp','button':0}]}]});q('DELETE',r+'/actions')
 assert js('return document.querySelector("#undo").disabled')==False
 js('document.querySelector("#undo").click()');assert js('return document.querySelector("#redo").disabled')==False
 js('document.querySelector("#redo").click();document.querySelector("#draft").click()');time.sleep(1);m=js('return document.querySelector("#message").textContent');assert '저장 ' in m,m
 print(json.dumps({'browser_render':'PASS','zoom_coordinate':'PASS','pointer_stroke_undo_redo':'PASS','save':'PASS','reload_data':'check server test output'}))
finally:
 if s:q('DELETE','/session/'+s)
