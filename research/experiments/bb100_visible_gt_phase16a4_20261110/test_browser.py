import urllib.request,json,time
B='http://127.0.0.1:45671'
def q(m,p,v=None):
 req=urllib.request.Request(B+p,data=None if v is None else json.dumps(v).encode(),headers={'Content-Type':'application/json'},method=m)
 with urllib.request.urlopen(req,timeout=45) as f:return json.load(f)['value']
s=None
try:
 s=q('POST','/session',{'capabilities':{'alwaysMatch':{'browserName':'firefox','moz:firefoxOptions':{'args':['-headless']}}}})['sessionId'];r='/session/'+s;q('POST',r+'/url',{'url':'http://127.0.0.1:7869/'})
 def js(t):return q('POST',r+'/execute/sync',{'script':t,'args':[]})
 time.sleep(3);v=js('return {n:document.querySelector("#frame").options.length,frame:document.querySelector("#frame").value,candidate:document.querySelector("#candidate").value,msg:document.querySelector("#message").textContent}')
 assert v['n']==100 and v['candidate']=='D_SAM_ROBOT',v
 js('document.querySelector("#peek_B_SAM").click()');assert 'B_SAM' in js('return document.querySelector("#previewLabel").textContent')
 v=js('let z=document.querySelector("#zoom");z.value=200;z.dispatchEvent(new Event("input"));let r=document.querySelector("#editor").getBoundingClientRect();let p=pos({clientX:r.left+200,clientY:r.top+240});return {x:p.x,y:p.y}');assert abs(v['x']-100)<.01 and abs(v['y']-120)<.01
 js('document.querySelector("#ignore").click();document.querySelector(".viewport").scrollTop=0;document.querySelector(".viewport").scrollLeft=0;document.querySelector("#editor").scrollIntoView()');time.sleep(.2);xy=js('let r=document.querySelector("#editor").getBoundingClientRect();return {x:Math.round(r.left+200),y:Math.round(r.top+240),h:innerHeight,w:innerWidth}');assert 0<=xy['x']<xy['w'] and 0<=xy['y']<xy['h'],xy
 q('POST',r+'/actions',{'actions':[{'type':'pointer','id':'m','parameters':{'pointerType':'mouse'},'actions':[{'type':'pointerMove','duration':0,'x':xy['x'],'y':xy['y'],'origin':'viewport'},{'type':'pointerDown','button':0},{'type':'pointerUp','button':0}]}]});q('DELETE',r+'/actions');assert not js('return document.querySelector("#undo").disabled');js('document.querySelector("#undo").click()');assert not js('return document.querySelector("#redo").disabled');js('document.querySelector("#redo").click();document.querySelector("#draft").click()');time.sleep(1);assert '저장 ' in js('return document.querySelector("#message").textContent')
 print(json.dumps({'100_frames':'PASS','D_default':'PASS','preview':'PASS','zoom_pointer_undo_redo_save':'PASS'}))
finally:
 if s:q('DELETE','/session/'+s)
