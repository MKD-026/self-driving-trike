"""Read-only HTTP dashboard for the live modular autonomy pipeline."""
from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2

DASHBOARD_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Trike autonomy</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;background:#07101b;color:#e7eff8}header{padding:13px 18px;background:#101d2c;border-bottom:1px solid #2a3b50;display:flex;justify-content:space-between}h1{font-size:19px;margin:0}.live,.ok{color:#35dc95}.stale,.danger{color:#ffbd55}main{padding:14px;display:grid;grid-template-columns:minmax(0,3fr) minmax(290px,1fr);gap:14px}.card{background:#101c2a;border:1px solid #293c52;border-radius:12px;overflow:hidden}img{width:100%;display:block;background:#020509}.side{padding:14px}h2{margin:0 0 12px;font-size:15px}dl{display:grid;grid-template-columns:1fr 1.3fr;gap:8px 10px;margin:0}dt{color:#91a6bd}dd{margin:0;font:13px ui-monospace,monospace;overflow-wrap:anywhere}.wide{grid-column:1/-1;border-top:1px solid #2a3b50;padding-top:9px}@media(max-width:900px){main{grid-template-columns:1fr}}
</style></head><body><header><h1>Trike autonomy monitor</h1><div id="connection" class="stale">CONNECTING</div></header><main><section class="card"><img src="/stream.mjpg" alt="Live perception"></section><aside class="card side"><h2>Live state</h2><dl>
<dt>Pipeline</dt><dd id="pipeline">—</dd><dt>Control</dt><dd id="control">—</dd><dt>Planner</dt><dd id="planner">—</dd><dt>Steering</dt><dd id="steering">—</dd><dt>Wheel target</dt><dd id="servo">—</dd><dt>Haptic</dt><dd id="haptic">—</dd><dt>Reason</dt><dd id="reason">—</dd><dt>Road</dt><dd id="road">—</dd><dt>Obstacles</dt><dd id="obstacles">—</dd><dt>Nearest</dt><dd id="nearest">—</dd><dt>Route</dt><dd id="route">—</dd><dt>GPS</dt><dd id="gps">—</dd><dt>Goal</dt><dd id="goal">—</dd><dt>Models parallel</dt><dd id="models">—</dd><dt>Detection</dt><dd id="detect">—</dd><dt>Segmentation</dt><dd id="segment">—</dd><dt>Depth</dt><dd id="depth">—</dd><dt>Frame age</dt><dd id="age">—</dd><dt class="wide">Detected objects</dt><dd id="objects" class="wide">—</dd></dl></aside></main><script>
const e=id=>document.getElementById(id),v=(x,s='')=>(x===null||x===undefined)?'—':`${x}${s}`,ms=x=>x==null?'—':`${(1000*x).toFixed(1)} ms`;
async function refresh(){try{const s=await (await fetch('/api/state',{cache:'no-store'})).json(),a=Math.max(0,Date.now()/1000-s.updated_at);e('connection').textContent=a<2?'LIVE':`STALE ${a.toFixed(1)}s`;e('connection').className=a<2?'live':'stale';e('pipeline').textContent=v(s.state);e('control').textContent=s.auto_enabled&&s.controller_armed?'ARMED':'DISARMED';e('planner').textContent=v(s.planner);e('steering').textContent=s.steering_normalized==null?'—':`${Number(s.steering_normalized).toFixed(3)} ${s.steering_direction}`;e('servo').textContent=s.steering_target==null?'—':`${Number(s.steering_target).toFixed(1)}°`;e('haptic').textContent=v(s.haptic);e('reason').textContent=v(s.reason);e('road').textContent=s.road_visible?(s.path_blocked?'BLOCKED':'VISIBLE'):'NOT VISIBLE';e('road').className=s.road_visible&&!s.path_blocked?'ok':'danger';e('obstacles').textContent=v(s.obstacle_count);e('nearest').textContent=s.nearest_forward_m==null?'—':`${Number(s.nearest_forward_m).toFixed(2)} m`;e('route').textContent=v(s.route);e('gps').textContent=s.gps_valid?`${Number(s.latitude).toFixed(7)}, ${Number(s.longitude).toFixed(7)}`:'STALE / UNAVAILABLE';e('goal').textContent=s.goal_forward_m==null?'—':`${Number(s.goal_forward_m).toFixed(2)}m forward, ${Number(s.goal_left_m).toFixed(2)}m left`;const t=s.timings_s||{};e('models').textContent=ms(t.parallel_wall_s);e('detect').textContent=ms(t.detection_model_s);e('segment').textContent=ms(t.segmentation_model_s);e('depth').textContent=ms(t.depth_model_s);e('age').textContent=`${a.toFixed(2)} s`;e('objects').textContent=(s.objects||[]).map(o=>`${o.label} ${Number(o.confidence).toFixed(2)}`).join(', ')||'none'}catch(_){e('connection').textContent='DISCONNECTED';e('connection').className='stale'}}refresh();setInterval(refresh,200);
</script></body></html>'''

class LiveDashboard:
    def __init__(self, host="0.0.0.0", port=8766, jpeg_quality=80):
        self.jpeg_quality=max(30,min(95,int(jpeg_quality))); self._condition=threading.Condition(); self._jpeg=None; self._generation=0
        self._state={"updated_at":time.time(),"state":"STARTING"}; self._server=ThreadingHTTPServer((host,int(port)),self._handler()); self._server.daemon_threads=True
        self._thread=threading.Thread(target=self._server.serve_forever,name="autonomy-dashboard",daemon=True)

    def _handler(self):
        dashboard=self; html=DASHBOARD_HTML.encode()
        class Handler(BaseHTTPRequestHandler):
            def send_payload(self,payload,content_type,status=HTTPStatus.OK):
                self.send_response(status); self.send_header("Content-Type",content_type); self.send_header("Content-Length",str(len(payload))); self.send_header("Cache-Control","no-store"); self.send_header("X-Content-Type-Options","nosniff"); self.end_headers(); self.wfile.write(payload)
            def do_GET(self):
                path=urlparse(self.path).path
                if path in ("/","/index.html"): return self.send_payload(html,"text/html; charset=utf-8")
                if path=="/api/state":
                    with dashboard._condition: payload=json.dumps(dashboard._state,separators=(",",":")).encode()
                    return self.send_payload(payload,"application/json")
                if path=="/health": return self.send_payload(b"ok\n","text/plain")
                if path!="/stream.mjpg": return self.send_error(HTTPStatus.NOT_FOUND)
                self.send_response(HTTPStatus.OK); self.send_header("Content-Type","multipart/x-mixed-replace; boundary=frame"); self.send_header("Cache-Control","no-store"); self.end_headers(); generation=-1
                try:
                    while True:
                        with dashboard._condition:
                            dashboard._condition.wait_for(lambda:dashboard._generation!=generation,timeout=2.0); generation,frame=dashboard._generation,dashboard._jpeg
                        if frame is not None: self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "+str(len(frame)).encode()+b"\r\n\r\n"+frame+b"\r\n")
                except (BrokenPipeError,ConnectionResetError): pass
            def log_message(self,*_args): pass
        return Handler

    def start(self): self._thread.start()
    def publish(self,image,state):
        ok,encoded=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,self.jpeg_quality])
        if not ok: return
        payload=dict(state); payload["updated_at"]=time.time()
        with self._condition: self._jpeg=encoded.tobytes(); self._state=payload; self._generation+=1; self._condition.notify_all()
    def close(self): self._server.shutdown(); self._server.server_close(); self._thread.join(timeout=2.0)
