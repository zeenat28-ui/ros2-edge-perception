import base64
import textwrap

html = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>ROS 2 3D Mission Control</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
body{margin:0;overflow:hidden;background:#0a0e17;color:#fff;font-family:sans-serif}
#c{width:100vw;height:100vh;display:block}
.hud{position:absolute;top:10px;left:10px;background:rgba(15,23,42,0.9);border:1px solid #1e293b;padding:10px 15px;border-radius:8px;pointer-events:none}
.hud h2{margin:0;font-size:15px;color:#00e5ff}
.badge{position:absolute;top:10px;right:10px;padding:6px 14px;border-radius:6px;font-weight:bold;font-size:12px}
.badge-ok{background:#00e676;color:#000}
.badge-aeb{background:#ff1744;color:#fff;animation:b .8s infinite}
@keyframes b{50%{opacity:.4}}
.tele{position:absolute;top:70px;left:10px;background:rgba(15,23,42,0.9);border:1px solid #1e293b;padding:10px;border-radius:8px;font-size:12px;line-height:1.6}
.ctrl{position:absolute;bottom:10px;right:10px;background:rgba(15,23,42,0.9);border:1px solid #1e293b;padding:10px;border-radius:8px;text-align:center}
.btn{width:42px;height:38px;margin:2px;background:#1e293b;border:1px solid #334155;color:#00e5ff;font-weight:bold;border-radius:4px;cursor:pointer}
.btn:hover{background:#00e5ff;color:#000}
.pip{position:absolute;bottom:10px;left:10px;width:180px;height:115px;background:#000;border:2px solid #00e5ff;border-radius:6px;overflow:hidden}
.pip img{width:100%;height:100%;object-fit:cover}
</style></head>
<body>
<div class="hud"><h2>ROS 2 AUTONOMOUS ROBOT 3D</h2><p style="margin:3px 0 0;font-size:10px;color:#94a3b8">ASIL-B Edge AI Perception &bull; 3D LiDAR &bull; AEB</p></div>
<div id="badge" class="badge badge-ok">SYSTEM NOMINAL</div>
<div class="tele">Speed: <b id="v">0.00 m/s</b><br>Closest Hazard: <b id="d">-- m</b><br>Est. TTC: <b id="ttc">-- s</b><br>AEB Interventions: <b id="i">0</b></div>
<div class="pip"><img src="/stream.mjpg"></div>
<div class="ctrl">
  <div style="font-size:10px;color:#94a3b8;margin-bottom:3px">DRIVE ROBOT</div>
  <div><button class="btn" onclick="cmd(1,0)">▲</button></div>
  <div><button class="btn" onclick="cmd(0,1.5)">◄</button><button class="btn" style="color:#ff1744" onclick="cmd(0,0)">■</button><button class="btn" onclick="cmd(0,-1.5)">►</button></div>
  <div><button class="btn" onclick="cmd(-0.8,0)">▼</button></div>
</div>
<canvas id="c"></canvas>
<script>
let scene,cam,ren,ctrl,bot,lidar,obs=[];
function init(){
  scene=new THREE.Scene();scene.background=new THREE.Color(0x0a0e17);
  cam=new THREE.PerspectiveCamera(60,innerWidth/innerHeight,0.1,100);cam.position.set(0,5,6);
  ren=new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});ren.setSize(innerWidth,innerHeight);
  ctrl=new THREE.OrbitControls(cam,ren.domElement);ctrl.enableDamping=true;
  scene.add(new THREE.AmbientLight(0xffffff,0.7));
  const l=new THREE.DirectionalLight(0x00e5ff,1.2);l.position.set(5,10,7);scene.add(l);
  scene.add(new THREE.GridHelper(20,20,0x00e5ff,0x1e293b));
  bot=new THREE.Group();
  const ch=new THREE.Mesh(new THREE.BoxGeometry(0.8,0.25,1.1),new THREE.MeshStandardMaterial({color:0x1e293b,metalness:0.8}));
  ch.position.y=0.25;bot.add(ch);
  const ring=new THREE.Mesh(new THREE.RingGeometry(1.48,1.52,32),new THREE.MeshBasicMaterial({color:0xff1744,side:THREE.DoubleSide}));
  ring.rotation.x=Math.PI/2;ring.position.y=0.02;bot.add(ring);
  scene.add(bot);
  const geo=new THREE.BufferGeometry();const pos=new Float32Array(600);
  for(let i=0;i<200;i++){let a=(i/200)*Math.PI*2,r=1.5+Math.random()*3.5;pos[i*3]=Math.cos(a)*r;pos[i*3+1]=0.1+Math.random()*0.3;pos[i*3+2]=Math.sin(a)*r;}
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  lidar=new THREE.Points(geo,new THREE.PointsMaterial({color:0x00e5ff,size:0.07}));bot.add(lidar);
  window.onresize=()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();ren.setSize(innerWidth,innerHeight);};
  (function anim(){requestAnimationFrame(anim);if(lidar)lidar.rotation.y+=0.03;ctrl.update();ren.render(scene,cam);})();
}
function cmd(v,w){fetch('/drive?v='+v+'&w='+w).catch(()=>{});}
setInterval(()=>{
  fetch('/telemetry').then(r=>r.json()).then(d=>{
    document.getElementById('v').innerText=Math.abs(d.speed).toFixed(2)+' m/s';
    document.getElementById('d').innerText=d.closest>0?d.closest.toFixed(2)+' m':'-- m';
    document.getElementById('ttc').innerText=d.ttc>0&&d.ttc<50?d.ttc.toFixed(2)+' s':'-- s';
    document.getElementById('i').innerText=d.interventions||0;
    const b=document.getElementById('badge');
    if(d.brake){b.className='badge badge-aeb';b.innerText='! AEB BRAKE ACTIVE !';}
    else{b.className='badge badge-ok';b.innerText='SYSTEM NOMINAL';}
    if(bot){bot.position.x=d.x||0;bot.position.z=-(d.y||0);bot.rotation.y=(d.theta||0)+Math.PI/2;}
    obs.forEach(o=>scene.remove(o));obs=[];
    (d.obstacles||[]).forEach(o=>{
      const m=new THREE.Mesh(new THREE.BoxGeometry(0.6,1.7,0.4),new THREE.MeshStandardMaterial({color:o.class_name==='person'?0xff9100:0x38bdf8}));
      m.position.set(o.x,0.85,-o.y);
      const w=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.7,1.8,0.5)),new THREE.LineBasicMaterial({color:d.brake?0xff1744:0x00e676}));
      w.position.set(o.x,0.85,-o.y);scene.add(m);scene.add(w);obs.push(m,w);
    });
  }).catch(()=>{});
},250);
window.onload=init;
</script></body></html>"""

b64_html = base64.b64encode(html.encode("utf-8")).decode("utf-8")
wrapped_html = "\n".join(textwrap.wrap(b64_html, width=76))
print("=== WRAPPED_HTML ===")
print(wrapped_html)

