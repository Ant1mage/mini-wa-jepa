const fs=require('fs'),os=require('os'),path=require('path'),crypto=require('crypto');
const {spawn}=require('child_process');
const root='artifacts/restart_stage7/codex_review';
const profile=fs.mkdtempSync(path.join(os.tmpdir(),'mini-wa-stage7-browser-'));
const log=fs.openSync(root+'/chrome.log','w');
const chrome=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--remote-debugging-port=2738','--user-data-dir='+profile,'--window-size=1500,1100','about:blank'],{stdio:['ignore',log,log]});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
let socket,id=0;const pending=new Map();
function call(method,params={}){return new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});socket.send(JSON.stringify({id:n,method,params}));});}
async function evaluate(expression){const x=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(x.exceptionDetails)throw new Error(JSON.stringify(x.exceptionDetails));return x.result.value;}
async function wait(expression,timeout=60000){const end=Date.now()+timeout;while(Date.now()<end){if(await evaluate(expression))return;await sleep(250);}throw new Error('Timed out: '+expression);}
const deep=`function all(selector,root=document){let result=[...root.querySelectorAll(selector)];for(const e of root.querySelectorAll('*'))if(e.shadowRoot)result.push(...all(selector,e.shadowRoot));return result;}`;
async function images(){return evaluate(`(()=>{${deep};return all('img').map(e=>e.src).join('|')})()`);}
async function key(key,code,v){await call('Input.dispatchKeyEvent',{type:'keyDown',key,code,windowsVirtualKeyCode:v});await call('Input.dispatchKeyEvent',{type:'keyUp',key,code,windowsVirtualKeyCode:v});}
async function sliderStep(){await evaluate(`(()=>{${deep};const e=all('[role=slider],input[type=range]')[0];if(!e)throw Error('No slider');e.focus();return true;})()`);await key('ArrowRight','ArrowRight',39);await key('Tab','Tab',9);}
async function screenshot(name){const x=await call('Page.captureScreenshot',{format:'png'});fs.writeFileSync(root+'/'+name,Buffer.from(x.data,'base64'));}
(async()=>{
 const result={passed:false,checks:[]};
 try{
  let pages;for(let n=0;n<100;n++){try{pages=await(await fetch('http://127.0.0.1:2738/json/list')).json();break;}catch{}await sleep(200);}
  if(!pages)throw Error('Chrome debugging endpoint unavailable');
  socket=new WebSocket(pages.find(x=>x.type==='page').webSocketDebuggerUrl);
  await new Promise((r,j)=>{socket.onopen=r;socket.onerror=j});
  socket.onmessage=event=>{const m=JSON.parse(event.data);if(m.id&&pending.has(m.id)){const q=pending.get(m.id);pending.delete(m.id);m.error?q.reject(new Error(JSON.stringify(m.error))):q.resolve(m.result);}};
  await call('Page.enable');await call('Emulation.setDeviceMetricsOverride',{width:1500,height:1100,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate',{url:'http://127.0.0.1:2737'});
  await wait(`document.body && document.body.innerText.includes('Chapter 0')`);
  await evaluate(`(()=>{${deep};const r=all('[role=radio],input[type=radio]').filter(e=>(e.getAttribute('aria-label')||e.closest('label')?.innerText||e.parentElement?.innerText||e.innerText||'').includes('Chapter 7'));if(!r.length)throw Error('Chapter 7 radio not found');r[0].click();return r.length;})()`);
  await wait(`document.body.innerText.includes('同一个场景，八份噪声')`);
  await wait(`(()=>{${deep};return all('img').some(x=>x.src.length>100)})()`);
  const before=await images();await screenshot('chapter7_flow_before.png');await sliderStep();
  const end=Date.now()+30000;let after='';while(Date.now()<end){after=await images();if(after.length>100&&after!==before)break;await sleep(200);}
  if(after===before||after.length<100)throw Error('Flow image did not react to keyboard slider');
  result.checks.push({view:'flow',imageChanged:true,before:crypto.createHash('sha256').update(before).digest('hex'),after:crypto.createHash('sha256').update(after).digest('hex')});
  await wait(`(()=>{${deep};return all('strong').some(e=>e.textContent.includes('1/10'))})()`);await sleep(500);await screenshot('chapter7_flow_after.png');
  const tab=await evaluate(`(()=>{${deep};const t=all('[role=tab]').find(e=>e.innerText.includes('2 看评分'));if(!t)throw Error('Planning tab missing');const r=t.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);await call('Input.dispatchMouseEvent',{type:'mousePressed',...tab,button:'left',clickCount:1});await call('Input.dispatchMouseEvent',{type:'mouseReleased',...tab,button:'left',clickCount:1});
  await wait(`(()=>{${deep};return all('h2').some(e=>e.textContent.includes('这一轮为什么选它'))&&all('strong').some(e=>e.textContent.includes('本轮实际评分'))})()`);
  await sleep(500);const planningBefore=await images();await sliderStep();
  const end2=Date.now()+30000;let planningAfter='';while(Date.now()<end2){planningAfter=await images();if(planningAfter.length>100&&planningAfter!==planningBefore)break;await sleep(200);}
  if(planningAfter===planningBefore||planningAfter.length<100)throw Error('Planning replay did not advance');
  const text=await evaluate(`(()=>{${deep};return all('h2,p,strong').map(e=>e.textContent).join('\n')})()`);if(/MarimoExceptionRaisedError|Traceback \(most recent call/.test(text))throw Error('Marimo runtime error');
  result.checks.push({view:'planning',imageChanged:true,scoreTableVisible:text.includes('本轮实际评分'),fallbackExplained:text.includes('规则生成的刹停')});
  await screenshot('chapter7_planning.png');result.passed=true;
 }catch(error){result.error=String(error);try{await screenshot('browser_failure.png');fs.writeFileSync(root+'/browser_failure_body.txt',await evaluate('document.body.innerText'));}catch{}}
 finally{fs.writeFileSync(root+'/browser_checks.json',JSON.stringify(result,null,2)+'\n');if(socket)socket.close();chrome.kill('SIGTERM');setTimeout(()=>{fs.rmSync(profile,{recursive:true,force:true});process.exit(result.passed?0:1)},500);}
})();
