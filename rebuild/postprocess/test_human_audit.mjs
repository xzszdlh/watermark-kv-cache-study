// Headless Chrome UI smoke in an isolated profile; never writes real annotations.
import {spawn} from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const build=path.join(root,'human_audit','build');await fs.mkdir(build,{recursive:true});
const profile=path.join(build,'browser-qa-profile');
const chrome='C:/Program Files/Google/Chrome/Application/chrome.exe';
const port=19527;
const proc=spawn(chrome,['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
    '--disable-extensions',`--remote-debugging-port=${port}`,`--user-data-dir=${profile}`,'about:blank'],{stdio:'ignore'});
let ws,nextId=0;const pending=new Map();
async function call(method,params={}){const id=++nextId;return new Promise((resolve,reject)=>{const timeout=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout '+method));},20000);pending.set(id,{resolve:r=>{clearTimeout(timeout);resolve(r);},reject:e=>{clearTimeout(timeout);reject(e);}});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
try{
 let tabs;
 for(let i=0;i<100;i++){try{const r=await fetch(`http://127.0.0.1:${port}/json/list`);tabs=await r.json();if(tabs.some(t=>t.type==='page'))break;}catch{}await new Promise(r=>setTimeout(r,100));}
 assert(tabs,'Chrome debugging endpoint unavailable');
 const tab=tabs.find(t=>t.type==='page');ws=new WebSocket(tab.webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
 ws.onmessage=e=>{const msg=JSON.parse(e.data);if(msg.id){const p=pending.get(msg.id);if(p){pending.delete(msg.id);msg.error?p.reject(Error(JSON.stringify(msg.error))):p.resolve(msg.result);}}};
 await call('Page.enable');await call('Runtime.enable');
 await call('Emulation.setDeviceMetricsOverride',{width:1450,height:1000,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:pathToFileURL(path.join(root,'human_audit','开始人工核验.html')).href});
 for(let i=0;i<100;i++){if(await evaluate('document.readyState==="complete" && !!document.getElementById("audit-data")'))break;await new Promise(r=>setTimeout(r,100));}
 assert.equal(await evaluate('items.length'),260);
 assert.equal(await evaluate('items.filter(complete).length'),0);
 assert.equal(await evaluate('document.querySelectorAll("input[name=label]:checked").length'),0);
 assert.equal(await evaluate('Object.keys(items[0]).sort().join(",")'),'item_id,source_visible_to_model,summary_context,summary_sentence');
 // Screenshot the working form with the initially expanded guide collapsed.
 await evaluate('document.querySelector("details.help").open=false');
 const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 await fs.writeFile(path.join(build,'preview.png'),Buffer.from(shot.data,'base64'));
 await evaluate(`byId('annotator').value='QA_ONLY_NOT_HUMAN';byId('notes').value='Automated UI test, not an annotation';document.querySelector('input[value=unclear]').checked=true;save();`);
 assert.equal(await evaluate('items.filter(complete).length'),1);
 await evaluate('move(1);move(0)');
 assert.equal(await evaluate('byId("notes").value'),'Automated UI test, not an annotation');
 await call('Page.reload');
 await new Promise(r=>setTimeout(r,500));
 assert.equal(await evaluate('items.filter(complete).length'),1);
 assert.equal(await evaluate('byId("annotator").value'),'QA_ONLY_NOT_HUMAN');
 // Intercept download so no synthetic label file can be mistaken for a result.
 await evaluate('globalThis.qaExport=null;download=(text,type,filename)=>{globalThis.qaExport={text,type,filename};};exportResult();');
 assert.equal(await evaluate('JSON.parse(qaExport.text).items.length'),260);
 assert.equal(await evaluate('JSON.parse(qaExport.text).dataset_id===data.dataset_id'),true);
 await evaluate(`(async()=>{globalThis.qaBackup=qaExport.text;state={dataset_id:data.dataset_id,index:4,annotator:'',answers:{}};render();globalThis.confirm=()=>true;globalThis.alert=message=>{throw Error(message)};const transfer=new DataTransfer();transfer.items.add(new File([qaBackup],'qa.json',{type:'application/json'}));byId('fileInput').files=transfer.files;await byId('fileInput').onchange({target:byId('fileInput')});})()`);
 assert.equal(await evaluate('items.filter(complete).length'),1);
 assert.equal(await evaluate('state.index'),0);
 // Clear the isolated profile state; real default browser was never touched.
 await evaluate('localStorage.removeItem(storageKey);state={dataset_id:data.dataset_id,index:0,annotator:"",answers:{}};render();');
 assert.equal(await evaluate('items.filter(complete).length'),0);
 await fs.writeFile(path.join(build,'ui_checks.json'),JSON.stringify({passed:true,items:260,initial_labels_blank:true,
   no_group_or_model_scores_in_public_data:true,navigation:true,local_storage_reload:true,export_payload:true,import_restore:true,
   synthetic_UI_test_labels_removed:true,profile_isolated:true},null,2));
 console.log('AUDIT_UI_QA_PASS: 260 blank items; navigation/save/reload/export checked in isolated profile');
 await call('Browser.close');
}catch(e){console.error(e);process.exitCode=1;}
finally{if(ws)ws.close();if(proc.exitCode===null)proc.kill();await new Promise(r=>setTimeout(r,500));
 try{await fs.rm(profile,{recursive:true,force:true,maxRetries:6,retryDelay:300});}catch(e){console.error('QA profile cleanup pending',e.message);}}
