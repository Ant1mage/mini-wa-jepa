const{chromium}=require('/tmp/mini-wa-jepa-browser-check/node_modules/playwright');
const fs=require('fs');
const dir='artifacts/learning_interactions_20261019';
(async()=>{
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
 const p=await browser.newPage({viewport:{width:1440,height:1050}});
 const check=(ok,msg)=>{if(!ok)throw Error(msg)};
 const text=()=>p.locator('marimo-callout-output').evaluateAll(es=>es.map(e=>e.shadowRoot?.textContent||e.textContent).join(''));
 const stable=async()=>{await p.waitForTimeout(650);await p.locator('img').first().waitFor();};
 try{
 await p.goto('http://127.0.0.1:2718');await p.getByRole('heading',{name:/^Chapter 0 ·/}).waitFor({timeout:60000});
 await p.getByRole('radio',{name:/^Chapter 6 ·/}).check();await p.getByRole('heading',{name:/^Chapter 6 ·/}).waitFor();await stable();
 await p.getByRole('slider').press('End');await stable();
 check((await text()).includes('2.0秒'),'时间滑块未到2秒');
 await p.getByRole('heading',{name:/^Chapter 6 ·/}).scrollIntoViewIfNeeded();
 await p.screenshot({path:dir+'/chapter6_replay_final.png'});
 await p.getByRole('tab').nth(1).click();await stable();
 await p.getByRole('radio',{name:'只给历史',exact:true}).check();await stable();
 let baseline=null;
 for(const name of ['A 保持路线','B 轻微左偏','C 轻微右偏','D 减速']){
  await p.getByRole('radio',{name,exact:true}).check();await stable();
  const t=await text(),found=t.match(/最接近\s*([ABCD])/);
  check(!!found,'没有后果匹配解释');if(baseline===null)baseline=found[1];
  check(found[1]===baseline,'只给历史的输出随候选改变');
 }
 await p.getByRole('radio',{name:'告诉模型方案',exact:true}).check();await stable();
 await p.screenshot({path:dir+'/chapter6_matching_final.png'});
 await p.getByRole('tab').nth(2).click();await stable();
 const right=await text();await p.getByRole('switch').click();await stable();const wrong=await text();
 check(right!==wrong,'错方案未改变实际结果说明');
 check(wrong.includes('C 轻微右偏'),'D的错命令应为C');
 await p.screenshot({path:dir+'/chapter6_wrong_final.png'});
 fs.writeFileSync(dir+'/chapter6_semantic_browser.json',JSON.stringify({passed:true,fullTimeReplay:true,allFourHistoryOnlyMatch:baseline,wrongCommandMapping:true},null,2));
 console.log('Chapter6 semantic browser checks passed');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
