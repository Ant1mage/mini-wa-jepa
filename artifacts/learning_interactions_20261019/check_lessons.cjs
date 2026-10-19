const{chromium}=require('/tmp/mini-wa-jepa-browser-check/node_modules/playwright');
const fs=require('fs'),crypto=require('crypto');
const out='artifacts/learning_interactions_20261019';
const count=Number(process.env.CHAPTER_COUNT||6),url=process.env.LESSON_URL||'http://127.0.0.1:2730';
const digest=x=>crypto.createHash('sha256').update(x).digest('hex');
async function output(p){
 const images=await p.locator('img').evaluateAll(els=>els.map(e=>e.getAttribute('src')).join('|'));
 const explanations=await p.locator('marimo-callout-output').evaluateAll(els=>els.map(e=>e.shadowRoot?.textContent||e.innerHTML).join('|'));
 return digest(images+'|'+explanations);
}
async function changed(p,before,label){
 const end=Date.now()+20000;
 while(Date.now()<end){await p.waitForTimeout(180);if(await output(p)!==before){await p.locator('img').first().waitFor();return;}}
 throw new Error('输出未联动：'+label);
}
(async()=>{
 const b=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
 const p=await b.newPage({viewport:{width:1440,height:1050}}),results=[],errors=[];
 p.on('pageerror',e=>errors.push(e.message));
 try{
  await p.goto(url);await p.getByRole('heading',{name:/^Chapter 0 ·/}).waitFor({timeout:60000});
  for(let ch=0;ch<count;ch++){
   await p.getByRole('radio',{name:new RegExp('^Chapter '+ch+' ·')}).check();
   await p.getByRole('heading',{name:new RegExp('^Chapter '+ch+' ·')}).waitFor();
   await p.locator('img').first().waitFor();await p.waitForTimeout(300);
   const tabs=await p.getByRole('tab').count();
   for(let t=0;t<tabs;t++){
    await p.getByRole('tab').nth(t).click();await p.waitForTimeout(350);await p.locator('img').first().waitFor();
    let n=await p.getByRole('slider').count();
    for(let i=0;i<n;i++){
     const control=p.getByRole('slider').nth(i),before=await output(p);
     const v=Number(await control.getAttribute('aria-valuenow')),max=Number(await control.getAttribute('aria-valuemax'));
     await control.press(v>=max?'ArrowLeft':'ArrowRight');
     await changed(p,before,`chapter${ch} section${t} slider${i}`);
     results.push({chapter:ch,section:t,control:'slider',index:i,passed:true});
    }
    for(let i=0;i<await p.getByRole('switch').count();i++){
     const before=await output(p);await p.getByRole('switch').nth(i).click();await changed(p,before,`chapter${ch} switch${i}`);
     results.push({chapter:ch,section:t,control:'switch',index:i,passed:true});
    }
    for(let i=0;i<await p.getByRole('combobox').count();i++){
     const c=p.getByRole('combobox').nth(i),before=await output(p),tag=await c.evaluate(e=>e.tagName);
     if(tag==='SELECT')await c.selectOption({index:1});else{await c.click();await p.getByRole('option').nth(1).click();}
     await changed(p,before,`chapter${ch} dropdown${i}`);
     results.push({chapter:ch,section:t,control:'dropdown',index:i,passed:true});
    }
    const localRadios=p.getByRole('radio').filter({hasNotText:/^Chapter/});
    // 章节导航之外，通道/方案的radio也必须让实际图示变化。
    const radios=await p.getByRole('radio').count();
    if(radios>count){
     const before=await output(p);
     let r=p.getByRole('radio').nth(count);
     if(await r.isChecked())r=p.getByRole('radio').nth(count+1);
     await r.check();await changed(p,before,`chapter${ch} layer`);results.push({chapter:ch,section:t,control:'radio',passed:true});
     if(ch===6&&t===1){
      const beforeModel=await output(p);
      await p.getByRole('radio',{name:'只给历史',exact:true}).check();
      await changed(p,beforeModel,'chapter6 history-only');
      results.push({chapter:ch,section:t,control:'model-radio',passed:true});
     }
    }
    await p.getByText('看看这一步的代码 · 中文注释',{exact:true}).click();
    await p.waitForTimeout(200);
    const code=await p.locator('marimo-accordion').first().evaluate(e=>{function read(n){return (n.nodeType===3?n.textContent:'')+Array.from(n.childNodes||[]).map(read).join('')+(n.shadowRoot?read(n.shadowRoot):'')}return read(e)});
    if(!code.includes('#')||!/[\u4e00-\u9fff]/.test(code))throw new Error('缺少中文注释代码 chapter'+ch+' section'+t);
    await p.getByText('看看这一步的代码 · 中文注释',{exact:true}).click();
    if(t===0||ch>=5)await p.screenshot({path:`${out}/chapter${ch}_section${t}.png`});
    console.log(JSON.stringify({chapter:ch,section:t,passed:true}));
   }
  }
  // 返回既有章节，确保控件与小节位置没有因为导航重建。
  await p.getByRole('radio',{name:/^Chapter 1 ·/}).check();await p.getByRole('heading',{name:/^Chapter 1 ·/}).waitFor();
  const retained=await p.getByRole('tab').nth(2).getAttribute('aria-selected');
  if(retained!=='true')throw new Error('章节小节状态未保留');
  if(errors.length)throw new Error(errors.join('\n'));
  fs.writeFileSync(`${out}/browser_${count}_chapters.json`,JSON.stringify({passed:true,chapterCount:count,results,stateRetained:true,pageErrors:errors},null,2));
  console.log('ALL_PASSED',results.length);
 }catch(e){
  await p.screenshot({path:`${out}/browser_failure_${count}.png`});
  fs.writeFileSync(`${out}/browser_failure_${count}.json`,JSON.stringify({error:String(e),results,pageErrors:errors},null,2));throw e;
 }finally{await b.close();}
})().catch(e=>{console.error(e);process.exit(1)});
