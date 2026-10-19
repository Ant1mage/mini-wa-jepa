// 验证真实浏览器事件：逐章选择，按键调整滑块，等待 Python 重新绘图。
const { chromium } = require('/tmp/mini-wa-jepa-browser-check/node_modules/playwright');
const fs = require('fs');
const crypto = require('crypto');
const root = 'artifacts/tutorial_rework_20261018';

async function lessonSnapshot(page, chapter) {
  const text = await page.locator('body').innerText();
  const start = text.indexOf('Chapter ' + chapter + ' ·');
  const images = await page.locator('img').evaluateAll(elements => elements.map(el => el.getAttribute('src')).join('|'));
  return text.slice(start) + images;
}

(async () => {
  const browser = await chromium.launch({executablePath: '/usr/bin/google-chrome', headless: true,
                                        args: ['--no-sandbox']});
  const page = await browser.newPage({viewport: {width: 1500, height: 1100}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const results = [];
  try {
    await page.goto('http://127.0.0.1:2729');
    for (let chapter = 0; chapter < 6; chapter++) {
      await page.getByRole('radio').nth(chapter).check();
      await page.getByRole('heading', {name: new RegExp('^Chapter ' + chapter + ' ·')}).waitFor({timeout: 60000});
      await page.getByRole('heading', {name: '这张图是怎样算出来的？', exact: true}).waitFor();
      await page.waitForTimeout(500);
      const images = page.locator('img');
      await images.first().waitFor({timeout: 30000});
      const before = await images.evaluateAll(elements => elements.map(el => el.getAttribute('src')).join('|'));
      const slider = page.getByRole('slider').first();
      const previous = await slider.getAttribute('aria-valuenow');
      await slider.press('ArrowRight');
      await slider.press('Tab');
      await page.waitForTimeout(1500);
      // Playwright locator 会穿透 Marimo 组件的 shadow DOM；原生 querySelectorAll 不会。
      const deadline = Date.now() + 30000;
      while (Date.now() < deadline) {
        const candidate = await images.evaluateAll(elements => elements.map(el => el.getAttribute('src')).join('|'));
        if (candidate.length > 100 && candidate !== before) break;
        await page.waitForTimeout(200);
      }
      await page.waitForTimeout(700);
      // 等待输出 cell 完成，避免把运行期间的暂时空输出误认成新图。
      await page.getByRole('heading', {name: new RegExp('^Chapter ' + chapter + ' ·')}).waitFor();
      await images.first().waitFor();
      const after = await images.evaluateAll(elements => elements.map(el => el.getAttribute('src')).join('|'));
      if (!after || before === after) throw new Error('Chapter ' + chapter + ' 图像未更新');
      const text = await page.locator('body').innerText();
      if (!text.includes('这张图是怎样算出来的')) throw new Error('缺少可阅读的实验代码');
      if (/Traceback|MarimoExceptionRaisedError/.test(text)) throw new Error('页面有运行错误');
      const current = await page.getByRole('slider').first().getAttribute('aria-valuenow');
      const result = {chapter, previous, current, imageChanged: true,
        before: crypto.createHash('sha256').update(before).digest('hex'),
        after: crypto.createHash('sha256').update(after).digest('hex')};
      results.push(result);
      console.log(JSON.stringify(result));
      await page.screenshot({path: root + '/chapter' + chapter + '_interactive.png', fullPage: true});
    }
    await page.getByRole('radio').nth(1).check();
    await page.getByRole('heading', {name: /^Chapter 1 ·/}).waitFor();
    const retained = await page.getByRole('slider').first().getAttribute('aria-valuenow');
    if (retained !== results[1].current) throw new Error('切换章节后参数被重置');
    const allControls = [];
    // 所有控件都必须触发消费 cell 的输出变化；某些参数仅改变数值而非图像。
    for (let chapter = 0; chapter < 6; chapter++) {
      await page.getByRole('radio').nth(chapter).check();
      await page.getByRole('heading', {name: new RegExp('^Chapter ' + chapter + ' ·')}).waitFor();
      await page.waitForTimeout(400);
      const sliders = page.getByRole('slider');
      const count = await sliders.count();
      for (let index = 0; index < count; index++) {
        const prior = await lessonSnapshot(page, chapter);
        const slider = sliders.nth(index);
        const beforeValue = Number(await slider.getAttribute('aria-valuenow'));
        const maximum = Number(await slider.getAttribute('aria-valuemax'));
        await slider.press(beforeValue >= maximum ? 'ArrowLeft' : 'ArrowRight');
        await slider.press('Tab');
        await page.waitForTimeout(800);
        const after = await lessonSnapshot(page, chapter);
        if (after === prior) throw new Error('Chapter ' + chapter + ' slider ' + index + ' 未更新结果');
        allControls.push({chapter, kind: 'slider', index, passed: true});
      }
      const selects = page.getByRole('combobox');
      for (let index = 0; index < await selects.count(); index++) {
        const prior = await lessonSnapshot(page, chapter);
        await selects.nth(index).selectOption({index: 1});
        await page.waitForTimeout(800);
        if (await lessonSnapshot(page, chapter) === prior) throw new Error('下拉框未更新结果');
        allControls.push({chapter, kind: 'dropdown', index, passed: true});
      }
    }
    if (errors.length) throw new Error(errors.join('\n'));
    fs.writeFileSync(root + '/browser_checks.json', JSON.stringify({passed: true, results,
      allControls, parameterRetention: true, pageErrors: errors}, null, 2));
    console.log('ALL_CONTROLS_PASSED', allControls.length);
  } catch (error) {
    fs.writeFileSync(root + '/browser_failure_body.txt', await page.locator('body').innerText());
    await page.screenshot({path: root + '/browser_failure.png', fullPage: true});
    fs.writeFileSync(root + '/browser_checks_failed.json', JSON.stringify({passed: false, results,
      error: String(error), pageErrors: errors}, null, 2));
    throw error;
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error);process.exit(1)});
