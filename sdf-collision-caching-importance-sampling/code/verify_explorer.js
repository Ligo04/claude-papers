// 交给 playwright-cli run-code --filename 运行，验证真实控件与 sandbox 行为。
async page => {
  const errors = [];
  const externalRequests = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('request', request => {
    if (!request.url().startsWith('http://127.0.0.1:37431/') && !request.url().startsWith('data:')) {
      externalRequests.push(request.url());
    }
  });
  await page.reload();
  const originalHtml = await (await page.request.get(page.url())).text();
  const data = await page.locator('#paperData').textContent().then(JSON.parse);
  let checks = 0;
  const assert = (condition, message) => {
    if (!condition) throw new Error(message);
    checks++;
  };
  const number = text => Number(text.replace(/,/g, '').match(/[\d.]+/)[0]);
  let previousSvg = '';
  for (let i = 0; i < 5; i++) {
    await page.locator('.step').nth(i).click();
    const current = await page.locator('#algorithmDiagram').innerHTML();
    assert(current !== previousSvg, '算法按钮未改变示意图：' + i);
    assert(await page.locator('.step').nth(i).getAttribute('aria-pressed') === 'true', '按钮激活状态错误');
    previousSvg = current;
  }
  for (let i = 0; i < data.table1.length; i++) {
    await page.locator('#dataset').selectOption('gpu');
    await page.locator('#gpuScene').selectOption(String(i));
    assert((await page.locator('#chartHeading').textContent()).includes(data.table1[i].scene), 'GPU 场景未切换');
    assert(number(await page.locator('#bestValue').textContent()) === Math.min(...['NM','Ellipsoid','SGD'].map(k=>data.table1[i][k])), 'GPU 数值不符');
  }
  for (const [dataset, key] of [['boxes','table3'],['gears','table4']]) {
    await page.locator('#dataset').selectOption(dataset);
    for (const ablated of [false, true]) {
      await page.locator('#ablations').setChecked(ablated);
      assert(await page.locator('#dataTable tbody tr').count() === (ablated ? 12 : 6), '消融条目数不符');
      for (const metric of ['ms','contacts','iterations','queries']) {
        await page.locator('#metric').selectOption(metric);
        const rows = data[key].filter(r=>ablated || !r.method.includes('/'));
        const expected = Math.min(...rows.map(r=>r[metric]).filter(Number.isFinite));
        assert(number(await page.locator('#bestValue').textContent()) === expected, 'CPU 统计项没有更新');
      }
    }
    if (dataset === 'gears') {
      assert((await page.locator('#barChart').getAttribute('aria-label')).includes('SGD / no cache FAIL'), 'FAIL 被误当有效数值');
    }
  }
  await page.locator('#dataset').selectOption('distortion');
  const operators = await page.locator('#distortionScene option').evaluateAll(options=>options.map(o=>o.value));
  for (const operator of operators) {
    await page.locator('#distortionScene').selectOption(operator);
    const rows = data.table5.filter(r=>r.operator+'|'+r.k===operator);
    for (const metric of ['ms','contacts','iterations','queries']) {
      await page.locator('#metric').selectOption(metric);
      assert(number(await page.locator('#bestValue').textContent()) === Math.min(...rows.map(r=>r[metric])), '失真场数值不符');
    }
  }
  for (const row of data.table2) {
    await page.locator('#cacheBudget').selectOption(String(row.cache));
    await page.locator('#randomBudget').selectOption(String(row.random));
    assert(number(await page.locator('#budgetValue').textContent()) === row.ms, '预算组合数值不符');
    assert((await page.locator('#heatMap').getAttribute('aria-label')).includes(`缓存 ${row.cache}，新种子 ${row.random}`), '热力图选择没有更新');
  }
  const sandbox = await page.context().newPage();
  const sandboxErrors = [];
  const sandboxRequests = [];
  sandbox.on('pageerror', error => sandboxErrors.push(String(error)));
  sandbox.on('request', request => sandboxRequests.push(request.url()));
  await page.context().setOffline(true);
  try {
    await sandbox.setContent('<iframe title="论文独立 sandbox" sandbox="allow-scripts" style="width:1100px;height:900px"></iframe>');
    await sandbox.locator('iframe').evaluate((frame, html) => { frame.srcdoc = html; }, originalHtml);
    const inner = sandbox.frameLocator('iframe');
    await inner.locator('#cacheBudget').selectOption('16');
    await inner.locator('#randomBudget').selectOption('4');
    assert(number(await inner.locator('#budgetValue').textContent()) === 8.9, '离线 sandbox 预算控件失败');
    await inner.locator('#dataset').selectOption('gears');
    await inner.locator('#ablations').check();
    assert(await inner.locator('#dataTable tbody tr').count() === 12, '离线 sandbox 消融控件失败');
    assert(sandboxRequests.length === 0, 'sandbox 发起了网络请求');
    assert(sandboxErrors.length === 0, 'sandbox JavaScript 异常');
  } finally {
    await page.context().setOffline(false);
    await sandbox.close();
  }
  await page.setViewportSize({width:1280,height:1000});
  await page.locator('#dataset').selectOption('gears');
  await page.locator('#metric').selectOption('ms');
  await page.locator('#ablations').uncheck();
  await page.locator('#cacheBudget').selectOption('4');
  await page.locator('#randomBudget').selectOption('1');
  await page.locator('.step').nth(3).click();
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path:'/home/ligo/claude-papers/papers/sdf-collision-caching-importance-sampling/output/playwright/explorer-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.reload();
  const mobileLayout = await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
  assert(mobileLayout.scroll <= mobileLayout.width + 1, '移动端出现整页水平溢出');
  await page.screenshot({path:'/home/ligo/claude-papers/papers/sdf-collision-caching-importance-sampling/output/playwright/explorer-mobile.png',fullPage:true});
  await page.setViewportSize({width:1280,height:1000});
  await page.reload();
  assert(errors.length === 0, '主页面 JavaScript 异常');
  assert(externalRequests.length === 0, '主页面发起外部网络请求');
  return {status:'passed',assertions:checks,algorithmSteps:5,gpuScenes:5,cpuAblationModes:4,
          distortionConfigurations:operators.length,budgetCombinations:data.table2.length,
          offlineSandbox:true,opaqueOrigin:true,mobileLayout,errors,externalRequests,
          scope:'验证本地 HTML 交互、数据选择和离线 iframe；不验证原作者仿真。'};
}
