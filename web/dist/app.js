import {predictForest} from './forest.js';

const $ = (selector, parent = document) => parent.querySelector(selector);
const $$ = (selector, parent = document) => [...parent.querySelectorAll(selector)];
const icon = name => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
const pct = (n, digits = 2) => `${(n * 100).toFixed(digits)}%`;
const fixed = (n, digits = 5) => Number(n).toFixed(digits);
const labels = {default: 'RF mặc định', random: 'RF · Random Search', tpe: 'RF · TPE', selected: 'Mô hình được chọn'};
const colors = {default: '#abb8a2', random: '#9b83c9', tpe: '#2a8b72', selected: '#4b7349'};
const titles = {overview: 'Tổng quan', comparison: 'So sánh mô hình', search: 'Quá trình tìm kiếm', curves: 'Train & Validation', sandbox: 'Thử nhận dạng'};
const state = {view: 'overview', seed: 22, prefix: 30, timer: null, metric: 'accuracy', zoom: false, pixels: new Array(64).fill(0), revision: 0, expected: null, samplePositions: new Array(10).fill(0)};
let data, modelPromise;
const content = $('#content');
async function fetchJSON(path) {
  const response = await fetch(new URL(path, import.meta.url));
  if (!response.ok) throw new Error(`Không đọc được dữ liệu (${response.status}).`);
  return response.json();
}
function toast(message) {
  const element = $('#toast');
  element.textContent = message;
  element.classList.add('show');
  clearTimeout(toast.timeout);
  toast.timeout = setTimeout(() => element.classList.remove('show'), 4200);
}
function heading(eyebrow, title, subtitle, actions = '') {
  return `<div class="page-heading"><div><p class="eyebrow"><i></i>${eyebrow}</p><h1>${title}</h1><p class="lead">${subtitle}</p></div>${actions}</div>`;
}
function miniature(pixels, className = 'digit-mini') {
  return `<div class="${className}" aria-hidden="true">${pixels.map(p => `<span style="background:rgb(${[0,1,2].map(() => Math.round(246 - p / 16 * 190)).join(',')})"></span>`).join('')}</div>`;
}
function legend(items) {
  return `<div class="legend">${items.map(([label, color]) => `<span><i style="background:${color}"></i>${label}</span>`).join('')}</div>`;
}
function chart(element, series, options = {}) {
  const all = series.flatMap(s => s.values);
  const w = Math.max(300, Math.round(element.getBoundingClientRect().width));
  const h = w < 500 ? 245 : 285, left = 54, right = 20, top = 18, bottom = 42;
  const minX = options.minX ?? 1, maxX = options.maxX ?? Math.max(...all.map(p => p.x));
  const min = options.min ?? Math.min(...all.map(p => p.y)) - .002;
  const max = options.max ?? Math.max(...all.map(p => p.y)) + .002;
  const x = v => left + (v - minX) / Math.max(1, maxX - minX) * (w - left - right);
  const y = v => h - bottom - (v - min) / Math.max(1e-9, max - min) * (h - top - bottom);
  const format = options.format ?? (n => n.toFixed(3));
  let result = '';
  for (let i = 0; i <= 4; i++) {
    const value = min + (max - min) * i / 4;
    result += `<line x1="${left}" x2="${w-right}" y1="${y(value)}" y2="${y(value)}" stroke="#e9eee6" stroke-width="1"/><text x="${left-12}" y="${y(value)+4}" text-anchor="end">${format(value)}</text>`;
  }
  const ticks = [...new Set([minX, ...[5,10,15,20,25,30,40,60,80,100].filter(n => n > minX && n < maxX), maxX])];
  const tickStep = Math.ceil(ticks.length / (w < 500 ? 5 : 8));
  const visibleTicks = ticks.filter((_, i) => i % tickStep === 0 || i === ticks.length - 1);
  visibleTicks.forEach(t => {result += `<text x="${x(t)}" y="${h-21}" text-anchor="middle">${t}</text>`;});
  for (const line of series) {
    const path = line.values.map((p, i) => `${i ? 'L' : 'M'}${x(p.x).toFixed(2)},${y(p.y).toFixed(2)}`).join(' ');
    result += `<path d="${path}" fill="none" stroke="${line.color}" stroke-width="${line.dashed ? 1.3 : 2.6}" ${line.dashed ? 'stroke-dasharray="5 5"' : ''}/>`;
    const end = line.values.at(-1);
    if (end && !line.dashed) result += `<circle cx="${x(end.x)}" cy="${y(end.y)}" r="3.3" fill="${line.color}" stroke="white" stroke-width="1.5"/>`;
  }
  element.innerHTML = `<svg class="chart-svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${options.label ?? 'Biểu đồ thực nghiệm'}"><title>${options.label ?? 'Biểu đồ thực nghiệm'}</title>${result}</svg>`;
}
function bestCurve(method, seed, prefix = 30) {
  const rows = data.trials.filter(r => r.method === method && r.seed === seed).sort((a,b) => a.trial-b.trial);
  let best = -Infinity;
  return rows.slice(0, prefix).map((row, i) => ({x: i+1, y: best = Math.max(best, row.cv_f1)}));
}
function meanCurve(method) {
  const curves = data.config.search_seeds.map(seed => bestCurve(method, seed));
  return curves[0].map((point, i) => ({x: point.x, y: curves.reduce((total, curve) => total + curve[i].y, 0) / curves.length}));
}
function modelTable(full = false) {
  return `<div class="table-wrap"><table><thead><tr><th>Cấu hình</th>${full ? '<th>Train accuracy</th>' : ''}<th>Validation F1</th><th>Test accuracy</th><th>Model (MB)</th></tr></thead><tbody>${data.metrics.map(r => `<tr class="${r.label === 'selected' ? 'selected-row' : ''}"><td><span class="method-name"><i class="method-dot" style="background:${colors[r.label]}"></i>${labels[r.label]}${r.label === 'selected' ? '<span class="pill">CHỌN</span>' : ''}</span></td>${full ? `<td>${pct(r.train_accuracy)}</td>` : ''}<td>${fixed(r.validation_f1_macro_mean)}</td><td>${pct(r.test_accuracy)}</td><td>${fixed(r.model_mb,3)}</td></tr>`).join('')}</tbody></table></div>`;
}
function overview() {
  const selected = data.metrics.find(r => r.label === 'selected');
  const random = data.metrics.find(r => r.label === 'random');
  const reduction = (1-selected.model_mb/random.model_mb)*100;
  content.innerHTML = `<div class="page-heading"><div class="hero-title"><p class="eyebrow"><i></i>NHẬP MÔN TỐI ƯU HÓA · DIGITS</p><h1>Chất lượng tốt.<br>Mô hình gọn hơn.</h1><p class="lead">Khám phá cách Random Search và TPE tìm siêu tham số Random Forest,<br>cân bằng chất lượng dự đoán với kích thước mô hình.</p></div><div class="hero-art">${miniature(data.samples.find(s => s.label === 8).pixels, 'hero-pixel')}<div class="hero-art-text"><strong>8 × 8</strong><p>Chữ số viết tay<br>10 lớp · 64 đặc trưng</p></div></div></div>
  <div class="stats">
    <article class="stat"><div class="stat-top">Test accuracy<span class="stat-icon">${icon('check')}</span></div><div class="stat-value">${fixed(selected.test_accuracy*100,2)}<small>%</small></div><div class="stat-foot good">${icon('check')}350 / 360 mẫu chính xác</div></article>
    <article class="stat"><div class="stat-top">Validation Macro-F1<span class="stat-icon">${icon('chart')}</span></div><div class="stat-value">${fixed(selected.validation_f1_macro_mean)}</div><div class="stat-foot">Trung bình 3 folds development</div></article>
    <article class="stat"><div class="stat-top">Kích thước mô hình<span class="stat-icon">${icon('leaf')}</span></div><div class="stat-value">${fixed(selected.model_mb,2)}<small>MB</small></div><div class="stat-foot good">↓ ${fixed(reduction,2)}% so với RF Random tốt nhất</div></article>
    <article class="stat"><div class="stat-top">Ngân sách tìm kiếm<span class="stat-icon">${icon('sliders')}</span></div><div class="stat-value">180<small>trials</small></div><div class="stat-foot">2 phương pháp × 3 seeds × 30 trials</div></article>
  </div>
  <div class="panels"><section class="panel"><div class="panel-header"><div><h2>Hành trình tìm kiếm</h2><p class="panel-subtitle">Best-so-far Macro-F1 · trung bình qua 3 search seeds</p></div><span class="tiny-tag">CV · DEVELOPMENT</span></div><div id="overview-chart" class="chart-box"></div>${legend([['Random Search',colors.random],['Optuna TPE',colors.tpe],['RF mặc định','#a7b59f']])}<div class="panel-footer"><span>Trục ngang: số trials hoàn thành mỗi run</span><button class="text-button" data-go="search">Khám phá ${icon('arrow')}</button></div></section>
  <section class="panel selection-panel"><div class="panel-header"><div><h2>Lựa chọn sau tối ưu</h2><p class="panel-subtitle">Model nhỏ nhất trong ngưỡng CV chấp nhận</p></div><span class="tiny-tag">ε = 0.005</span></div><div class="panel-inner"><div class="forest-count"><strong>100</strong><span>cây quyết định</span></div><dl class="param-list"><div><dt>max_depth</dt><dd>Không giới hạn</dd></div><div><dt>min_samples_leaf</dt><dd>1</dd></div><div><dt>max_features</dt><dd>sqrt</dd></div></dl><p class="selection-note">RF mặc định đạt ngưỡng <strong>${fixed(data.selection.minimum_acceptable_cv_f1)}</strong>, với kích thước nhỏ nhất trong các ứng viên đã quan sát.</p><button class="text-button" data-go="comparison">Xem so sánh chi tiết ${icon('arrow')}</button></div><div class="selection-badge">${icon('check')}Chọn bằng CV, không chọn bằng test</div></section></div>
  <section class="panel full-panel"><div class="panel-header"><div><h2>Các mô hình trên cùng một protocol</h2><p class="panel-subtitle">Test được báo cáo sau khi khóa lựa chọn · MB thập phân, joblib không nén</p></div><button class="text-button" data-go="comparison">Xem tất cả ${icon('arrow')}</button></div>${modelTable()}<div class="split-row"><span class="split-info"><strong>1.437</strong> development</span><div class="split-bar" aria-label="80% development, 20% test"><i></i></div><span class="split-info"><strong>360</strong> test</span><span class="tiny-tag">STRATIFIED · SEED 42</span></div></section>
  <div class="info-strip">${icon('file')}<span><strong>Đọc kết quả đúng ngữ cảnh.</strong> Mô hình được chọn trùng RF mặc định theo quy tắc epsilon. Điểm CV cao nhất thuộc Random Search; điều này không chứng minh Random luôn tốt hơn TPE.</span></div>`;
  const base = data.trials.find(r => r.method === 'default').cv_f1;
  chart($('#overview-chart'), [{values: meanCurve('random'),color: colors.random},{values: meanCurve('tpe'),color: colors.tpe}, {values:[{x:1,y:base},{x:30,y:base}],color:'#a7b59f',dashed:true}], {label:'Best-so-far Macro-F1 trung bình 3 search seeds, Random và TPE'});
}
function comparison() {
  const selected = data.metrics.find(r => r.label === 'selected');
  content.innerHTML = heading('KẾT QUẢ / BASELINE', 'So sánh bằng cùng một thước đo.', 'Bốn đại diện được định trước. Validation là trung bình 3 folds;<br>test là holdout gốc sau khi khóa cấu hình.', `<a class="button" id="download-csv" href="data/comparison.csv" download="digits-baseline-comparison.csv">${icon('file')}Tải bảng CSV</a>`)+`
    <div class="summary-grid"><article class="summary-card"><p>TRAIN · IN-SAMPLE</p><strong>${pct(selected.train_accuracy)}</strong><small>Đúng 1.437 / 1.437 mẫu đã học.<br>Không phải đánh giá trên dữ liệu mới.</small></article><article class="summary-card"><p>VALIDATION · ACCURACY</p><strong>${pct(selected.validation_accuracy_mean)}</strong><small>Trung bình 3 folds.<br>Khác với Macro-F1 ở bảng dưới.</small></article><article class="summary-card"><p>TEST · ACCURACY</p><strong>${pct(selected.test_accuracy)}</strong><small>Đúng 350 / 360 mẫu test gốc.<br>Giữ nguyên lựa chọn theo epsilon.</small></article></div>
    <section class="panel full-panel"><div class="panel-header"><div><h2>RF mặc định, Random Search và TPE</h2><p class="panel-subtitle">Train refit trên toàn development · Validation F1 = mean Macro-F1 CV</p></div><span class="tiny-tag">CÙNG SPLIT · CÙNG FOLDS</span></div>${modelTable(true)}<div class="panel-footer">Train 100% có thể xảy ra với cây không giới hạn độ sâu. Cần đọc cùng validation/test để đánh giá khả năng tổng quát hóa.</div></section>
    <div class="definition-grid"><div class="definition"><h3>Vì sao không chọn CV cao nhất?</h3><p>Quy tắc đã định trước cho phép giảm tối đa 0.005 Macro-F1 CV để ưu tiên model nhỏ. RF mặc định đạt ngưỡng; đây là lựa chọn nhỏ nhất trong tập ứng viên đã quan sát, không phải tối ưu toàn cục.</p></div><div class="definition"><h3>Test hơn nhau một mẫu nghĩa là gì?</h3><p>Random đúng 351 mẫu, selected đúng 350 mẫu. Chênh lệch trên test nhỏ này không được dùng để thay đổi quy tắc chọn model hay kết luận một thuật toán luôn tốt hơn.</p></div></div>
    <section class="panel full-panel" style="margin-top:22px"><div class="panel-header"><div><h2>10 mẫu mô hình selected nhận nhầm</h2><p class="panel-subtitle">Nhãn thật → dự đoán đã lưu · xem lại toàn bộ lỗi trên test gốc, không dự đoán mới</p></div></div><div class="error-grid">${data.errors.map(e => `<div class="error-card">${miniature(e.pixels)}<span class="error-label">${e.actual} → <strong>${e.predicted}</strong></span><span class="error-index">#${e.index}</span></div>`).join('')}</div></section>`;
}
function search() {
  content.innerHTML = heading('PHƯƠNG PHÁP / TÌM KIẾM', 'Từng trial, một bước tiến.', 'Phát lại log Random Search và TPE để quan sát best-so-far.<br>Đây là lịch sử thực nghiệm đã lưu, không chạy huấn luyện mới.')+`
    <section class="panel chart-large full-panel"><div class="panel-header"><div><h2>Quá trình tối ưu Macro-F1 CV</h2><p class="panel-subtitle">Chỉ các trial đã hoàn thành trong phần log đang xem</p></div><span class="tiny-tag">NGÂN SÁCH 30 TRIALS / RUN</span></div><div class="toolbar"><label for="seed-select">Search seed</label><select id="seed-select">${data.config.search_seeds.map(s => `<option ${s===state.seed?'selected':''}>${s}</option>`).join('')}</select><div class="range-control"><input id="trial-range" aria-label="Số trial được xem" type="range" min="1" max="30" value="${state.prefix}"><span class="range-number" id="prefix-count">${state.prefix} / 30</span></div><button class="button soft" id="play-replay">▶ Phát lại</button></div><div id="search-chart" class="chart-box"></div>${legend([['Random Search',colors.random],['Optuna TPE',colors.tpe]])}<div class="panel-footer"><span id="search-best"></span><span>Giữ RF seed = 123 và CV seed = 42</span></div></section>
    <section class="panel full-panel"><div class="panel-header"><div><h2>Các trial vừa hoàn thành</h2><p class="panel-subtitle">8 bản ghi gần nhất của seed đang xem · trial hiển thị từ 1</p></div><span class="tiny-tag" id="trial-record-count"></span></div><div class="table-wrap"><table><thead><tr><th>Phương pháp</th><th>Trial</th><th>Số cây</th><th>Độ sâu</th><th>Leaf</th><th>Features</th><th>CV Macro-F1</th></tr></thead><tbody id="trial-rows"></tbody></table></div><div class="panel-footer">TPE có 10 startup trials ngẫu nhiên trong protocol gốc. Cả hai phương pháp dùng cùng folds, miền tìm kiếm và ngân sách.</div></section>`;
  $('#seed-select').addEventListener('change', e => {state.seed=Number(e.target.value);updateSearch();});
  $('#trial-range').addEventListener('input', e => {stopReplay();state.prefix=Number(e.target.value);updateSearch();});
  $('#play-replay').addEventListener('click', () => {
    if (state.timer) {stopReplay(); return;}
    if (state.prefix>=30) state.prefix=0;
    $('#play-replay').textContent='Ⅱ Tạm dừng';
    const tick = () => {state.prefix++;$('#trial-range').value=state.prefix;updateSearch();if(state.prefix>=30)stopReplay();};
    tick(); if(state.prefix<30)state.timer=setInterval(tick,420);
  });
  updateSearch();
}
function updateSearch() {
  const random = bestCurve('random',state.seed,state.prefix), tpe=bestCurve('tpe',state.seed,state.prefix);
  chart($('#search-chart'),[{values:random,color:colors.random},{values:tpe,color:colors.tpe}],{label:`CV best-so-far seed ${state.seed}, ${state.prefix} trial mỗi phương pháp`});
  $('#prefix-count').textContent=`${state.prefix} / 30`;
  $('#search-best').textContent=`CV cao nhất trong phần log này: ${fixed(Math.max(random.at(-1).y,tpe.at(-1).y))}`;
  const rows=data.trials.filter(r => r.method!=='default'&&r.seed===state.seed&&r.trial<state.prefix).sort((a,b)=>b.trial-a.trial||a.method.localeCompare(b.method));
  $('#trial-record-count').textContent=`${rows.length} / 60 BẢN GHI`;
  $('#trial-rows').innerHTML=rows.slice(0,8).map(r=>`<tr><td><span class="method-name"><i class="method-dot" style="background:${colors[r.method]}"></i>${r.method==='random'?'Random':'TPE'}</span></td><td>${r.trial+1}</td><td>${r.params.n_estimators}</td><td>${r.params.max_depth??'None'}</td><td>${r.params.min_samples_leaf}</td><td>${r.params.max_features}</td><td>${fixed(r.cv_f1)}</td></tr>`).join('');
}
function stopReplay() {
  clearInterval(state.timer);state.timer=null;
  const button=$('#play-replay');if(button)button.textContent='▶ Phát lại';
}
function curves() {
  content.innerHTML=heading('CHẨN ĐOÁN / LEARNING CURVES','Hai tập dữ liệu, hai góc nhìn.','Theo dõi train và validation khi tăng số cây của cấu hình selected.<br>Mỗi điểm là trung bình của 3 folds đã cố định.')+`
    <section class="panel chart-large full-panel"><div class="panel-header curves-header"><div><h2 id="curve-title"></h2><p class="panel-subtitle">Đủ mốc số cây · không làm mượt · không dùng test</p></div><div class="curve-controls"><div class="segmented" aria-label="Chọn metric"><button data-metric="accuracy">Accuracy</button><button data-metric="log_loss">Loss</button></div><label class="checkbox-control"><input type="checkbox" id="zoom-loss" ${state.zoom?'checked':''}>Zoom 20–100 cây</label></div></div><div id="curve-chart" class="chart-box"></div>${legend([['Train','#356c50'],['Validation','#4885aa']])}<div class="panel-footer"><span id="curve-endpoint"></span><span>Trục ngang: số cây, không phải epoch</span></div></section>
    <div class="definition-grid"><div class="definition"><h3>Đường train đo trên tập nào?</h3><p>Trong mỗi fold, model học trên 958 mẫu rồi đo train trên chính các mẫu đó; validation đo trên 479 mẫu còn lại. Các đường là mean qua 3 folds, khác với loss sau refit trên toàn bộ 1.437 mẫu.</p></div><div class="definition"><h3>Vì sao train có thể đạt 100%?</h3><p>Các cây không giới hạn độ sâu có thể ghi nhớ tập học. Validation khoảng 97% cho thấy kết quả trên mẫu chưa học thấp hơn; train 100% không có nghĩa mô hình hoàn hảo.</p></div></div>`;
  $$('[data-metric]').forEach(button=>button.addEventListener('click',()=>{state.metric=button.dataset.metric;updateCurves();}));
  $('#zoom-loss').addEventListener('change',e=>{state.zoom=e.target.checked;updateCurves();});
  updateCurves();
}
function updateCurves() {
  const loss=state.metric==='log_loss', start=loss&&state.zoom?20:1;
  $$('[data-metric]').forEach(b=>{b.classList.toggle('active',b.dataset.metric===state.metric);b.setAttribute('aria-pressed',String(b.dataset.metric===state.metric));});
  $('#zoom-loss').disabled=!loss;
  $('#curve-title').textContent=loss?'Cross-entropy / log loss':'Accuracy trên train và validation';
  const series=['train','validation'].map((split,i)=>({color:i?'#4885aa':'#356c50',values:data.curves.filter(r=>r.split===split&&r.trees>=start).map(r=>({x:r.trees,y:r[state.metric]}))}));
  const max=loss?Math.max(...series.flatMap(s=>s.values.map(v=>v.y)))*1.07:1.015;
  chart($('#curve-chart'),series,{min:loss?0:.65,max,minX:start,maxX:100,format:loss?n=>n.toFixed(2):n=>pct(n,0),label:`Hai đường train và validation ${loss?'log loss':'accuracy'}, ${start} đến 100 cây`});
  const endpoints=series.map(s=>s.values.at(-1).y);
  $('#curve-endpoint').textContent=`100 cây · Train ${loss?fixed(endpoints[0],4):pct(endpoints[0])} · Validation ${loss?fixed(endpoints[1],4):pct(endpoints[1])}`;
}
function sandbox() {
  content.innerHTML=heading('TRẢI NGHIỆM / NHẬN DẠNG','Một nét vẽ. Một dự đoán.','Chọn ảnh Digits có sẵn hoặc vẽ trực tiếp trên lưới 8×8.<br>Model selected gồm 100 cây chạy ngay trong trình duyệt.')+`
    <div class="sandbox-grid"><section class="panel"><div class="panel-header"><div><h2>Ảnh đầu vào</h2><p class="panel-subtitle" id="input-description">64 pixel · cường độ 0–16</p></div><span class="tiny-tag">8 × 8 PIXEL</span></div><div class="drawing-area"><div class="canvas-frame"><canvas id="digit-canvas" width="256" height="256" role="img" tabindex="0" aria-label="Lưới vẽ chữ số 8 nhân 8. Có thể chọn mẫu bằng các nút số bên dưới."></canvas></div><p class="canvas-hint"><strong>Vẽ một chữ số<br>ở giữa khung.</strong><br><br>Nét sáng, nền tối.<br>Giữ hình gọn trong lưới.</p></div><div class="drawing-actions"><button class="button primary big" id="predict-digit">${icon('pen')}Nhận dạng</button><button class="button" id="clear-digit">Xóa nét vẽ</button></div><div class="sample-picker"><p>HOẶC CHỌN MẪU DIGITS</p><div class="sample-buttons">${Array.from({length:10},(_,i)=>`<button data-digit="${i}" aria-label="Chọn mẫu chữ số ${i}">${i}</button>`).join('')}</div><div class="sample-footer"><span id="sample-source"></span><button class="text-button" id="next-sample">Mẫu khác ↻</button></div></div></section>
    <section class="panel"><div class="panel-header"><div><h2>Kết quả nhận dạng</h2><p class="panel-subtitle">Random Forest selected · cấu hình đã khóa</p></div><span class="tiny-tag">100 CÂY</span></div><div class="prediction-main"><div class="prediction-digit" id="prediction-label">–</div><div class="prediction-meta"><p id="prediction-title">Sẵn sàng nhận dạng</p><span id="prediction-detail">Vẽ hoặc chọn một mẫu,<br>rồi nhấn Nhận dạng.</span></div></div><div class="prob-list" id="probabilities"></div><p class="prediction-caption">Điểm số là trung bình xác suất của 100 cây, chưa được hiệu chuẩn. Không coi là độ tin cậy đã được bảo đảm.</p><div class="sandbox-info">Mẫu có sẵn thuộc development, nên dự đoán đúng không phải kết quả test mới. Chữ tự vẽ có thể khác phân phối Digits và bị nhận dạng sai.</div></section></div>
    <div class="info-strip">${icon('check')}<span><strong>Mô hình thật, số liệu thật.</strong> Phép duyệt cây trong trình duyệt được đối chiếu với sklearn trên toàn bộ 1.437 mẫu development và 34 đầu vào tổng hợp. Bản web không huấn luyện lại hoặc sửa model.</span></div>`;
  const canvas=$('#digit-canvas');let drawing=false,last=null;
  const position=e=>{const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left)/r.width*8,y:(e.clientY-r.top)/r.height*8};};
  const stamp=p=>{for(let y=0;y<8;y++)for(let x=0;x<8;x++){const d=(x+.5-p.x)**2+(y+.5-p.y)**2;if(d<1.5)state.pixels[y*8+x]=Math.max(state.pixels[y*8+x],Math.round(16*Math.exp(-d/.42)));}};
  canvas.addEventListener('pointerdown',e=>{e.preventDefault();drawing=true;canvas.setPointerCapture(e.pointerId);last=position(e);state.expected=null;state.revision++;stamp(last);resetPrediction();drawCanvas();$('#input-description').textContent='Nét tự vẽ · 64 pixel';$('#sample-source').textContent='Đầu vào tự vẽ';$$('[data-digit]').forEach(b=>b.classList.remove('active'));});
  canvas.addEventListener('pointermove',e=>{if(!drawing)return;const p=position(e),distance=Math.hypot(p.x-last.x,p.y-last.y),steps=Math.max(1,Math.ceil(distance/.15));for(let i=1;i<=steps;i++)stamp({x:last.x+(p.x-last.x)*i/steps,y:last.y+(p.y-last.y)*i/steps});last=p;state.revision++;drawCanvas();});
  const end=()=>{drawing=false;last=null;};canvas.addEventListener('pointerup',end);canvas.addEventListener('pointercancel',end);canvas.addEventListener('lostpointercapture',end);
  canvas.addEventListener('keydown',e=>{if(/^\d$/.test(e.key))loadSample(Number(e.key));});
  $('#clear-digit').addEventListener('click',()=>{state.pixels.fill(0);state.expected=null;state.revision++;drawCanvas();resetPrediction();$('#input-description').textContent='Nét tự vẽ · 64 pixel';$('#sample-source').textContent='Khung trống';$$('[data-digit]').forEach(b=>b.classList.remove('active'));});
  $$('[data-digit]').forEach(b=>b.addEventListener('click',()=>{loadSample(Number(b.dataset.digit));predictCanvas();}));
  $('#next-sample').addEventListener('click',()=>{const digit=state.expected?.label??3;state.samplePositions[digit]=(state.samplePositions[digit]+1)%3;loadSample(digit);predictCanvas();});
  $('#predict-digit').addEventListener('click',predictCanvas);
  loadSample(state.expected?.label??3);predictCanvas();
}
function drawCanvas() {
  const context=$('#digit-canvas').getContext('2d');
  state.pixels.forEach((p,i)=>{const t=p/16;context.fillStyle=`rgb(${Math.round(18+t*220)},${Math.round(38+t*205)},${Math.round(30+t*210)})`;context.fillRect(i%8*32,Math.floor(i/8)*32,32,32);});
}
function loadSample(label) {
  const samples=data.samples.filter(s=>s.label===label), sample=samples[state.samplePositions[label]];
  state.pixels=[...sample.pixels];state.expected={label,index:sample.index};state.revision++;drawCanvas();resetPrediction();
  $('#input-description').textContent=`Mẫu Digits #${sample.index} · nhãn thật ${label}`;
  $('#sample-source').textContent=`Development · mẫu ${state.samplePositions[label]+1}/3`;
  $$('[data-digit]').forEach(b=>{b.classList.toggle('active',Number(b.dataset.digit)===label);b.setAttribute('aria-pressed',String(Number(b.dataset.digit)===label));});
}
function resetPrediction() {
  $('#prediction-label').textContent='–';$('#prediction-title').textContent='Ảnh đầu vào đã cập nhật';$('#prediction-detail').textContent='Nhấn Nhận dạng để xem kết quả.';$('#probabilities').innerHTML='';
}
async function predictCanvas() {
  if(state.pixels.every(p=>p===0)){toast('Bạn hãy vẽ một chữ số hoặc chọn mẫu có sẵn.');return;}
  const button=$('#predict-digit'), revision=state.revision, input=[...state.pixels];
  button.disabled=true;
  try {
    modelPromise??=fetchJSON('data/forest.json').catch(error=>{modelPromise=null;throw error;});
    const model=await modelPromise;
    if(state.view!=='sandbox'||revision!==state.revision)return;
    const result=predictForest(model,input);
    $('#prediction-label').textContent=result.label;
    $('#prediction-title').textContent=`Dự đoán: chữ số ${result.label}`;
    $('#prediction-detail').textContent=state.expected?`Nhãn thật ${state.expected.label} · ${result.label===state.expected.label?'khớp với mẫu đã chọn':'khác nhãn mẫu đã chọn'}`:'Kết quả từ nét tự vẽ của bạn.';
    $('#probabilities').innerHTML=result.probabilities.map((p,i)=>({p,label:model.classes[i]})).sort((a,b)=>b.p-a.p||a.label-b.label).slice(0,3).map(r=>`<div class="prob-row"><span>Số ${r.label}</span><div class="prob-track"><div class="prob-fill" style="width:${r.p*100}%"></div></div><strong>${pct(r.p,0)}</strong></div>`).join('');
  } catch(error){toast(`Không chạy được model: ${error.message}`);}finally{if(button.isConnected)button.disabled=false;}
}
function navigate(view) {
  if(!(view in titles))view='overview';
  stopReplay();state.view=view;
  $$('.nav-item').forEach(button=>{button.classList.toggle('active',button.dataset.view===view);if(button.dataset.view===view)button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');});
  $('#breadcrumb-title').textContent=titles[view];
  $('#sidebar').classList.remove('open');$('#backdrop').classList.remove('open');$('#menu-toggle').setAttribute('aria-expanded','false');
  ({overview,comparison,search,curves,sandbox})[view]();
  window.scrollTo({top:0,behavior:'instant'});
}
document.addEventListener('click',event=>{const button=event.target.closest('[data-view],[data-go]');if(button)location.hash=button.dataset.view??button.dataset.go;});
$('#menu-toggle').addEventListener('click',()=>{const open=$('#sidebar').classList.toggle('open');$('#backdrop').classList.toggle('open',open);$('#menu-toggle').setAttribute('aria-expanded',String(open));});
$('#backdrop').addEventListener('click',()=>{$('#sidebar').classList.remove('open');$('#backdrop').classList.remove('open');$('#menu-toggle').setAttribute('aria-expanded','false');});
document.addEventListener('keydown',e=>{if(e.key==='Escape')$('#backdrop').click();});
window.addEventListener('hashchange',()=>{if(data)navigate(location.hash.slice(1));});
let resizeTimeout;
window.addEventListener('resize',()=>{clearTimeout(resizeTimeout);resizeTimeout=setTimeout(()=>{if(!data)return;if(state.view==='overview')overview();else if(state.view==='search')updateSearch();else if(state.view==='curves')updateCurves();},100);});
try {data=await fetchJSON('data/dashboard.json');navigate(location.hash.slice(1)||'overview');}
catch(error){content.innerHTML='<div class="fatal"><h1>Chưa mở được dữ liệu.</h1><p>Hãy mở bằng địa chỉ web, thay vì nhấp trực tiếp file HTML.</p><button class="button" id="retry-load">Thử lại</button></div>';$('#retry-load').addEventListener('click',()=>location.reload());console.error(error);}
