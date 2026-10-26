const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const NS = 'http://www.w3.org/2000/svg';
const node = (name, attrs = {}, text = '') => {
  const el = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text) el.textContent = text;
  return el;
};
const colors = {
  'Coh-F': '#7b837b',
  'Coh-R': '#32718c',
  'Bond-F': '#cfaa61',
  'Bond-R': '#aa708e'
};
let evidence, toastTimer;
const groups = [];
let reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
try {
  reduced ||= localStorage.getItem('tapesim-reduced') === 'true';
} catch {}
document.documentElement.classList.add('js');

function toast(message) {
  clearTimeout(toastTimer);
  $('#toast').textContent = message;
  $('#toast').classList.add('visible');
  toastTimer = setTimeout(() => $('#toast').classList.remove('visible'), 4000);
}

function loadVideo(video) {
  if (video.dataset.src && !video.getAttribute('src')) {
    video.src = video.dataset.src;
    video.load();
  }
}

function ready(video) {
  video.preload = 'auto';
  loadVideo(video);
  if (video.readyState >= 2) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error('Video loading timed out. Please try again.'));
    }, 18000);
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener('loadeddata', done);
      video.removeEventListener('canplay', done);
      video.removeEventListener('seeked', done);
      video.removeEventListener('error', fail);
    };
    const done = () => {
      if (video.readyState < 2) return;
      cleanup();
      resolve();
    };
    const fail = () => {
      cleanup();
      reject(new Error('This video could not load. Please reload and try again.'));
    };
    video.addEventListener('loadeddata', done);
    video.addEventListener('canplay', done);
    video.addEventListener('seeked', done);
    video.addEventListener('error', fail, {
      once: true
    });
  });
}

function setPressed(buttons, selected) {
  for (const b of buttons) {
    const yes = b === selected;
    b.classList.toggle('is-active', yes);
    b.setAttribute('aria-pressed', yes);
  }
}

class SynchronizedPlayer {
  constructor({
    videos,
    button,
    range,
    duration,
    rate = 1,
    onUpdate = () => {},
    region
  }) {
    Object.assign(this, {
      videos,
      button,
      range,
      duration,
      rate,
      onUpdate,
      region
    });
    this.time = 0;
    this.playing = false;
    this.busy = false;
    this.version = 0;
    groups.push(this);
    videos.forEach(v => { v.controls = false; });
    button.addEventListener('click', () => this.playing ? this.pause() : this.play());
    range.addEventListener('input', () => {
      this.seek(+range.value);
    });
    this.update(0, true);
  }
  async prepare() {
    await Promise.all(this.videos.map(ready));
  }
  async play() {
    if (this.busy) return;
    if (this.time >= this.duration - .03) this.time = 0;
    this.busy = true;
    this.button.disabled = true;
    const token = this.version;
    try {
      await this.prepare();
      if (token !== this.version) return;
      for (const g of groups)
        if (g !== this) g.pause();
      hero.pause();
      stopRoll();
      this.base = this.time;
      this.started = performance.now();
      this.playing = true;
      this.update(this.time, true);
      await Promise.all(this.videos.map(v => {
        if (this.time < v.duration - .05) {
          v.playbackRate = this.rate;
          return v.play();
        }
      }));
      this.button.textContent = 'Ⅱ';
      this.button.setAttribute('aria-label', 'Pause comparison');
      this.tick();
    } catch (error) {
      this.pause();
      toast(error.message);
    } finally {
      this.busy = false;
      this.button.disabled = false;
    }
  }
  pause() {
    this.version++;
    this.playing = false;
    cancelAnimationFrame(this.raf);
    for (const v of this.videos) v.pause();
    this.button.textContent = '▶';
    this.button.setAttribute('aria-label', 'Play comparison');
  }
  async seek(t) {
    this.time = clamp(t, 0, this.duration);
    this.base = this.time;
    this.started = performance.now();
    this.update(this.time, true);
    try {
      await this.prepare();
      this.update(this.time, true);
      if (this.playing)
        for (const v of this.videos) {
          if (this.time < v.duration - .05) {
            v.playbackRate = this.rate;
            v.play().catch(() => {});
          }
        }
    } catch (error) {
      toast(error.message);
    }
  }
  setRate(rate) {
    this.base = this.time;
    this.started = performance.now();
    this.rate = rate;
    for (const v of this.videos) v.playbackRate = rate;
  }
  update(t, force = false) {
    for (const v of this.videos) {
      if (v.readyState >= 1) {
        const wanted = clamp(t, 0, Math.max(0, v.duration - 1 / 30));
        if (force || Math.abs(v.currentTime - wanted) > .1) v.currentTime = wanted;
        if (t >= v.duration - .033) v.pause();
      }
    }
    this.range.value = t;
    this.onUpdate(t);
  }
  tick() {
    if (!this.playing) return;
    this.time = Math.min(this.duration, this.base + (performance.now() - this.started) / 1000 * this.rate);
    this.update(this.time);
    if (this.time >= this.duration) {
      this.pause();
      return;
    }
    this.raf = requestAnimationFrame(() => this.tick());
  }
  replace(videos, duration) {
    this.version++;
    this.pause();
    this.videos = videos;
    videos.forEach(v => { v.controls = false; });
    this.duration = duration;
    this.time = 0;
    this.range.max = duration;
    this.update(0, true);
  }
}

// Videos and schematics animate only after explicit interaction. The display
// preference stops them without disabling subsequent manual playback.
function setMotion(value, persist = false) {
  reduced = value;
  document.documentElement.classList.toggle('reduced', value);
  $('#motion-setting').setAttribute('aria-pressed', value);
  $('#motion-setting').innerHTML = `<span aria-hidden="true">${value?'◎':'◉'}</span> ${value?'Motion reduced':'Reduce motion'}`;
  if (value) {
    $('#research-video').pause();
    stopRoll();
  }
  if (persist) try {
    localStorage.setItem('tapesim-reduced', String(value));
  } catch {}
}
$('#motion-setting').addEventListener('click', () => setMotion(!reduced, true));
matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change', e => setMotion(e.matches));
$('#menu-toggle').addEventListener('click', () => {
  const open = $('#menu-toggle').getAttribute('aria-expanded') !== 'true';
  $('#menu-toggle').setAttribute('aria-expanded', open);
  $('#menu-toggle').setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
  $('#nav').classList.toggle('is-open', open);
});
for (const link of $$('#nav a')) link.addEventListener('click', () => {
  $('#nav').classList.remove('is-open');
  $('#menu-toggle').setAttribute('aria-expanded', 'false');
  $('#menu-toggle').setAttribute('aria-label', 'Open navigation');
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    $('#nav').classList.remove('is-open');
    $('#menu-toggle').setAttribute('aria-expanded', 'false');
  }
});
let scrollRAF;
addEventListener('scroll', () => {
  if (scrollRAF) return;
  scrollRAF = requestAnimationFrame(() => {
    let active = '';
    for (const id of ['video', 'abstract', 'method', 'dynamics', 'reality', 'policies', 'bibtex'])
      if ($('#' + id).getBoundingClientRect().top < innerHeight * .43) active = id;
    for (const a of $$('#nav a')) {
      a.classList.toggle('active', a.hash === '#' + active);
      if (a.hash === '#' + active) a.setAttribute('aria-current', 'location');
      else a.removeAttribute('aria-current');
    }
    scrollRAF = null;
  });
}, {
  passive: true
});
const lazy = new IntersectionObserver(entries => {
  for (const e of entries)
    if (e.isIntersecting) {
      loadVideo(e.target);
      lazy.unobserve(e.target);
    }
}, {
  rootMargin: '260px'
});
$$('video[data-src]').forEach(v => lazy.observe(v));
const hero = $('#research-video');

const chapterButtons = $$('[data-film-time]');
const chapterStarts = chapterButtons.map(b => Number(b.dataset.filmTime));
let chapterRequest = 0;
function updateFilmChapters() {
  const t = hero.currentTime || 0;
  const active = Math.max(0, chapterStarts.findLastIndex(start => t >= start));
  chapterButtons.forEach((button, i) => {
    const selected = i === active;
    button.classList.toggle('is-active', selected);
    if (selected) button.setAttribute('aria-current', 'true');
    else button.removeAttribute('aria-current');
    const end = chapterStarts[i + 1] ?? (Number.isFinite(hero.duration) ? hero.duration : 179);
    const progress = clamp((t - chapterStarts[i]) / (end - chapterStarts[i]), 0, 1);
    button.style.setProperty('--chapter-progress', `${progress * 100}%`);
  });
}
chapterButtons.forEach(button => button.addEventListener('click', () => {
  const request = ++chapterRequest;
  const seek = () => {
    if (request !== chapterRequest) return;
    hero.currentTime = Number(button.dataset.filmTime);
    updateFilmChapters();
  };
  if (hero.readyState >= 1) seek();
  else hero.addEventListener('loadedmetadata', seek, {once: true});
  hero.preload = 'auto';
  // Start from the user gesture, also when metadata has not loaded yet.
  hero.play().catch(error => {
    if (request === chapterRequest && error.name !== 'AbortError') {
      toast('Chapter selected. Use the video play button to start playback.');
    }
  });
}));
['loadedmetadata', 'timeupdate', 'seeking', 'ended'].forEach(event =>
  hero.addEventListener(event, updateFilmChapters));
updateFilmChapters();
$('#video-chapters').hidden = false;

new IntersectionObserver(entries => {
  for (const e of entries) {
    if (!e.isIntersecting) hero.pause();
  }
}, {
  threshold: .2
}).observe(hero);

// Continuous spiral and tangent free strip. Purple is still-wound collar, not
// a connector inserted into the roll. This animation is explanatory only.
let rollFrame = null,
  rollRunning = false,
  rollLast = 0,
  rollPosition = 20,
  activeRegion = 'cluster';

function drawRoll(value) {
  const p = value / 100,
    cx = 434 + 90 * p,
    cy = 230,
    r = 109 - 20 * p,
    turns = 7.4 - 1.7 * p;
  const layer = $('#roll-drawing');
  layer.replaceChildren();
  const annulus = node('path', {
    d: `M ${cx-r} ${cy} a ${r} ${r} 0 1 0 ${2*r} 0 a ${r} ${r} 0 1 0 ${-2*r} 0 M ${cx-45} ${cy} a 45 45 0 1 1 90 0 a 45 45 0 1 1 -90 0`,
    'fill-rule': 'evenodd',
    fill: '#d1ad69',
    opacity: activeRegion === 'cluster' ? 1 : .65
  });
  layer.append(annulus);
  let gold = '',
    purple = '';
  const count = 950;
  for (let i = 0; i <= count; i++) {
    const q = i / count,
      theta = -Math.PI / 2 + turns * 2 * Math.PI * (1 - q),
      rad = 46 + (r - 46) * q,
      x = cx + Math.cos(theta) * rad,
      y = cy + Math.sin(theta) * rad;
    const point = `${x.toFixed(2)} ${y.toFixed(2)}`;
    gold += (i ? 'L' : 'M') + point;
    if (theta <= -Math.PI / 2 + .8) purple += (purple ? 'L' : 'M') + point;
  }
  layer.append(node('path', {
    d: gold,
    fill: 'none',
    stroke: '#967b46',
    'stroke-width': 1.5,
    opacity: activeRegion === 'cluster' ? .65 : .25
  }));
  layer.append(node('circle', {
    cx,
    cy,
    r: 43,
    fill: '#ffffff',
    stroke: '#a4ad9f',
    'stroke-width': 7
  }));
  const theta = -p * 2.7;
  layer.append(node('path', {
    d: `M${cx+30*Math.cos(theta)} ${cy+30*Math.sin(theta)}L${cx+39*Math.cos(theta)} ${cy+39*Math.sin(theta)}`,
    stroke: '#76866d',
    'stroke-width': 3
  }));
  layer.append(node('path', {
    d: purple,
    fill: 'none',
    stroke: '#a96e8e',
    'stroke-width': 8,
    'stroke-linecap': 'round',
    opacity: activeRegion === 'collar' ? 1 : .78
  }));
  const endY = cy - r;
  layer.append(node('path', {
    d: `M${cx} ${endY} C${cx-100} ${endY} 220 ${endY-2} 89 ${endY-8*p}`,
    fill: 'none',
    stroke: '#32718c',
    'stroke-width': 7,
    'stroke-linecap': 'round',
    opacity: activeRegion === 'free' ? 1 : .8
  }));
  layer.append(node('path', {
    d: `M78 ${endY-8*p-14}v29M70 ${endY-8*p-9}h8M70 ${endY-8*p+1}h8M70 ${endY-8*p+11}h8`,
    stroke: '#7c8775',
    'stroke-width': 2,
    fill: 'none'
  }));
  layer.append(node('text', {
    x: cx,
    y: cy + 4,
    'text-anchor': 'middle',
    fill: '#687765',
    'font-size': 11,
    'font-family': 'DM'
  }, 'HUB'));
  const leaders = $$('.diagram-labels path');
  leaders[0].setAttribute('d', `M${cx+r*.72} ${cy+r*.68}L610 319H660`);
  leaders[1].setAttribute('d', `M${cx+r*.35} ${cy-r*.91}L580 61H650`);
  leaders[2].setAttribute('d', `M203 ${endY-1}V217H146`);
  $('#roll-progress').value = Math.round(value) + '%';
}

function stopRoll() {
  rollRunning = false;
  cancelAnimationFrame(rollFrame);
  $('#roll-play').textContent = '▶';
  $('#roll-play').setAttribute('aria-label', 'Animate dispensing');
}

function animateRoll(now) {
  if (!rollRunning) return;
  const delta = rollLast ? Math.min(now - rollLast, 100) : 0;
  rollLast = now;
  let value = rollPosition + delta / 90;
  rollPosition = value;
  if (value >= 100) {
    value = 100;
    stopRoll();
  }
  $('#roll-range').value = value;
  drawRoll(value);
  if (rollRunning) rollFrame = requestAnimationFrame(animateRoll);
}
$('#roll-play').addEventListener('click', () => {
  if (rollRunning) {
    stopRoll();
    return;
  }
  if (+$('#roll-range').value >= 100) $('#roll-range').value = 0;
  rollRunning = true;
  rollLast = 0;
  rollPosition = +$('#roll-range').value;
  $('#roll-play').textContent = 'Ⅱ';
  $('#roll-play').setAttribute('aria-label', 'Pause dispensing animation');
  rollFrame = requestAnimationFrame(animateRoll);
});
$('#roll-range').addEventListener('input', () => {
  stopRoll();
  drawRoll(+$('#roll-range').value);
});
$$('[data-region]').forEach(b => b.addEventListener('click', () => {
  activeRegion = b.dataset.region;
  setPressed($$('[data-region]'), b);
  drawRoll(+$('#roll-range').value);
}));

function svgDrag(svg, input, project, callback) {
  let active = false;
  svg.addEventListener('pointerdown', e => {
    active = true;
    svg.setPointerCapture(e.pointerId);
    if (e.pointerType === 'mouse') e.preventDefault();
  });
  svg.addEventListener('pointermove', e => {
    if (!active) return;
    const r = svg.getBoundingClientRect();
    input.value = project(e, r);
    callback();
  });
  svg.addEventListener('pointerup', () => active = false);
  svg.addEventListener('pointercancel', () => active = false);
}
svgDrag($('#roll-svg'), $('#roll-range'), (e, r) => clamp((e.clientX - r.left) / r.width * 160 - 30, 0, 100), () => {
  stopRoll();
  drawRoll(+$('#roll-range').value);
});
drawRoll(20);

let bonded = true;

function drawBond() {
  const v = +$('#bond-range').value,
    y = 119 - v * .55;
  if (bonded && v > 70) bonded = false;
  if (!bonded && v <= 0) bonded = true;
  const links = $('#bond-links');
  links.replaceChildren();
  if (bonded) {
    links.append(node('path', {
      d: `M122 210L275 ${y}L418 210L330 254Z`,
      fill: '#b9799b20',
      stroke: 'none'
    }));
    for (const [x, yy] of [
        [122, 210],
        [418, 210],
        [330, 254]
      ]) links.append(node('path', {
      d: `M275 ${y}L${x} ${yy}`,
      fill: 'none',
      stroke: '#a9708d',
      'stroke-width': 2
    }));
  }
  $('#bond-point').setAttribute('cy', y);
  $('#bond-point').style.fill = bonded ? '#955571' : '#32718c';
  const tip = Math.max(47, y - 47);
  $('#force-arrow').setAttribute('d', `M275 ${y-17}V${tip}m-6 7 6-7 6 7`);
  $('#force-arrow').style.opacity = v < 0 ? '0' : '.8';
  $('#bond-state').classList.toggle('released', !bonded);
  $('#bond-state').textContent = !bonded ? 'Released' : v < 0 ? 'Compressed · still bonded' : 'Bonded';
  $('#bond-message').textContent = !bonded ? 'Opening + force threshold exceeded: the bond releases.' : v < 0 ? 'Compression does not trigger force-based release.' : v > 10 ? 'The bond resists opening until its force threshold is exceeded.' : 'Nearby eligible surfaces form a bond.';
  $('#bond-hint').textContent = !bonded ? 'Return to rest to reattach' : v < 0 ? 'Compression keeps the bond' : 'Pull upward to test the bond';
}
$('#bond-range').addEventListener('input', drawBond);
$('#bond-reset').addEventListener('click', () => {
  $('#bond-range').value = 0;
  bonded = true;
  drawBond();
});
svgDrag($('#bond-svg'), $('#bond-range'), (e, r) => clamp((.45 - (e.clientY - r.top) / r.height) * 210, -60, 100), drawBond);
drawBond();
const variantText = {
  'Coh-F': 'Coh-F resolves cohesive contact while every wound layer stays deformable.',
  'Coh-R': 'Coh-R keeps cohesive interfaces near the release front and uses the rigid cluster and collar.',
  'Bond-F': 'Bond-F uses releasable internal bonds while every wound layer stays deformable.',
  'Bond-R': 'Bond-R combines releasable internal bonds with the rigid cluster and advancing collar.'
};
$$('[data-model]').forEach(b => b.addEventListener('click', () => {
  setPressed($$('[data-model]'), b);
  $('#variant-description').textContent = variantText[b.dataset.model];
}));

// Dialog: the shipped real files are already physically trimmed to the whitelist.
let dialogTrigger = null,
  dialogVersion = 0;
async function openMedia(id, title, description, at = 0) {
  const request = ++dialogVersion;
  for (const g of groups) g.pause();
  hero.pause();
  stopRoll();
  dialogTrigger = document.activeElement;
  const d = $('#media-dialog'),
    v = $('#dialog-video');
  v.pause();
  v.replaceChildren();
  v.src = `assets/media/${id}.mp4`;
  v.poster = `assets/media/${id}.webp`;
  $('#dialog-title').textContent = title;
  $('#dialog-description').textContent = description;
  $('#dialog-kicker').textContent = 'TAPESIM / ORIGINAL FOOTAGE';
  d.showModal();
  document.body.classList.add('modal-open');
  v.load();
  try {
    await ready(v);
    if (!d.open || request !== dialogVersion) return;
    v.currentTime = at;
    await v.play();
  } catch {
    if (d.open) toast('Use the video controls to start playback.');
  }
}
$$('.media-open').forEach(b => b.addEventListener('click', () => openMedia(b.dataset.media, b.dataset.title, b.dataset.description)));
$('#dialog-close').addEventListener('click', () => $('#media-dialog').close());
$('#media-dialog').addEventListener('click', e => {
  if (e.target === $('#media-dialog')) {
    const r = e.target.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) e.target.close();
  }
});
$('#media-dialog').addEventListener('close', () => {
  dialogVersion++;
  $('#dialog-video').pause();
  document.body.classList.remove('modal-open');
  dialogTrigger?.focus();
});

// Data-dependent interactions are isolated so a failed JSON request cannot hide
// the page or stop native film controls from working.
function drawMotion(t) {
  const svg = $('#motion-chart');
  const width = Math.max(280, svg.clientWidth || 1040),
    height = width < 600 ? 180 : 205;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.replaceChildren();
  const samples = evidence.motion.samples;
  const frame = Math.floor((t + 1e-5) * 30) + 165;
  const current = samples.find(s => s.video_frame === frame) || samples.at(-1);
  $('#motion-readout').textContent = `Source t = ${current.time_s.toFixed(2)} s`;
  const left = 40,
    right = width - 20,
    bottom = height - 40;
  const X = x => left + (x - 5.5) / 8.5 * (right - left);
  const Y = y => bottom - y / 140 * (bottom - 24);
  for (const y of [0, 40, 80, 120]) {
    svg.append(node('path', {
      d: `M${left} ${Y(y)}H${right}`,
      stroke: '#d8ded4',
      'stroke-width': 1
    }));
    svg.append(node('text', {
      x: left - 9,
      y: Y(y) + 4,
      'text-anchor': 'end',
      fill: '#586271',
      'font-size': 11,
      'font-family': 'DM'
    }, String(y)));
  }
  for (const x of [5.5, 7.5, 10, 12, 14]) {
    svg.append(node('text', {
      x: X(x),
      y: bottom + 19,
      'text-anchor': 'middle',
      fill: '#586271',
      'font-size': 11,
      'font-family': 'DM'
    }, String(x)));
  }
  svg.append(node('text', {
    x: (left + right) / 2,
    y: height - 3,
    'text-anchor': 'middle',
    fill: '#586271',
    'font-size': 10,
    'font-family': 'DM'
  }, 'Simulation time (s)'));
  svg.append(node('path', {
    d: `M${X(7.5)} 16V${bottom}`,
    stroke: '#7c8b78',
    'stroke-dasharray': '3 4',
    'stroke-width': 1
  }));
  svg.append(node('text', {
    x: X(7.5) + 6,
    y: 13,
    fill: '#586271',
    'font-size': 10,
    'font-family': 'DM'
  }, 'Release'));
  for (const model of ['Coh-F', 'Coh-R']) {
    const shown = samples.filter(s => s.video_frame <= frame && s.time_s >= 5.5 && s.time_s <= 14);
    if (!shown.length) continue;
    const path = shown.map((s, i) => `${i ? 'L' : 'M'}${X(s.time_s).toFixed(2)} ${Y(s.theta_deg[model]).toFixed(2)}`).join(' ');
    svg.append(node('path', {
      d: path,
      fill: 'none',
      stroke: colors[model],
      'stroke-width': 2.2,
      'stroke-dasharray': model === 'Coh-R' ? '7 4' : 'none'
    }));
    const last = shown.at(-1);
    svg.append(node('circle', {
      cx: X(last.time_s),
      cy: Y(last.theta_deg[model]),
      r: 3.5,
      fill: colors[model]
    }));
  }
  svg.setAttribute('aria-label', `Saved roll orientation through simulation time ${current.time_s.toFixed(2)} seconds. Coh-F ${current.theta_deg['Coh-F'].toFixed(1)} degrees; Coh-R ${current.theta_deg['Coh-R'].toFixed(1)} degrees.`);
}
let turns = 32,
  family = 'Coh';

function drawCosts() {
  const rows = evidence.costs.filter(r => r.turns === turns);
  for (const m of evidence.models) {
    let row = $(`[data-cost-model="${m}"]`);
    if (!row) {
      row = document.createElement('div');
      row.className = 'cost-row';
      row.dataset.costModel = m;
      row.innerHTML = `<span class="cost-row-label">${m}</span><div class="cost-bar-track"><div class="cost-bar"></div></div><span class="cost-value"></span>`;
      $('#cost-chart').append(row);
    }
    const cost = rows.find(r => r.model === m).physics_mean_ms;
    row.classList.toggle('highlight', m.startsWith(family));
    $('.cost-bar', row).style.setProperty('--bar-color', colors[m]);
    $('.cost-bar', row).style.width = `${cost/330*100}%`;
    $('.cost-value', row).textContent = cost.toFixed(2);
    row.setAttribute('aria-label', `${m}: ${cost.toFixed(2)} milliseconds per physics step`);
  }
  const full = rows.find(r => r.model === family + '-F').physics_mean_ms,
    rigid = rows.find(r => r.model === family + '-R').physics_mean_ms;
  $('#speedup').innerHTML = (full / rigid).toFixed(1) + '<span>×</span>';
  $('#speedup-description').innerHTML = `faster physics steps<br>${family}-F → ${family}-R · ${turns} turns`;
  $('#cost-note').textContent = `${turns} turns · comparable rotation range · RTX PRO 6000.`;
  $('#cost-chart').setAttribute('aria-label', `${turns}-turn mean physics-step costs: ${rows.map(r=>r.model+' '+r.physics_mean_ms+' milliseconds').join(', ')}.`);
}

function drawReplayPlot() {
  const svg = $('#replay-chart');
  const width = Math.max(280, svg.clientWidth || 570),
    height = Math.min(290, Math.max(235, width / 2));
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.replaceChildren();
  const left = 30,
    right = width - 8,
    bottom = height - 34;
  const X = i => left + (i + .5) * (right - left) / 4;
  const Y = v => bottom - v / 40 * (bottom - 35);
  for (const y of [0, 10, 20, 30, 40]) {
    svg.append(node('path', {
      d: `M${left} ${Y(y)}H${right}`,
      stroke: '#d1d8ca',
      'stroke-width': 1
    }));
    svg.append(node('text', {
      x: left - 8,
      y: Y(y) + 4,
      'text-anchor': 'end',
      fill: '#586271',
      'font-size': 11,
      'font-family': 'DM'
    }, String(y)));
  }
  evidence.models.forEach((m, i) => {
    const mean = evidence.replay.models[m].active_mean_px;
    svg.append(node('text', {
      x: X(i),
      y: bottom + 25,
      'text-anchor': 'middle',
      fill: '#222933',
      'font-size': 12,
      'font-family': 'DM'
    }, m));
    svg.append(node('path', {
      d: `M${X(i)-25} ${Y(mean)}H${X(i)+25}`,
      stroke: colors[m],
      'stroke-width': 4
    }));
    svg.append(node('text', {
      x: X(i),
      y: 20,
      'text-anchor': 'middle',
      fill: '#222933',
      'font-size': 14,
      'font-family': 'DM'
    }, mean.toFixed(2)));
    Object.entries(evidence.replay.episodes).forEach(([ep, models], k) => {
      const v = models[m].active_mean_px;
      const point = node('circle', {
        cx: X(i) + (k - 2) * 10,
        cy: Y(v),
        r: 4.6,
        fill: colors[m],
        opacity: .82,
        tabindex: 0,
        class: 'plot-point',
        role: 'img',
        'aria-label': `${m}, episode ${ep.split('__')[1]}, ${v.toFixed(2)} pixels`
      });
      const read = () => {
        $('#point-detail').textContent = `${m} · episode ${ep.split('__')[1]} · ${v.toFixed(2)} px`;
      };
      point.addEventListener('mouseenter', read);
      point.addEventListener('focus', read);
      point.addEventListener('click', read);
      svg.append(point);
    });
  });
}
const taskData = {
  peel: {
    models: ['Real', 'Bond-F', 'Bond-R'],
    case: 34,
    start: 0,
    duration: 4.6,
    ends: [3.1364667, 4.6, 4.6],
    outcomes: [
      ['Real failure', 'neutral'],
      ['× False positive', 'error'],
      ['✓ Correct failure', 'success']
    ],
    count: 100,
    title: 'Peel: predicting detachment outcomes',
    copy: 'Here, Bond-F predicts success while Bond-R correctly predicts the real failure.',
    caveat: 'On 100 paired Peel cases, balanced accuracy improves from 50.0% for Coh-F to 72.9% for Coh-R and 76.3% for Bond-R.'
  },
  stick: {
    models: ['Real', 'Coh-R', 'Bond-R'],
    case: 19,
    start: 2.5,
    duration: 7.5,
    ends: [7.5, 7.5, 7.5],
    outcomes: [
      ['✓ Real success', 'success'],
      ['✓ Sim success', 'success'],
      ['✓ Sim success', 'success']
    ],
    count: 71,
    title: 'Stick: aggregate rates and paired outcomes',
    copy: 'The roll follows the controlled free end. Aggregate rates are close, but predicting each individual outcome remains difficult.',
    caveat: 'Real success is 70.4%; Coh-R and Bond-R achieve 67.6% and 69.0%. Their case-level agreement is 57.7% and 59.2%, respectively.'
  },
  wrap: {
    models: ['Real', 'Coh-F', 'Bond-R'],
    case: 49,
    start: 2,
    duration: 8.3666666667,
    ends: [8.36035, 8, 8],
    outcomes: [
      ['✓ Final: 2 turns', 'success'],
      ['◐ Final: 1 turn', 'partial'],
      ['✓ Final: 2 turns', 'success']
    ],
    count: 55,
    title: 'Wrap: attachment and progressive dispensing',
    copy: 'Bond-R matches two real turns in this example. The task combines attachment to a target with release from the roll.',
    caveat: 'This example is not the aggregate ranking: Coh-R has higher overall outcome agreement than Bond-R at both the one-turn and two-turn criteria.'
  }
};
let activeTask = 'wrap',
  policyGroup;

function drawPolicyTable(task) {
  const rows = evidence.closed_loop.filter(r => r.task.toLowerCase() === task);
  $('#policy-table').innerHTML = '<table><thead><tr><th>Model / target</th><th>Real success</th><th>Sim success</th><th>Agreement</th><th>Balanced accuracy</th></tr></thead><tbody>' + rows.map(r => `<tr><td>${r.model}${r.wrap_turns?' / '+r.wrap_turns+' turn'+(r.wrap_turns===1?'':'s'):''}</td><td>${(100*r.real_successes/r.n).toFixed(1)}%</td><td>${(100*r.sim_successes/r.n).toFixed(1)}%</td><td>${(100*r.agreement).toFixed(1)}%</td><td>${(100*r.balanced_accuracy).toFixed(1)}%</td></tr>`).join('') + '</tbody></table>';
  $('#policy-caveat').textContent = taskData[task].caveat;
}

function updatePolicy(t) {
  const d = taskData[activeTask];
  $('#policy-time').value = (d.start + Math.min(t, d.duration)).toFixed(2) + ' s';
  $$('.policy-video').forEach((card, i) => {
    const ended = t >= d.ends[i] - .09;
    $('.outcome', card).classList.add('visible');
    $('.outcome', card).textContent = d.outcomes[i][0];
    $('.end-note', card).textContent = ended ? (i === 0 ? 'Recording excerpt ended' : 'Display held · full-trial result') : 'Full-trial outcome';
  });
}

function chooseTask(task) {
  activeTask = task;
  const d = taskData[task];
  if (policyGroup) policyGroup.pause();
  $$('[data-task]').forEach(b => {
    const chosen = b.dataset.task === task;
    b.classList.toggle('is-active', chosen);
    b.setAttribute('aria-selected', chosen);
    b.tabIndex = chosen ? 0 : -1;
  });
  $('#task-panel').setAttribute('aria-labelledby', 'tab-' + task);
  $('#policy-case').textContent = `${task.toUpperCase()} / CASE ${d.case}`;
  $('#policy-conclusion').textContent = d.title;
  $('#policy-explanation').textContent = d.copy;
  $('#policy-count').textContent = d.count + ' paired cases';
  $('#policy-videos').replaceChildren();
  d.models.forEach((model, i) => {
    const key = task + '-' + model.toLowerCase(),
      card = document.createElement('div');
    card.className = 'policy-video';
    card.innerHTML = `<div class="model-title">${model}<small>${i===0?'REAL ROBOT':'SIMULATION'}</small></div><div class="policy-media"><video muted playsinline preload="none" poster="assets/media/${key}.webp" src="assets/media/${key}.mp4" aria-label="${task} ${model} case ${d.case}"></video><button class="policy-expand" aria-label="Enlarge ${model} ${task} video">↗</button></div><div class="outcome ${d.outcomes[i][1]}">${d.outcomes[i][0]}</div><p class="end-note"></p>`;
    $('#policy-videos').append(card);
    const v = $('video', card);
    $('.policy-expand', card).addEventListener('click', () => openMedia(key, `${task[0].toUpperCase()+task.slice(1)} · ${model} · case ${d.case}`, `${i===0?'Real robot':'Original simulation'} excerpt from the current film. Source ${d.start.toFixed(2)}–${(d.start+d.ends[i]).toFixed(2)} s. Outcome labels in the comparison refer to the complete trial.`, v.currentTime));
    v.muted = true;
    lazy.observe(v);
  });
  const videos = $$('#policy-videos video');
  if (policyGroup) policyGroup.replace(videos, d.duration + 1.8);
  else policyGroup = new SynchronizedPlayer({
    videos,
    button: $('#policy-play'),
    range: $('#policy-range'),
    duration: d.duration + 1.8,
    onUpdate: updatePolicy,
    region: $('#task-panel')
  });
  $('#policy-range').max = d.duration + 1.8;
  drawPolicyTable(task);
  updatePolicy(0);
}

async function initializeEvidence() {
  try {
    const response = await fetch('assets/data/evidence.json');
    if (!response.ok) throw Error('Could not load research data.');
    evidence = await response.json();
    const motionGroup = new SynchronizedPlayer({
      videos: [$('#motion-video')],
      button: $('#motion-play'),
      range: $('#motion-range'),
      duration: 8.5,
      rate: .5,
      onUpdate: t => drawMotion(Math.min($('#motion-video').currentTime, 8.5 - 1 / 30)),
      region: $('#motion-lab')
    });
    $('#motion-video').addEventListener('seeked', () => drawMotion(Math.min($('#motion-video').currentTime, 8.5 - 1 / 30)));
    $('#motion-rate').addEventListener('click', () => {
      const rate = motionGroup.rate === .5 ? 1 : .5;
      motionGroup.setRate(rate);
      $('#motion-rate').textContent = rate + '×';
    });
    drawCosts();
    $$('[data-turns]').forEach(b => b.addEventListener('click', () => {
      turns = +b.dataset.turns;
      setPressed($$('[data-turns]'), b);
      drawCosts();
    }));
    $$('[data-family]').forEach(b => b.addEventListener('click', () => {
      family = b.dataset.family;
      setPressed($$('[data-family]'), b);
      drawCosts();
    }));
    const replayGroup = new SynchronizedPlayer({
      videos: [$('#replay-real'), $('#replay-sim')],
      button: $('#replay-play'),
      range: $('#replay-range'),
      duration: 7,
      onUpdate: t => {
        $('#replay-time').value = (t + 3).toFixed(2) + ' s';
      },
      region: $('.replay-lab')
    });
    let replayRequest = 0;
    $$('[data-replay]').forEach(b => b.addEventListener('click', async () => {
      const request = ++replayRequest;
      const wasPlaying = replayGroup.playing,
        t = replayGroup.time;
      replayGroup.pause();
      const m = b.dataset.replay;
      setPressed($$('[data-replay]'), b);
      $('#replay-model-label').textContent = m;
      const v = $('#replay-sim');
      v.src = `assets/media/replay-${m.toLowerCase()}.mp4`;
      v.load();
      v.poster = `assets/media/replay-${m.toLowerCase()}.webp`;
      try {
        await ready(v);
        if (request !== replayRequest) return;
        await replayGroup.seek(t);
        if (request !== replayRequest) return;
        if (wasPlaying) replayGroup.play();
      } catch (e) {
        toast(e.message);
      }
    }));
    $('#wipe-range').addEventListener('input', () => $('#replay-stage').style.setProperty('--wipe', $('#wipe-range').value + '%'));
    $('#replay-layout').addEventListener('click', () => {
      const split = $('#replay-stage').classList.toggle('split');
      $('#replay-layout').setAttribute('aria-pressed', split);
      $('#replay-layout').innerHTML = split ? 'Wipe view <span aria-hidden="true">↔</span>' : 'Split view <span aria-hidden="true">▥</span>';
    });
    drawReplayPlot();
    let plotResizeFrame;
    const plotResize = new ResizeObserver(() => {
      cancelAnimationFrame(plotResizeFrame);
      plotResizeFrame = requestAnimationFrame(() => {
        drawMotion(Math.min($('#motion-video').currentTime, 8.5 - 1 / 30));
        drawReplayPlot();
      });
    });
    plotResize.observe($('#motion-lab'));
    plotResize.observe($('.replay-chart-panel'));
    chooseTask('wrap');
    $$('[data-task]').forEach(b => b.addEventListener('click', () => chooseTask(b.dataset.task)));
    $('.task-tabs').addEventListener('keydown', e => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
      e.preventDefault();
      const buttons = $$('[data-task]'),
        current = buttons.findIndex(b => b.getAttribute('aria-selected') === 'true');
      const next = e.key === 'Home' ? 0 : e.key === 'End' ? 2 : (current + (e.key === 'ArrowRight' ? 1 : 2)) % 3;
      chooseTask(buttons[next].dataset.task);
      buttons[next].focus();
    });
    // Pause expensive media outside the viewport; no automatic real-video playback.
    const offscreen = new IntersectionObserver(entries => {
      for (const e of entries)
        if (!e.isIntersecting) {
          const group = groups.find(g => g.region === e.target);
          group?.pause();
        }
    }, {
      threshold: .01
    });
    groups.forEach(g => offscreen.observe(g.region));
    document.documentElement.dataset.ready = 'true';
  } catch (error) {
    console.error(error);
    $$('video').forEach(v => { v.controls = true; });
    toast('Interactive data could not load. The native video controls, success plots and citation remain available.');
    document.documentElement.dataset.ready = 'error';
  }
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    hero.pause();
    groups.forEach(g => g.pause());
    stopRoll();
  }
});
// Native film playback also respects the one-demonstration-at-a-time rule.
hero.addEventListener('play', () => {
  groups.forEach(g => g.pause());
  stopRoll();
});
$('#copy-bibtex').addEventListener('click', async () => {
  const text = $('#bibtex-code').textContent + '\n';
  try {
    await navigator.clipboard.writeText(text);
    $('#copy-status').textContent = 'Citation copied.';
    $('#copy-bibtex').textContent = 'Copied';
    setTimeout(() => { $('#copy-bibtex').textContent = 'Copy citation'; }, 2200);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents($('#bibtex-code'));
    selection.removeAllRanges();
    selection.addRange(range);
    $('#bibtex-code').parentElement.focus();
    $('#copy-status').textContent = 'Copy is unavailable in this browser. The citation is selected; copy it or use Download .bib.';
  }
});
setMotion(reduced);
initializeEvidence();
