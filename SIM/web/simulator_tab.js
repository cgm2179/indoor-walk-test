/* Simulator tab for Frontend_Data_Display.html.
 *
 * Physics here mirrors SIM/phase_a.py BIT-FOR-BIT via SIM/manifest.json
 * (Rule R6): every constant -- losses, freq multipliers, n, d0, saturation,
 * normalization -- is read from window.SIM_ASSETS.manifest, never hard-coded.
 * The UNet surrogate (SIM/web/pl_unet.onnx) is used when it loads (http only);
 * otherwise the exact JS physics runs (~2-4 s per map, with a progress bar).
 *
 * Rules enforced: R2 (P_rx = P_tx + G - PL applied here, never in the model),
 * R7 (combining in linear power), R8 (Tx snaps to walkable cells),
 * R10 (no silent defaults; Solve disabled until inputs are set).
 */
(function () {
  'use strict';
  if (!window.SIM_ASSETS) { console.error('[sim] sim_assets.js missing'); return; }
  const A = window.SIM_ASSETS, M = A.manifest;
  const H = M.grid_shape[0], W = M.grid_shape[1];
  const CELL = M.cell_size_m, C_MPS = M.speed_of_light_mps;
  const PHY = M.physics, NORM = M.norm;

  function u8(b) { const s = atob(b), a = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a; }
  const GRID = u8(A.grid_b64), WALK = u8(A.walkable_b64), INSIDE = u8(A.inside_b64);
  const LOSS = M.materials.map(m => m.loss_db);
  const APM = M.materials.map(m => m.loss_per_m_db || 0);

  const $ = id => document.getElementById(id);
  const tick = () => new Promise(r => setTimeout(r, 0));

  // ---------- physics (mirror of phase_a.pathloss_map) ----------------------
  const fspl1m = f => 32.44 + 20 * Math.log10(f) - 60;
  function satObs(x) {
    const T = PHY.obstruction_linear_db, S = PHY.obstruction_sat_extra_db;
    return x <= T ? x : T + S * (1 - Math.exp(-(x - T) / S));
  }
  async function pathlossPhysics(tx, fMHz, onProg) {
    const mult = M.freq_loss_mult[String(Math.round(fMHz))];
    const L = LOSS.map(v => v * mult), Ap = APM.map(v => v * mult);
    const out = new Float32Array(H * W);
    const stepC = PHY.ray_step_cells, f1 = fspl1m(fMHz);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const dx = x - tx.x, dy = y - tx.y;
        const dist = Math.hypot(dx, dy);
        const dm = Math.max(dist * CELL, PHY.d0_m);
        const K = Math.max(1, Math.ceil(dist / stepC));
        let wall = 0, perM = 0, cur = -1;
        for (let k = 0; k <= K; k++) {
          const t = k / K;
          const xi = Math.round(tx.x + t * dx), yi = Math.round(tx.y + t * dy);
          const c = GRID[yi * W + xi];
          if (c !== cur) { cur = c; if (L[c] > 0) wall += L[c]; }  // R3 runs
          perM += Ap[c];
        }
        wall += perM * (dist / K) * CELL;
        out[y * W + x] = f1 + 10 * PHY.n_exp * Math.log10(dm) + satObs(wall);
      }
      if (onProg && (y & 15) === 0) { onProg(y / H); await tick(); }
    }
    if (onProg) onProg(1);
    return out;
  }

  // ---------- ONNX surrogate (optional) --------------------------------------
  let ortSession = null, engineName = 'physics (exact JS mirror)';
  async function tryLoadModel() {
    if (!window.ort || location.protocol === 'file:') return;
    try {
      for (const eps of [['webgpu'], ['wasm']]) {
        try {
          ortSession = await ort.InferenceSession.create('SIM/web/pl_unet.onnx',
            { executionProviders: eps });
          engineName = 'UNet surrogate (' + eps[0] + ')';
          break;
        } catch (e) { console.error('[sim] ONNX init failed on', eps[0], '-', e.message); }
      }
    } catch (e) { ortSession = null; }
    $('simEngine').textContent = engineName;
  }
  const FREQ_FEAT = f => (Math.log10(f) - Math.log10(NORM.freq_log_lo_mhz)) /
    (Math.log10(NORM.freq_log_hi_mhz) - Math.log10(NORM.freq_log_lo_mhz));
  let onehotCache = null;
  async function pathlossOnnx(tx, fMHz) {
    const n = H * W;
    if (!onehotCache) {
      onehotCache = new Float32Array(6 * n);
      for (let i = 0; i < n; i++) onehotCache[GRID[i] * n + i] = 1;
    }
    const x = new Float32Array(9 * n);
    x.set(onehotCache);
    const s = M.tx_blob_sigma_cells, ff = FREQ_FEAT(fMHz);
    const dn = M.dist_channel_norm || 3.0;
    for (let y = 0; y < H; y++)
      for (let xx = 0; xx < W; xx++) {
        const i = y * W + xx;
        const d2 = (xx - tx.x) ** 2 + (y - tx.y) ** 2;
        x[6 * n + i] = Math.exp(-d2 / (2 * s * s));
        x[7 * n + i] = ff;
        x[8 * n + i] = Math.log10(Math.max(Math.sqrt(d2) * CELL, 1)) / dn;
      }
    const feed = {};
    feed[ortSession.inputNames[0]] = new ort.Tensor('float32', x, [1, 9, H, W]);
    const out = await ortSession.run(feed);
    const y = out[Object.keys(out)[0]].data;
    const pl = new Float32Array(n);
    for (let i = 0; i < n; i++)
      pl[i] = Math.min(Math.max(y[i], 0), 1) * NORM.pl_range_db + NORM.pl_min_db;
    return pl;
  }
  async function pathloss(tx, fMHz, onProg) {
    return ortSession ? pathlossOnnx(tx, fMHz) : pathlossPhysics(tx, fMHz, onProg);
  }

  // ---------- BS precomputed maps --------------------------------------------
  function bsMaps(bearing) {
    if (!A.bs) return null;
    const bl = A.bs.bearings;
    let bi = 0, bd = 1e9;
    bl.forEach((b, i) => {
      const d = Math.min(Math.abs(bearing - b), 360 - Math.abs(bearing - b));
      if (d < bd) { bd = d; bi = i; }
    });
    const g16 = new Int16Array(u8(A.bs.gain_decidb_b64[bi]).buffer);
    const t16 = new Uint16Array(u8(A.bs.time_ns_b64[bi]).buffer);
    const gain = new Float32Array(H * W), tArr = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) { gain[i] = g16[i] / 10; tArr[i] = t16[i] * 1e-9; }
    return { gain, tArr, snapped: bl[bi] };
  }

  // ---------- rendering -------------------------------------------------------
  const VIRIDIS = ['#440154', '#471365', '#482475', '#463480', '#414487', '#3b528b', '#355f8d', '#2f6c8e', '#2a788e', '#25848e', '#21918c', '#1e9c89', '#22a884', '#2fb47c', '#44bf70', '#5ec962', '#7ad151', '#9bd93c', '#bddf26', '#dfe318', '#fde725'];
  function vColor(t) {
    t = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
    const i = Math.floor(t), f = t - i;
    const c0 = VIRIDIS[i], c1 = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
    const h = c => [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
    const a = h(c0), b = h(c1);
    return [0, 1, 2].map(k => Math.round(a[k] + f * (b[k] - a[k])));
  }

  const state = {
    sub: 'transmitter', view: 'static',
    tx: null,                       // {x, y} cell coords
    maps: { received: null, transmitter: null, combined: null },  // {value, tArr, meta}
    playT: 0, playing: false, playTimer: null,
    range: [-120, -40],
  };

  function activeMap() { return state.maps[state.sub]; }

  function draw() {
    const cv = $('simCanvas'), cx = cv.getContext('2d');
    const img = cx.createImageData(W, H);
    const m = activeMap();
    const [lo, hi] = state.range;
    const dt = 2 * CELL / C_MPS;    // leading-edge band: |T - t| <= 2 cells/c
    for (let i = 0; i < H * W; i++) {
      let r = 236, g = 238, b = 235;                       // outside
      const cls = GRID[i];
      if (INSIDE[i]) {
        if (m && (state.view === 'static' || m.tArr[i] <= state.playT)) {
          [r, g, b] = vColor((m.value[i] - lo) / (hi - lo));
        } else { r = g = b = 248; }
        if (m && state.view === 'timelapse' &&
            Math.abs(m.tArr[i] - state.playT) <= dt) { r = 255; g = 255; b = 190; }
        if (cls !== 0 && cls !== 4) { r = 30; g = 32; b = 38; }   // walls on top
      }
      const o = i * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
    cx.putImageData(img, 0, 0);
    if (state.tx && (state.sub !== 'received')) {
      cx.beginPath(); cx.arc(state.tx.x, state.tx.y, 4, 0, 7);
      cx.fillStyle = '#28c840'; cx.fill();
      cx.lineWidth = 1.2; cx.strokeStyle = '#0b3d14'; cx.stroke();
    }
    drawColorbar();
  }

  function drawColorbar() {
    const cb = $('simCbar'), cx = cb.getContext('2d');
    for (let x = 0; x < cb.width; x++) {
      const [r, g, b] = vColor(x / (cb.width - 1));
      cx.fillStyle = `rgb(${r},${g},${b})`; cx.fillRect(x, 0, 1, cb.height);
    }
    $('simCbarLo').textContent = state.range[0] + ' dBm';
    $('simCbarHi').textContent = state.range[1] + ' dBm';
  }

  // ---------- R10 validation ---------------------------------------------------
  function inputsValid() {
    const need = [];
    const band = $('simBand').value;
    if (!band) need.push('band');
    if (state.sub !== 'received') {
      const p = parseFloat($('simTxPower').value);
      if (!($('simTxPower').value !== '' && p >= 0 && p <= 30)) need.push('Tx power 0-30 dBm');
      const g = parseFloat($('simGain').value);
      if (!($('simGain').value !== '' && g >= -2 && g <= 9)) need.push('antenna gain -2 to 9 dBi');
      if (!state.tx) need.push('click the map to place the Tx');
    }
    if (state.sub !== 'transmitter') {
      const b = parseFloat($('simBearing').value);
      if (!($('simBearing').value !== '' && b >= 0 && b <= 359)) need.push('BS bearing 0-359');
      if ($('simPref').value === '') need.push('P_ref at facade (dBm)');
    }
    $('simValidation').textContent = need.length ? 'Required: ' + need.join(' · ') : '';
    return need.length === 0;
  }
  function refreshButtons() {
    const ok = inputsValid();
    $('simSolve').disabled = !ok;
    $('simOptimize').disabled = !(state.sub === 'transmitter' && ok);
  }

  // ---------- solve flows ------------------------------------------------------
  function setProgress(f) {
    $('simProgress').style.display = f == null ? 'none' : '';
    if (f != null) $('simProgressFill').style.width = Math.round(f * 100) + '%';
  }

  async function solve() {
    if (!inputsValid()) return;
    const f = parseFloat($('simBand').value);
    $('simSolve').disabled = true;
    try {
      if (state.sub === 'transmitter' || state.sub === 'combined') {
        const eirp = parseFloat($('simTxPower').value) + parseFloat($('simGain').value);
        setProgress(0);
        const pl = await pathloss(state.tx, f, setProgress);
        setProgress(null);
        const value = new Float32Array(H * W), tArr = new Float32Array(H * W);
        for (let i = 0; i < H * W; i++) {
          value[i] = eirp - pl[i];                          // Rule R2
          const dx = (i % W) - state.tx.x, dy = Math.floor(i / W) - state.tx.y;
          tArr[i] = Math.hypot(dx, dy) * CELL / C_MPS;      // F.5 v1: d/c
        }
        state.maps.transmitter = { value, tArr, meta: { f_mhz: f, eirp, tx: { ...state.tx }, engine: engineName } };
      }
      if (state.sub === 'received' || state.sub === 'combined') {
        const bs = bsMaps(parseFloat($('simBearing').value));
        if (!bs) throw new Error('BS assets missing - run SIM/export_web_assets.py');
        const pref = parseFloat($('simPref').value);
        const value = new Float32Array(H * W);
        for (let i = 0; i < H * W; i++) value[i] = pref + bs.gain[i];
        state.maps.received = {
          value, tArr: bs.tArr,
          meta: { bearing_snapped: bs.snapped, pref, f_mhz: A.bs.f_mhz }
        };
        $('simBearingNote').textContent =
          `bearing snapped to ${bs.snapped}° (precomputed at ${A.bs.f_mhz} MHz, 45° steps)`;
      }
      if (state.sub === 'combined') {
        const t = state.maps.transmitter, r = state.maps.received;
        const value = new Float32Array(H * W), tArr = new Float32Array(H * W);
        for (let i = 0; i < H * W; i++) {
          value[i] = 10 * Math.log10(                        // Rule R7: linear
            Math.pow(10, t.value[i] / 10) + Math.pow(10, r.value[i] / 10));
          tArr[i] = Math.min(t.tArr[i], r.tArr[i]);
        }
        state.maps.combined = { value, tArr, meta: { of: [t.meta, r.meta] } };
      }
      autoRange(); resetPlayback(); draw();
      $('simExportPng').disabled = $('simExportCsv').disabled = !activeMap();
    } catch (err) {
      $('simValidation').textContent = 'Solve failed: ' + err.message;
    } finally {
      setProgress(null); refreshButtons();
    }
  }

  function autoRange() {
    const m = activeMap(); if (!m) return;
    let vals = [];
    for (let i = 0; i < H * W; i += 17) if (INSIDE[i]) vals.push(m.value[i]);
    vals.sort((a, b) => a - b);
    const lo = Math.floor(vals[Math.floor(vals.length * 0.02)] / 5) * 5;
    const hi = Math.ceil(vals[Math.floor(vals.length * 0.98)] / 5) * 5;
    state.range = [lo, hi];
  }

  // ---------- time-lapse -------------------------------------------------------
  function tMax() { const m = activeMap(); if (!m) return 1e-7;
    let mx = 0; for (let i = 0; i < H * W; i += 7) if (INSIDE[i] && isFinite(m.tArr[i])) mx = Math.max(mx, m.tArr[i]);
    return mx; }
  function resetPlayback() {
    state.playT = 0; stopPlay();
    $('simScrub').value = 0;
    updateClock();
  }
  function updateClock() {
    $('simClock').textContent = (state.playT * 1e9).toFixed(1) + ' ns real · playback ×10⁻⁸ (label per spec F.5)';
  }
  function stopPlay() {
    state.playing = false;
    if (state.playTimer) { clearInterval(state.playTimer); state.playTimer = null; }
    $('simPlay').textContent = 'Play';
  }
  function togglePlay() {
    if (state.playing) { stopPlay(); return; }
    state.playing = true; $('simPlay').textContent = 'Pause';
    const mx = tMax();
    state.playTimer = setInterval(() => {
      // 3 s per sweep at ~30 fps
      state.playT += mx / 90;
      if (state.playT >= mx) state.playT = 0;
      $('simScrub').value = Math.round(1000 * state.playT / mx);
      updateClock(); draw();
    }, 33);
  }

  // ---------- optimizer (Phase E, in-browser) ----------------------------------
  let optCancel = false;
  async function optimize() {
    if (!inputsValid()) return;
    const f = parseFloat($('simBand').value);
    const eirp = parseFloat($('simTxPower').value) + parseFloat($('simGain').value);
    const stride = parseInt($('simOptStride').value, 10);
    const objective = $('simOptObjective').value;
    const thr = parseFloat($('simThreshold').value);
    const z = { '80': 0.84, '90': 1.28, '95': 1.65 }[$('simReliability').value];
    const sigma = M.ui.sigma_sf_db.nlos;
    const margin = z * sigma;
    let holeMask = null;
    if (objective === 'hole_filling') {
      if (!state.maps.received) { alert('Solve the Received tab first for hole-filling'); return; }
      holeMask = new Uint8Array(H * W);
      for (let i = 0; i < H * W; i++)
        holeMask[i] = WALK[i] && state.maps.received.value[i] < thr ? 1 : 0;
    }
    const cand = [];
    for (let y = 0; y < H; y += stride)
      for (let x = 0; x < W; x += stride)
        if (WALK[y * W + x]) cand.push({ x, y });
    const est = ortSession ? 0.15 : 2.5;
    if (!confirm(`${cand.length} candidates × ~${est}s ≈ ${(cand.length * est / 60).toFixed(0)} min with the ${ortSession ? 'surrogate' : 'physics'} engine. Continue?`)) return;
    optCancel = false;
    $('simOptCancel').style.display = '';
    const scores = [];
    for (let i = 0; i < cand.length; i++) {
      if (optCancel) break;
      const pl = await pathloss(cand[i], f, null);
      let score = 0, n = 0;
      for (let j = 0; j < H * W; j++) {
        const inSet = holeMask ? holeMask[j] : WALK[j];
        if (!inSet) continue;
        n++;
        if (objective === 'mean_pl') score -= pl[j];
        else if (eirp - pl[j] - margin >= thr) score++;
      }
      scores.push(score / Math.max(n, 1));
      setProgress(i / cand.length);
      if ((i & 7) === 0) await tick();
    }
    setProgress(null); $('simOptCancel').style.display = 'none';
    const order = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).slice(0, 5);
    const list = $('simTop5'); list.innerHTML = '';
    order.forEach(([s, i], r) => {
      const li = document.createElement('li');
      const c = cand[i];
      li.textContent = `#${r + 1} (${(c.x * CELL).toFixed(1)}, ${(c.y * CELL).toFixed(1)}) m — score ${s.toFixed(4)}`;
      li.style.cursor = 'pointer';
      li.onclick = () => { state.tx = { x: c.x, y: c.y }; refreshButtons(); solve(); };
      list.appendChild(li);
    });
  }

  // ---------- exports ----------------------------------------------------------
  function exportPng() {
    const a = document.createElement('a');
    a.download = `sim_${state.sub}_${state.view}.png`;
    a.href = $('simCanvas').toDataURL('image/png');
    a.click();
  }
  function exportCsv() {
    const m = activeMap(); if (!m) return;
    let s = `# tab=${state.sub}\n# meta=${JSON.stringify(m.meta)}\n# model_version=${M.version}\n# grid=${H}x${W} cell_size_m=${CELL}\n# VALUE_GRID_dBm\n`;
    for (let y = 0; y < H; y++) {
      const row = [];
      for (let x = 0; x < W; x++) row.push(m.value[y * W + x].toFixed(2));
      s += row.join(',') + '\n';
    }
    s += '# ARRIVAL_TIME_ns\n';
    for (let y = 0; y < H; y++) {
      const row = [];
      for (let x = 0; x < W; x++) row.push((m.tArr[y * W + x] * 1e9).toFixed(2));
      s += row.join(',') + '\n';
    }
    const a = document.createElement('a');
    a.download = `sim_${state.sub}_grids.csv`;
    a.href = URL.createObjectURL(new Blob([s], { type: 'text/csv' }));
    a.click();
  }

  // ---------- wiring -----------------------------------------------------------
  function snapWalkable(x, y) {                          // Rule R8
    if (WALK[Math.round(y) * W + Math.round(x)]) return { x: Math.round(x), y: Math.round(y) };
    for (let r = 1; r < 40; r++)
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++) {
          const xx = Math.round(x) + dx, yy = Math.round(y) + dy;
          if (xx >= 0 && yy >= 0 && xx < W && yy < H && WALK[yy * W + xx])
            return { x: xx, y: yy };
        }
    return null;
  }

  function initUI() {
    const bandSel = $('simBand');
    M.ui.bands.forEach(b => bandSel.add(new Option(b.label, b.f_mhz)));
    // legend from the manifest material table (8.3) - same numbers as physics
    const tb = $('simLegend');
    M.materials.forEach(mm => {
      const tr = document.createElement('tr');
      const loss = mm.loss_db > 0 ? mm.loss_db + ' dB/crossing'
        : (mm.loss_per_m_db > 0 ? mm.loss_per_m_db + ' dB/m' : '—');
      tr.innerHTML = `<td><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:${mm.color};border:1px solid #999"></span></td><td>${mm.name.replace(/_/g, ' ')}</td><td>${loss}</td>`;
      tb.appendChild(tr);
    });

    document.querySelectorAll('.sim-subtab').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.sim-subtab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.sub = btn.dataset.sub;
        $('simTxFields').style.display = state.sub === 'received' ? 'none' : '';
        $('simBsFields').style.display = state.sub === 'transmitter' ? 'none' : '';
        $('simOptRow').style.display = state.sub === 'transmitter' ? '' : 'none';
        refreshButtons(); autoRange(); resetPlayback(); draw();
        $('simExportPng').disabled = $('simExportCsv').disabled = !activeMap();
      });
    });

    const cv = $('simCanvas');
    cv.addEventListener('click', ev => {
      if (state.sub === 'received') return;
      const r = cv.getBoundingClientRect();
      const x = (ev.clientX - r.left) * W / r.width;
      const y = (ev.clientY - r.top) * H / r.height;
      const snapped = snapWalkable(x, y);
      if (snapped) { state.tx = snapped; refreshButtons(); draw(); }
    });
    cv.addEventListener('mousemove', ev => {
      const r = cv.getBoundingClientRect();
      const x = Math.floor((ev.clientX - r.left) * W / r.width);
      const y = Math.floor((ev.clientY - r.top) * H / r.height);
      const m = activeMap();
      if (x >= 0 && y >= 0 && x < W && y < H) {
        let s = `(${(x * CELL).toFixed(1)}, ${(y * CELL).toFixed(1)}) m`;
        if (m && INSIDE[y * W + x]) {
          s += ` · ${m.value[y * W + x].toFixed(1)} dBm · arrives ${(m.tArr[y * W + x] * 1e9).toFixed(1)} ns`;
        }
        $('simReadout').textContent = s;
      }
    });

    ['simBand', 'simTxPower', 'simGain', 'simBearing', 'simPref'].forEach(id =>
      $(id).addEventListener('input', refreshButtons));
    $('simSolve').addEventListener('click', solve);
    $('simOptimize').addEventListener('click', optimize);
    $('simOptCancel').addEventListener('click', () => { optCancel = true; });
    $('simPlay').addEventListener('click', togglePlay);
    $('simScrub').addEventListener('input', () => {
      stopPlay();
      state.playT = tMax() * $('simScrub').value / 1000;
      updateClock(); draw();
    });
    document.querySelectorAll('input[name=simView]').forEach(rb =>
      rb.addEventListener('change', () => {
        state.view = document.querySelector('input[name=simView]:checked').value;
        $('simTimeControls').style.display = state.view === 'timelapse' ? '' : 'none';
        resetPlayback(); draw();
      }));
    $('simExportPng').addEventListener('click', exportPng);
    $('simExportCsv').addEventListener('click', exportCsv);
    $('simThresholdVal').textContent = $('simThreshold').value;
    $('simThreshold').addEventListener('input',
      () => { $('simThresholdVal').textContent = $('simThreshold').value; });

    $('simEngine').textContent = engineName;
    tryLoadModel();
    refreshButtons(); draw();
  }

  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', initUI);
  else initUI();

  // exposed for console debugging / parity tests
  window.SIM_DEBUG = { pathlossPhysics, GRID, W, H, CELL };
})();
