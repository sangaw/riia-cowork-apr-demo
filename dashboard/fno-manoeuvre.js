// ── Position Manoeuvre — fno-manoeuvre.js ─────────────────────────────────────
// Three-section redesign:
//   Top    : 6 KPI tiles (3 expiry months × NIFTY + BANKNIFTY)
//   Middle : 5 named groups with tab navigation + P&L sparklines
//   Bottom : 2-column position pool (filtered to selected month)
//
// Depends on globals from fno.html main script:
//   positions[], scenarioLevels{}, marketData{},
//   currentUnd, fmtPnl(), pnlClass()

// ── Constants ─────────────────────────────────────────────────────────────────
const MAN_STORE_PFX = 'rita_man_v6_';   // + month key e.g. rita_man_v6_APR
const MAN_LOT = { NIFTY: 65, BANKNIFTY: 30 };

const MAN_GROUPS = [
  { id: 'anchor',      name: 'Monthly Anchor',   defaultView: 'bull', icon: '⚓' },
  { id: 'directional', name: 'Directional',       defaultView: 'bull', icon: '◈' },
  { id: 'futures',     name: 'Futures',           defaultView: 'bull', icon: '⇄' },
  { id: 'spread',      name: 'Spread',            defaultView: 'bull', icon: '≋' },
  { id: 'hedge',       name: 'Hedge',             defaultView: 'bear', icon: '⛨' },
];

// ── State ─────────────────────────────────────────────────────────────────────
let manSelectedMonth = null;   // e.g. 'APR'
let manSelectedUnd   = 'NIFTY'; // 'NIFTY' | 'BANKNIFTY' — one tile selected at a time
let manGroupState    = {};     // { view: 'bull'|'bear', name: string } keyed by group id — per month
let manAssign        = {};     // lotKey → group id — per month
let manActiveTab     = 'anchor';
let manPnlHistory    = [];     // [{date, groups:[{id,pnl_now,sl_pnl,target_pnl}]}]
let _manDrag         = null;
let _manSparkCharts  = {};     // Chart.js instances keyed by group id

// ── NSE Expiry Logic ──────────────────────────────────────────────────────────
// NSE monthly expiry = last Thursday of the month.
// If today > last Thursday of current calendar month → window slides forward.

const MONTH_NAMES = ['JAN','FEB','MAR','APR','MAY','JUN',
                     'JUL','AUG','SEP','OCT','NOV','DEC'];

function lastThursdayOf(year, month0) {
  // month0: 0-based JS month index
  const d = new Date(year, month0 + 1, 0); // last day of month
  while (d.getDay() !== 4) d.setDate(d.getDate() - 1); // walk back to Thursday
  return d;
}

function manActiveMonths() {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  let year  = today.getFullYear();
  let month = today.getMonth(); // 0-based

  // If today is past (or on) the last Thursday → this month has expired, move forward
  const expiry = lastThursdayOf(year, month);
  if (today > expiry) {
    month += 1;
    if (month > 11) { month = 0; year += 1; }
  }

  const result = [];
  for (let i = 0; i < 3; i++) {
    let m = month + i;
    let y = year;
    if (m > 11) { m -= 12; y += 1; }
    result.push(MONTH_NAMES[m]);
  }
  return result; // e.g. ['APR','MAY','JUN']
}

// ── DTE + distance helpers ────────────────────────────────────────────────────
function manDte(month) {
  // Days to expiry for a given 3-letter NSE month code.
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const mIdx = MONTH_NAMES.indexOf((month || '').toUpperCase());
  if (mIdx === -1) return null;
  let year = today.getFullYear();
  // If the month index is before the current calendar month it wraps to next year
  if (mIdx < today.getMonth()) year += 1;
  const expiry = lastThursdayOf(year, mIdx);
  return Math.max(0, Math.round((expiry - today) / 86400000));
}

function manPctFromLevel(spot, level) {
  // Signed % distance of spot from level, rounded to 2dp.
  // Positive = spot above level; negative = below.
  if (!spot || !level) return null;
  return Math.round(((spot - level) / level) * 10000) / 100;
}

// ── Per-month Persistence ─────────────────────────────────────────────────────
function manStoreKey(month) { return MAN_STORE_PFX + month; }

function manSave(month) {
  // 1. localStorage (fast, local fallback)
  try {
    localStorage.setItem(manStoreKey(month),
      JSON.stringify({ groupState: manGroupState, assign: manAssign }));
  } catch(e) {}
  // 2. Server-side persistence (fire-and-forget — enables cron + cross-device)
  fetch('/api/v1/portfolio/man-groups', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ month, groupState: manGroupState, assign: manAssign }),
  }).catch(() => {});
}

async function manLoad(month) {
  // Try server first (authoritative, survives browser cache clears)
  try {
    const res = await fetch(`/api/v1/portfolio/man-groups?month=${month}`);
    if (res.ok) {
      const d = await res.json();
      manGroupState = d.groupState || {};
      manAssign     = d.assign     || {};
      // Keep localStorage in sync
      try {
        localStorage.setItem(manStoreKey(month),
          JSON.stringify({ groupState: manGroupState, assign: manAssign }));
      } catch(e) {}
      return;
    }
  } catch(e) {}
  // Fall back to localStorage
  try {
    const s = localStorage.getItem(manStoreKey(month));
    if (s) {
      const d = JSON.parse(s);
      manGroupState = d.groupState || {};
      manAssign     = d.assign     || {};
      return;
    }
  } catch(e) {}
  manGroupState = {};
  manAssign     = {};
}

function manGroupEffective(gid) {
  // Returns {name, view} for a group id, falling back to defaults
  const saved = manGroupState[gid] || {};
  const def   = MAN_GROUPS.find(g => g.id === gid);
  return {
    name: saved.name ?? def.name,
    view: saved.view ?? def.defaultView,
  };
}

// ── Lot Expansion ─────────────────────────────────────────────────────────────
function manLots(p) {
  const lotSz = MAN_LOT[p.und] || 1;
  const nLots = Math.max(1, Math.round(p.qty / lotSz));
  const pnlPerLot = p.pnl / nLots;
  return Array.from({ length: nLots }, (_, i) => ({
    ...p,
    lotKey:  p.instrument + '_L' + (i + 1),
    lotIdx:  i + 1,
    nLots,
    lotSz,
    lotPnl: pnlPerLot,
  }));
}

// ── At-expiry Intrinsic P&L for one lot ───────────────────────────────────────
function manPayoff(lot, price) {
  const sign = lot.side === 'Long' ? 1 : -1;
  let intrinsic;
  if      (lot.type === 'FUT') intrinsic = price;
  else if (lot.type === 'CE')  intrinsic = Math.max(0, price - (lot.strike_val || 0));
  else                         intrinsic = Math.max(0, (lot.strike_val || 0) - price);
  return sign * lot.lotSz * (intrinsic - lot.avg);
}

// ── Section 1: Month Tiles ────────────────────────────────────────────────────
function renderMonthTiles() {
  const months = manActiveMonths();
  const pnlByMonthUnd = (und, exp) =>
    positions.filter(p => p.und === und && p.exp === exp).reduce((s, p) => s + p.pnl, 0);
  const cntByMonthUnd = (und, exp) =>
    positions.filter(p => p.und === und && p.exp === exp).length;

  const tileHtml = (und, exp) => {
    const pnl    = pnlByMonthUnd(und, exp);
    const cnt    = cntByMonthUnd(und, exp);
    const isSelected = exp === manSelectedMonth && und === manSelectedUnd;
    const isEmpty    = cnt === 0;
    const undLabel   = und === 'BANKNIFTY' ? 'BANKNIFTY' : 'NIFTY';
    const undColor   = und === 'NIFTY' ? 'var(--p02)' : 'var(--p04)';
    return `
      <div class="kpi man-month-tile${isSelected ? ' selected' : ''}"
           style="${isEmpty ? 'opacity:.55;' : ''}"
           onclick="manSelectTile('${und}','${exp}')">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
          <span style="font-family:var(--fm);font-size:9px;font-weight:700;color:${undColor};
                       text-transform:uppercase;letter-spacing:.06em;">${undLabel}</span>
          <span class="exp-badge ${exp.toLowerCase()}">${exp}</span>
        </div>
        <div class="kpi-value ${isEmpty ? '' : pnlClass(pnl)}" style="font-size:18px;">
          ${isEmpty ? '—' : fmtPnl(pnl)}
        </div>
        <div class="kpi-sub">${cnt} position${cnt !== 1 ? 's' : ''}${isSelected ? ' · selected' : ''}</div>
      </div>`;
  };

  let html = '<div class="kpi-row man-month-grid">';
  for (const exp of months) {
    html += tileHtml('NIFTY', exp);
    html += tileHtml('BANKNIFTY', exp);
  }
  html += '</div>';

  const el = document.getElementById('man-month-tiles');
  if (el) el.innerHTML = html;
}

async function manSelectTile(und, month) {
  if (manSelectedMonth) manSave(manSelectedMonth);
  manSelectedMonth = month;
  manSelectedUnd   = und;
  await manLoad(month);
  renderMonthTiles();
  renderTabStrip();
  renderActiveGroupCard();
  renderPool();
}

// ── Section 2: Tab Strip + Group Cards ────────────────────────────────────────
function renderTabStrip() {
  const el = document.getElementById('man-tab-strip');
  if (!el) return;
  el.innerHTML = MAN_GROUPS.map(g => {
    const eff = manGroupEffective(g.id);
    const allLots = positions.filter(p => p.exp === manSelectedMonth).flatMap(manLots);
    const count = allLots.filter(lot => manAssign[lot.lotKey] === g.id).length;
    const isActive = g.id === manActiveTab;
    return `<button class="man-tab${isActive ? ' active' : ''}"
              onclick="manSwitchTab('${g.id}')">
              <span class="man-tab-icon">${g.icon}</span>
              ${eff.name}
              ${count ? `<span class="man-tab-badge">${count}</span>` : ''}
            </button>`;
  }).join('');
}

function manSwitchTab(gid) {
  manActiveTab = gid;
  renderTabStrip();
  renderActiveGroupCard();
}

function renderActiveGroupCard() {
  const el = document.getElementById('man-group-panel');
  if (!el) return;
  const g   = MAN_GROUPS.find(x => x.id === manActiveTab);
  if (!g) return;
  const eff = manGroupEffective(g.id);
  el.innerHTML = renderGroupCard(g.id, eff);
  renderSparkline(g.id);
}

// ── Group Card ────────────────────────────────────────────────────────────────
function renderGroupCard(gid, eff) {
  const allLots = positions.filter(p => p.exp === manSelectedMonth).flatMap(manLots);
  const gLots   = allLots.filter(lot => manAssign[lot.lotKey] === gid);
  const isBull  = eff.view === 'bull';

  const rows = gLots.map(lot => {
    const sc_    = (scenarioLevels[lot.und] || {})[eff.view] || {};
    const pnlSL  = sc_.sl     != null ? manPayoff(lot, sc_.sl)     : null;
    const pnlTgt = sc_.target != null ? manPayoff(lot, sc_.target) : null;
    const undClr = lot.und === 'NIFTY' ? 'var(--p02)' : 'var(--p04)';
    const undBg  = lot.und === 'NIFTY' ? 'var(--p02-bg)' : 'var(--p04-bg)';
    const lotBadge = lot.nLots > 1
      ? `<span style="font-family:var(--fm);font-size:9px;font-weight:700;padding:1px 4px;
                      border-radius:2px;background:var(--surface2);border:1px solid var(--border);
                      color:var(--t3);margin-right:3px;">L${lot.lotIdx}</span>`
      : '';
    return `<tr>
      <td style="white-space:nowrap;">
        <span style="font-family:var(--fm);font-size:9px;font-weight:600;padding:1px 4px;
                     border-radius:2px;background:${undBg};color:${undClr};margin-right:3px;">
          ${lot.und === 'BANKNIFTY' ? 'BNK' : lot.und}
        </span>
        ${lotBadge}
        <span style="font-family:var(--fm);font-size:11px;font-weight:500;">${lot.full}</span>
      </td>
      <td><span class="exp-badge ${lot.exp.toLowerCase()}">${lot.exp}</span></td>
      <td><span class="side-badge ${lot.side.toLowerCase()}">${lot.side}</span></td>
      <td class="val">${lot.lotSz}</td>
      <td class="val">${lot.type === 'FUT'
        ? lot.avg.toLocaleString('en-IN', { minimumFractionDigits: 2 })
        : lot.avg.toFixed(2)}</td>
      <td class="${pnlSL  != null && pnlSL  >= 0 ? 'pos' : 'neg'} val">${pnlSL  != null ? fmtPnl(pnlSL)  : '—'}</td>
      <td class="${pnlTgt != null && pnlTgt >= 0 ? 'pos' : 'neg'} val">${pnlTgt != null ? fmtPnl(pnlTgt) : '—'}</td>
      <td class="${pnlClass(lot.lotPnl)} val">${fmtPnl(lot.lotPnl)}</td>
      <td style="text-align:center;padding:3px 6px;">
        <button onclick="manRemove('${lot.lotKey}')"
          style="font-family:var(--fm);font-size:10px;padding:1px 6px;border-radius:3px;
                 border:1px solid var(--border);background:var(--surface);cursor:pointer;color:var(--t3);"
          title="Return to pool">↩</button>
      </td>
    </tr>`;
  }).join('');

  // Column header label helper
  const undsInGroup = [...new Set(gLots.map(l => l.und))];
  function scLabel(key) {
    if (!undsInGroup.length) return '—';
    return undsInGroup.map(u => {
      const v = ((scenarioLevels[u] || {})[eff.view] || {})[key];
      return v != null
        ? (undsInGroup.length > 1
            ? u.replace('BANKNIFTY', 'BNK') + ' ' + v.toLocaleString('en-IN')
            : v.toLocaleString('en-IN'))
        : '—';
    }).join(' / ');
  }

  const totSL  = gLots.reduce((s, lot) => { const v = ((scenarioLevels[lot.und] || {})[eff.view] || {}).sl;     return v != null ? s + manPayoff(lot, v) : s; }, 0);
  const totTgt = gLots.reduce((s, lot) => { const v = ((scenarioLevels[lot.und] || {})[eff.view] || {}).target; return v != null ? s + manPayoff(lot, v) : s; }, 0);
  const totNow = gLots.reduce((s, lot) => s + lot.lotPnl, 0);

  const tableSection = gLots.length ? `
    <div style="overflow-x:auto;max-height:184px;overflow-y:auto;">
      <table style="width:100%;border-collapse:collapse;font-size:11px;">
        <thead><tr>
          <th class="man-th" style="text-align:left;position:sticky;top:0;z-index:1;">Instrument</th>
          <th class="man-th" style="position:sticky;top:0;z-index:1;">Exp</th>
          <th class="man-th" style="position:sticky;top:0;z-index:1;">Side</th>
          <th class="man-th" style="position:sticky;top:0;z-index:1;">Qty</th>
          <th class="man-th" style="position:sticky;top:0;z-index:1;">Entry</th>
          <th class="man-th" style="color:var(--neg);background:var(--neg-bg);border-bottom-color:var(--neg-bd);position:sticky;top:0;z-index:1;">@SL ${scLabel('sl')}</th>
          <th class="man-th" style="color:var(--p01);background:var(--p01-bg);border-bottom-color:var(--p01-bd);position:sticky;top:0;z-index:1;">@Tgt ${scLabel('target')}</th>
          <th class="man-th" style="position:sticky;top:0;z-index:1;">P&amp;L Now</th>
          <th style="background:var(--surface2);border-bottom:1px solid var(--border);width:30px;position:sticky;top:0;z-index:1;"></th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>` : '<div class="man-empty-zone">Drop lots from pool below</div>';

  const g = MAN_GROUPS.find(x => x.id === gid);

  return `
    <div class="man-group-layout"
         ondragover="event.preventDefault();document.getElementById('man-group-drop-${gid}').style.outline='2px dashed var(--p02)'"
         ondragleave="document.getElementById('man-group-drop-${gid}').style.outline=''"
         ondrop="manDropToGroup(event,'${gid}');document.getElementById('man-group-drop-${gid}').style.outline=''">
      <div class="man-group-drop-target" id="man-group-drop-${gid}">
        <!-- Header -->
        <div class="man-group-hdr">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <span style="font-size:16px;">${g.icon}</span>
            <input class="man-name-input" value="${eff.name.replace(/"/g, '&quot;')}"
              onchange="manSaveName('${gid}',this.value)"
              ondragstart="event.stopPropagation()" title="Click to rename">
            <span class="man-view-badge ${isBull ? 'bull' : 'bear'}">${isBull ? 'BULL' : 'BEAR'}</span>
            ${gLots.length ? `<span style="font-family:var(--fm);font-size:10px;color:var(--t3);">${gLots.length} lot${gLots.length !== 1 ? 's' : ''}</span>` : ''}
          </div>
          <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
            ${gLots.length ? `
              <span style="font-family:var(--fm);font-size:10px;color:var(--t3);">@SL&nbsp;<span class="${totSL >= 0 ? 'pos' : 'neg'}" style="font-weight:600;">${fmtPnl(totSL)}</span></span>
              <span style="font-family:var(--fm);font-size:10px;color:var(--t3);">@Tgt&nbsp;<span class="${totTgt >= 0 ? 'pos' : 'neg'}" style="font-weight:600;">${fmtPnl(totTgt)}</span></span>
              <span style="font-family:var(--fm);font-size:10px;color:var(--t3);">Now&nbsp;<span class="${totNow >= 0 ? 'pos' : 'neg'}" style="font-weight:600;">${fmtPnl(totNow)}</span></span>
            ` : ''}
            <button class="man-view-btn" onclick="manToggleView('${gid}')">⇄ ${isBull ? 'Bear' : 'Bull'} view</button>
          </div>
        </div>
        <!-- Lots table -->
        ${tableSection}
      </div>
      <!-- Sparkline panel -->
      <div class="man-spark-panel">
        <div class="man-spark-title">P&amp;L History</div>
        <div class="man-spark-wrap">
          <canvas id="man-spark-${gid}"></canvas>
        </div>
        <div id="man-spark-empty-${gid}" class="man-spark-empty" style="display:none;">
          No history yet.<br>Save a snapshot to start tracking.
        </div>
      </div>
    </div>`;
}

// ── Sparkline ──────────────────────────────────────────────────────────────────
function renderSparkline(gid) {
  const canvas = document.getElementById('man-spark-' + gid);
  const emptyEl = document.getElementById('man-spark-empty-' + gid);
  if (!canvas) return;

  // Destroy existing chart instance
  if (_manSparkCharts[gid]) { _manSparkCharts[gid].destroy(); delete _manSparkCharts[gid]; }

  // Filter history to selected month and this group
  const days = manPnlHistory
    .map(day => ({ date: day.date, g: (day.groups || []).find(x => x.id === gid) }))
    .filter(x => x.g);

  if (!days.length) {
    canvas.style.display = 'none';
    if (emptyEl) emptyEl.style.display = 'block';
    return;
  }
  canvas.style.display = 'block';
  if (emptyEl) emptyEl.style.display = 'none';

  const labels   = days.map(x => x.date);
  const pnlData  = days.map(x => x.g.pnl_now);
  const slData   = days.map(x => x.g.sl_pnl);
  const tgtData  = days.map(x => x.g.target_pnl);

  const hasSL  = slData.some(v => v != null);
  const hasTgt = tgtData.some(v => v != null);

  const cssVar = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  const datasets = [
    {
      label: 'P&L Now',
      data: pnlData,
      borderColor: cssVar('--p02'),
      backgroundColor: cssVar('--p02') + '22',
      borderWidth: 2,
      tension: 0.35,
      fill: true,
      pointRadius: 3,
      pointHoverRadius: 5,
    },
  ];
  if (hasSL) datasets.push({
    label: '@SL',
    data: slData,
    borderColor: cssVar('--neg'),
    borderWidth: 1.5,
    borderDash: [4, 3],
    tension: 0.35,
    fill: false,
    pointRadius: 2,
    pointHoverRadius: 4,
  });
  if (hasTgt) datasets.push({
    label: '@Target',
    data: tgtData,
    borderColor: cssVar('--pos'),
    borderWidth: 1.5,
    borderDash: [4, 3],
    tension: 0.35,
    fill: false,
    pointRadius: 2,
    pointHoverRadius: 4,
  });

  _manSparkCharts[gid] = new Chart(canvas, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            font: { family: 'var(--fm)', size: 10 },
            color: cssVar('--t2'),
            boxWidth: 12,
            padding: 8,
          },
        },
        tooltip: {
          callbacks: {
            label: ctx => ' ' + fmtPnl(ctx.parsed.y),
          },
        },
      },
      scales: {
        x: {
          ticks: { font: { family: 'var(--fm)', size: 9 }, color: cssVar('--t3'), maxRotation: 0 },
          grid:  { color: cssVar('--border') + '55' },
        },
        y: {
          ticks: {
            font: { family: 'var(--fm)', size: 9 },
            color: cssVar('--t3'),
            callback: v => fmtPnl(v),
          },
          grid: { color: cssVar('--border') + '55' },
        },
      },
    },
  });
}

// ── Snapshot Save ─────────────────────────────────────────────────────────────
async function manSaveSnapshot() {
  if (!manSelectedMonth) { alert('Select a month first.'); return; }

  const today = marketData.NIFTY && marketData.NIFTY.date
    ? marketData.NIFTY.date
    : new Date().toLocaleDateString('en-IN', { day:'2-digit', month:'short', year:'numeric' }).replace(/ /g,'-');

  const niftySpot     = (marketData.NIFTY     || {}).close || null;
  const bankniftySpot = (marketData.BANKNIFTY || {}).close || null;
  const dte           = manDte(manSelectedMonth);

  const notesEl = document.getElementById('man-snapshot-notes');
  const notes   = notesEl ? notesEl.value.trim() : '';

  const allLots = positions.filter(p => p.exp === manSelectedMonth).flatMap(manLots);

  const groupPayload = MAN_GROUPS.map(g => {
    const eff   = manGroupEffective(g.id);
    const gLots = allLots.filter(lot => manAssign[lot.lotKey] === g.id);

    const totSL  = gLots.reduce((s, lot) => { const v = ((scenarioLevels[lot.und]||{})[eff.view]||{}).sl;     return v != null ? s + manPayoff(lot,v) : s; }, 0);
    const totTgt = gLots.reduce((s, lot) => { const v = ((scenarioLevels[lot.und]||{})[eff.view]||{}).target; return v != null ? s + manPayoff(lot,v) : s; }, 0);
    const totNow = gLots.reduce((s, lot) => s + lot.lotPnl, 0);

    // Market context: use first lot's underlying for pct calculations
    const refUnd  = gLots.length ? gLots[0].und : 'NIFTY';
    const refSpot = refUnd === 'BANKNIFTY' ? bankniftySpot : niftySpot;
    const sc      = (scenarioLevels[refUnd] || {})[eff.view] || {};

    // Per-lot detail for man_position_snapshot.csv
    const lots = gLots.map(lot => {
      const sc_ = (scenarioLevels[lot.und] || {})[eff.view] || {};
      return {
        lot_key:    lot.lotKey,
        instrument: lot.instrument,
        und:        lot.und,
        type:       lot.type,
        side:       lot.side,
        lot_sz:     lot.lotSz,
        avg:        lot.avg,
        pnl_now:    Math.round(lot.lotPnl),
        pnl_sl:     sc_.sl     != null ? Math.round(manPayoff(lot, sc_.sl))     : null,
        pnl_target: sc_.target != null ? Math.round(manPayoff(lot, sc_.target)) : null,
      };
    });

    return {
      id:              g.id,
      name:            eff.name,
      view:            eff.view,
      pnl_now:         Math.round(totNow),
      sl_pnl:          gLots.length ? Math.round(totSL)  : null,
      target_pnl:      gLots.length ? Math.round(totTgt) : null,
      lot_count:       gLots.length,
      pct_from_sl:     manPctFromLevel(refSpot, sc.sl),
      pct_from_target: manPctFromLevel(refSpot, sc.target),
      lots,
    };
  });

  try {
    const btn = document.getElementById('man-snapshot-btn');
    if (btn) { btn.disabled = true; btn.textContent = 'Saving…'; }

    const res = await fetch('/api/v1/portfolio/man-snapshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        month:          manSelectedMonth,
        date:           today,
        nifty_spot:     niftySpot,
        banknifty_spot: bankniftySpot,
        dte,
        notes,
        groups:         groupPayload,
      }),
    });
    if (!res.ok) throw new Error(await res.text());

    if (btn) { btn.disabled = false; btn.textContent = '⬇ Save Snapshot'; }

    await manLoadHistory(manSelectedMonth);
    renderSparkline(manActiveTab);
    // Clear notes field after successful save
    if (notesEl) notesEl.value = '';
    alert(`Snapshot saved for ${manSelectedMonth} · ${today}`);
  } catch(e) {
    alert('Snapshot failed: ' + e.message);
    const btn = document.getElementById('man-snapshot-btn');
    if (btn) { btn.disabled = false; btn.textContent = '⬇ Save Snapshot'; }
  }
}

async function manLoadHistory(month) {
  try {
    const data = await fetch(`/api/v1/portfolio/man-pnl-history?month=${month}`).then(r => r.json());
    manPnlHistory = (data.days || []).slice(-30); // last 30 days
  } catch(e) {
    manPnlHistory = [];
  }
}

// ── Action Logger ─────────────────────────────────────────────────────────────
// Fire-and-forget: captures every drag-drop decision in market context.
// Errors are swallowed — never blocks the UI.
function manLogAction(action, lotKey, fromGroup, toGroup) {
  fetch('/api/v1/portfolio/man-action', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      date:           (marketData.NIFTY || {}).date  || '',
      month:          manSelectedMonth || '',
      action,
      lot_key:        lotKey,
      from_group:     fromGroup || '',
      to_group:       toGroup   || '',
      nifty_spot:     (marketData.NIFTY     || {}).close || null,
      banknifty_spot: (marketData.BANKNIFTY || {}).close || null,
    }),
  }).catch(() => {});
}

// ── Section 3: Pool ───────────────────────────────────────────────────────────
function renderPool() {
  const el = document.getElementById('man-pool-container');
  if (!el) return;

  const allLots = positions.filter(p => p.exp === manSelectedMonth && p.und === manSelectedUnd).flatMap(manLots);
  const poolLots = allLots.filter(lot => !manAssign[lot.lotKey]);
  const assignedCount = allLots.filter(lot => manAssign[lot.lotKey]).length;

  // Split pool into four columns
  const col1 = poolLots.filter((_, i) => i % 4 === 0);
  const col2 = poolLots.filter((_, i) => i % 4 === 1);
  const col3 = poolLots.filter((_, i) => i % 4 === 2);
  const col4 = poolLots.filter((_, i) => i % 4 === 3);

  const undClr = u => u === 'NIFTY' ? 'var(--p02)' : 'var(--p04)';
  const undBg  = u => u === 'NIFTY' ? 'var(--p02-bg)' : 'var(--p04-bg)';

  function lotRow(lot) {
    const lotBadge = lot.nLots > 1
      ? `<span style="font-family:var(--fm);font-size:8px;font-weight:700;padding:0 3px;
                      border-radius:2px;background:var(--surface);border:1px solid var(--border);
                      color:var(--t3);flex-shrink:0;">L${lot.lotIdx}</span>`
      : '';
    return `<div class="man-row" draggable="true"
        ondragstart="manDragStart(event,'${lot.lotKey}')" ondragend="manDragEnd(event)">
      <span class="man-drag-handle" style="font-size:11px;">⠿</span>
      <span style="font-family:var(--fm);font-size:8px;font-weight:700;padding:1px 4px;border-radius:2px;
                   background:${undBg(lot.und)};color:${undClr(lot.und)};flex-shrink:0;">
        ${lot.und === 'BANKNIFTY' ? 'BNK' : 'NF'}
      </span>
      ${lotBadge}
      <span style="font-family:var(--fm);font-size:10px;font-weight:500;flex:1;min-width:0;
                   white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${lot.full}</span>
      <span class="inst-badge ${lot.type.toLowerCase()}" style="flex-shrink:0;font-size:9px;padding:1px 4px;">${lot.type}</span>
      <span class="side-badge ${lot.side.toLowerCase()}" style="flex-shrink:0;font-size:9px;padding:1px 4px;">${lot.side === 'Long' ? 'L' : 'S'}</span>
      <span class="${pnlClass(lot.lotPnl)}" style="font-family:var(--fm);font-size:10px;flex-shrink:0;">${fmtPnl(lot.lotPnl)}</span>
    </div>`;
  }

  function colHtml(lots) {
    return lots.length
      ? lots.map(lotRow).join('')
      : `<div style="padding:16px 8px;font-family:var(--fm);font-size:11px;color:var(--t4);text-align:center;">—</div>`;
  }

  el.innerHTML = `
    <div class="card">
      <div class="card-hdr">
        <span class="card-title">Position Pool — ${manSelectedUnd || ''} ${manSelectedMonth || '—'}</span>
        <span class="card-sub">${poolLots.length} unassigned · ${assignedCount} in groups</span>
      </div>
      <div class="man-pool-2col"
           ondragover="event.preventDefault();this.style.background='var(--surface2)'"
           ondragleave="this.style.background=''"
           ondrop="manDropToPool(event);this.style.background=''">
        <div class="man-pool-col">${colHtml(col1)}</div>
        <div class="man-pool-col">${colHtml(col2)}</div>
        <div class="man-pool-col">${colHtml(col3)}</div>
        <div class="man-pool-col">${colHtml(col4)}</div>
      </div>
    </div>`;
}

// ── Drag-drop ─────────────────────────────────────────────────────────────────
function manDragStart(e, lotKey) {
  _manDrag = lotKey;
  e.dataTransfer.setData('text/plain', lotKey);
  e.dataTransfer.effectAllowed = 'move';
  setTimeout(() => { if (e.target) e.target.classList.add('dragging'); }, 0);
}
function manDragEnd(e) {
  _manDrag = null;
  if (e.target) e.target.classList.remove('dragging');
}
function manDropToGroup(e, gid) {
  e.preventDefault();
  const lotKey = e.dataTransfer.getData('text/plain') || _manDrag;
  if (!lotKey) return;
  const prevGroup = manAssign[lotKey] || '';
  manAssign[lotKey] = gid;
  manLogAction('assign', lotKey, prevGroup, gid);
  manSave(manSelectedMonth);
  renderTabStrip();
  renderActiveGroupCard();
  renderPool();
}
function manDropToPool(e) {
  e.preventDefault();
  const lotKey = e.dataTransfer.getData('text/plain') || _manDrag;
  if (!lotKey) return;
  const prevGroup = manAssign[lotKey] || '';
  delete manAssign[lotKey];
  manLogAction('unassign', lotKey, prevGroup, '');
  manSave(manSelectedMonth);
  renderTabStrip();
  renderActiveGroupCard();
  renderPool();
}
function manRemove(lotKey) {
  const prevGroup = manAssign[lotKey] || '';
  delete manAssign[lotKey];
  manLogAction('remove', lotKey, prevGroup, '');
  manSave(manSelectedMonth);
  renderTabStrip();
  renderActiveGroupCard();
  renderPool();
}

// ── Name / View ───────────────────────────────────────────────────────────────
function manSaveName(gid, name) {
  if (!manGroupState[gid]) manGroupState[gid] = {};
  manGroupState[gid].name = name;
  manSave(manSelectedMonth);
}
function manToggleView(gid) {
  if (!manGroupState[gid]) manGroupState[gid] = {};
  const current = manGroupEffective(gid).view;
  manGroupState[gid].view = current === 'bull' ? 'bear' : 'bull';
  manSave(manSelectedMonth);
  renderActiveGroupCard();
}

// ── CSV Export (kept for compatibility) ───────────────────────────────────────
function manSaveCsv() {
  if (!manSelectedMonth) { alert('Select a month first.'); return; }
  const today = new Date().toISOString().slice(0, 10);
  const hdr = ['Date','Month','Group','View','Instrument','Lot','Underlying','Expiry',
                'Type','Side','Qty','Entry','@SL_PnL','@Target_PnL','PnL_Now'];
  const rowData = [];
  const allLots = positions.filter(p => p.exp === manSelectedMonth).flatMap(manLots);
  for (const g of MAN_GROUPS) {
    const eff   = manGroupEffective(g.id);
    const gLots = allLots.filter(lot => manAssign[lot.lotKey] === g.id);
    for (const lot of gLots) {
      const sc_    = (scenarioLevels[lot.und] || {})[eff.view] || {};
      const pnlSL  = sc_.sl     != null ? manPayoff(lot, sc_.sl).toFixed(0)     : '';
      const pnlTgt = sc_.target != null ? manPayoff(lot, sc_.target).toFixed(0) : '';
      rowData.push([today, manSelectedMonth, eff.name, eff.view.toUpperCase(),
        lot.instrument, 'L'+lot.lotIdx, lot.und, lot.exp, lot.type, lot.side,
        lot.lotSz, lot.avg.toFixed(2), pnlSL, pnlTgt, lot.lotPnl.toFixed(0)]);
    }
  }
  if (!rowData.length) { alert('No lots assigned to groups yet.'); return; }
  const csv = [hdr, ...rowData].map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `manoeuvre_${manSelectedMonth}_${today}.csv`;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(a.href);
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function initManoeuvre() {
  const months = manActiveMonths();
  manSelectedMonth = months[0]; // default: current expiry month
  manSelectedUnd   = 'NIFTY';  // default: NIFTY
  await manLoad(manSelectedMonth);
  manActiveTab = 'anchor';

  await manLoadHistory(manSelectedMonth);

  renderMonthTiles();
  renderTabStrip();
  renderActiveGroupCard();
  renderPool();

  // Auto-snapshot: if no snapshot exists for today's date, silently save one
  manAutoSnapshot();
}

function manAutoSnapshot() {
  if (!manSelectedMonth || !manPnlHistory.length) return;
  const todayDate = (marketData.NIFTY || {}).date || '';
  if (!todayDate) return;
  const alreadySaved = manPnlHistory.some(d => d.date === todayDate);
  if (alreadySaved) return;
  // Silent save — no alert, no notes
  const notesEl = document.getElementById('man-snapshot-notes');
  const prevVal = notesEl ? notesEl.value : '';
  // Trigger full snapshot silently
  manSaveSnapshot().then(() => {
    // Restore notes field to whatever it was (should be empty but be safe)
    if (notesEl) notesEl.value = prevVal;
  }).catch(() => {});
}
