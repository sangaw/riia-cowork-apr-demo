// ── Position Manoeuvre — fno-manoeuvre.js ────────────────
// Split-pane: left = unassigned lot pool, right = 7 groups.
// Depends on globals: positions, currentUnd, currentExpiry,
//   scenarioLevels, fmtPnl(), pnlClass()  (all in fno.html main script)

const MAN_STORE = 'rita_man_v5';
const MAN_LOT   = { NIFTY: 65, BANKNIFTY: 30 };
let manGroups   = null;   // [{id, name, view:'bull'|'bear'}]
let manAssign   = {};     // lotKey → group id  (absent = unassigned)
let _manDrag    = null;

// ── Persistence ──────────────────────────────────────────
function manSave() {
  try { localStorage.setItem(MAN_STORE, JSON.stringify({ groups: manGroups, assign: manAssign })); } catch(e) {}
}

function manLoadStore() {
  try {
    const s = localStorage.getItem(MAN_STORE);
    if (s) { const d = JSON.parse(s); manGroups = d.groups || null; manAssign = d.assign || {}; }
  } catch(e) {}
}

function manBuildDefaults() {
  manGroups = [1,2,3,4,5,6,7].map(i => ({ id: 'g'+i, name: 'Group '+i, view: i <= 4 ? 'bull' : 'bear' }));
  manAssign = {};
}

// ── Lot expansion ────────────────────────────────────────
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
    lotPnl:  pnlPerLot,
  }));
}

// ── At-expiry intrinsic P&L for one lot ──────────────────
function manPayoff(lot, price) {
  const sign = lot.side === 'Long' ? 1 : -1;
  let intrinsic;
  if      (lot.type === 'FUT') intrinsic = price;
  else if (lot.type === 'CE')  intrinsic = Math.max(0, price - (lot.strike_val || 0));
  else                         intrinsic = Math.max(0, (lot.strike_val || 0) - price);
  return sign * lot.lotSz * (intrinsic - lot.avg);
}

// ── Pool row (left panel vertical list) ──────────────────
function manRowHtml(lot) {
  const undClr = lot.und === 'NIFTY' ? 'var(--p02)' : 'var(--p04)';
  const undBg  = lot.und === 'NIFTY' ? 'var(--p02-bg)' : 'var(--p04-bg)';
  const lotBadge = lot.nLots > 1
    ? `<span style="font-family:var(--fm);font-size:9px;font-weight:700;padding:1px 4px;border-radius:2px;background:var(--surface);border:1px solid var(--border);color:var(--t3);">L${lot.lotIdx}</span>`
    : '';
  return `<div class="man-row" draggable="true"
      ondragstart="manDragStart(event,'${lot.lotKey}')" ondragend="manDragEnd(event)">
    <span class="man-drag-handle">⠿</span>
    <span style="font-family:var(--fm);font-size:9px;font-weight:700;padding:1px 5px;border-radius:2px;background:${undBg};color:${undClr};">${lot.und==='BANKNIFTY'?'BNK':lot.und}</span>
    ${lotBadge}
    <span style="font-family:var(--fm);font-size:11px;font-weight:500;flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${lot.full}</span>
    <span class="exp-badge ${lot.exp.toLowerCase()}" style="flex-shrink:0;">${lot.exp}</span>
    <span class="inst-badge ${lot.type.toLowerCase()}" style="flex-shrink:0;">${lot.type}</span>
    <span class="side-badge ${lot.side.toLowerCase()}" style="flex-shrink:0;">${lot.side}</span>
    <span style="font-family:var(--fm);font-size:10px;color:var(--t3);flex-shrink:0;">×${lot.lotSz}</span>
    <span class="${pnlClass(lot.lotPnl)}" style="font-family:var(--fm);font-size:11px;flex-shrink:0;">${fmtPnl(lot.lotPnl)}</span>
  </div>`;
}

// ── Group card (right panel) ──────────────────────────────
function renderGroupCard(g) {
  const allLots = positions.flatMap(manLots);
  const gLots   = allLots.filter(lot => manAssign[lot.lotKey] === g.id);
  const isBull  = g.view === 'bull';

  const rows = gLots.map(lot => {
    const sc_    = (scenarioLevels[lot.und] || {})[g.view] || {};
    const pnlSL  = sc_.sl     != null ? manPayoff(lot, sc_.sl)     : null;
    const pnlTgt = sc_.target != null ? manPayoff(lot, sc_.target) : null;
    const undClr = lot.und === 'NIFTY' ? 'var(--p02)' : 'var(--p04)';
    const undBg  = lot.und === 'NIFTY' ? 'var(--p02-bg)' : 'var(--p04-bg)';
    const lotBadge = lot.nLots > 1
      ? `<span style="font-family:var(--fm);font-size:9px;font-weight:700;padding:1px 4px;border-radius:2px;background:var(--surface2);border:1px solid var(--border);color:var(--t3);margin-right:3px;">L${lot.lotIdx}</span>`
      : '';
    return `<tr>
      <td style="white-space:nowrap;">
        <span style="font-family:var(--fm);font-size:9px;font-weight:600;padding:1px 4px;border-radius:2px;background:${undBg};color:${undClr};margin-right:3px;">${lot.und==='BANKNIFTY'?'BNK':lot.und}</span>
        ${lotBadge}<span style="font-family:var(--fm);font-size:11px;font-weight:500;">${lot.full}</span>
      </td>
      <td><span class="exp-badge ${lot.exp.toLowerCase()}">${lot.exp}</span></td>
      <td><span class="side-badge ${lot.side.toLowerCase()}">${lot.side}</span></td>
      <td class="val">${lot.lotSz}</td>
      <td class="val">${lot.type==='FUT' ? lot.avg.toLocaleString('en-IN',{minimumFractionDigits:2}) : lot.avg.toFixed(2)}</td>
      <td class="${pnlSL !=null&&pnlSL >=0?'pos':'neg'} val">${pnlSL !=null?fmtPnl(pnlSL) :'—'}</td>
      <td class="${pnlTgt!=null&&pnlTgt>=0?'pos':'neg'} val">${pnlTgt!=null?fmtPnl(pnlTgt):'—'}</td>
      <td class="${pnlClass(lot.lotPnl)} val">${fmtPnl(lot.lotPnl)}</td>
      <td style="text-align:center;padding:3px 6px;">
        <button onclick="manRemove('${lot.lotKey}')"
          style="font-family:var(--fm);font-size:10px;padding:1px 6px;border-radius:3px;border:1px solid var(--border);background:var(--surface);cursor:pointer;color:var(--t3);"
          title="Return to pool">↩</button>
      </td>
    </tr>`;
  }).join('');

  // Column header labels — handles mixed underlyings
  const undsInGroup = [...new Set(gLots.map(lot => lot.und))];
  function scLabel(key) {
    if (!undsInGroup.length) return '—';
    return undsInGroup.map(u => {
      const v = ((scenarioLevels[u] || {})[g.view] || {})[key];
      return v != null
        ? (undsInGroup.length > 1 ? u.replace('BANKNIFTY','BNK')+' '+v.toLocaleString('en-IN') : v.toLocaleString('en-IN'))
        : '—';
    }).join(' / ');
  }

  const totSL  = gLots.reduce((s,lot) => { const v=((scenarioLevels[lot.und]||{})[g.view]||{}).sl;     return v!=null?s+manPayoff(lot,v):s; }, 0);
  const totTgt = gLots.reduce((s,lot) => { const v=((scenarioLevels[lot.und]||{})[g.view]||{}).target; return v!=null?s+manPayoff(lot,v):s; }, 0);
  const totNow = gLots.reduce((s,lot) => s+lot.lotPnl, 0);

  const tableSection = gLots.length ? `
    <div style="overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;font-size:11px;">
        <thead><tr>
          <th style="font-family:var(--fm);font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:.05em;padding:5px 10px;background:var(--surface2);border-bottom:1px solid var(--border);text-align:left;white-space:nowrap;">Instrument</th>
          <th style="font-family:var(--fm);font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:.05em;padding:5px 8px;background:var(--surface2);border-bottom:1px solid var(--border);">Exp</th>
          <th style="font-family:var(--fm);font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:.05em;padding:5px 8px;background:var(--surface2);border-bottom:1px solid var(--border);">Side</th>
          <th style="font-family:var(--fm);font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:.05em;padding:5px 8px;background:var(--surface2);border-bottom:1px solid var(--border);">Qty</th>
          <th style="font-family:var(--fm);font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:.05em;padding:5px 10px;background:var(--surface2);border-bottom:1px solid var(--border);">Entry</th>
          <th style="font-family:var(--fm);font-size:9px;color:var(--neg);text-transform:uppercase;letter-spacing:.05em;padding:5px 10px;background:var(--neg-bg);border-bottom:1px solid var(--neg-bd);white-space:nowrap;">@SL ${scLabel('sl')}</th>
          <th style="font-family:var(--fm);font-size:9px;color:var(--p01);text-transform:uppercase;letter-spacing:.05em;padding:5px 10px;background:var(--p01-bg);border-bottom:1px solid var(--p01-bd);white-space:nowrap;">@Tgt ${scLabel('target')}</th>
          <th style="font-family:var(--fm);font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:.05em;padding:5px 10px;background:var(--surface2);border-bottom:1px solid var(--border);">P&amp;L Now</th>
          <th style="background:var(--surface2);border-bottom:1px solid var(--border);width:30px;"></th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <div class="tbl-footer" style="font-size:11px;">
      <span class="lbl">${gLots.length} lot${gLots.length!==1?'s':''}</span>
      <span class="lbl" style="margin-left:12px;">@SL:</span>
      <span class="val ${totSL >=0?'pos':'neg'}">${fmtPnl(totSL)}</span>
      <span class="lbl" style="margin-left:12px;">@Tgt:</span>
      <span class="val ${totTgt>=0?'pos':'neg'}">${fmtPnl(totTgt)}</span>
      <span class="lbl" style="margin-left:12px;">Now:</span>
      <span class="val ${totNow>=0?'pos':'neg'}">${fmtPnl(totNow)}</span>
    </div>` : '<div class="man-empty-zone">Drop lots here</div>';

  return `<div class="man-group-card"
      ondragover="event.preventDefault();this.style.outline='2px dashed var(--p02)'"
      ondragleave="this.style.outline=''"
      ondrop="manDropToGroup(event,'${g.id}');this.style.outline=''">
    <div class="man-group-hdr">
      <div style="display:flex;align-items:center;gap:7px;flex-wrap:wrap;">
        <input class="man-name-input" value="${g.name.replace(/"/g,'&quot;')}"
          onchange="manSaveName('${g.id}',this.value)"
          ondragstart="event.stopPropagation()" title="Click to rename">
        <span style="background:${isBull?'var(--p01-bg)':'var(--neg-bg)'};color:${isBull?'var(--p01)':'var(--neg)'};border:1px solid ${isBull?'var(--p01-bd)':'var(--neg-bd)'};font-family:var(--fm);font-size:9px;font-weight:700;padding:1px 7px;border-radius:3px;">${isBull?'BULL':'BEAR'}</span>
        ${gLots.length ? `<span style="font-family:var(--fm);font-size:10px;color:var(--t3);">${gLots.length} lot${gLots.length!==1?'s':''}</span>` : ''}
      </div>
      <button class="man-view-btn" onclick="manToggleView('${g.id}')">⇄ ${isBull?'Bear':'Bull'}</button>
    </div>
    ${tableSection}
  </div>`;
}

// ── Main render (split-pane layout) ──────────────────────
function renderManoeuvre() {
  const el = document.getElementById('manoeuvre-container');
  if (!el || !manGroups) return;

  const allLots = positions.flatMap(manLots);

  // Pool: unassigned lots filtered by current pill selection
  const poolLots = allLots.filter(lot =>
    !manAssign[lot.lotKey] &&
    (currentUnd    === 'ALL' || lot.und === currentUnd) &&
    (currentExpiry === 'ALL' || lot.exp === currentExpiry)
  );

  const totalAssigned = allLots.filter(lot => manAssign[lot.lotKey]).length;

  const leftPanel = `
    <div class="man-left-panel">
      <div class="card">
        <div class="card-hdr">
          <span class="card-title">Pool</span>
          <span class="card-sub">${poolLots.length} unassigned · ${totalAssigned} in groups</span>
        </div>
        <div class="man-pool-scroll" id="man-pool"
            ondragover="event.preventDefault();this.style.background='var(--surface2)'"
            ondragleave="this.style.background=''"
            ondrop="manDropToPool(event);this.style.background=''">
          ${poolLots.length
            ? poolLots.map(manRowHtml).join('')
            : `<div style="padding:20px 12px;font-family:var(--fm);font-size:11px;color:var(--t4);text-align:center;">All lots assigned</div>`}
        </div>
      </div>
    </div>`;

  const rightPanel = `
    <div>
      <div style="display:flex;justify-content:flex-end;margin-bottom:10px;gap:8px;">
        <label class="man-load-btn" title="Upload a previously saved manoeuvre CSV to restore group assignments">
          ⬆ Load CSV
          <input type="file" accept=".csv" style="display:none;" onchange="manLoadCsv(this)">
        </label>
        <button class="man-save-btn" onclick="manSaveCsv()">⬇ Save CSV</button>
      </div>
      <div class="man-right-panel">
        ${manGroups.map(renderGroupCard).join('')}
      </div>
    </div>`;

  el.innerHTML = `<div class="man-layout">${leftPanel}${rightPanel}</div>`;
}

// ── CSV export ───────────────────────────────────────────
function manSaveCsv() {
  const today = new Date().toISOString().slice(0, 10);
  const hdr = ['Date','Group','View','Instrument','Lot','Underlying','Expiry','Type','Side','Qty','Entry','@SL_PnL','@Target_PnL','PnL_Now'];
  const rowData = [];

  for (const g of manGroups) {
    const allLots = positions.flatMap(manLots);
    const gLots = allLots.filter(lot => manAssign[lot.lotKey] === g.id);
    for (const lot of gLots) {
      const sc_    = (scenarioLevels[lot.und] || {})[g.view] || {};
      const pnlSL  = sc_.sl     != null ? manPayoff(lot, sc_.sl).toFixed(0)     : '';
      const pnlTgt = sc_.target != null ? manPayoff(lot, sc_.target).toFixed(0) : '';
      rowData.push([
        today, g.name, g.view.toUpperCase(),
        lot.instrument, 'L'+lot.lotIdx, lot.und, lot.exp, lot.type, lot.side,
        lot.lotSz, lot.avg.toFixed(2), pnlSL, pnlTgt, lot.lotPnl.toFixed(0)
      ]);
    }
  }

  if (!rowData.length) { alert('No lots assigned to groups yet.'); return; }

  const csv = [hdr, ...rowData].map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `manoeuvre_${today}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(a.href);
}

// ── CSV import ───────────────────────────────────────────
function manLoadCsv(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    const lines = e.target.result.trim().split('\n');
    if (lines.length < 2) { alert('Empty or invalid CSV.'); return; }

    const hdr = lines[0].split(',');
    const col  = name => hdr.indexOf(name);
    const iGrp = col('Group'), iView = col('View'), iInst = col('Instrument'), iLot = col('Lot');
    if (iGrp < 0 || iView < 0 || iInst < 0 || iLot < 0) {
      alert('CSV format not recognised. Please use a file saved from this page.');
      return;
    }

    // Collect unique groups in order of appearance
    const csvGroups = [];
    const seenNames = new Set();
    for (let i = 1; i < lines.length; i++) {
      const row = lines[i].split(',');
      if (row.length <= Math.max(iGrp, iView, iInst, iLot)) continue;
      const name = row[iGrp].trim();
      const view = (row[iView] || '').trim().toUpperCase() === 'BEAR' ? 'bear' : 'bull';
      if (!seenNames.has(name)) { seenNames.add(name); csvGroups.push({ name, view }); }
    }

    // Apply group names/views to existing group slots in order
    const nameToId = {};
    for (let i = 0; i < manGroups.length; i++) {
      if (i < csvGroups.length) {
        manGroups[i].name = csvGroups[i].name;
        manGroups[i].view = csvGroups[i].view;
      }
      nameToId[manGroups[i].name] = manGroups[i].id;
    }

    // Restore lot assignments
    manAssign = {};
    let loaded = 0;
    for (let i = 1; i < lines.length; i++) {
      const row = lines[i].split(',');
      if (row.length <= Math.max(iGrp, iInst, iLot)) continue;
      const inst   = row[iInst].trim();
      const lotNum = row[iLot].trim();           // 'L1', 'L2', …
      const gName  = row[iGrp].trim();
      const lotKey = inst + '_' + lotNum;
      const gId    = nameToId[gName];
      if (gId) { manAssign[lotKey] = gId; loaded++; }
    }

    input.value = '';                            // reset so same file can be re-uploaded
    manSave();
    renderManoeuvre();
    alert(`Restored ${loaded} lot assignment${loaded !== 1 ? 's' : ''} from "${file.name}".`);
  };
  reader.readAsText(file);
}

// ── Drag-drop handlers ────────────────────────────────────
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
  manAssign[lotKey] = gid;
  manSave();
  renderManoeuvre();
}

function manDropToPool(e) {
  e.preventDefault();
  const lotKey = e.dataTransfer.getData('text/plain') || _manDrag;
  if (!lotKey) return;
  delete manAssign[lotKey];
  manSave();
  renderManoeuvre();
}

function manRemove(lotKey) {
  delete manAssign[lotKey];
  manSave();
  renderManoeuvre();
}

// ── Name / view persistence ───────────────────────────────
function manSaveName(gid, name) {
  const g = manGroups.find(x => x.id === gid);
  if (g) { g.name = name; manSave(); }
}

function manToggleView(gid) {
  const g = manGroups.find(x => x.id === gid);
  if (!g) return;
  g.view = g.view === 'bull' ? 'bear' : 'bull';
  manSave();
  renderManoeuvre();
}

// ── Init ─────────────────────────────────────────────────
function initManoeuvre() {
  manLoadStore();
  if (!manGroups || !manGroups.length) {
    manBuildDefaults();
    manSave();
  }
  renderManoeuvre();
}
