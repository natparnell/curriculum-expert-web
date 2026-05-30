#!/usr/bin/env python3
"""Generate curriculum-map.html: an interactive map of every subject and NC strand,
linking to the apps that service each strand, plus sections for the non-curriculum
app collections (CPD, Data). Data-driven from coverage-master.json and the catalogue
so it can be regenerated as apps are added."""
import json, os

REPO = '/Users/nathanaelparnell/CCP2/curriculum-expert-web'
MASTER = os.path.join(REPO, 'curriculum/schema/coverage-master.json')
CAT = os.path.join(REPO, 'edu-apps-catalogue.json')
OUT = os.path.join(REPO, 'curriculum-map.html')

ACTIVE = {'English', 'Mathematics', 'Science', 'History', 'Geography',
          'Religious Education', 'Modern Foreign Languages', 'Drama',
          'PSHE, RSHE and Behaviour/Pastoral'}
PROFESSIONAL = {'CPD & Staff Tools', 'Data & Analytics'}
# flat fallback sections for any non-curriculum collection NOT yet in coverage-master
NON_NC = []


def main():
    d = json.load(open(MASTER))
    cat = json.load(open(CAT))
    names = {c['file']: c['name'] for c in cat}

    tc = tp = tn = 0
    model = []
    for s in d:
        ks_list = []
        sc = sp = sn = 0
        for ks in s['keyStages']:
            strands = []
            for st in ks['strands']:
                apps = [{'file': f, 'name': names.get(f, f)} for f in st.get('apps', [])]
                strands.append({'strand': st['strand'], 'coverage': st['coverage'],
                                'apps': apps, 'gap': st.get('gapNote', '')})
                c = st['coverage']
                sc += c == 'covered'; sp += c == 'partial'; sn += c == 'none'
            ks_list.append({'keyStage': ks['keyStage'], 'note': ks.get('statutoryNote', ''),
                            'strands': strands,
                            'cov': sum(1 for x in strands if x['coverage'] == 'covered'),
                            'tot': len(strands)})
        tc += sc; tp += sp; tn += sn
        cells = sc + sp + sn
        cat = 'active' if s['subject'] in ACTIVE else ('professional' if s['subject'] in PROFESSIONAL else 'new')
        model.append({'subject': s['subject'], 'nc': True, 'active': s['subject'] in ACTIVE,
                      'category': cat,
                      'pct': round((sc + 0.5 * sp) / cells * 100) if cells else 0,
                      'covered': sc, 'partial': sp, 'none': sn, 'keyStages': ks_list})
    order = {'active': 0, 'professional': 1, 'new': 2}
    model.sort(key=lambda m: (order[m['category']], -m['pct']))

    # non-NC collections
    for subj in NON_NC:
        apps = [{'file': c['file'], 'name': c['name'], 'desc': c.get('desc', '')}
                for c in cat if c['subject'] == subj]
        apps.sort(key=lambda a: a['name'])
        if apps:
            model.append({'subject': subj, 'nc': False, 'count': len(apps), 'apps': apps})

    total = tc + tp + tn
    pct = round((tc + 0.5 * tp) / total * 100) if total else 0
    nonnc_apps = sum(m['count'] for m in model if not m['nc'])
    data_json = json.dumps(model, ensure_ascii=False).replace('</', '<\\/')

    page = (TEMPLATE.replace('__PCT__', str(pct)).replace('__SUBJ__', str(len(d)))
            .replace('__STRANDS__', str(total)).replace('__COV__', str(tc))
            .replace('__PART__', str(tp)).replace('__NONE__', str(tn))
            .replace('__TOOLS__', str(nonnc_apps)).replace('__DATA__', data_json))
    open(OUT, 'w', encoding='utf-8').write(page)
    print('wrote', OUT, '(%d NC subjects, %d strands, %d%% coverage, %d non-NC tools)'
          % (len(d), total, pct, nonnc_apps))


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Curriculum Coverage Map | Curriculum Expert</title>
<style>
  :root{ --ink:#1a1a2e; --muted:#64748b; --line:#e2e8f0; --bg:#f8fafc;
    --cov:#16a34a; --covbg:#dcfce7; --part:#d97706; --partbg:#fef3c7; --none:#dc2626; --nonebg:#fee2e2; }
  *{ box-sizing:border-box; }
  body{ margin:0; font:16px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    color:var(--ink); background:var(--bg); }
  header{ background:linear-gradient(135deg,#1e3a8a,#1d4ed8); color:#fff; padding:32px 24px 26px; }
  header h1{ margin:0 0 6px; font-size:1.8rem; letter-spacing:-0.02em; }
  header p{ margin:0; opacity:.9; max-width:760px; }
  .stats{ display:flex; flex-wrap:wrap; gap:14px; margin-top:18px; }
  .stat{ background:rgba(255,255,255,.14); border-radius:10px; padding:10px 16px; min-width:88px; }
  .stat b{ display:block; font-size:1.5rem; line-height:1.1; }
  .stat span{ font-size:.78rem; opacity:.85; text-transform:uppercase; letter-spacing:.05em; }
  .wrap{ max-width:1100px; margin:0 auto; padding:22px 18px 60px; }
  .controls{ position:sticky; top:0; background:var(--bg); padding:14px 0; z-index:10;
    display:flex; flex-wrap:wrap; gap:10px; align-items:center; border-bottom:1px solid var(--line); }
  .controls input[type=search]{ flex:1; min-width:200px; padding:9px 12px; border:1px solid var(--line);
    border-radius:8px; font-size:.95rem; }
  .controls label{ font-size:.9rem; color:var(--muted); display:flex; align-items:center; gap:6px; cursor:pointer; }
  .subject{ background:#fff; border:1px solid var(--line); border-radius:14px; margin-top:18px; overflow:hidden; }
  .subject>summary{ list-style:none; cursor:pointer; padding:16px 20px; display:flex; align-items:center;
    gap:14px; flex-wrap:wrap; }
  .subject>summary::-webkit-details-marker{ display:none; }
  .subject>summary::before{ content:'\25B8'; color:var(--muted); font-size:.9rem; transition:transform .15s; }
  .subject[open]>summary::before{ transform:rotate(90deg); }
  .subject h2{ margin:0; font-size:1.2rem; flex:1; min-width:160px; }
  .tag{ font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
    padding:3px 8px; border-radius:999px; }
  .tag.new{ background:#fee2e2; color:#991b1b; } .tag.active{ background:#dcfce7; color:#166534; }
  .tag.tools,.tag.prof{ background:#e2e8f0; color:#475569; }
  .bar{ flex-basis:150px; height:10px; background:var(--line); border-radius:999px; overflow:hidden; }
  .bar i{ display:block; height:100%; background:linear-gradient(90deg,#16a34a,#65a30d); }
  .pct{ font-weight:700; font-variant-numeric:tabular-nums; min-width:44px; text-align:right; }
  .ks{ border-top:1px solid var(--line); }
  .ks>summary{ list-style:none; cursor:pointer; padding:11px 20px; display:flex; align-items:center; gap:10px;
    background:#fbfcfe; font-size:.95rem; }
  .ks>summary::-webkit-details-marker{ display:none; }
  .ks>summary::before{ content:'\25B8'; color:var(--muted); font-size:.8rem; transition:transform .15s; }
  .ks[open]>summary::before{ transform:rotate(90deg); }
  .ks .kslabel{ font-weight:700; text-transform:uppercase; letter-spacing:.04em; color:#334155; }
  .ks .ksnote{ color:var(--muted); font-size:.82rem; font-weight:400; }
  .ks .kscount{ margin-left:auto; font-size:.78rem; color:var(--muted); font-variant-numeric:tabular-nums; }
  .ksbody{ padding:2px 20px 10px; }
  .strand{ padding:9px 0; border-top:1px dashed var(--line); display:flex; gap:12px; align-items:flex-start; }
  .strand:first-child{ border-top:none; }
  .dot{ flex-shrink:0; font-size:.7rem; font-weight:700; padding:3px 8px; border-radius:6px; margin-top:2px; }
  .dot.covered{ background:var(--covbg); color:var(--cov); }
  .dot.partial{ background:var(--partbg); color:var(--part); }
  .dot.none{ background:var(--nonebg); color:var(--none); }
  .strand .body{ flex:1; }
  .strand .name{ font-weight:600; }
  .strand .gap{ font-size:.85rem; color:var(--muted); margin-top:2px; }
  .apps{ margin-top:5px; display:flex; flex-wrap:wrap; gap:6px; }
  .apps a{ font-size:.82rem; text-decoration:none; background:#eff6ff; color:#1d4ed8;
    border:1px solid #bfdbfe; border-radius:6px; padding:3px 9px; }
  .apps a:hover{ background:#dbeafe; }
  .toollist{ padding:6px 20px 14px; }
  .toollist .tool{ padding:9px 0; border-top:1px dashed var(--line); }
  .toollist .tool:first-child{ border-top:none; }
  .toollist a{ font-weight:600; color:#1d4ed8; text-decoration:none; }
  .toollist a:hover{ text-decoration:underline; }
  .toollist .d{ font-size:.85rem; color:var(--muted); }
  .note{ padding:10px 20px; font-size:.85rem; color:var(--muted); background:#fbfcfe; border-top:1px solid var(--line); }
  footer{ text-align:center; color:var(--muted); font-size:.82rem; padding:30px; }
</style>
</head>
<body>
<header>
  <h1>Curriculum Coverage Map</h1>
  <p>Every subject and National Curriculum strand across EYFS to KS5, mapped to the interactive
  apps that teach it. Subjects without a statutory National Curriculum (PSHE, RSHE and Behaviour,
  and the strand frameworks for newly added subjects) use a curriculum of comparable scope. The
  professional toolkits (CPD and Data) are listed separately as they are not pupil curricula.</p>
  <div class="stats">
    <div class="stat"><b>__SUBJ__</b><span>Subjects</span></div>
    <div class="stat"><b>__STRANDS__</b><span>Strands</span></div>
    <div class="stat"><b>__PCT__%</b><span>Covered</span></div>
    <div class="stat"><b>__COV__</b><span>Full</span></div>
    <div class="stat"><b>__PART__</b><span>Partial</span></div>
    <div class="stat"><b>__NONE__</b><span>Gaps</span></div>
    <div class="stat"><b>__TOOLS__</b><span>Pro tools</span></div>
  </div>
</header>
<div class="wrap">
  <div class="controls">
    <input type="search" id="q" placeholder="Search subjects, strands or apps...">
    <label><input type="checkbox" id="gapsOnly"> Show gaps only</label>
    <label><input type="checkbox" id="expand"> Expand all</label>
  </div>
  <div id="list"></div>
  <footer>Generated from the curriculum coverage schema. Tap a subject, then a key stage, to expand.</footer>
</div>
<script type="application/json" id="map-data">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('map-data').textContent);
const list = document.getElementById('list');
const q = document.getElementById('q'), gapsOnly = document.getElementById('gapsOnly'), expand = document.getElementById('expand');

function el(tag, cls, html){ const x=document.createElement(tag); if(cls)x.className=cls; if(html!=null)x.innerHTML=html; return x; }
function appLink(a){ const x=el('a'); x.href='apps/'+a.file; x.target='_blank'; x.textContent=a.name; return x; }

function render(){
  const term = q.value.trim().toLowerCase();
  const gaps = gapsOnly.checked;
  const openAll = expand.checked || !!term;
  list.innerHTML='';
  DATA.forEach(s=>{
    const det=el('details','subject'); if(openAll) det.open=true;
    const sum=el('summary');
    let right;
    if(s.nc){ const tm={active:['active','Active'],professional:['prof','Professional'],new:['new','New']};
      const tg=tm[s.category]||['new','New'];
      right=`<span class="tag ${tg[0]}">${tg[1]}</span>`+
      `<span class="bar"><i style="width:${s.pct}%"></i></span><span class="pct">${s.pct}%</span>`; }
    else { right=`<span class="tag tools">Pro tools</span><span class="pct">${s.count}</span>`; }
    sum.innerHTML=`<h2>${s.subject}</h2>`+right;
    det.appendChild(sum);
    let anyShown=false;

    if(!s.nc){
      if(gaps) return;            // professional tools have no curriculum gaps
      const matched=s.apps.filter(a=>!term || (a.name+' '+a.desc+' '+s.subject).toLowerCase().includes(term));
      if(!matched.length) return;
      det.appendChild(el('div','note','Professional toolkit, not a pupil curriculum. '+s.count+' apps.'));
      const tl=el('div','toollist');
      matched.forEach(a=>{ const t=el('div','tool'); const link=appLink(a); t.appendChild(link); t.appendChild(el('div','d',a.desc||'')); tl.appendChild(t); });
      det.appendChild(tl); list.appendChild(det); return;
    }

    s.keyStages.forEach(ks=>{
      const rows=ks.strands.filter(st=>{
        if(gaps && st.coverage==='covered') return false;
        if(term){ const hay=(s.subject+' '+ks.keyStage+' '+st.strand+' '+st.apps.map(a=>a.name).join(' ')+' '+(st.gap||'')).toLowerCase(); if(!hay.includes(term)) return false; }
        return true;
      });
      if(!rows.length) return;
      anyShown=true;
      const ksd=el('details','ks'); if(openAll) ksd.open=true;
      const kss=el('summary');
      kss.innerHTML=`<span class="kslabel">${ks.keyStage}</span>`+(ks.note?`<span class="ksnote">${ks.note}</span>`:'')+`<span class="kscount">${ks.cov}/${ks.tot} covered</span>`;
      ksd.appendChild(kss);
      const body=el('div','ksbody');
      rows.forEach(st=>{
        const row=el('div','strand');
        const lbl={covered:'Covered',partial:'Partial',none:'Gap'}[st.coverage];
        row.appendChild(el('span','dot '+st.coverage,lbl));
        const b=el('div','body'); b.appendChild(el('div','name',st.strand));
        if(st.gap && st.coverage!=='covered') b.appendChild(el('div','gap',st.gap));
        if(st.apps.length){ const ap=el('div','apps'); st.apps.forEach(a=>ap.appendChild(appLink(a))); b.appendChild(ap); }
        row.appendChild(b); body.appendChild(row);
      });
      ksd.appendChild(body); det.appendChild(ksd);
    });
    if(anyShown) list.appendChild(det);
  });
  if(!list.children.length) list.innerHTML='<p style="color:#64748b;padding:20px">No matches.</p>';
}
q.addEventListener('input',render); gapsOnly.addEventListener('change',render); expand.addEventListener('change',render);
render();
</script>
</body>
</html>
"""

if __name__ == '__main__':
    main()
