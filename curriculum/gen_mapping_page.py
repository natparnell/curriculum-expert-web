#!/usr/bin/env python3
"""Generate curriculum-mapping.html: a friendly Subject -> NC domain -> strand (key stage +
assessment ref) -> app links view, with deep-link highlighting (?app=<file>)."""
import json, os

REPO = '/Users/nathanaelparnell/CCP2/curriculum-expert-web'
MASTER = os.path.join(REPO, 'curriculum/schema/coverage-master.json')
DOMAINS = os.path.join(REPO, 'curriculum/schema/domain-grouping.json')
CAT = os.path.join(REPO, 'edu-apps-catalogue.json')
OUT = os.path.join(REPO, 'curriculum-mapping.html')

KS_ORDER = {'EYFS': 0, 'KS1': 1, 'KS2': 2, 'KS3': 3, 'KS4': 4, 'KS5': 5}


def main():
    master = {s['subject']: s for s in json.load(open(MASTER))}
    domains = {d['subject']: d['domains'] for d in json.load(open(DOMAINS))}
    names = {c['file']: c['name'] for c in json.load(open(CAT))}

    model = []
    unmatched = 0
    for subject, dlist in domains.items():
        s = master.get(subject)
        if not s:
            continue
        # strand text -> domain name
        s2d = {}
        for d in dlist:
            for st in d.get('strands', []):
                s2d[st.strip()] = d['name']
        # collect coverage strand entries
        entries = []
        for ks in s['keyStages']:
            for st in ks['strands']:
                dom = s2d.get(st['strand'].strip())
                if dom is None:
                    dom = 'Other'
                    unmatched += 1
                apps = [{'file': f, 'name': names.get(f, f)} for f in st.get('apps', [])]
                entries.append({'domain': dom, 'ks': ks['keyStage'],
                                'note': ks.get('statutoryNote', ''), 'strand': st['strand'],
                                'apps': apps})
        # group by domain in the agent's order (+ Other last)
        order = [d['name'] for d in dlist]
        if any(e['domain'] == 'Other' for e in entries):
            order.append('Other')
        doms = []
        for dn in order:
            rows = [e for e in entries if e['domain'] == dn]
            if not rows:
                continue
            rows.sort(key=lambda e: (KS_ORDER.get(e['ks'], 9), e['strand']))
            doms.append({'name': dn, 'rows': rows, 'apps': sum(len(r['apps']) for r in rows)})
        model.append({'subject': subject, 'domains': doms,
                      'apps': sum(d['apps'] for d in doms)})
    model.sort(key=lambda m: m['subject'])

    data_json = json.dumps(model, ensure_ascii=False).replace('</', '<\\/')
    total_apps = sum(m['apps'] for m in model)
    page = TEMPLATE.replace('__DATA__', data_json).replace('__SUBJ__', str(len(model))).replace('__APPS__', str(total_apps))
    open(OUT, 'w', encoding='utf-8').write(page)
    print('wrote', OUT, '(%d subjects, %d app links, %d unmatched strands)' % (len(model), total_apps, unmatched))


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>National Curriculum Mapping | Curriculum Expert</title>
<style>
  :root{ --ink:#1a1a2e; --muted:#64748b; --line:#e2e8f0; --bg:#f8fafc; --blue:#1d4ed8; }
  *{ box-sizing:border-box; }
  body{ margin:0; font:16px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; color:var(--ink); background:var(--bg); }
  header{ background:linear-gradient(135deg,#1e3a8a,#1d4ed8); color:#fff; padding:30px 24px 24px; }
  header h1{ margin:0 0 6px; font-size:1.7rem; letter-spacing:-0.02em; }
  header p{ margin:0; opacity:.92; max-width:760px; }
  .stats{ margin-top:14px; font-size:.85rem; opacity:.9; }
  .wrap{ max-width:1080px; margin:0 auto; padding:20px 18px 70px; }
  .controls{ position:sticky; top:0; background:var(--bg); padding:12px 0; z-index:10; border-bottom:1px solid var(--line); display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  .controls input{ flex:1; min-width:220px; padding:9px 12px; border:1px solid var(--line); border-radius:8px; font-size:.95rem; }
  .controls label{ font-size:.88rem; color:var(--muted); display:flex; gap:6px; align-items:center; cursor:pointer; }
  .subject{ background:#fff; border:1px solid var(--line); border-radius:14px; margin-top:16px; overflow:hidden; }
  .subject>summary{ list-style:none; cursor:pointer; padding:15px 20px; display:flex; align-items:center; gap:12px; }
  .subject>summary::-webkit-details-marker{ display:none; }
  .subject>summary::before{ content:'\25B8'; color:var(--muted); transition:transform .15s; }
  .subject[open]>summary::before{ transform:rotate(90deg); }
  .subject h2{ margin:0; font-size:1.2rem; flex:1; }
  .scount{ font-size:.8rem; color:var(--muted); }
  .domain{ border-top:1px solid var(--line); }
  .domain>summary{ list-style:none; cursor:pointer; padding:11px 20px; display:flex; align-items:center; gap:10px; background:#fbfcfe; }
  .domain>summary::-webkit-details-marker{ display:none; }
  .domain>summary::before{ content:'\25B8'; color:var(--muted); font-size:.85rem; transition:transform .15s; }
  .domain[open]>summary::before{ transform:rotate(90deg); }
  .domain h3{ margin:0; font-size:1rem; font-weight:700; color:#334155; flex:1; }
  .dcount{ font-size:.78rem; color:var(--muted); }
  .rows{ padding:4px 20px 12px; }
  .row{ padding:10px 0; border-top:1px dashed var(--line); display:flex; gap:12px; align-items:flex-start; }
  .row:first-child{ border-top:none; }
  .ks{ flex-shrink:0; font-size:.7rem; font-weight:800; padding:3px 8px; border-radius:6px; color:#fff; min-width:46px; text-align:center; }
  .ks.EYFS{ background:#db2777; } .ks.KS1{ background:#ea580c; } .ks.KS2{ background:#ca8a04; }
  .ks.KS3{ background:#16a34a; } .ks.KS4{ background:#0891b2; } .ks.KS5{ background:#6d28d9; }
  .body{ flex:1; }
  .strand{ font-weight:600; }
  .note{ font-size:.8rem; color:var(--muted); margin-top:1px; }
  .apps{ margin-top:6px; display:flex; flex-wrap:wrap; gap:6px; }
  .apps a{ font-size:.82rem; text-decoration:none; background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; border-radius:6px; padding:3px 10px; }
  .apps a:hover{ background:#dbeafe; }
  .row.hl{ background:#fffbeb; border-radius:8px; box-shadow:0 0 0 2px #f59e0b; padding:10px 10px; }
  footer{ text-align:center; color:var(--muted); font-size:.82rem; padding:30px; }
</style>
</head>
<body>
<header>
  <h1>National Curriculum Mapping</h1>
  <p>Find any app's place in the curriculum: pick a subject, then a curriculum area, to see the learning
  journey across the key stages with a direct link to each app. Curriculum area sits above key stage, so you
  can follow a thread (for example "Number") all the way from Reception to A level.</p>
  <div class="stats">__SUBJ__ subjects &middot; __APPS__ app links</div>
</header>
<div class="wrap">
  <div class="controls">
    <input type="search" id="q" placeholder="Search subject, curriculum area, strand or app...">
    <label><input type="checkbox" id="expand"> Expand all</label>
  </div>
  <div id="list"></div>
  <footer>Each app also carries a "Curriculum mapping" link back to this page.</footer>
</div>
<script type="application/json" id="map-data">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('map-data').textContent);
const list = document.getElementById('list');
const q = document.getElementById('q'), expand = document.getElementById('expand');
const params = new URLSearchParams(location.search);
const focusApp = params.get('app');

function el(t,c,h){ const x=document.createElement(t); if(c)x.className=c; if(h!=null)x.innerHTML=h; return x; }

function render(){
  const term = q.value.trim().toLowerCase();
  const openAll = expand.checked || !!term;
  list.innerHTML='';
  DATA.forEach(s=>{
    const sd=el('details','subject'); if(openAll) sd.open=true;
    const ss=el('summary'); ss.innerHTML=`<h2>${s.subject}</h2><span class="scount">${s.apps} apps</span>`;
    sd.appendChild(ss);
    let anyS=false;
    s.domains.forEach(d=>{
      const rows=d.rows.filter(r=>{
        if(!term) return true;
        const hay=(s.subject+' '+d.name+' '+r.strand+' '+r.ks+' '+r.apps.map(a=>a.name).join(' ')).toLowerCase();
        return hay.includes(term);
      });
      if(!rows.length) return;
      anyS=true;
      const dd=el('details','domain'); if(openAll) dd.open=true;
      const ds=el('summary'); ds.innerHTML=`<h3>${d.name}</h3><span class="dcount">${rows.length}</span>`;
      dd.appendChild(ds);
      const rc=el('div','rows');
      rows.forEach(r=>{
        const row=el('div','row'); row.dataset.apps=r.apps.map(a=>a.file).join(',');
        row.appendChild(el('span','ks '+r.ks, r.ks));
        const b=el('div','body');
        b.appendChild(el('div','strand', r.strand));
        if(r.note) b.appendChild(el('div','note', r.note));
        if(r.apps.length){ const ap=el('div','apps'); r.apps.forEach(a=>{ const x=el('a'); x.href='/apps/'+a.file; x.target='_blank'; x.textContent=a.name; ap.appendChild(x); }); b.appendChild(ap); }
        row.appendChild(b); rc.appendChild(row);
      });
      dd.appendChild(rc); sd.appendChild(dd);
    });
    if(anyS) list.appendChild(sd);
  });
  if(!list.children.length) list.innerHTML='<p style="color:#64748b;padding:20px">No matches.</p>';
}
q.addEventListener('input',render); expand.addEventListener('change',render);
render();

// deep link: ?app=<file> -> expand ancestors, scroll, highlight
if(focusApp){
  let target=null;
  document.querySelectorAll('.row').forEach(r=>{
    if((r.dataset.apps||'').split(',').includes(focusApp)){
      target=r;
      let p=r.closest('details.domain'); if(p) p.open=true;
      let s=r.closest('details.subject'); if(s) s.open=true;
    }
  });
  if(target){ target.classList.add('hl'); setTimeout(()=>target.scrollIntoView({behavior:'smooth',block:'center'}),120); }
}
</script>
</body>
</html>
"""

if __name__ == '__main__':
    main()
