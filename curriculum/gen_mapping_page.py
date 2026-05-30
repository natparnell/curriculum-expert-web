#!/usr/bin/env python3
"""Generate curriculum-mapping.html: a friendly Subject -> NC domain -> strand (key stage +
assessment ref) -> app links view, styled to match the apps portal. Deep-link via ?app=<file>."""
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
        s2d = {}
        for d in dlist:
            for st in d.get('strands', []):
                s2d[st.strip()] = d['name']
        entries = []
        for ks in s['keyStages']:
            for st in ks['strands']:
                dom = s2d.get(st['strand'].strip())
                if dom is None:
                    dom = 'Other'; unmatched += 1
                apps = [{'file': f, 'name': names.get(f, f)} for f in st.get('apps', [])]
                entries.append({'domain': dom, 'ks': ks['keyStage'],
                                'note': ks.get('statutoryNote', ''), 'strand': st['strand'], 'apps': apps})
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
        model.append({'subject': subject, 'domains': doms, 'apps': sum(d['apps'] for d in doms)})
    model.sort(key=lambda m: m['subject'])

    # stats
    n_subjects = len(model)
    n_domains = sum(len(m['domains']) for m in model)
    n_strands = sum(len(d['rows']) for m in model for d in m['domains'])
    appset, ksset = set(), set()
    for m in model:
        for d in m['domains']:
            for r in d['rows']:
                ksset.add(r['ks'])
                for a in r['apps']:
                    appset.add(a['file'])

    data_json = json.dumps(model, ensure_ascii=False).replace('</', '<\\/')
    page = (TEMPLATE.replace('__DATA__', data_json)
            .replace('__SUBJ__', str(n_subjects)).replace('__DOMAINS__', str(n_domains))
            .replace('__STRANDS__', str(n_strands)).replace('__APPS__', str(len(appset)))
            .replace('__KS__', str(len(ksset))))
    open(OUT, 'w', encoding='utf-8').write(page)
    print('wrote', OUT, '(%d subjects, %d domains, %d strands, %d apps, %d unmatched)'
          % (n_subjects, n_domains, n_strands, len(appset), unmatched))


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>National Curriculum Mapping | Curriculum Expert</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body{ font-family:'Inter',system-ui,-apple-system,sans-serif; background:#faf7f2; color:#3d2e1f; min-height:100vh; padding:2rem; }
  .wrap{ max-width:1400px; margin:0 auto; }
  header{ display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem; margin-bottom:1.2rem; }
  .header-left h1{ font-size:1.5rem; font-weight:800; letter-spacing:-0.02em; }
  .header-left p{ font-size:0.85rem; color:#8b7355; margin-top:0.2rem; max-width:680px; }
  .header-actions{ display:flex; align-items:center; gap:0.6rem; flex-wrap:wrap; }
  .nav-link{ display:inline-flex; align-items:center; gap:0.35rem; background:#fff; color:#6b5744; font-size:0.8rem; font-weight:600; padding:0.4rem 0.9rem; border:1.5px solid #e0d5c5; border-radius:999px; text-decoration:none; transition:all 0.15s; white-space:nowrap; }
  .nav-link:hover{ border-color:#e07020; color:#e07020; }
  .actions-divider{ width:1px; height:22px; background:#e0d5c5; margin:0 0.2rem; }
  .ce-link{ display:inline-flex; align-items:center; gap:0.35rem; background:linear-gradient(135deg,#e07020,#c96830); color:#fff; font-size:0.8rem; font-weight:700; padding:0.4rem 0.9rem; border-radius:999px; text-decoration:none; transition:all 0.15s; white-space:nowrap; }
  .ce-link:hover{ background:linear-gradient(135deg,#c96830,#b05a28); box-shadow:0 2px 8px rgba(224,112,32,0.25); }
  .stats{ display:flex; flex-wrap:wrap; gap:0.8rem; margin-bottom:1.4rem; }
  .stat{ background:#fff; border:1px solid #e8ddd0; border-radius:12px; padding:0.7rem 1.1rem; min-width:96px; }
  .stat b{ display:block; font-size:1.5rem; font-weight:800; line-height:1.1; color:#e07020; }
  .stat span{ font-size:0.72rem; color:#8b7355; text-transform:uppercase; letter-spacing:0.05em; font-weight:600; }
  .controls{ position:sticky; top:0; background:#faf7f2; padding:0.8rem 0; z-index:10; display:flex; gap:0.7rem; flex-wrap:wrap; align-items:center; border-bottom:1px solid #e8ddd0; }
  .controls input{ flex:1; min-width:220px; padding:0.6rem 1rem; border:1px solid #e8ddd0; border-radius:8px; font-family:inherit; font-size:0.9rem; background:#fff; color:#3d2e1f; }
  .controls label{ font-size:0.85rem; color:#8b7355; display:flex; gap:6px; align-items:center; cursor:pointer; font-weight:500; }
  .subject{ background:#fff; border:1px solid #e8ddd0; border-radius:14px; margin-top:1rem; overflow:hidden; }
  .subject>summary{ list-style:none; cursor:pointer; padding:0.95rem 1.2rem; display:flex; align-items:center; gap:0.75rem; }
  .subject>summary::-webkit-details-marker{ display:none; }
  .subject>summary::before{ content:'\25B8'; color:#b9a88f; transition:transform .15s; }
  .subject[open]>summary::before{ transform:rotate(90deg); }
  .subject h2{ margin:0; font-size:1.15rem; font-weight:700; flex:1; }
  .scount{ font-size:0.78rem; color:#8b7355; font-weight:600; }
  .domain{ border-top:1px solid #f0e8dc; }
  .domain>summary{ list-style:none; cursor:pointer; padding:0.65rem 1.2rem; display:flex; align-items:center; gap:0.6rem; background:#fdfbf7; }
  .domain>summary::-webkit-details-marker{ display:none; }
  .domain>summary::before{ content:'\25B8'; color:#b9a88f; font-size:0.85rem; transition:transform .15s; }
  .domain[open]>summary::before{ transform:rotate(90deg); }
  .domain h3{ margin:0; font-size:0.98rem; font-weight:700; color:#6b5744; flex:1; }
  .dcount{ font-size:0.76rem; color:#a0896e; font-weight:600; }
  .rows{ padding:0.2rem 1.2rem 0.7rem; }
  .row{ padding:0.6rem 0; border-top:1px dashed #f0e8dc; display:flex; gap:0.75rem; align-items:flex-start; }
  .row:first-child{ border-top:none; }
  .ks{ flex-shrink:0; font-size:0.68rem; font-weight:800; padding:3px 8px; border-radius:6px; color:#fff; min-width:46px; text-align:center; }
  .ks.EYFS{ background:#db2777; } .ks.KS1{ background:#ea580c; } .ks.KS2{ background:#ca8a04; }
  .ks.KS3{ background:#16a34a; } .ks.KS4{ background:#0891b2; } .ks.KS5{ background:#6d28d9; }
  .body{ flex:1; }
  .strand{ font-weight:600; font-size:0.92rem; }
  .note{ font-size:0.78rem; color:#a0896e; margin-top:1px; }
  .apps{ margin-top:6px; display:flex; flex-wrap:wrap; gap:6px; }
  .apps a{ font-size:0.8rem; text-decoration:none; background:#fbeee3; color:#c96830; border:1px solid #f0d9c5; border-radius:6px; padding:3px 10px; font-weight:600; }
  .apps a:hover{ background:#f7e1cf; }
  .row.hl{ background:#fff7ed; border-radius:8px; box-shadow:0 0 0 2px #e07020; padding:0.6rem 0.6rem; }
  footer{ text-align:center; color:#a0896e; font-size:0.8rem; padding:2rem; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="header-left">
      <h1>National Curriculum Mapping</h1>
      <p>Find any app's place in the curriculum: pick a subject, then a curriculum area, to follow the
      learning journey across the key stages with a direct link to each app. The curriculum area sits
      above key stage, so you can trace a thread (for example "Number") from Reception to A level.</p>
    </div>
    <div class="header-actions">
      <a href="/apps" class="nav-link">&#128218; Educational Apps</a>
      <span class="actions-divider"></span>
      <a href="/" class="ce-link">&#129302; Curriculum Expert</a>
    </div>
  </header>

  <div class="stats">
    <div class="stat"><b>__SUBJ__</b><span>Subjects</span></div>
    <div class="stat"><b>__DOMAINS__</b><span>Curriculum areas</span></div>
    <div class="stat"><b>__STRANDS__</b><span>Strands</span></div>
    <div class="stat"><b>__KS__</b><span>Key stages</span></div>
    <div class="stat"><b>__APPS__</b><span>Apps</span></div>
  </div>

  <div class="controls">
    <input type="search" id="q" placeholder="Search subject, curriculum area, strand or app...">
    <label><input type="checkbox" id="expand"> Expand all</label>
  </div>
  <div id="list"></div>
  <footer>Every app also carries a "Curriculum mapping" link back to this page.</footer>
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
  if(!list.children.length) list.innerHTML='<p style="color:#8b7355;padding:1.2rem">No matches.</p>';
}
q.addEventListener('input',render); expand.addEventListener('change',render);
render();
if(focusApp){
  let target=null;
  document.querySelectorAll('.row').forEach(r=>{
    if((r.dataset.apps||'').split(',').includes(focusApp)){
      target=r;
      const p=r.closest('details.domain'); if(p) p.open=true;
      const s=r.closest('details.subject'); if(s) s.open=true;
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
