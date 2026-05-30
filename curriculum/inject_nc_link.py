#!/usr/bin/env python3
"""Inject a small 'Curriculum mapping' chip into every edu app, deep-linking to that
app's place in /curriculum-mapping. Idempotent via a sentinel. Mechanical sweep, no agents."""
import pathlib

SENTINEL = "<!-- NC-MAP-LINK v1 -->"

BLOCK = SENTINEL + """
<style>
.nc-map-link{position:fixed;right:14px;bottom:14px;z-index:99990;background:#1d4ed8;color:#fff;
  font:600 0.8rem/1 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:9px 13px;
  border-radius:999px;text-decoration:none;box-shadow:0 2px 10px rgba(0,0,0,0.22);
  display:inline-flex;align-items:center;gap:6px;transition:background 0.15s;}
.nc-map-link:hover{background:#1e40af;}
.nc-map-link span.ncm-x{display:none;}
@media (max-width:520px){.nc-map-link .ncm-t{display:none;}.nc-map-link span.ncm-x{display:inline;}}
@media print{.nc-map-link{display:none;}}
</style>
<a class="nc-map-link" id="ncMapLink" href="/curriculum-mapping" target="_blank" rel="noopener"
   title="See where this app fits the National Curriculum">
  <span aria-hidden="true">&#128218;</span><span class="ncm-t">Curriculum mapping</span><span class="ncm-x">NC</span>
</a>
<script>
(function(){var a=document.getElementById('ncMapLink');if(!a)return;
 var f=(location.pathname.split('/').pop()||'');
 if(f && /\\.html$/.test(f)) a.href='/curriculum-mapping?app='+encodeURIComponent(f);})();
</script>
"""


def inject(path: pathlib.Path) -> str:
    text = path.read_text(encoding='utf-8')
    if SENTINEL in text:
        return 'skip-already'
    import re
    m = re.search(r'<body[^>]*>', text)
    if not m:
        return 'skip-no-body'
    at = m.end()
    path.write_text(text[:at] + '\n' + BLOCK + text[at:], encoding='utf-8')
    return 'injected'


SKIP = {'data-daily-cashflow.html', 'data-idsr-analyser.html'}  # real-data tools, not NC-mapped


def main():
    roots = [pathlib.Path('/Users/nathanaelparnell/CCP2/apps'),
             pathlib.Path('/Users/nathanaelparnell/CCP2/curriculum-expert-web/apps')]
    counts = {}
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.glob('*.html')):
            if p.name in SKIP:
                counts['skip-list'] = counts.get('skip-list', 0) + 1
                continue
            s = inject(p)
            counts[s] = counts.get(s, 0) + 1
            if s == 'skip-no-body':
                print('no-body:', p.name)
    print('---')
    for k, v in counts.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
