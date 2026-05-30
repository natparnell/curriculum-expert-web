#!/usr/bin/env python3
"""Mark strands covered in coverage-master.json given new apps.
Usage: python3 update_coverage.py <meta.json>  (entries: file, subject, ks, strand)"""
import sys, json, os
MASTER='curriculum/schema/coverage-master.json'
def main():
    meta=json.load(open(sys.argv[1]))
    d=json.load(open(MASTER))
    by={s['subject']:s for s in d}
    upd=0; miss=[]
    for m in meta:
        s=by.get(m['subject'])
        if not s: miss.append(m['file']); continue
        hit=False
        for ks in s['keyStages']:
            if ks['keyStage']!=m['ks']: continue
            for st in ks['strands']:
                if st['strand']==m['strand']:
                    st['coverage']='covered'
                    st.setdefault('apps',[])
                    if m['file'] not in st['apps']: st['apps'].append(m['file'])
                    hit=True; upd+=1
        if not hit: miss.append(m['file'])
    json.dump(d,open(MASTER,'w'),indent=2,ensure_ascii=False)
    print(f"coverage-master: marked {upd} strands covered; unmatched: {miss or 'none'}")
if __name__=='__main__': main()
