#!/usr/bin/env python3
"""Drive the app build pipeline for one or more subjects.

  python3 pipeline.py briefs <MasterSubject> [...]      -> /tmp/build_briefs.json
  python3 pipeline.py script <briefs.json> <out.js>     -> workflow script embedding briefs
  python3 pipeline.py post   <briefs.json> <wf_output>  -> inject + coverage + menus + catalogue + mirror + map

MasterSubject = exact 'subject' value in coverage-master.json.
Builds one app per strand whose coverage is 'none'.
"""
import sys, os, re, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = '/Users/nathanaelparnell/CCP2/curriculum-expert-web'
MASTER = os.path.join(REPO, 'curriculum/schema/coverage-master.json')
APPS = os.path.join(REPO, 'apps')

# masterSubject -> (filename prefix, menu/catalogue subject)
SMETA = {
    'English': ('english', 'English'),
    'Mathematics': ('maths', 'Maths'),
    'Science': ('science', 'Science'),
    'History': ('history', 'History'),
    'Geography': ('geography', 'Geography'),
    'Religious Education': ('re', 'Religious Education'),
    'Modern Foreign Languages': ('mfl', 'Modern Foreign Languages'),
    'Drama': ('drama', 'Drama'),
    'PSHE, RSHE and Behaviour/Pastoral': ('behaviour', 'Behaviour & Pastoral'),
    'Art and Design': ('art', 'Art and Design'),
    'Music': ('music', 'Music'),
    'Computing': ('computing', 'Computing'),
    'Design and Technology': ('dt', 'Design and Technology'),
    'Physical Education': ('pe', 'Physical Education'),
    'Citizenship': ('citizenship', 'Citizenship'),
    'CPD & Staff Tools': ('cpd', 'CPD & Staff Tools'),
    'Data & Analytics': ('data', 'Data & Analytics'),
}


def slug(t):
    return re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', t.lower())).strip('-')


def cmd_briefs(subjects, coverage='none'):
    d = json.load(open(MASTER))
    by = {s['subject']: s for s in d}
    if not subjects:  # empty => every master subject that has a prefix mapping
        subjects = [s['subject'] for s in d if s['subject'] in SMETA]
    existing = {f for f in os.listdir(APPS) if f.endswith('.html')}
    briefs = []
    used = set()
    for subj in subjects:
        if subj not in by:
            print('!! unknown subject:', subj); continue
        if subj not in SMETA:
            print('!! no prefix mapping:', subj); continue
        prefix, menu = SMETA[subj]
        s = by[subj]
        for ks in s['keyStages']:
            for st in ks['strands']:
                if st['coverage'] != coverage:
                    continue
                sug = st.get('suggestedApps', [])
                idea = sug[0] if sug else {'title': st['strand'][:40], 'concept': st.get('gapNote', '')}
                fn = '%s-%s-%s.html' % (prefix, slug(ks['keyStage']), slug(idea['title']))
                base = fn; i = 2
                while fn in existing or fn in used:
                    fn = base.replace('.html', '-%d.html' % i); i += 1
                used.add(fn)
                briefs.append({
                    'file': fn, 'masterSubject': subj, 'menuSubject': menu,
                    'ks': ks['keyStage'], 'strand': st['strand'],
                    'gapNote': st.get('gapNote', ''), 'title': idea['title'],
                    'concept': idea['concept'],
                    'altIdeas': [a['title'] for a in sug[1:]],
                })
    json.dump(briefs, open('/tmp/build_briefs.json', 'w'), indent=2, ensure_ascii=False)
    from collections import Counter
    c = Counter(b['masterSubject'] for b in briefs)
    print('briefs:', len(briefs), dict(c))


PROMPT = r'''Author ONE brand-new, self-contained interactive teaching app as a single HTML file for an England school trust (WeST). A real, deployable classroom demo.

WRITE THE FILE TO: `apps/@@FILE@@`

## What to build
App name: @@TITLE@@
Subject: @@SUBJECT@@  |  Key stage: @@KS@@
National Curriculum strand it must cover: @@STRAND@@
Gap it fills: @@GAP@@
Concept to implement: @@CONCEPT@@
@@ALT@@

## Non-negotiable quality bar
1. Single self-contained HTML file. All CSS and JS inline. NO external CDNs, fonts, libraries or network calls. Works fully offline.
2. Genuinely interactive with immediate, specific feedback (not reveal-and-read). The pupil manipulates something and the app responds with targeted feedback. Depth over breadth: one excellent interaction beats five shallow ones.
3. Pitched precisely at @@KS@@. Accurate and on-strand for the England National Curriculum (or for newly added subjects, the standard programme of study / exam content for that subject and phase).
4. Born accessible and touch-ready: Pointer Events (pointerdown/move/up + setPointerCapture) for any drag, touch-action:none on draggables, so it works on tablets and interactive whiteboards. Keyboard operation and ARIA labels/live regions for any canvas/SVG or drag interaction. Mobile-first responsive.
5. UK English. NO em dashes in visible text (use commas, colons, semicolons, full stops, parentheses).
6. Demonstration data only. Never imply real pupils; use clearly fictional names. Do NOT add a demo-data banner/modal yourself (one is injected centrally afterwards); write a clean normal page.
7. Emoji: use HTML hex entities in HTML body; the \u escape only inside <script> JS strings; CSS content uses a CSS hex escape. Never use the \u escape in HTML outside scripts.
8. Editorial clean design: clear hierarchy, whitespace, a short teacher-facing aim line, projector-readable.

After writing, re-read the file: valid HTML, balanced <script> tags, ends with </html>, no syntax errors, no external requests.

Return the structured summary (built=true, a menu title, a one-line catalogue summary, key features, any risks).'''


def tts_guidance():
    snip = open(os.path.join(HERE, 'west-tts-snippet.html')).read()
    return ('\n\n## Audio (ONLY if the app reads text aloud)\n'
            'If the app speaks, do NOT call SpeechSynthesis directly. Inject this helper block ONCE '
            'immediately after the <body> tag, then produce all speech via '
            "window.westTTS.speak(text, { lang: 'en-GB', rate: 0.85, pitch: 1.1 }). It selects the "
            'high-quality Google UK English neural voice (free, online) and falls back to the best local '
            'voice offline. Use the correct BCP-47 lang for other languages (fr-FR, de-DE, es-ES). Keep a '
            'gentle pace for young pupils (rate ~0.85, pitch ~1.1).\n\nHELPER BLOCK (inject verbatim):\n' + snip)


def build_prompt(b):
    alt = ('Related ideas you may fold in if natural: ' + '; '.join(b['altIdeas'])) if b.get('altIdeas') else ''
    return (PROMPT.replace('@@FILE@@', b['file']).replace('@@TITLE@@', b['title'])
            .replace('@@SUBJECT@@', b['menuSubject']).replace('@@KS@@', b['ks'])
            .replace('@@STRAND@@', b['strand']).replace('@@GAP@@', b['gapNote'])
            .replace('@@CONCEPT@@', b['concept']).replace('@@ALT@@', alt))


def cmd_script(briefs_path, out_path):
    briefs = json.load(open(briefs_path))
    tasks = [{'file': b['file'], 'prompt': build_prompt(b)} for b in briefs]
    tj = json.dumps(tasks, ensure_ascii=False)
    ttsj = json.dumps(tts_guidance(), ensure_ascii=False)   # embedded ONCE, appended at call time
    script = '''export const meta = {
  name: 'build-apps-batch',
  description: 'Build new apps for uncovered NC strands (one agent per app)',
  phases: [{ title: 'Build', detail: 'one agent authors each new interactive app' }],
}

const TTS = ''' + ttsj + ''';
const TASKS = ''' + tj + ''';

const SCHEMA = {
  type: 'object',
  required: ['file', 'built', 'summary'],
  properties: {
    file: { type: 'string' },
    built: { type: 'boolean' },
    title: { type: 'string' },
    summary: { type: 'string' },
    keyFeatures: { type: 'array', items: { type: 'string' } },
    risks: { type: 'string' },
  },
}

phase('Build')
const results = await parallel(TASKS.map((t) => () => agent(t.prompt + TTS, { label: `build:${t.file}`, phase: 'Build', schema: SCHEMA })))
const ok = results.filter(Boolean)
log(`Built ${ok.length}/${TASKS.length} apps`)
return ok
'''
    open(out_path, 'w').write(script)
    print('wrote', out_path, '(%d tasks)' % len(tasks))


def cmd_post(briefs_path, wf_output):
    import integrate_apps, update_coverage, gen_map_page
    briefs = {b['file']: b for b in json.load(open(briefs_path))}
    out = json.load(open(wf_output))
    res = out.get('result', out) if isinstance(out, dict) else out
    titles = {}
    for r in (res or []):
        if not r:
            continue
        fn = os.path.basename(r.get('file', ''))
        if fn:
            titles[fn] = r
    # only integrate apps whose file actually exists on disk
    meta = []   # for coverage (masterSubject)
    entries = []  # for catalogue/menus (menuSubject)
    missing = []
    for fn, b in briefs.items():
        if not os.path.exists(os.path.join(APPS, fn)):
            missing.append(fn); continue
        r = titles.get(fn, {})
        name = (r.get('title') or b['title'])
        desc = (r.get('summary') or b['concept'])
        desc = desc.rstrip('.').strip()[:150]
        meta.append({'file': fn, 'subject': b['masterSubject'], 'ks': b['ks'], 'strand': b['strand']})
        entries.append({'file': fn, 'name': name, 'desc': desc,
                        'subject': b['menuSubject'], 'ks': b['ks']})
    print('to integrate: %d (missing/not built: %d)' % (len(entries), len(missing)))
    if missing:
        print('  missing:', missing)
    if not entries:
        return
    # 1) inject demo notice (global, idempotent)
    os.system('python3 /Users/nathanaelparnell/CCP2/scripts/inject_demo_notice.py >/dev/null')
    # 2) coverage
    json.dump(meta, open('/tmp/_post_meta.json', 'w'), ensure_ascii=False)
    sys.argv = ['x', '/tmp/_post_meta.json']
    update_coverage.main()
    # 3) catalogue + menus + mirror
    ca, tot = integrate_apps.update_catalogue(entries)
    print('  catalogue: +%d (now %d)' % (ca, tot))
    ok, msg = integrate_apps.update_menu_file(integrate_apps.PORTAL, entries)
    print('  apps-portal.html:', 'OK' if ok else 'FAIL', '|', msg)
    ok2, msg2 = integrate_apps.update_menu_file(integrate_apps.MENU, entries)
    print('  menu.html:', 'OK' if ok2 else 'FAIL', '|', msg2)
    m = integrate_apps.mirror_files(entries)
    print('  mirrored to CCP2/apps:', m)
    # 4) regenerate map
    gen_map_page.main()


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'briefs':
        cmd_briefs(sys.argv[2:])
    elif cmd == 'briefs-partial':
        cmd_briefs(sys.argv[2:], coverage='partial')   # no subjects => all
    elif cmd == 'script':
        cmd_script(sys.argv[2], sys.argv[3])
    elif cmd == 'post':
        cmd_post(sys.argv[2], sys.argv[3])
    else:
        print('unknown command', cmd)
