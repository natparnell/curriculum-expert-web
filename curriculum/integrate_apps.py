#!/usr/bin/env python3
"""Integrate new edu apps into the catalogue and both eduApps menus, then mirror files.

Usage: python3 integrate_apps.py <entries.json>
entries.json: list of {file, name, desc, subject, ks}
  - subject = catalogue/menu subject name (e.g. 'English', 'Maths', 'Science', 'Art and Design')
  - ks = 'EYFS' | 'KS1' | 'KS2' | 'KS3' | 'KS4' | 'KS5'

Safe: backs up each menu file, validates with Node (eduApps must still eval and the
app count must rise by exactly the number inserted), and reverts that file on failure.
Idempotent: an app whose file is already present in a target is skipped there.
"""
import sys, os, json, shutil, subprocess, tempfile

REPO = '/Users/nathanaelparnell/CCP2/curriculum-expert-web'
PARENT = '/Users/nathanaelparnell/CCP2'
CATALOGUE = os.path.join(REPO, 'edu-apps-catalogue.json')
PORTAL = os.path.join(REPO, 'apps-portal.html')
MENU = os.path.join(PARENT, 'menu.html')

# style for subjects that may need a new group created (new NC subjects)
SUBJECT_STYLE = {
    'Art and Design':        {'icon': '\\u{1F3A8}', 'color': '#db2777', 'bg': '#fce7f3'},
    'Music':                 {'icon': '\\u{1F3B5}', 'color': '#7c3aed', 'bg': '#ede9fe'},
    'Computing':             {'icon': '\\u{1F4BB}', 'color': '#0891b2', 'bg': '#cffafe'},
    'Design and Technology': {'icon': '\\u{1F6E0}', 'color': '#ea580c', 'bg': '#ffedd5'},
    'Physical Education':    {'icon': '\\u{26BD}',  'color': '#16a34a', 'bg': '#dcfce7'},
    'Citizenship':           {'icon': '\\u{2696}',  'color': '#4f46e5', 'bg': '#e0e7ff'},
}
KS_ORDER = ['EYFS', 'KS1', 'KS2', 'KS3', 'KS4', 'KS5']


def js_str(s):
    """Valid JS double-quoted string literal."""
    return json.dumps(s, ensure_ascii=False)


def find_eduapps_span(text):
    """Return (start_index_of_open_bracket, end_index_after_close_bracket) for `const eduApps = [ ... ]`."""
    anchor = text.index('const eduApps')
    open_idx = text.index('[', anchor)
    i = open_idx
    depth = 0
    instr = None
    esc = False
    while i < len(text):
        c = text[i]
        if instr:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == instr:
                instr = None
        else:
            if c in '"\'`':
                instr = c
            elif c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    return open_idx, i + 1
        i += 1
    raise ValueError('eduApps array not closed')


def count_apps_node(menu_text):
    span = find_eduapps_span(menu_text)
    arr = menu_text[span[0]:span[1]]
    js = ('const eduApps = ' + arr + ';\nlet c=0;eduApps.forEach(g=>{'
          'if(g.keystages){g.keystages.forEach(k=>c+=k.apps.length);}'
          'else if(g.apps){c+=g.apps.length;}});'
          'console.log(JSON.stringify({groups:eduApps.length,apps:c}));')
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(js)
        tmp = f.name
    try:
        out = subprocess.run(['node', tmp], capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None, out.stderr.strip()
        return json.loads(out.stdout.strip()), None
    finally:
        os.unlink(tmp)


def insert_into_menu(text, subject, ks, entry_obj):
    """Return new text with entry inserted at (subject, ks). Creates keystage/subject as needed."""
    entry = '{ name: %s, desc: %s, file: %s }' % (
        js_str(entry_obj['name']), js_str(entry_obj['desc']), js_str(entry_obj['file']))

    span = find_eduapps_span(text)
    arr_start, arr_end = span
    region = text[arr_start:arr_end]

    subj_marker = "subject: '%s'" % subject
    if subj_marker in text and text.index(subj_marker) < arr_end:
        s_abs = text.index(subj_marker)
        # subject region ends at next "\n    subject: '" or array end
        nxt = text.find("\n    subject: '", s_abs + 1)
        s_region_end = nxt if (nxt != -1 and nxt < arr_end) else arr_end
        # flat-apps subject (no keystages, e.g. Behaviour/CPD/Data): append to its apps array
        if 'keystages:' not in text[s_abs:s_region_end]:
            close = text.find('\n    ]', s_abs)
            if close == -1 or close > s_region_end:
                raise ValueError('flat apps close not found for %s' % subject)
            return text[:close] + ',\n      ' + entry + text[close:]
        label_marker = "label: '%s'" % ks
        lpos = text.find(label_marker, s_abs, s_region_end)
        if lpos != -1:
            # append to existing keystage apps array: insert before its "\n      ]}"
            close = text.find('\n      ]}', lpos)
            if close == -1 or close > s_region_end:
                raise ValueError('keystage close not found for %s/%s' % (subject, ks))
            ins = ',\n        ' + entry
            return text[:close] + ins + text[close:]
        else:
            # create a new keystage block inside this subject's keystages array.
            # keystages array closes with "\n    ]" within subject region.
            ks_close = text.find('\n    ]', s_abs)
            if ks_close == -1 or ks_close > s_region_end:
                raise ValueError('keystages close not found for %s' % subject)
            block = ",\n      { label: '%s', apps: [\n        %s\n      ]}" % (ks, entry)
            return text[:ks_close] + block + text[ks_close:]
    else:
        # create a whole new subject group; insert before eduApps array close "]"
        style = SUBJECT_STYLE.get(subject, {'icon': '\\u{1F4E6}', 'color': '#6b7280', 'bg': '#f3f4f6'})
        group = (",\n  {\n    subject: '%s',\n    icon: '%s',\n    color: '%s',\n    bg: '%s',\n"
                 "    keystages: [\n      { label: '%s', apps: [\n        %s\n      ]}\n    ]\n  }" % (
                     subject, style['icon'], style['color'], style['bg'], ks, entry))
        # insert just before the final ] of the array (arr_end-1 points at ']')
        return text[:arr_end - 1] + group + '\n' + text[arr_end - 1:]


def update_menu_file(path, entries):
    text = open(path, encoding='utf-8').read()
    before, err = count_apps_node(text)
    if before is None:
        return False, 'pre-validate failed: %s' % err
    added = 0
    for e in entries:
        if ("file: '%s'" % e['file']) in text or ('file: "%s"' % e['file']) in text:
            continue  # already present
        text = insert_into_menu(text, e['subject'], e['ks'], e)
        added += 1
    if added == 0:
        return True, 'all already present'
    after, err = count_apps_node(text)
    if after is None:
        return False, 'post-validate failed (not written): %s' % err
    if after['apps'] != before['apps'] + added:
        return False, 'count mismatch: %d + %d != %d (not written)' % (before['apps'], added, after['apps'])
    bak = path + '.bak'
    shutil.copy2(path, bak)
    open(path, 'w', encoding='utf-8').write(text)
    return True, 'added %d (groups %d->%d, apps %d->%d)' % (added, before['groups'], after['groups'], before['apps'], after['apps'])


def update_catalogue(entries):
    d = json.load(open(CATALOGUE))
    have = {x['file'] for x in d}
    added = 0
    for e in entries:
        if e['file'] in have:
            continue
        d.append({'name': e['name'], 'desc': e['desc'], 'file': e['file'],
                  'subject': e['subject'], 'keystage': e['ks']})
        added += 1
    json.dump(d, open(CATALOGUE, 'w'), indent=2, ensure_ascii=False)
    open(CATALOGUE, 'a').write('\n')
    return added, len(d)


def mirror_files(entries):
    dst = os.path.join(PARENT, 'apps')
    n = 0
    if not os.path.isdir(dst):
        return 0
    for e in entries:
        src = os.path.join(REPO, 'apps', e['file'])
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, e['file']))
            n += 1
    return n


def main():
    entries = json.load(open(sys.argv[1]))
    print('Integrating %d entries' % len(entries))
    ca, total = update_catalogue(entries)
    print('  catalogue: +%d (now %d)' % (ca, total))
    ok, msg = update_menu_file(PORTAL, entries)
    print('  apps-portal.html: %s | %s' % ('OK' if ok else 'FAIL', msg))
    ok2, msg2 = update_menu_file(MENU, entries)
    print('  menu.html: %s | %s' % ('OK' if ok2 else 'FAIL', msg2))
    m = mirror_files(entries)
    print('  mirrored to CCP2/apps: %d files' % m)


if __name__ == '__main__':
    main()
