# Curriculum Expert — Claude Instructions

## Overview
- **Live**: www.curriculumexpert.co.uk
- **Hosting**: Railway (Docker, auto-deploys on push to main)
- **Stack**: Python 3.11 + Flask + Gunicorn + ChromaDB
- **Key file**: `rag_server.py` — all Flask routes, RAG pipeline, streaming responses
- **Scale**: 704 self-contained HTML edu apps in `apps/`, mapped to the full England
  National Curriculum (100% strand coverage across 17 subjects/areas).

## Educational Apps (CRITICAL)

### Dual Menu System
The edu apps menu exists in TWO places that MUST stay in sync:

| File | Location | Purpose |
|---|---|---|
| `apps-portal.html` | This repo | Live site menu at curriculumexpert.co.uk/apps |
| `menu.html` | Parent CCP2 repo (`ccp2-dashboard`) | Local dashboard menu |

Both contain an identical `eduApps[]` JavaScript array. **When you add or modify apps,
update both.** The eduApps array shape: most subject groups use
`{ subject, icon, color, bg, keystages:[{label, apps:[{name,desc,file}]}] }`; the
flat groups (`Behaviour & Pastoral`, `CPD & Staff Tools`, `Data & Analytics`) use
`{ subject, icon, color, bg, apps:[...] }` (no keystages).

### Use the pipeline, do not hand-edit menus for batches
Reusable tooling lives in `curriculum/` (see below). To add apps in bulk:
`curriculum/integrate_apps.py` updates the catalogue + BOTH eduApps menus + mirrors
files to `CCP2/apps`, with **Node-eval validation and auto-revert** so a bad edit
can't corrupt a menu. It handles keystage groups, flat-apps groups, and creating
new subject groups/keystages. Hand-editing the 700-app arrays is error-prone; prefer
the integrator.

### Adding a New Edu App (single)
1. Create the HTML file in `apps/`.
2. Run `python3 ../scripts/inject_demo_notice.py` (demo-data banner+modal, idempotent)
   and `python3 curriculum/inject_nc_link.py` (the "Curriculum mapping" chip, idempotent).
3. Add it to `eduApps[]` in BOTH `apps-portal.html` and parent `menu.html`
   (or run `curriculum/integrate_apps.py` with a one-entry list).
4. Update `edu-apps-catalogue.json`. Mirror the file to `CCP2/apps`.
5. Commit and push to deploy.

### Mandatory blocks on EVERY app (two idempotent sentinels)
- `<!-- DEMO-DATA-NOTICE v1 -->` — amber "Demonstration data only" banner + once-per-session
  acknowledgement modal. Injector: `../scripts/inject_demo_notice.py`.
- `<!-- NC-MAP-LINK v1 -->` — a small fixed "Curriculum mapping" chip that deep-links to the
  app's place in `/curriculum-mapping` (derives the filename at runtime). Injector:
  `curriculum/inject_nc_link.py`. Both injectors sweep `apps/` AND `CCP2/apps/`, skip
  `data-daily-cashflow.html` and `data-idsr-analyser.html` (real uploaded data).

### Text-to-speech (audio apps)
Speaking apps must use the shared `westTTS` helper (`curriculum/west-tts-snippet.html`,
sentinel `<!-- WEST-TTS v1 -->`), injected after `<body>`. It prefers the **Google UK
English** neural voice (free, network) and falls back to the best local voice offline.
Speak via `window.westTTS.speak(text, {lang, rate, pitch})`. Never call SpeechSynthesis
directly. The build pipeline bakes this in automatically.

### Emoji in HTML
- In raw HTML content: use `&#xHEX;` entities (e.g. `&#x1F30A;`)
- Inside `<script>` JS strings: use `\u{HEX}`
- NEVER use `\u{HEX}` in HTML outside script tags — browsers render it literally

## Curriculum coverage & mapping system (`curriculum/`)

Data-driven; pages regenerate from JSON, so update data then regenerate.

- **`schema/coverage-master.json`** — the master schema: per subject → keyStages → strands,
  each with coverage (covered/partial/none) and the app files that teach it.
- **`schema/domain-grouping.json`** — strands grouped into NC domains per subject (so the
  mapping page can show domain ABOVE key stage).
- **`schema/app-rankings.json` / `.md`** — every app scored against the v1 rubric
  (4 dimensions /20, tiers A–D). `schema/rubric.md` defines it.
- **`curriculum-map.html`** (route `/curriculum-map`) — coverage view (covered/partial/gap
  badges + app links). Generator: `gen_map_page.py`.
- **`curriculum-mapping.html`** (route `/curriculum-mapping`) — the friendly public view:
  Subject → domain → strand (KS tag + assessment ref) → app links; styled like the apps
  portal; deep-links via `?app=<file>`. Generator: `gen_mapping_page.py`.
- **`pipeline.py`** — brief-gen → build-script → post-integrate for bulk app builds.
- **`integrate_apps.py`**, **`update_coverage.py`**, **`inject_nc_link.py`**, **`BUILD-QUEUE.md`**.

After changing coverage data or adding apps, regenerate both pages:
`python3 curriculum/gen_map_page.py && python3 curriculum/gen_mapping_page.py`.

## Routes
- `/` — main Curriculum Expert (RAG) app
- `/apps`, `/apps/` — apps portal (`apps-portal.html`)
- `/apps/<filename>` — individual app files
- `/curriculum-map`, `/curriculum-map/` — coverage map (`curriculum-map.html`)
- `/curriculum-mapping`, `/curriculum-mapping/` — NC mapping page (`curriculum-mapping.html`)
- `/dynamic`, `/dynamic/apps/<filename>` — backwards-compat redirects

## Dockerfile gotcha (IMPORTANT)
The Dockerfile **COPYs root HTML files individually** (not a wildcard). Any NEW root-level
page (like `curriculum-map.html`, `curriculum-mapping.html`) MUST get its own `COPY x.html .`
line or the route 500s in production (the file won't be in the image). `apps/` is copied as
a directory, so new apps are fine.

## apps-portal.html notes
- Header: two nav-link buttons (Educational Apps, Curriculum Mapping; current page gets
  `.active`), then a divider, then the distinct orange `.ce-link` Curriculum Expert pill.
  Stat boxes (`#stat-row`) show Apps / Subjects / Key stages.
- The "Suggest an App" feature was **removed** (it collected name/email/school via Firebase);
  do not reintroduce visitor-data collection without explicit sign-off.

## File Structure
- `apps/` — 704 self-contained HTML edu app files
- `apps-portal.html` — apps menu page (served at `/apps`)
- `curriculum-map.html`, `curriculum-mapping.html` — coverage + mapping pages
- `edu-apps-catalogue.json` — catalogue of all apps (704 entries)
- `curriculum/` — coverage schema, rankings, page generators, build pipeline
- `rag_server.py` — Flask app serving everything

## Deploy
`git push` to main triggers Railway auto-deploy (2-3 min). **Never push until reviewed.**
After a new root HTML page, confirm its Dockerfile COPY line. Verify live with
`curl -sI https://www.curriculumexpert.co.uk/<route>` (write to a temp file then grep —
piping curl through `grep -c` can return blank counts in zsh).

## Never Commit
`.env`, API keys, `__pycache__/`, `venv/`
