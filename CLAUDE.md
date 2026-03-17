# Curriculum Expert — Claude Instructions

## Overview
- **Live**: www.curriculumexpert.co.uk
- **Hosting**: Railway (Docker, auto-deploys on push to main)
- **Stack**: Python 3.11 + Flask + Gunicorn + ChromaDB
- **Key file**: `rag_server.py` — all Flask routes, RAG pipeline, streaming responses

## Educational Apps (CRITICAL)

### Dual Menu System
The edu apps menu exists in TWO places that MUST stay in sync:

| File | Location | Purpose |
|---|---|---|
| `apps-portal.html` | This repo | Live site menu at curriculumexpert.co.uk/apps |
| `C:\CCP2\menu.html` | Parent CCP2 repo | Local dashboard menu |

Both contain an identical `eduApps[]` JavaScript array. **When you add or modify apps in either file, update the other.**

### Adding a New Edu App
1. Create the HTML file in `C:\CCP2\apps/` (source of truth)
2. Copy it to `curriculum-expert-web/apps/` (for Railway deployment)
3. Add the app entry to `eduApps[]` in BOTH:
   - `C:\CCP2\menu.html`
   - `curriculum-expert-web/apps-portal.html`
4. Optionally update `edu-apps-catalogue.json`
5. Commit and push this repo to deploy

### Emoji in HTML
- In raw HTML content: use `&#xHEX;` entities (e.g. `&#x1F30A;`)
- Inside `<script>` JS strings: use `\u{HEX}` (e.g. `\u{1F30A}`)
- NEVER use `\u{HEX}` in HTML outside script tags — browsers render it as literal text

### File Structure
- `apps/` — 200+ self-contained HTML edu app files
- `apps-portal.html` — the apps menu page (served at `/apps`)
- `edu-apps-catalogue.json` — JSON catalogue of all apps (197 entries)
- `rag_server.py` — Flask app serving everything

### Routes
- `/apps` — apps portal/menu (serves `apps-portal.html`)
- `/apps/<filename>` — individual app files
- `/dynamic` — redirects to `/apps` (backwards compatibility)
- `/dynamic/apps/<filename>` — redirects to `/apps/<filename>`

## Deploy
`git push` to main triggers Railway auto-deploy (2-3 min). **Never push until reviewed.**

## Never Commit
`.env`, API keys, `__pycache__/`
