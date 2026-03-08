# Curriculum Expert

AI-powered curriculum assistant for WeST multi-academy trust, with an integrated library of 149 interactive educational apps.

**Live:** https://www.curriculumexpert.co.uk

## Architecture

- **Backend:** Python/Flask with ChromaDB RAG pipeline
- **Frontend:** Static HTML/JS/CSS (no build step)
- **Deployment:** Railway (auto-deploy on push to `main`, Dockerfile-based)
- **LLM:** Anthropic Claude API (Haiku for short, Sonnet for medium, Opus for extended responses)

## Routes

| Route | File | Description |
|---|---|---|
| `/` | `curriculum-expert.html` | AI curriculum chat |
| `/apps` | `apps-portal.html` | Educational Apps portal |
| `/apps/<file>` | `apps/*.html` | Individual interactive apps |
| `/admin` | `admin.html` | Admin panel |
| `/feedback` | `feedback.html` | User feedback |
| `/dynamic` | — | 301 redirect to `/apps` (legacy) |

## Key Files

| File | Purpose |
|---|---|
| `rag_server.py` | Flask server — routes, RAG queries, LLM streaming, app catalogue injection |
| `rag_pipeline.py` | ChromaDB RAG indexing and retrieval from `knowledge/` docs |
| `usage_tracker.py` | Query usage logging |
| `app_tracker.py` | App usage analytics |
| `curriculum-agent-config.json` | System prompt configuration per subject |
| `edu-apps-catalogue.json` | Flat JSON catalogue of all 149 apps for LLM prompt injection |
| `apps-portal.html` | Educational Apps portal page |
| `apps/` | 149 standalone HTML educational apps |
| `knowledge/` | PDF/document files for RAG indexing |

## Educational Apps

149 interactive single-file HTML apps across 11 subjects:

- **Drama** (12 apps, EYFS–KS5) — Emotion masks, freeze frames, improvisation, devising, practitioners, theatre review
- **English** (12 apps, KS3–KS4) — Rhetoric, sentence structure, narrative perspective, etymology
- **Maths** (29 apps, EYFS–KS5) — Place value, fractions, algebra, calculus, statistics
- **Science** (12 apps, EYFS–KS5) — Cells, circuits, forces, periodic table, genetics
- **Geography** (12 apps, EYFS–KS5) — Maps, weather, rivers, plate tectonics, urbanisation
- **History** (12 apps, EYFS–KS5) — Timelines, sources, causation, historiography
- **MFL** (12 apps, EYFS–KS5) — Vocabulary, grammar, conjugation, pronunciation
- **RE** (12 apps, KS3–KS4) — Worldviews, ethics, sacred texts, moral philosophy
- **Behaviour & Pastoral** (12 apps) — De-escalation, restorative practice, emotional regulation
- **CPD & Staff Tools** (12 apps) — Observation feedback, Bloom's taxonomy, lesson planning
- **Data & Analytics** (12 apps) — Cohort tracking, grade boundaries, value-added analysis

## App Suggestion Feature

Users can suggest new apps via a modal form on the portal. Suggestions are saved to Firebase Firestore (`edu_app_suggestions` collection on `west-analytics-47c83`). The LLM can also suggest apps via prepopulated URL parameters.

## LLM Integration with Apps

For medium and extended responses, the app catalogue (~2,400 tokens) is injected into the system prompt. The LLM can:
- Link to relevant existing apps in its responses
- Suggest new apps via a prepopulated suggestion form URL

Markdown links in LLM responses are rendered as styled `.app-link` elements in the frontend.

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python rag_server.py

# Deploy (auto on push)
git push origin main
```

## Environment Variables (Railway)

- `ANTHROPIC_API_KEY` — Claude API key
- `CLOUD_MODE=true` — Enables cloud deployment paths
- `WORKSPACE_DIR=/app` — Working directory in container
