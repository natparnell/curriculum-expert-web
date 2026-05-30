export const meta = {
  name: 'app-quality-ranking',
  description: 'Score edu apps against the v1 rubric, one agent per app',
  phases: [{ title: 'Score', detail: 'one agent reads and scores each app' }],
}

// args = array of app entries: { file, name, ks, desc }  (one batch, usually one subject)
// Pass via Workflow({ scriptPath: 'curriculum/ranking-workflow.js', args: [ ...apps ] })
const APPS = Array.isArray(args) ? args : []

const SCHEMA = {
  type: 'object',
  required: ['file', 'pedagogy', 'interactivity', 'alignment', 'technical', 'total', 'tier', 'rationale'],
  properties: {
    file: { type: 'string' },
    pedagogy: { type: 'integer', minimum: 1, maximum: 5 },
    interactivity: { type: 'integer', minimum: 1, maximum: 5 },
    alignment: { type: 'integer', minimum: 1, maximum: 5 },
    technical: { type: 'integer', minimum: 1, maximum: 5 },
    total: { type: 'integer', minimum: 4, maximum: 20 },
    tier: { type: 'string', enum: ['A', 'B', 'C', 'D'] },
    rationale: { type: 'string', description: 'one to two sentences' },
    fixIt: { type: 'string', description: 'short improvement note for any dimension under 3; empty if none' },
  },
}

function prompt(a) {
  return `Score one interactive teaching app against the WeST app-quality rubric (v1). UK English.

App file: \`apps/${a.file}\`
Catalogue name: ${a.name}
Catalogue key stage: ${a.ks}
Catalogue description: ${a.desc}

**Read the actual file** at \`apps/${a.file}\` - the HTML AND the JavaScript - before scoring. Judge what it really does, not what the description claims.

Score each dimension 1 to 5 (integers):
1. **pedagogy** - clear learning purpose, cognitive demand, misconception-aware, worth lesson time.
2. **interactivity** - genuine manipulation and immediate meaningful feedback vs static display.
3. **alignment** - maps onto the England National Curriculum at the right key-stage pitch; content accurate.
4. **technical** - robust (no obvious bugs/console errors), responsive, accessible, age-appropriate, clean design.

Then:
- total = sum of the four.
- tier = A (17-20), B (13-16), C (9-12), D (8 or below).
- rationale = one to two sentences justifying the scores.
- fixIt = a short, specific improvement note for any dimension scoring under 3 (empty string if all are 3+).

Be a discerning critic: most apps should not be 5s. Reserve 5 for genuinely excellent work. Return only the structured object.`
}

phase('Score')

const results = await parallel(
  APPS.map((a) => () => agent(prompt(a), { label: `score:${a.file}`, phase: 'Score', schema: SCHEMA }))
)

const ok = results.filter(Boolean).sort((x, y) => y.total - x.total)
log(`Scored ${ok.length}/${APPS.length} apps`)
return ok
