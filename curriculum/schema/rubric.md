# App quality ranking rubric (v1)

Every app in `../../apps/` is scored against four equally weighted dimensions,
each marked **1 to 5**, giving a total out of **20**. An agent reads the actual
HTML and JavaScript of the app before scoring.

## Dimensions

### 1. Pedagogical value (1 to 5)
Does it teach or assess something worthwhile, and well?
- **5:** sharp learning purpose, demands real thinking, misconception-aware, worth lesson time.
- **3:** sound but ordinary; a useful illustration with limited depth.
- **1:** no clear learning point, or pedagogically confused.

### 2. Interactivity and engagement (1 to 5)
Genuine manipulation and feedback, not a static display.
- **5:** rich interaction with immediate, meaningful feedback; strong pupil agency; real "wow" demo.
- **3:** some interaction but shallow, or feedback is thin.
- **1:** static page or trivial single click.

### 3. Curriculum alignment (1 to 5)
Maps onto the National Curriculum at the right pitch.
- **5:** bang-on an NC strand, correct key-stage pitch, content accurate.
- **3:** broadly relevant but loosely aligned or slightly mis-pitched.
- **1:** off-spec, wrong key stage, or contains content errors.

### 4. Technical and UX quality (1 to 5)
Robust, usable, accessible, well designed.
- **5:** polished, no bugs or console errors, responsive, accessible, age-appropriate, clean editorial design.
- **3:** works but rough edges (layout issues, minor bugs, dated styling).
- **1:** broken, buggy, or unusable.

## Output per app

- The four scores and the total out of 20.
- A **tier**: A (17 to 20), B (13 to 16), C (9 to 12), D (8 or below).
- A one to two sentence rationale.
- A short "fix-it" note for any dimension scoring under 3 (feeds future improvement work).

## How tiers are used

- **A / B:** keep; promote as flagship demos.
- **C:** improve; usually a quick win on one weak dimension.
- **D:** rebuild or retire candidates.

The backlog therefore covers improving existing apps, not only building new ones.

## Running the ranking

Run in **per-subject batches** to stay within subscription limits: one agent
per app, scored against this rubric, results appended to
`app-rankings.json` and summarised in `app-rankings.md`.
