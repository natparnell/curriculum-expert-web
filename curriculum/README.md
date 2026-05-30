# Curriculum coverage and app quality

This folder maps the **England National Curriculum** against the interactive
teaching apps in `../apps/`, shows where coverage is strong and where the gaps
are, and ranks the existing apps by quality. It is the planning base for
building more excellent teacher demos over time.

## What lives here

```
curriculum/
  README.md                     this file
  subjects/                     one markdown file per NC subject:
                                broad strands x key stage, with apps mapped and gaps flagged
  schema/
    coverage-master.json        machine-readable cross-reference (the schema)
    coverage-report.md          human-readable gap analysis and headline numbers
    rubric.md                   the app-quality ranking rubric (v1)
    app-rankings.json           per-app scores (built in staggered batches)
    app-rankings.md             ranked tables, per subject and overall
    generation-backlog.md       prioritised "ready to build" list (NOT yet built)
```

## Scope decisions (agreed 29/05/2026)

- **Subject scope:** the full England National Curriculum, including subjects we
  currently build nothing for (Art and Design, Music, Computing, Design and
  Technology, Physical Education, Citizenship), so whole-subject white space is
  visible.
- **Source:** hybrid. Reuse the structured NC summaries already in
  `../knowledge/*/01_national_curriculum/` for the 7 subjects that have them;
  fetch fresh DfE material for the rest.
- **Granularity:** broad strand level, not individual statutory statements.
- **Phases:** EYFS through KS5, covering each subject where it is actually taught
  (some phases are non-statutory or exam-board defined; this is noted per cell).
- **Not pupil curricula:** `CPD & Staff Tools` and `Data & Analytics` apps are
  teacher/admin tools. They are excluded from the coverage map but are still
  scored in the ranking.
- **Behaviour and Pastoral:** no National Curriculum exists. It is mapped
  alongside statutory PSHE/RSHE as a WeST-defined strand set of comparable size.

## How it is produced

- **Coverage map:** one agent per subject builds the strand x key-stage grid and
  cross-references the existing apps. A consolidation step writes the master
  schema and report.
- **Ranking:** one agent per app scores it against `schema/rubric.md`. Run in
  per-subject batches across sessions to stay within subscription limits.
