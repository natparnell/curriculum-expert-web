# Agentic build job list

The pipeline builds one app per uncovered ("none") strand, per subject, then
integrates (inject demo notice, update coverage, catalogue, both eduApps menus,
mirror to CCP2/apps, regenerate the map). Run order and status:

| # | Subject / area | Type | Status |
|---|---|---|---|
| 1 | English | NC active | DONE (17 apps) |
| 2 | Mathematics | NC active | DONE (8 apps) |
| 3 | Science | NC active | DONE |
| 4 | History | NC active | DONE |
| 5 | Geography | NC active | DONE |
| 6 | Religious Education | NC active | DONE |
| 7 | Modern Foreign Languages | NC active | DONE |
| 8 | Drama | NC active | DONE |
| 9 | PSHE, RSHE and Behaviour/Pastoral | non-NC framework | DONE |
| 10 | Art and Design | NC new | DONE |
| 11 | Music | NC new | DONE |
| 12 | Computing | NC new | DONE |
| 13 | Design and Technology | NC new | DONE |
| 14 | Physical Education | NC new | DONE |
| 15 | Citizenship | NC new | DONE |
| 16 | CPD & Staff Tools | non-NC framework | DONE |
| 17 | Data & Analytics | non-NC framework | DONE |

## How to run the next batch

```
python3 curriculum/pipeline.py briefs '<MasterSubject>' ['<MasterSubject>' ...]
cp /tmp/build_briefs.json curriculum/_briefs_<label>.json
python3 curriculum/pipeline.py script curriculum/_briefs_<label>.json curriculum/run-build-<label>.js
# launch the workflow on that script, wait, then:
python3 curriculum/pipeline.py post curriculum/_briefs_<label>.json <workflow_output_file>
```

Keep going until the usage limit halts agent calls or the queue is empty.
Nothing is pushed or deployed without Nat's review.

## QUEUED: P2 partials (146 apps) — deepen thinly-covered strands

Builds one app per **partial** strand across all 12 subjects with partials
(English 10, Maths 13, Science 26, History 12, Geography 13, RE 15, MFL 14,
Drama 13, Citizenship 3, PSHE 13, CPD 5, Data 9). Closes the amber on the map
toward ~100%. Briefs: `curriculum/_briefs_p2.json`; script: `curriculum/run-build-p2.js`.
**Hold until the re-rank + lift pass finishes**, then launch and integrate with:
`python3 curriculum/pipeline.py post curriculum/_briefs_p2.json <output>`

## Final phase (only after #1-17 finish, if usage limit remains)

18. Re-run the ranking across the WHOLE library (regenerate app-inventory from the
    catalogue first), then patch any app with a sub-3 dimension or low total to lift
    its score (library-wide C-tier lift), preserving the demo-data block. Also finish
    earlier deferred fixes (e.g. english-unreliable-narrator drag lacks a non-Pointer
    fallback) and any original fix-it notes only the C-tier received.
