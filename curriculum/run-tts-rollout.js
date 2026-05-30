export const meta = {
  name: 'tts-google-rollout',
  description: 'Upgrade all speech apps to prefer Google UK English neural voice via shared westTTS helper',
  phases: [{ title: 'TTS', detail: 'inject westTTS and route speech through it' }],
}
const FILES = ["behaviour-perspective-swap.html", "computing-ks2-decompose-it.html", "drama-eyfs-story-stage.html", "drama-ks2-improvisation-spinner.html", "english-eyfs-magic-letter-tracer.html", "english-eyfs-rhyming-pairs.html", "english-ks1-phonics-blender.html", "english-ks1-sentence-builder.html", "french_verb_conjugation_explorer.html", "french_word_level_planner.html", "geography-eyfs-my-world-map.html", "geography-ks1-seaside-fieldwork.html", "geography-ks2-biomes-explorer.html", "geography-ks2-map-skills.html", "geography-ks5-global-governance.html", "geography-weather-systems.html", "history-eyfs-family-timeline.html", "history-ks1-famous-people.html", "history-ks2-civilisation-compare.html", "history-ks5-historiography-debate.html", "maths-ks2-coordinates.html", "mfl-eyfs-counting-languages.html", "mfl-eyfs-hello-world.html", "mfl-ks1-action-song-builder.html", "mfl-ks1-colours-languages.html", "mfl-ks5-essay-connectives.html", "mfl-ks5-literary-analysis.html", "mfl-listening-discrimination.html", "mfl-sentence-complexity-ladder.html", "re-eyfs-celebrations.html", "re-eyfs-special-places.html", "re-ks1-belonging-signs.html", "re-ks1-religious-stories.html", "re-ks2-prayer-practices.html", "re-ks5-medical-ethics.html", "science-ks1-materials-sorter.html", "science-ks5-organic-mechanisms.html"];
const SNIPPET = "<!-- WEST-TTS v1 : shared voice selection. Prefers Google UK English (female) / matching\n     Google voice per language (free network neural), falls back to best local voice offline. -->\n<script>\nwindow.westTTS = (function () {\n  var synth = window.speechSynthesis, voices = [];\n  function load() { if (synth) voices = synth.getVoices() || []; }\n  if (synth) { load(); try { synth.addEventListener('voiceschanged', load); } catch (e) {} }\n  function score(v, lang) {\n    var base = lang.slice(0, 2).toLowerCase(), s = 0, n = (v.name || '').toLowerCase();\n    if ((v.lang || '').toLowerCase() === lang.toLowerCase()) s += 25;\n    else if ((v.lang || '').toLowerCase().slice(0, 2) === base) s += 12;\n    if (n.indexOf('google') >= 0) s += 40;           // chosen preference: Google neural\n    if (v.localService === false) s += 18;           // network neural quality\n    if (base === 'en' && n.indexOf('uk english female') >= 0) s += 30;\n    if (base === 'en' && (v.lang || '').toLowerCase() === 'en-gb') s += 8;\n    if (/natural|neural|enhanced|premium|siri/.test(n)) s += 8;\n    return s;\n  }\n  function pick(lang) {\n    lang = lang || 'en-GB';\n    var base = lang.slice(0, 2).toLowerCase();\n    var pool = voices.filter(function (v) { return (v.lang || '').toLowerCase().slice(0, 2) === base; });\n    if (!pool.length) pool = voices.slice();\n    pool.sort(function (a, b) { return score(b, lang) - score(a, lang); });\n    return pool[0] || null;\n  }\n  function speak(text, opts) {\n    if (!synth || !text) return null;\n    opts = opts || {};\n    synth.cancel();\n    var u = new SpeechSynthesisUtterance(text);\n    u.lang = opts.lang || 'en-GB';\n    var v = pick(u.lang); if (v) u.voice = v;\n    u.rate = opts.rate != null ? opts.rate : 0.9;\n    u.pitch = opts.pitch != null ? opts.pitch : 1.1;\n    synth.speak(u);\n    return u;\n  }\n  return { speak: speak, pick: pick, cancel: function () { if (synth) synth.cancel(); }, ready: function () { return !!synth; } };\n})();\n</script>\n";
const SCHEMA = {
  type:'object', required:['file','changes','demoNoticePreserved'],
  properties:{ file:{type:'string'}, changes:{type:'array',items:{type:'string'}},
    helperInjected:{type:'boolean'}, demoNoticePreserved:{type:'boolean'}, risks:{type:'string'} },
}
function prompt(f){
  return `Upgrade the text-to-speech VOICE in ONE existing app so it prefers the high-quality Google UK English neural voice. UK English; no em dashes.

File to edit IN PLACE: \`apps/${f}\`

This app uses the Web Speech API (window.speechSynthesis / SpeechSynthesisUtterance). The problem: it sets a language but never selects a .voice, so the browser uses its poor default compact voice. Fix it to use a shared helper that picks the best available voice (Google UK English Female first, then any Google / network neural voice, falling back to the best local voice offline).

## Steps
1. If the page does not already contain "WEST-TTS v1", inject this EXACT block once, immediately after the <body> tag (or just before </script> of head is wrong; put it right after <body>):
${SNIPPET}
2. For EVERY utterance the app speaks, ensure the chosen voice is applied. Either:
   (a) set the voice on the existing utterance: after the app creates its SpeechSynthesisUtterance and sets its .lang, add a line that sets \`<utter>.voice = window.westTTS.pick(<utter>.lang || 'en-GB')\` (guard with if(window.westTTS)); OR
   (b) replace the app's speak path with \`window.westTTS.speak(text, { lang: <existingLang>, rate: <existingRate>, pitch: <existingPitch> })\`.
   PRESERVE the app's existing lang codes, rate and pitch exactly (do not change the pace or pitch the app already uses). For non-English apps (fr-FR, de-DE, es-ES etc.) pass that same lang so the helper picks the matching Google voice.
3. Voices load asynchronously; the helper already handles voiceschanged, so no extra wiring is needed, but if the app caches voices itself at load, make sure speaking still works after voices populate (the helper re-reads voices each speak).
4. Do not change anything else: no content, layout, pace, or behaviour beyond voice selection. Keep the \`<!-- DEMO-DATA-NOTICE v1 -->\` block untouched (demoNoticePreserved=true).
5. Verify: balanced <script> tags, ends with </html>, the WEST-TTS block present once, no external/network resource added (the helper uses only the browser API).

Return the structured summary.`
}
phase('TTS')
const results = await parallel(FILES.map((f)=>()=>agent(prompt(f),{label:`tts:${f}`,phase:'TTS',schema:SCHEMA})))
const ok=results.filter(Boolean)
log(`TTS upgraded ${ok.length}/${FILES.length}`)
return ok
