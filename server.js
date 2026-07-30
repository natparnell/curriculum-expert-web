const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 8080;

app.set('trust proxy', 1);
app.use(cors());
app.use(express.json({ limit: '16kb' }));

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const APP_PASSWORD = process.env.APP_PASSWORD;

// --- Abuse protection for the paid Anthropic endpoint ---
// Without this, /api/chat was public + unmetered: anyone could run unlimited
// LLM calls billed to the Anthropic key (cost-based DoS).
const MAX_MESSAGE_CHARS = 4000;      // cap prompt size (cost + latency)
const PER_IP_WINDOW_MS = 60 * 1000;  // per-IP window: 1 minute
const PER_IP_MAX = 10;               // max requests per IP per window
const DAILY_CAP = 500;               // global safety cap across all users per day

const ipHits = new Map();            // ip -> { count, resetAt }
let dailyCount = 0;
let dailyResetAt = Date.now() + 24 * 60 * 60 * 1000;

function withinLimits(req, res) {
  const now = Date.now();
  if (now > dailyResetAt) { dailyCount = 0; dailyResetAt = now + 24 * 60 * 60 * 1000; }
  if (dailyCount >= DAILY_CAP) {
    res.status(429).json({ error: 'Daily request limit reached. Please try again tomorrow.' });
    return false;
  }
  const ip = req.ip || (req.connection && req.connection.remoteAddress) || 'unknown';
  let h = ipHits.get(ip);
  if (!h || now > h.resetAt) { h = { count: 0, resetAt: now + PER_IP_WINDOW_MS }; ipHits.set(ip, h); }
  if (h.count >= PER_IP_MAX) {
    res.status(429).json({ error: 'Too many requests, please slow down.' });
    return false;
  }
  h.count++;
  dailyCount++;
  return true;
}

app.use(express.static(path.join(__dirname, 'public')));

app.post('/api/chat', async (req, res) => {
  // Fail closed: if no password is configured the endpoint is unusable, never open.
  if (!APP_PASSWORD) {
    return res.status(503).json({ error: 'Service not configured.' });
  }
  if ((req.get('x-app-password') || '') !== APP_PASSWORD) {
    return res.status(401).json({ error: 'Unauthorised.' });
  }
  if (!withinLimits(req, res)) return;

  const { message } = req.body || {};
  if (typeof message !== 'string' || !message.trim()) {
    return res.status(400).json({ error: 'Message required.' });
  }
  if (message.length > MAX_MESSAGE_CHARS) {
    return res.status(413).json({ error: `Message too long (max ${MAX_MESSAGE_CHARS} characters).` });
  }

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [{ role: 'user', content: message }]
      })
    });
    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error('Chat error:', err);
    res.status(500).json({ error: 'Request failed' });
  }
});

app.listen(PORT, () => {
  console.log(`Curriculum Expert listening on http://localhost:${PORT}`);
});
