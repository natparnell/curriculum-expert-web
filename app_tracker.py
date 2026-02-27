#!/usr/bin/env python3
"""
App-level usage tracker for Curriculum Expert.

Logs each query to app_usage.jsonl (one JSON object per line) and
provides aggregation for the /admin dashboard.
"""

import json
import os
import threading
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

APP_DIR = Path(__file__).parent

# DATA_DIR env var lets Railway point this at a persistent volume.
# Falls back to the app directory for local development.
_data_dir = Path(os.environ.get('DATA_DIR', str(APP_DIR)))
_data_dir.mkdir(parents=True, exist_ok=True)
LOG_PATH = _data_dir / "app_usage.jsonl"

_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_query(subject, question, length, has_file, cite_thinkers,
              model, duration_ms, response_chars, success=True, endpoint='ask-stream'):
    """Append a query event to app_usage.jsonl."""
    entry = {
        "type": "query",
        "ts": datetime.utcnow().isoformat() + "Z",
        "subject": subject or "unknown",
        "question": (question or "")[:200],
        "length": length or "medium",
        "has_file": bool(has_file),
        "cite_thinkers": bool(cite_thinkers),
        "model": model or "unknown",
        "duration_ms": int(duration_ms) if duration_ms is not None else None,
        "response_chars": int(response_chars) if response_chars is not None else None,
        "success": success,
        "endpoint": endpoint,
    }
    _append(entry)


def log_infographic(question, subject=None, duration_ms=None, success=True):
    """Append an infographic generation event to app_usage.jsonl."""
    entry = {
        "type": "infographic",
        "ts": datetime.utcnow().isoformat() + "Z",
        "subject": subject or "unknown",
        "question": (question or "")[:200],
        "duration_ms": int(duration_ms) if duration_ms is not None else None,
        "success": success,
    }
    _append(entry)


def log_upload(filename, word_count, subject=None):
    """Append a file upload event to app_usage.jsonl."""
    entry = {
        "type": "upload",
        "ts": datetime.utcnow().isoformat() + "Z",
        "filename": filename or "",
        "word_count": int(word_count) if word_count else 0,
        "subject": subject or "unknown",
    }
    _append(entry)


def _append(entry):
    try:
        with _write_lock:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[app_tracker] write failed: {e}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def get_stats():
    """Read and aggregate app_usage.jsonl. Returns dict for /admin/stats."""
    entries = _read_all()
    queries      = [e for e in entries if e.get("type") == "query"]
    uploads      = [e for e in entries if e.get("type") == "upload"]
    infographics = [e for e in entries if e.get("type") == "infographic"]

    today = datetime.utcnow().strftime("%Y-%m-%d")
    week_ago = (datetime.utcnow() - timedelta(days=6)).strftime("%Y-%m-%d")

    today_count = sum(1 for q in queries if q.get("ts", "")[:10] == today)
    week_count  = sum(1 for q in queries if q.get("ts", "")[:10] >= week_ago)

    # --- By subject ---
    by_subject = defaultdict(int)
    for q in queries:
        by_subject[q.get("subject", "unknown")] += 1

    # --- By day (last 14 days) ---
    by_day_map = defaultdict(int)
    for q in queries:
        d = q.get("ts", "")[:10]
        if d:
            by_day_map[d] += 1
    by_day = []
    for i in range(13, -1, -1):
        d = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        by_day.append({"date": d, "count": by_day_map.get(d, 0)})

    # --- Length mode ---
    by_length = defaultdict(int)
    for q in queries:
        by_length[q.get("length", "medium")] += 1

    # --- Thinkers toggle ---
    thinkers_on  = sum(1 for q in queries if q.get("cite_thinkers", True))
    thinkers_off = len(queries) - thinkers_on

    # --- By model ---
    by_model = defaultdict(int)
    for q in queries:
        by_model[q.get("model", "unknown")] += 1

    # --- Avg duration (successful queries only) ---
    durations = [q["duration_ms"] for q in queries
                 if q.get("duration_ms") and q.get("success")]
    avg_duration = int(sum(durations) / len(durations)) if durations else None

    # --- With file ---
    with_file = sum(1 for q in queries if q.get("has_file"))

    # --- Recent activity (queries + infographics combined, last 30) ---
    activity = sorted(queries + infographics, key=lambda x: x.get("ts", ""), reverse=True)[:30]

    return {
        "summary": {
            "total":          len(queries),
            "today":          today_count,
            "this_week":      week_count,
            "uploads":        len(uploads),
            "with_file":      with_file,
            "infographics":   len(infographics),
            "avg_duration_ms": avg_duration,
        },
        "by_subject": dict(sorted(by_subject.items(), key=lambda x: -x[1])),
        "by_day":     by_day,
        "by_length":  dict(by_length),
        "thinkers":   {"on": thinkers_on, "off": thinkers_off},
        "by_model":   dict(sorted(by_model.items(), key=lambda x: -x[1])),
        "recent":     activity,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def _read_all():
    entries = []
    try:
        if LOG_PATH.exists():
            with open(LOG_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    except Exception:
        pass
    return entries


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = get_stats()
    import json as _json
    print(_json.dumps(data, indent=2))
