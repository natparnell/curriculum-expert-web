#!/usr/bin/env python3
"""
Usage Tracker — Aggregates model usage from OpenClaw JSONL session & cron logs.

Scans:
  - /home/node/.openclaw/agents/main/sessions/*.jsonl  (interactive sessions)
  - /home/node/.openclaw/cron/runs/*.jsonl              (cron job runs)

Returns aggregated token counts by model, provider, and time period.
Cached for 5 minutes to avoid re-parsing on every request.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

SESSIONS_DIR = Path("/home/node/.openclaw/agents/main/sessions")
CRON_DIR = Path("/home/node/.openclaw/cron/runs")

# Cost per 1M tokens (USD) — looked up Feb 2026
# Relative costs use Kimi K2.5 as the £1× baseline
MODEL_COSTS = {
    "kimi-k2.5":        {"input": 0.60, "output": 3.00, "cache_read": 0.15},          # £1×
    "claude-opus-4-5":  {"input": 5.00, "output": 25.00, "cache_read": 0.50},         # £8.3×
    "claude-sonnet-4-5":{"input": 3.00, "output": 15.00, "cache_read": 0.30},         # £5×
    "claude-haiku-3-5": {"input": 0.80, "output": 4.00, "cache_read": 0.08},          # £1.3×
    "sonar-reasoning":  {"input": 1.00, "output": 5.00, "cache_read": 0.0},           # £1.7×
    "sonar-pro":        {"input": 2.00, "output": 8.00, "cache_read": 0.0},           # £2.8×
    "gpt-4o":           {"input": 2.50, "output": 10.00, "cache_read": 0.0},          # £3.5×
    "whisper-1":        {"input": 0.0, "output": 0.0, "cache_read": 0.0, "per_min": 0.006},  # $0.006/min
    "text-embedding-3-small": {"input": 0.02, "output": 0.0, "cache_read": 0.0},     # embeddings
}

# Friendly display names
MODEL_NAMES = {
    "kimi-k2.5": "Kimi K2.5",
    "claude-opus-4-5": "Claude Opus 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "claude-haiku-3-5": "Claude Haiku 3.5",
    "sonar-reasoning": "Sonar Reasoning",
    "sonar-pro": "Sonar Pro",
    "gpt-4o": "GPT-4o",
    "whisper-1": "Whisper (Voice)",
    "text-embedding-3-small": "Embeddings",
}

# Cache
_usage_cache = None
_usage_cache_time = 0
CACHE_TTL = 300  # 5 minutes


def _normalise_model(model_id):
    """Normalise model ID variations to a canonical key."""
    if not model_id:
        return None
    m = model_id.lower().strip()
    # Strip provider prefix (e.g. "anthropic/claude-opus-4-5")
    if "/" in m:
        m = m.split("/", 1)[1]
    # Strip @profile suffix (e.g. "claude-sonnet-4-5-20250929@anthropic:claude-max")
    if "@" in m:
        m = m.split("@", 1)[0]
    # Skip internal/system models
    if m in ("delivery-mirror", "default"):
        return None
    if "kimi" in m or "k2.5" in m or "k2-5" in m:
        return "kimi-k2.5"
    if m == "opus" or ("opus" in m and ("4-5" in m or "4.5" in m)):
        return "claude-opus-4-5"
    if "sonnet" in m:
        return "claude-sonnet-4-5"
    if "haiku" in m:
        return "claude-haiku-3-5"
    if "sonar-pro" in m:
        return "sonar-pro"
    if "sonar" in m:
        return "sonar-reasoning"
    if "gpt-4o" in m:
        return "gpt-4o"
    if "whisper" in m:
        return "whisper-1"
    if "embedding" in m:
        return "text-embedding-3-small"
    return model_id


def _parse_session_file(filepath):
    """Parse a single JSONL file, yielding (model, date_str, input_tokens, output_tokens, cache_read) tuples."""
    results = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Session messages have usage nested in rec.message.usage
                msg = rec.get("message", {})
                usage = msg.get("usage", {})
                model = msg.get("model") or rec.get("model")
                ts = rec.get("timestamp") or msg.get("timestamp")

                if not model or not usage:
                    # Cron run summaries have usage at top level
                    if rec.get("usage") and rec.get("model"):
                        usage = rec["usage"]
                        model = rec["model"]
                        ts = rec.get("ts")
                    else:
                        continue

                model = _normalise_model(model)
                if not model:
                    continue

                # Extract token counts — handle both formats
                inp = usage.get("input_tokens") or usage.get("input", 0)
                out = usage.get("output_tokens") or usage.get("output", 0)
                cache = usage.get("cacheRead", 0) or usage.get("cache_read", 0)

                # Parse date
                date_str = None
                if isinstance(ts, str):
                    try:
                        date_str = ts[:10]  # "2026-02-21T..."
                    except Exception:
                        pass
                elif isinstance(ts, (int, float)):
                    if ts > 1e12:
                        ts = ts / 1000  # epoch ms → s
                    try:
                        date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                    except Exception:
                        pass

                if not date_str:
                    # Fallback: use file mtime
                    try:
                        date_str = datetime.utcfromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d")
                    except Exception:
                        date_str = "unknown"

                results.append((model, date_str, inp, out, cache))
    except Exception:
        pass
    return results


def aggregate_usage():
    """Aggregate all usage data from JSONL logs. Returns dict with models, daily, totals."""
    global _usage_cache, _usage_cache_time

    now = time.time()
    if _usage_cache is not None and (now - _usage_cache_time) < CACHE_TTL:
        return _usage_cache

    # Collect all entries
    all_entries = []
    for d in [SESSIONS_DIR, CRON_DIR]:
        if d.exists():
            for f in d.glob("*.jsonl"):
                all_entries.extend(_parse_session_file(f))

    # Aggregate by model (total)
    model_totals = defaultdict(lambda: {"input": 0, "output": 0, "cache_read": 0, "calls": 0})
    # Aggregate by model + date
    model_daily = defaultdict(lambda: defaultdict(lambda: {"input": 0, "output": 0, "cache_read": 0, "calls": 0}))
    # Aggregate by date (all models)
    daily_totals = defaultdict(lambda: {"input": 0, "output": 0, "cache_read": 0, "calls": 0})

    for model, date_str, inp, out, cache in all_entries:
        model_totals[model]["input"] += inp
        model_totals[model]["output"] += out
        model_totals[model]["cache_read"] += cache
        model_totals[model]["calls"] += 1

        model_daily[model][date_str]["input"] += inp
        model_daily[model][date_str]["output"] += out
        model_daily[model][date_str]["cache_read"] += cache
        model_daily[model][date_str]["calls"] += 1

        daily_totals[date_str]["input"] += inp
        daily_totals[date_str]["output"] += out
        daily_totals[date_str]["cache_read"] += cache
        daily_totals[date_str]["calls"] += 1

    # Calculate costs
    models_out = []
    grand_total_cost = 0.0
    for model_id, totals in sorted(model_totals.items(), key=lambda x: -x[1]["calls"]):
        costs = MODEL_COSTS.get(model_id, {"input": 0, "output": 0, "cache_read": 0})
        cost_input = (totals["input"] / 1_000_000) * costs["input"]
        cost_output = (totals["output"] / 1_000_000) * costs["output"]
        cost_cache = (totals["cache_read"] / 1_000_000) * costs["cache_read"]
        total_cost = cost_input + cost_output + cost_cache
        grand_total_cost += total_cost

        # Find first and last use dates
        dates = sorted(model_daily[model_id].keys())
        first_use = dates[0] if dates else None
        last_use = dates[-1] if dates else None

        # Last 7 days breakdown
        today = datetime.utcnow().strftime("%Y-%m-%d")
        last_7 = []
        for i in range(7):
            d = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            day_data = model_daily[model_id].get(d, {"input": 0, "output": 0, "calls": 0})
            last_7.append({"date": d, "calls": day_data["calls"], "tokens": day_data["input"] + day_data["output"]})

        models_out.append({
            "model_id": model_id,
            "display_name": MODEL_NAMES.get(model_id, model_id),
            "provider": _provider_for(model_id),
            "total_input_tokens": totals["input"],
            "total_output_tokens": totals["output"],
            "total_cache_read": totals["cache_read"],
            "total_tokens": totals["input"] + totals["output"] + totals["cache_read"],
            "total_calls": totals["calls"],
            "cost_usd": round(total_cost, 4),
            "cost_per_m_input": costs["input"],
            "cost_per_m_output": costs["output"],
            "first_use": first_use,
            "last_use": last_use,
            "last_7_days": last_7,
        })

    # Kimi baseline cost for relative pricing
    kimi_input = MODEL_COSTS.get("kimi-k2.5", {}).get("input", 0.60)
    kimi_output = MODEL_COSTS.get("kimi-k2.5", {}).get("output", 3.00)
    for m in models_out:
        # Relative cost based on blended rate (3:1 input:output)
        m_blend = (m["cost_per_m_input"] * 3 + m["cost_per_m_output"]) / 4
        k_blend = (kimi_input * 3 + kimi_output) / 4
        m["relative_cost"] = round(m_blend / k_blend, 1) if k_blend > 0 else 0

    result = {
        "models": models_out,
        "total_cost_usd": round(grand_total_cost, 4),
        "total_calls": sum(m["total_calls"] for m in models_out),
        "total_tokens": sum(m["total_tokens"] for m in models_out),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    _usage_cache = result
    _usage_cache_time = now
    return result


def _provider_for(model_id):
    """Map model ID to provider name."""
    if "kimi" in model_id:
        return "Moonshot AI"
    if "claude" in model_id:
        return "Anthropic"
    if "sonar" in model_id:
        return "Perplexity"
    if "gpt" in model_id or "whisper" in model_id or "embedding" in model_id:
        return "OpenAI"
    return "Unknown"


if __name__ == "__main__":
    import sys
    data = aggregate_usage()
    print(json.dumps(data, indent=2))
    print(f"\n--- Summary ---")
    print(f"Total calls: {data['total_calls']}")
    print(f"Total tokens: {data['total_tokens']:,}")
    print(f"Total cost: ${data['total_cost_usd']:.4f}")
    for m in data["models"]:
        print(f"  {m['display_name']}: {m['total_calls']} calls, {m['total_tokens']:,} tokens, ${m['cost_usd']:.4f}, {m['relative_cost']}× Kimi")
