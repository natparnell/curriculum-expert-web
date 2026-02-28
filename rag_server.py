#!/usr/bin/env python3
"""
Curriculum Expert — Flask API + Web Frontend
Serves the curriculum expert chat interface and provides RAG query,
file upload, and LLM answer endpoints.
"""

import os
import re
import sys
import json
import uuid
import time
from pathlib import Path
from datetime import datetime

import threading

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

# Add rag module to path
sys.path.insert(0, str(Path(__file__).parent))
from rag_pipeline import RAGPipeline
from app_tracker import log_query, log_upload, log_infographic, log_feedback, get_stats as get_app_stats

# Optional imports for file extraction
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# --- Config ---
CLOUD_MODE = os.environ.get('CLOUD_MODE', '').lower() in ('1', 'true', 'yes')
APP_DIR = Path(__file__).parent
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/app"))
CONFIG_PATH = APP_DIR / "curriculum-agent-config.json"
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Session store: {session_id: {filename, text, uploaded_at}}
file_sessions = {}
SESSION_TTL = 7200  # 2 hours

app = Flask(__name__)
CORS(app)

# Initialize pipeline once
pipeline = RAGPipeline()


def _startup_index():
    """Background thread to build ChromaDB index on first deploy."""
    if pipeline.collection.count() == 0:
        print("ChromaDB empty — starting initial indexing in background...")
        try:
            result = pipeline.index_knowledge_base()
            print(f"Initial indexing complete: {result['chunks_indexed']} chunks from {result['files_indexed']} files")
        except Exception as e:
            print(f"Initial indexing failed: {e}")


if CLOUD_MODE:
    threading.Thread(target=_startup_index, daemon=True).start()


_config_cache = None
_config_mtime = 0

def load_config():
    """Load curriculum agent config, cached in memory. Reloads if file changed."""
    global _config_cache, _config_mtime
    try:
        current_mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        current_mtime = 0
    if _config_cache is None or current_mtime != _config_mtime:
        with open(CONFIG_PATH) as f:
            _config_cache = json.load(f)
        _config_mtime = current_mtime
    return _config_cache


# --- Dave Status (cached reads of jobs.json, BUILD_ROTATION.md, watchdog.txt) ---

_JOBS_PATH = Path("/home/node/.openclaw/cron/jobs.json")
_ROTATION_PATH = WORKSPACE / "knowledge" / "BUILD_ROTATION.md"
_WATCHDOG_PATH = WORKSPACE / "memory" / "watchdog.txt"

_status_cache = None
_status_mtimes = (0, 0, 0)

# Human-friendly display names for cron jobs
_JOB_DISPLAY = {
    "morning-dashboard":       ("Morning Dashboard", "Preparing Nat's daily briefing"),
    "overnight-worker-2am":    ("Heartbeat Check",   "Running overnight health checks"),
    "overnight-worker-4am":    ("Task Worker",        "Processing overnight task queue"),
    "dave-workspace-backup":   ("Workspace Backup",   "Backing up workspace to GitHub"),
    "dave-git-backup":         ("Git Backup",         "Syncing git repositories"),
    "weekly-memory-consolidation": ("Memory Tidy",    "Consolidating long-term memory"),
    "copilot-updates-monitor": ("Copilot Monitor",    "Checking for Copilot updates"),
    "weekly-project-builder":  ("Project Builder",    "Building weekly project summary"),
    "evening-meeting-prep":    ("Meeting Prep",       "Preparing briefing notes"),
    "friday-weekly-summary":   ("Weekly Summary",     "Creating Friday weekly summary"),
    "watchdog":                ("Watchdog",            "System heartbeat check"),
    "rag-reindex":             ("Knowledge Indexer",   "Updating search index"),
}


def _safe_mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0


def _parse_rotation_table(text):
    """Parse the rotation table from BUILD_ROTATION.md.
    Returns dict keyed by subject: {task_id, task_desc, slot, phase}
    """
    rotation = {}
    for line in text.splitlines():
        # Match: | 1 | History | history | 1 | F-03 Similarity and Difference |
        m = re.match(
            r'\|\s*(\d+)\s*\|\s*\S.*?\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*(\S+)\s+(.*?)\s*\|',
            line
        )
        if m:
            slot, key, phase, task_id, task_desc = m.groups()
            # Clean parentheses from task_desc: "(Overview)" → "Overview"
            task_desc = task_desc.strip().strip('()')
            rotation[key] = {
                "slot": int(slot),
                "phase": int(phase),
                "task_id": task_id,
                "task_desc": task_desc,
            }
    return rotation


def _parse_build_queue_current(subject_key):
    """Parse a subject's BUILD_QUEUE.md for the in-progress [~] task.
    Returns dict with task_id, task_desc, output_file, or None.
    """
    queue_path = WORKSPACE / "knowledge" / f"{subject_key}-curriculum" / "BUILD_QUEUE.md"
    try:
        text = queue_path.read_text()
    except Exception:
        return None

    # Look for [~] markers indicating in-progress tasks
    # Pattern 1 (list format): "- [~] Similarity and Difference → `02_.../file.md`"
    for line in text.splitlines():
        if '[~]' not in line:
            continue
        # Extract description and optional output file
        # Strip the "- [~] " prefix
        content = re.sub(r'^[-*]\s*\[~\]\s*', '', line).strip()
        # Skip legend/key lines like "- [~] In Progress -- Currently being worked on"
        content_lower = content.lower()
        if content_lower.startswith('in progress') or 'currently being worked on' in content_lower:
            continue
        # Try to find output file: `path/file.md`
        out_match = re.search(r'`([^`]+\.md)`', content)
        output_file = out_match.group(1) if out_match else None
        # Clean description: remove output file reference, arrows, "← START HERE" etc.
        desc = re.sub(r'→.*$', '', content).strip()
        desc = re.sub(r'←.*$', '', desc).strip()
        desc = re.sub(r'\*\*([^*]+)\*\*', r'\1', desc)  # remove bold markers
        return {"task_desc": desc, "output_file": output_file}

    # Pattern 2 (table format): "| NC-01 | Description... | `file.md` | IN PROGRESS |"
    for line in text.splitlines():
        if 'IN PROGRESS' not in line.upper():
            continue
        cols = [c.strip() for c in line.split('|')]
        if len(cols) >= 5:
            task_id = cols[1] if len(cols) > 1 else ""
            task_desc = cols[2] if len(cols) > 2 else ""
            out_match = re.search(r'`([^`]+\.md)`', cols[3] if len(cols) > 3 else "")
            output_file = out_match.group(1) if out_match else None
            # Truncate long descriptions
            if len(task_desc) > 80:
                task_desc = task_desc[:77] + "..."
            return {"task_id": task_id, "task_desc": task_desc, "output_file": output_file}

    return None


def _mark_task_done(text, task_id):
    """Find task_id in BUILD_QUEUE.md text and flip its status to DONE.

    Returns (new_text, found_bool).

    Handles all format variations across subjects:
    - Table rows with [ ] TODO, ⬜ TODO, bare TODO, IN PROGRESS
    - Section headings (### T-05: Title) with forward-scanning
    - **Status:** lines (TODO/PENDING/IN PROGRESS → DONE)
    """
    lines = text.split('\n')
    today = datetime.now().strftime('%Y-%m-%d')
    found = False

    esc_id = re.escape(task_id)
    id_in_table = re.compile(r'^\|\s*' + esc_id + r'\s*\|')
    id_in_heading = re.compile(r'^#{2,3}\s+' + esc_id + r'[\s:.]')

    task_line_indices = []
    for i, line in enumerate(lines):
        if id_in_table.search(line) or id_in_heading.search(line):
            task_line_indices.append(i)

    if not task_line_indices:
        return text, False

    for idx in task_line_indices:
        line = lines[idx]

        # --- TABLE ROW ---
        if '|' in line and line.strip().startswith('|'):
            original = line
            # [~] IN PROGRESS -> [x] DONE
            line = re.sub(r'\[~\]\s*IN\s+PROGRESS', '[x] DONE', line)
            # [ ] TODO -> [x] DONE
            line = re.sub(r'\[\s*\]\s*TODO', '[x] DONE', line)
            # ⬜ TODO -> ✅ DONE
            line = re.sub(r'\u2b1c\s*TODO', '\u2705 DONE', line)
            # bare IN PROGRESS in cell
            line = re.sub(r'(?<=\|)\s*IN\s+PROGRESS\s*(?=\|)', ' DONE ', line)
            # bare TODO in cell (only if nothing above matched)
            if line == original:
                line = re.sub(r'(?<=\|)\s*TODO\s*(?=\|)', ' DONE ', line)
                line = re.sub(r'(?<=\|)\s*TODO\s*$', ' DONE |', line)
            # PENDING in cell
            line = re.sub(r'(?<=\|)\s*PENDING\s*(?=\|)', ' DONE ', line)
            line = re.sub(r'(?<=\|)\s*PENDING\s*$', ' DONE |', line)

            if line != original:
                lines[idx] = line
                found = True

        # --- SECTION HEADING ---
        elif id_in_heading.search(line):
            if 'DONE' not in line and 'COMPLETE' not in line:
                lines[idx] = line.rstrip() + f' \u2014 **DONE** {today}'
                found = True

            # Forward-scan for **Status:** lines and unchecked checkboxes
            for j in range(idx + 1, min(idx + 30, len(lines))):
                fwd = lines[j]
                if fwd.strip().startswith('###') or fwd.strip() == '---':
                    break
                if '**Status:**' in fwd and 'DONE' not in fwd and 'COMPLETE' not in fwd:
                    lines[j] = re.sub(
                        r'(\*\*Status:\*\*)\s*(TODO|PENDING|IN PROGRESS).*',
                        rf'\1 DONE ({today})',
                        fwd,
                    )
                    found = True
                if re.match(r'^-\s*\[\s*\]', fwd):
                    lines[j] = fwd.replace('[ ]', '[x]', 1)

    return '\n'.join(lines), found


def _append_progress(subject, task_id, output_file=None, words=None, notes=None):
    """Append a completion row to the subject's PROGRESS.md."""
    progress_path = WORKSPACE / "knowledge" / f"{subject}-curriculum" / "PROGRESS.md"
    today = datetime.now().strftime('%Y-%m-%d')

    words_str = f"{words:,}" if isinstance(words, int) else (str(words) if words else '\u2014')
    output_str = output_file or '\u2014'
    notes_str = notes or '\u2014'
    new_row = f"| {today} | {task_id} | {output_str} | {words_str} | {notes_str} |"

    try:
        if progress_path.is_file():
            text = progress_path.read_text().rstrip('\n')
            text = text + '\n' + new_row + '\n'
        else:
            text = (
                f"# Progress Log \u2014 {subject.capitalize()} Curriculum Knowledge Base\n\n"
                f"| Date | Task | File | Words | Notes |\n"
                f"|------|------|------|-------|-------|\n"
                f"{new_row}\n"
            )
        progress_path.write_text(text)
        return True
    except Exception as e:
        print(f"Failed to update PROGRESS.md for {subject}: {e}")
        return False


def _find_next_todo(subject):
    """Find the next unbuilt TODO task for a subject.

    Parses BUILD_QUEUE.md, finds TODO tasks in order, checks if the output
    file already exists on disk. If it does, auto-marks it DONE and moves on.
    Returns dict with task info or None if nothing left to build.
    """
    queue_path = WORKSPACE / "knowledge" / f"{subject}-curriculum" / "BUILD_QUEUE.md"
    if not queue_path.is_file():
        return None

    text = queue_path.read_text()
    lines = text.split('\n')

    # Collect all TODO tasks in order
    todos = []
    for i, line in enumerate(lines):
        # Skip non-table rows
        if not line.strip().startswith('|'):
            continue
        # Skip header / separator rows
        if re.match(r'^\|[-\s|]+\|$', line.strip()):
            continue
        if 'ID' in line and 'Task' in line and 'Status' in line:
            continue

        # Check if this row is TODO (any format)
        is_todo = False
        if re.search(r'\[ \] TODO|\bTODO\b|\u2b1c\s*TODO', line):
            # Make sure it is NOT a legend line
            if '= Not yet started' in line or '= Not Started' in line:
                continue
            is_todo = True

        if not is_todo:
            continue

        # Parse task ID (first cell)
        cols = [c.strip() for c in line.split('|')]
        # cols[0] is empty (before first |), cols[1] is task_id
        if len(cols) < 3:
            continue
        task_id = cols[1].strip()
        if not re.match(r'^[A-Z]+-\d+$', task_id):
            continue

        # Parse description (second cell)
        task_desc = cols[2].strip() if len(cols) > 2 else ''

        # Try to find output file in backticks anywhere in the row
        out_match = re.search(r'`([^`]+\.md)`', line)
        output_file = out_match.group(1) if out_match else None

        todos.append({
            'line_idx': i,
            'task_id': task_id,
            'task_desc': task_desc,
            'output_file': output_file,
        })

    # Also check section-heading format (maths-style: ### NC-05: Title)
    for i, line in enumerate(lines):
        m = re.match(r'^#{2,3}\s+([A-Z]+-\d+)[:\s.]\s*(.*)', line)
        if not m:
            continue
        task_id = m.group(1)
        task_desc = m.group(2).strip()

        # Skip if already in todos from table parsing
        if any(t['task_id'] == task_id for t in todos):
            continue

        # Check if already DONE (explicit marker in heading)
        if 'DONE' in line or 'COMPLETE' in line:
            continue

        # Forward-scan the section body for status & completion info
        done_below = False
        has_unchecked = False
        has_any_checkbox = False
        out_file = None
        for j in range(i + 1, min(i + 30, len(lines))):
            fwd = lines[j]
            if fwd.strip().startswith('###') or fwd.strip() == '---':
                break
            # Check for **Status:** DONE
            if '**Status:**' in fwd and ('DONE' in fwd or 'COMPLETE' in fwd):
                done_below = True
            # Track checkboxes: all [x] = done, any [ ] = not done
            if re.match(r'^-\s*\[x\]', fwd):
                has_any_checkbox = True
            if re.match(r'^-\s*\[\s*\]', fwd):
                has_any_checkbox = True
                has_unchecked = True
            # Find output file
            if out_file is None:
                om = re.search(r'`([^`]+\.md)`', fwd)
                if om:
                    out_file = om.group(1)

        # Section is done if Status says DONE, or all checkboxes are [x]
        if done_below:
            continue
        if has_any_checkbox and not has_unchecked:
            continue

        todos.append({
            'line_idx': i,
            'task_id': task_id,
            'task_desc': task_desc,
            'output_file': out_file,
        })

    if not todos:
        return None

    # Walk through TODOs, auto-marking any whose file already exists
    subj_dir = WORKSPACE / "knowledge" / f"{subject}-curriculum"
    auto_marked = []

    for task in todos:
        if task['output_file']:
            full_path = subj_dir / task['output_file']
            if full_path.is_file():
                # File already exists - auto-mark DONE and skip
                text_now = queue_path.read_text()
                new_text, found = _mark_task_done(text_now, task['task_id'])
                if found:
                    queue_path.write_text(new_text)
                    _append_progress(
                        subject, task['task_id'],
                        output_file=task['output_file'],
                        notes='auto-marked by /next-task (file already existed)'
                    )
                    auto_marked.append(task['task_id'])
                continue

        # This task is genuinely unbuilt - return it
        return {
            'task_id': task['task_id'],
            'task_desc': task['task_desc'],
            'output_file': task['output_file'],
            'auto_marked': auto_marked,
            'remaining': len(todos) - len(auto_marked) - 1,
        }

    # All TODOs were auto-marked (all files exist)
    return {
        'task_id': None,
        'all_done': True,
        'auto_marked': auto_marked,
        'remaining': 0,
    }

def _parse_completed_builds(text):
    """Parse the Completed Builds Log table from BUILD_ROTATION.md.
    Returns list of recent builds: [{date, subject, task, file, words}]
    """
    builds = []
    for line in text.splitlines():
        m = re.match(
            r'\|\s*(20\d{2}-\d{2}-\d{2})\s*\|\s*(\w+)\s*\|\s*(.*?)\s*\|\s*`?(.*?)`?\s*\|\s*([\d,]+)\s*\|',
            line
        )
        if m:
            builds.append({
                "date": m.group(1),
                "subject": m.group(2),
                "task": m.group(3).strip(),
                "file": m.group(4).strip(),
                "words": m.group(5).strip(),
            })
    return builds


def _enrich_builds_with_timestamps(builds):
    """Cross-reference completed builds with JSONL data to add real timestamps."""
    if not builds:
        return builds
    runs_dir = Path("/home/node/.openclaw/cron/runs")
    if not runs_dir.is_dir():
        return builds
    # Build a map of output_file -> timestamp from successful builds in ALL JSONL files
    file_ts_map = {}
    try:
        for jsonl_path in runs_dir.glob("*.jsonl"):
            try:
                with open(jsonl_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        if raw.get("status") != "ok":
                            continue
                        summary = raw.get("summary", "")
                        ts_val = raw.get("ts", 0)
                        # Check explicit "Build Complete" summaries
                        if "Build Complete" in summary:
                            m = re.search(r'\*\*Output(?:\s*file)?\*\*\s*\|\s*`([^`]+)`', summary)
                            if m:
                                fname = m.group(1).split("/")[-1]
                                file_ts_map[fname] = ts_val
                                continue
                        # Also check "Write failed" sessions that actually succeeded
                        if "Write failed" in summary:
                            session_id = raw.get("sessionId")
                            actual = _check_session_for_actual_success(session_id)
                            if actual and actual.get("output_file"):
                                fname = actual["output_file"].split("/")[-1]
                                file_ts_map[fname] = ts_val
                                continue
            except Exception:
                continue
    except Exception:
        return builds
    # Enrich builds
    for b in builds:
        fname = b["file"].split("/")[-1] if "/" in b["file"] else b["file"]
        if fname in file_ts_map:
            b["ts_ms"] = file_ts_map[fname]
    return builds


def load_status():
    """Load Dave's operational status, cached by file mtimes."""
    global _status_cache, _status_mtimes

    mtimes = (_safe_mtime(_JOBS_PATH), _safe_mtime(_ROTATION_PATH), _safe_mtime(_WATCHDOG_PATH))
    if _status_cache is not None and mtimes == _status_mtimes:
        # Update 'now_ms' even on cache hit so frontend gets fresh timestamps
        _status_cache["now_ms"] = int(time.time() * 1000)
        return _status_cache

    now_ms = int(time.time() * 1000)

    # 1. Parse jobs.json
    jobs_raw = []
    try:
        with open(_JOBS_PATH) as f:
            data = json.load(f)
        jobs_raw = data.get("jobs", []) if isinstance(data, dict) else data
    except Exception:
        pass

    # 2. Parse BUILD_ROTATION.md — subject, task IDs, completed builds
    builder_subject = "curriculum"
    rotation = {}
    completed_builds = []
    try:
        rot_text = _ROTATION_PATH.read_text()
        m = re.search(r"\*\*Current position:\*\*\s*Slot\s+\d+\s*\((\w+)\)", rot_text)
        if m:
            builder_subject = m.group(1).lower()
        rotation = _parse_rotation_table(rot_text)
        completed_builds = _parse_completed_builds(rot_text)
        # Enrich builds with real timestamps from JSONL data
        completed_builds = _enrich_builds_with_timestamps(completed_builds)
    except Exception:
        pass

    # 3. Get current/next task from rotation table + BUILD_QUEUE.md
    builder_detail = None
    if builder_subject != "curriculum":
        rot_info = rotation.get(builder_subject, {})
        task_id = rot_info.get("task_id", "")
        task_desc = rot_info.get("task_desc", "")
        output_file = None
        # Check BUILD_QUEUE.md for a live [~] in-progress marker
        queue_info = _parse_build_queue_current(builder_subject)
        if queue_info:
            if queue_info.get("task_desc"):
                task_desc = queue_info["task_desc"]
            if queue_info.get("task_id"):
                task_id = queue_info["task_id"]
            output_file = queue_info.get("output_file")
        builder_detail = {
            "subject": builder_subject.capitalize(),
            "task_id": task_id,
            "task_desc": task_desc,
            "output_file": output_file,
            "phase": rot_info.get("phase", 1),
        }

    # 4. Parse watchdog
    watchdog_ts = None
    try:
        watchdog_ts = _WATCHDOG_PATH.read_text().strip()
    except Exception:
        pass

    # 5. Process jobs into display format
    processed = []
    builder_jobs = []
    for j in jobs_raw:
        if not isinstance(j, dict):
            continue
        raw_name = j.get("name", "")
        state = j.get("state", {})
        enabled = j.get("enabled", True)

        if raw_name.startswith("curriculum-builder-") or re.match(r"^cb-\d", raw_name):
            builder_jobs.append({"raw": raw_name, "state": state, "enabled": enabled})
            continue

        display = _JOB_DISPLAY.get(raw_name, (raw_name, ""))
        processed.append({
            "name": display[0],
            "raw_name": raw_name,
            "description": display[1],
            "enabled": enabled,
            "last_status": state.get("lastStatus"),
            "last_run": state.get("lastRunAtMs"),
            "last_duration": state.get("lastDurationMs"),
            "next_run": state.get("nextRunAtMs"),
            "errors": state.get("consecutiveErrors", 0),
        })

    # 6. Collapse all curriculum-builder-* slots into one row with detail
    if builder_jobs:
        # Find the most recently run builder
        last_builder = max(builder_jobs, key=lambda b: b["state"].get("lastRunAtMs", 0))
        # Find next to run (only consider FUTURE nextRunAtMs to avoid stale past values)
        future_builder_slots = [
            b for b in builder_jobs
            if b["enabled"] and b["state"].get("nextRunAtMs", 0) > now_ms
        ]
        if future_builder_slots:
            next_builder = min(future_builder_slots, key=lambda b: b["state"]["nextRunAtMs"])
        else:
            next_builder = last_builder
        lb_state = last_builder["state"]
        nb_state = next_builder["state"]
        subj_cap = builder_subject.capitalize()
        # Build a rich description using builder_detail
        if builder_detail and builder_detail.get("task_id"):
            desc = f"{subj_cap}: {builder_detail['task_id']} {builder_detail.get('task_desc', '')}".strip()
        else:
            desc = f"Building {subj_cap} knowledge files"
        processed.append({
            "name": "Knowledge Builder",
            "raw_name": "curriculum-builder",
            "description": desc,
            "enabled": True,
            "last_status": lb_state.get("lastStatus"),
            "last_run": lb_state.get("lastRunAtMs"),
            "last_duration": lb_state.get("lastDurationMs"),
            "next_run": nb_state.get("nextRunAtMs"),
            "errors": lb_state.get("consecutiveErrors", 0),
            "slots_total": len(builder_jobs),
            "slots_ok": sum(1 for b in builder_jobs if b["state"].get("lastStatus") == "ok"),
        })

    # 6b. Tag dedup skips: ok status with duration < 90s means file already existed
    if builder_jobs and processed:
        builder_row = next((j for j in processed if j.get("raw_name") == "curriculum-builder"), None)
        if builder_row:
            lb_dur = builder_row.get("last_duration") or 0
            if builder_row.get("last_status") == "ok" and lb_dur and lb_dur < 90000:
                builder_row["is_dedup_skip"] = True
                builder_row["dedup_duration_s"] = round(lb_dur / 1000)

    # 7. Determine running / last / next
    running = []
    for job in processed:
        nr = job.get("next_run") or 0
        lr = job.get("last_run") or 0
        if nr and lr and nr <= now_ms and lr < nr:
            running.append(job)

    # Also check individual builder slots for a currently running one
    if builder_jobs and not any(j.get("raw_name") == "curriculum-builder" for j in running):
        for bj in builder_jobs:
            bs = bj["state"]
            bnr = bs.get("nextRunAtMs", 0)
            blr = bs.get("lastRunAtMs", 0)
            if bnr and blr and bnr <= now_ms and blr < bnr:
                collapsed = next((j for j in processed if j.get("raw_name") == "curriculum-builder"), None)
                if collapsed:
                    running.append(collapsed)
                break

    last = max(
        [j for j in processed if j.get("last_run")],
        key=lambda j: j["last_run"],
        default=None,
    )

    future = [j for j in processed if j.get("next_run") and j["next_run"] > now_ms]
    nxt = min(future, key=lambda j: j["next_run"], default=None)

    # 8. Build rotation summary for all subjects
    rotation_summary = []
    for key, info in sorted(rotation.items(), key=lambda x: x[1]["slot"]):
        rotation_summary.append({
            "subject": key.capitalize(),
            "key": key,
            "slot": info["slot"],
            "phase": info["phase"],
            "task_id": info["task_id"],
            "task_desc": info["task_desc"],
        })

    result = {
        "now_ms": now_ms,
        "running": running,
        "last": last,
        "next": nxt,
        "jobs": processed,
        "watchdog": watchdog_ts,
        "builder_subject": builder_subject,
        "builder_detail": builder_detail,
        "rotation": rotation_summary,
        "completed_builds": completed_builds[-5:],  # last 5 builds
    }

    _status_cache = result
    _status_mtimes = mtimes
    return result


# --- Singleton API clients ---
_anthropic_client = None
_openai_llm_client = None

def _get_anthropic_client():
    """Get or create singleton Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        key = _get_anthropic_key()
        if key and HAS_ANTHROPIC:
            _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client

def _get_openai_llm_client():
    """Get or create singleton OpenAI client for LLM calls."""
    global _openai_llm_client
    if _openai_llm_client is None:
        key = _get_openai_key()
        if key:
            import openai as oai
            _openai_llm_client = oai.OpenAI(api_key=key)
    return _openai_llm_client


def cleanup_sessions():
    """Remove expired file sessions."""
    now = time.time()
    expired = [k for k, v in file_sessions.items()
               if now - v['uploaded_at'] > SESSION_TTL]
    for k in expired:
        del file_sessions[k]


def extract_text(filepath: Path) -> str:
    """Extract text from uploaded file."""
    suffix = filepath.suffix.lower()

    if suffix in ('.txt', '.md'):
        return filepath.read_text(encoding='utf-8', errors='replace')

    elif suffix == '.pdf' and HAS_PDF:
        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return '\n\n'.join(text_parts)

    elif suffix == '.docx' and HAS_DOCX:
        doc = DocxDocument(str(filepath))
        return '\n\n'.join(p.text for p in doc.paragraphs if p.text.strip())

    else:
        return filepath.read_text(encoding='utf-8', errors='replace')


def build_system_prompt(config, subject_key):
    """Build the full system prompt for a subject from config."""
    subject_config = config['subjects'][subject_key]
    thinker_lines = "\n".join(
        f"- {t['name']} ({t['focus']})" for t in subject_config['thinkers']
    )
    return (
        config['base_system_prompt']
        .replace('{SUBJECT}', subject_config['name'])
        .replace('{THINKERS}', thinker_lines)
    )


def _get_anthropic_key():
    """Try to find an Anthropic API key."""
    key = os.environ.get('ANTHROPIC_API_KEY')
    if key:
        return key
    # Check Dave's auth-profiles.json (primary source)
    try:
        auth_path = Path("/home/node/.openclaw/agents/main/agent/auth-profiles.json")
        if auth_path.exists():
            data = json.loads(auth_path.read_text())
            profiles = data.get('profiles', {})
            profile = profiles.get('anthropic:default', {})
            k = profile.get('key', '').strip()
            if k and k.startswith('sk-ant'):
                return k
    except Exception:
        pass
    # Check local config file fallback
    for path in [
        Path("/home/node/.openclaw/anthropic_api_key"),
        APP_DIR / "anthropic_api_key"
    ]:
        try:
            if path.exists():
                return path.read_text().strip()
        except Exception:
            pass
    return None


def _get_openai_key():
    """Get the OpenAI API key from env or clawdbot.json."""
    key = os.environ.get('OPENAI_API_KEY')
    if key:
        return key
    try:
        with open('/home/node/.openclaw/clawdbot.json') as f:
            cj = json.load(f)
        return cj.get('env', {}).get('vars', {}).get('OPENAI_API_KEY')
    except Exception:
        return None


def call_llm(system_prompt, user_message, conversation_history=None, max_tokens=2000, model=None):
    """
    Call an LLM. Tries Anthropic (Claude) first, falls back to OpenAI (GPT-4.1).
    max_tokens controls response length — callers should set this based on length mode.
    model: override the default Anthropic model (e.g. 'claude-opus-4-5-20251101' for extended).
    """
    anthropic_model = model or "claude-sonnet-4-20250514"
    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    # --- Try Anthropic first ---
    client = _get_anthropic_client()
    if client:
        try:
            response = client.messages.create(
                model=anthropic_model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages
            )
            inp = response.usage.input_tokens if response.usage else None
            out = response.usage.output_tokens if response.usage else None
            return response.content[0].text, inp, out
        except Exception as e:
            print(f"Anthropic call failed: {e}")

    # --- Fall back to OpenAI ---
    oai_client = _get_openai_llm_client()
    if oai_client:
        try:
            oai_messages = [{"role": "system", "content": system_prompt}]
            oai_messages.extend(messages)
            response = oai_client.chat.completions.create(
                model="gpt-4.1",
                max_tokens=max_tokens,
                messages=oai_messages
            )
            inp = response.usage.prompt_tokens if response.usage else None
            out = response.usage.completion_tokens if response.usage else None
            return response.choices[0].message.content, inp, out
        except Exception as e:
            return f"Error calling OpenAI: {e}", None, None

    return "Error: No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.", None, None


# --- Routes ---

@app.route('/')
def index():
    """Serve the frontend."""
    html_path = APP_DIR / "curriculum-expert.html"
    if html_path.exists():
        return send_file(html_path)
    return "<h1>Curriculum Expert</h1><p>Frontend not found.</p>", 404


@app.route('/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "collection_count": pipeline.collection.count(),
        "has_pdf": HAS_PDF,
        "has_docx": HAS_DOCX,
        "has_anthropic": HAS_ANTHROPIC
    })


@app.route('/config', methods=['GET'])
def get_config():
    """Return subject list + UI config for the frontend."""
    config = load_config()
    subjects = {}
    for key, val in config['subjects'].items():
        subjects[key] = {
            "name": val['name'],
            "command": val['command'],
            "thinker_count": len(val['thinkers']),
            "accent_color": val.get('accent_color', '#718096'),
            "accent_light": val.get('accent_light', '#f7fafc'),
            "suggestions": val.get('suggestions', [
                f"What are the key curriculum requirements for {val['name']}?",
                f"How should {val['name']} be taught across key stages?",
                f"What do the leading thinkers say about {val['name']} pedagogy?",
                f"What does the National Curriculum require for {val['name']}?"
            ]),
        }
    return jsonify({"subjects": subjects})


@app.route('/query', methods=['POST'])
def query():
    """Raw RAG query — returns chunks only (no LLM call)."""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' field"}), 400

        config = load_config()
        rag = config['rag_settings']
        chunks = pipeline.query(
            data['question'],
            subject=data.get('subject'),
            top_k=rag['top_k'],
            max_per_source=rag['max_per_source']
        )
        formatted = pipeline.format_for_prompt(chunks)

        return jsonify({
            "question": data['question'],
            "subject": data.get('subject'),
            "chunks": chunks,
            "formatted_context": formatted
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload():
    """Upload a file for session-based Q&A."""
    cleanup_sessions()

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save temporarily
    session_id = str(uuid.uuid4())[:8]
    safe_name = f.filename.replace('/', '_').replace('\\', '_')
    save_path = UPLOAD_DIR / f"{session_id}_{safe_name}"
    f.save(str(save_path))

    # Extract text
    try:
        text = extract_text(save_path)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": f"Failed to extract text: {e}"}), 500

    word_count = len(text.split())
    preview = text[:500] + ("..." if len(text) > 500 else "")

    file_sessions[session_id] = {
        "filename": f.filename,
        "text": text,
        "word_count": word_count,
        "uploaded_at": time.time()
    }

    # Clean up saved file
    save_path.unlink(missing_ok=True)

    log_upload(f.filename, word_count)

    return jsonify({
        "session_id": session_id,
        "filename": f.filename,
        "word_count": word_count,
        "preview": preview
    })


@app.route('/ask', methods=['POST'])
def ask():
    """
    Main endpoint: question + subject + optional file session.
    Retrieves RAG chunks, builds prompt, calls Claude, returns answer.
    """
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' field"}), 400

        question = data['question']
        subject = data.get('subject', 'history')
        session_id = data.get('session_id')
        conversation_history = data.get('conversation_history', [])
        length = data.get('length', 'medium')  # 'short' | 'medium' | 'extended'
        cite_thinkers = data.get('cite_thinkers', True)

        # Length mode settings
        LENGTH_MODES = {
            'short':    {'top_k': 3, 'max_tokens': 500,  'instruction': 'Answer concisely in 2–3 short paragraphs. Be direct and focused. No preamble.'},
            'medium':   {'top_k': 5, 'max_tokens': 3500, 'instruction': None},
            'extended': {'top_k': 8, 'max_tokens': 8192, 'instruction': 'Answer comprehensively. Cover stages, nuance, and practical examples in full.'},
        }
        # Non-streaming endpoint uses call_llm which picks its own model
        mode = LENGTH_MODES.get(length, LENGTH_MODES['medium'])

        config = load_config()
        rag = config['rag_settings']

        # 1. Get RAG chunks from knowledge base
        chunks = pipeline.query(
            question,
            subject=subject,
            top_k=mode['top_k'],
            max_per_source=rag['max_per_source']
        )
        rag_context = pipeline.format_for_prompt(chunks)

        # 2. Build system prompt from config
        system_prompt = build_system_prompt(config, subject)

        # 3. Build the user message with context
        user_message_parts = []

        # Include uploaded file if present
        file_info = None
        if session_id and session_id in file_sessions:
            session = file_sessions[session_id]
            file_info = {
                "filename": session['filename'],
                "word_count": session['word_count']
            }
            # Truncate very large files to ~8000 words
            file_text = session['text']
            words = file_text.split()
            if len(words) > 8000:
                file_text = ' '.join(words[:8000]) + "\n\n[Document truncated — showing first 8,000 words]"

            user_message_parts.append(
                f"UPLOADED DOCUMENT ({session['filename']}):\n"
                f"---\n{file_text}\n---\n"
            )

        user_message_parts.append(
            f"KNOWLEDGE BASE CONTEXT:\n{rag_context}\n"
        )
        if mode['instruction']:
            user_message_parts.append(f"LENGTH INSTRUCTION: {mode['instruction']}")
        if cite_thinkers:
            user_message_parts.append(
                "STYLE INSTRUCTION: Reference and quote key curriculum thinkers and academics by name. "
                "Attribute ideas to the researchers who developed them (e.g. 'As Christine Counsell argues...'). "
                "This applies regardless of answer length — even short answers should name-check relevant thinkers."
            )
        else:
            user_message_parts.append(
                "STYLE INSTRUCTION: Do NOT reference specific curriculum thinkers, academics, or researchers by name — not even in passing. "
                "You may draw on their ideas from the knowledge base, but present them as straightforward professional advice "
                "without attributing them to any named theorist, academic, or researcher. "
                "Give practical, direct guidance that a teacher can immediately use."
            )

        user_message_parts.append(f"QUESTION: {question}")

        user_message = "\n\n".join(user_message_parts)

        # 4. Call LLM (Claude if available, else GPT-4o)
        # Use Opus for extended, Sonnet for medium/short
        llm_model = 'claude-opus-4-5-20251101' if length == 'extended' else None
        _t0 = time.time()
        answer, _inp_tokens, _out_tokens = call_llm(system_prompt, user_message, conversation_history, max_tokens=mode['max_tokens'], model=llm_model)
        _duration_ms = int((time.time() - _t0) * 1000)

        # 5. Build sources list
        sources = []
        for chunk in chunks:
            meta = chunk.get('metadata', {})
            sources.append({
                "file": meta.get('file_path', ''),
                "heading": meta.get('heading', ''),
                "thinker": meta.get('thinker', ''),
                "key_stage": meta.get('key_stage', '')
            })

        log_query(subject, question, length, bool(file_info), cite_thinkers,
                  llm_model or 'claude-sonnet', _duration_ms, len(answer or ''),
                  success=True, endpoint='ask',
                  input_tokens=_inp_tokens, output_tokens=_out_tokens)

        return jsonify({
            "answer": answer,
            "sources": sources,
            "file_context": file_info,
            "subject": subject
        })

    except Exception as e:
        log_query(subject if 'subject' in dir() else 'unknown',
                  data.get('question', '') if 'data' in dir() else '',
                  'unknown', False, True, 'unknown', None, None,
                  success=False, endpoint='ask')
        return jsonify({"error": str(e)}), 500


@app.route('/ask-stream', methods=['POST'])
def ask_stream():
    """
    Streaming version of /ask. Returns Server-Sent Events (SSE).

    Event types:
      event: phase    data: {"phase": "searching"|"generating"}
      event: token    data: {"text": "..."}
      event: sources  data: {"sources": [...]}
      event: done     data: {}
      event: error    data: {"error": "..."}
    """
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' field"}), 400

        question = data['question']
        subject = data.get('subject', 'history')
        session_id = data.get('session_id')
        conversation_history = data.get('conversation_history', [])
        length = data.get('length', 'medium')
        cite_thinkers = data.get('cite_thinkers', True)

        LENGTH_MODES = {
            'short':    {'top_k': 3, 'max_tokens': 500,  'instruction': 'Answer concisely in 2-3 short paragraphs. Be direct and focused. No preamble.', 'model': 'claude-haiku-4-5-20251001'},
            'medium':   {'top_k': 5, 'max_tokens': 3500, 'instruction': None, 'model': 'claude-sonnet-4-20250514'},
            'extended': {'top_k': 8, 'max_tokens': 8192, 'instruction': 'Answer comprehensively. Cover stages, nuance, and practical examples in full.', 'model': 'claude-opus-4-5-20251101'},
        }
        mode = LENGTH_MODES.get(length, LENGTH_MODES['medium'])

        def generate():
            import json as _json
            _t0 = time.time()
            _response_chars = 0
            _model_used = mode['model']

            def send_event(event_type, data_dict):
                return f"event: {event_type}\ndata: {_json.dumps(data_dict)}\n\n"

            # Phase 1: RAG retrieval
            yield send_event("phase", {"phase": "searching"})

            config = load_config()
            rag = config['rag_settings']
            chunks = pipeline.query(
                question, subject=subject,
                top_k=mode['top_k'],
                max_per_source=rag['max_per_source']
            )
            rag_context = pipeline.format_for_prompt(chunks)
            system_prompt = build_system_prompt(config, subject)

            # Build user message (same logic as /ask)
            user_message_parts = []
            file_info = None
            if session_id and session_id in file_sessions:
                session = file_sessions[session_id]
                file_info = {
                    "filename": session['filename'],
                    "word_count": session['word_count']
                }
                file_text = session['text']
                words = file_text.split()
                if len(words) > 8000:
                    file_text = ' '.join(words[:8000]) + "\n\n[Document truncated]"
                user_message_parts.append(
                    f"UPLOADED DOCUMENT ({session['filename']}):\n---\n{file_text}\n---\n"
                )

            user_message_parts.append(f"KNOWLEDGE BASE CONTEXT:\n{rag_context}\n")
            if mode['instruction']:
                user_message_parts.append(f"LENGTH INSTRUCTION: {mode['instruction']}")
            if cite_thinkers:
                user_message_parts.append(
                    "STYLE INSTRUCTION: Reference and quote key curriculum thinkers and academics by name. "
                    "Attribute ideas to the researchers who developed them (e.g. 'As Christine Counsell argues...'). "
                    "This applies regardless of answer length — even short answers should name-check relevant thinkers."
                )
            else:
                user_message_parts.append(
                    "STYLE INSTRUCTION: Do NOT reference specific curriculum thinkers, academics, or researchers by name — not even in passing. "
                    "You may draw on their ideas from the knowledge base, but present them as straightforward professional advice "
                    "without attributing them to any named theorist, academic, or researcher. "
                    "Give practical, direct guidance that a teacher can immediately use."
                )
            user_message_parts.append(f"QUESTION: {question}")
            user_message = "\n\n".join(user_message_parts)

            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": user_message})

            # Phase 2: LLM streaming
            yield send_event("phase", {"phase": "generating"})

            # Send sources early so frontend can display them
            sources = []
            for chunk in chunks:
                meta = chunk.get('metadata', {})
                sources.append({
                    "file": meta.get('file_path', ''),
                    "heading": meta.get('heading', ''),
                    "thinker": meta.get('thinker', ''),
                    "key_stage": meta.get('key_stage', '')
                })
            yield send_event("sources", {"sources": sources})

            # Try Anthropic streaming
            client = _get_anthropic_client()
            if client:
                try:
                    _inp_tokens = None
                    _out_tokens = None
                    with client.messages.stream(
                        model=mode['model'],
                        max_tokens=mode['max_tokens'],
                        system=system_prompt,
                        messages=messages
                    ) as stream:
                        for text in stream.text_stream:
                            _response_chars += len(text)
                            yield send_event("token", {"text": text})
                        _usage = stream.get_final_usage()
                        if _usage:
                            _inp_tokens = _usage.input_tokens
                            _out_tokens = _usage.output_tokens
                    _dur = int((time.time() - _t0) * 1000)
                    log_query(subject, question, length, bool(session_id and session_id in file_sessions),
                              cite_thinkers, _model_used, _dur, _response_chars, success=True, endpoint='ask-stream',
                              input_tokens=_inp_tokens, output_tokens=_out_tokens)
                    yield send_event("done", {})
                    return
                except Exception as e:
                    print(f"Anthropic stream failed: {e}")

            # Fallback: OpenAI streaming
            oai_client = _get_openai_llm_client()
            if oai_client:
                try:
                    oai_messages = [{"role": "system", "content": system_prompt}]
                    oai_messages.extend(messages)
                    _model_used = 'gpt-4.1'
                    stream = oai_client.chat.completions.create(
                        model="gpt-4.1",
                        max_tokens=mode['max_tokens'],
                        messages=oai_messages,
                        stream=True
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            _response_chars += len(delta.content)
                            yield send_event("token", {"text": delta.content})
                    _dur = int((time.time() - _t0) * 1000)
                    log_query(subject, question, length, bool(session_id and session_id in file_sessions),
                              cite_thinkers, _model_used, _dur, _response_chars, success=True, endpoint='ask-stream')
                    yield send_event("done", {})
                    return
                except Exception as e:
                    log_query(subject, question, length, False, cite_thinkers,
                              _model_used, None, None, success=False, endpoint='ask-stream')
                    yield send_event("error", {"error": f"OpenAI stream failed: {e}"})
                    return

            log_query(subject, question, length, False, cite_thinkers,
                      _model_used, None, None, success=False, endpoint='ask-stream')
            yield send_event("error", {"error": "No API key available"})

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/admin')
def admin():
    """Serve the usage analytics admin dashboard."""
    return send_file(APP_DIR / 'admin.html')


@app.route('/admin/stats')
def admin_stats():
    """Return aggregated usage stats as JSON."""
    try:
        return jsonify(get_app_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/feedback')
def feedback_page():
    """Serve the feedback form."""
    return send_file(APP_DIR / 'feedback.html')


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Receive and log user feedback."""
    try:
        data = request.get_json(force=True)
        log_feedback(
            overall=data.get('overall'),
            quality=data.get('quality'),
            subject=data.get('subject'),
            role=data.get('role'),
            went_well=data.get('went_well'),
            improve=data.get('improve'),
            recommend=data.get('recommend'),
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/dashboard')
def dashboard():
    """Serve the WeST Lab dashboard HTML."""
    if CLOUD_MODE:
        return jsonify({"error": "Not available in cloud deployment"}), 404
    dashboard_path = Path("/home/node/.openclaw/workspace/apps/menu/index.html")
    if dashboard_path.is_file():
        return send_file(dashboard_path)
    return "Dashboard not found", 404


@app.route('/reindex', methods=['POST'])
def reindex():
    """Trigger a full reindex of the knowledge base from disk. Safe to call at any time."""
    if CLOUD_MODE:
        return jsonify({"error": "Not available in cloud deployment"}), 404
    try:
        result = pipeline.index_knowledge_base()

        # Refresh all 00_INDEX.md files to match disk reality
        config = load_config()
        for subj in config.get('subjects', {}):
            try:
                _refresh_index_file(subj)
            except Exception:
                pass  # Non-fatal — index refresh is best-effort

        # Clear knowledge status cache so dashboard picks up changes
        _knowledge_cache.clear()

        return jsonify({
            "status": "ok",
            "files_indexed": result["files_indexed"],
            "files_skipped": result.get("files_skipped", 0),
            "chunks_indexed": result["chunks_indexed"],
            "subjects": result["subjects"],
            "collection_count": pipeline.collection.count()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/mark-done', methods=['POST'])
def mark_done():
    """Mark a task as DONE in a subject's BUILD_QUEUE.md.

    POST body (JSON):
        subject     (str, required): e.g. "history", "geography"
        task_id     (str, required): e.g. "T-05", "NC-06", "P-02"
        words       (int, optional): word count of output file
        output_file (str, optional): relative path to output file
        notes       (str, optional): freeform note for PROGRESS.md

    Returns:
        200: {"status": "ok", ...}
        400/404/500: {"error": "..."}
    """
    if CLOUD_MODE:
        return jsonify({"error": "Not available in cloud deployment"}), 404
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    subject = (data.get('subject') or '').strip().lower()
    task_id = (data.get('task_id') or '').strip()
    words = data.get('words')
    output_file = data.get('output_file')
    notes = data.get('notes')

    if not subject or not task_id:
        return jsonify({"error": "Both 'subject' and 'task_id' are required"}), 400

    queue_path = WORKSPACE / "knowledge" / f"{subject}-curriculum" / "BUILD_QUEUE.md"
    if not queue_path.is_file():
        return jsonify({"error": f"BUILD_QUEUE.md not found for subject '{subject}'"}), 404

    try:
        text = queue_path.read_text()
    except Exception as e:
        return jsonify({"error": f"Failed to read BUILD_QUEUE.md: {e}"}), 500

    new_text, found = _mark_task_done(text, task_id)

    if not found:
        return jsonify({
            "error": f"Task '{task_id}' not found or already DONE in {subject} BUILD_QUEUE.md"
        }), 404

    try:
        queue_path.write_text(new_text)
    except Exception as e:
        return jsonify({"error": f"Failed to write BUILD_QUEUE.md: {e}"}), 500

    progress_ok = False
    if words or output_file or notes:
        progress_ok = _append_progress(
            subject, task_id, output_file=output_file, words=words, notes=notes
        )

    return jsonify({
        "status": "ok",
        "subject": subject,
        "task_id": task_id,
        "progress_updated": progress_ok,
    })


@app.route('/next-task', methods=['GET'])
def next_task():
    """Return the next unbuilt TODO task for a subject.

    GET /next-task?subject=science

    Parses BUILD_QUEUE.md, skips tasks whose output files already exist
    (auto-marking them DONE), and returns the first genuinely unbuilt task.

    Returns:
        200: {"subject": "...", "task_id": "...", "task_desc": "...",
              "output_file": "...", "auto_marked": [...], "remaining": N}
        200: {"subject": "...", "all_done": true} if no tasks remain
        400/404: {"error": "..."}
    """
    if CLOUD_MODE:
        return jsonify({"error": "Not available in cloud deployment"}), 404
    subject = (request.args.get('subject') or '').strip().lower()
    if not subject:
        return jsonify({"error": "'subject' query parameter is required"}), 400

    subj_dir = WORKSPACE / "knowledge" / f"{subject}-curriculum"
    if not subj_dir.is_dir():
        return jsonify({"error": f"Subject directory not found: {subject}"}), 404

    result = _find_next_todo(subject)
    if result is None:
        return jsonify({"subject": subject, "all_done": True, "remaining": 0})

    result["subject"] = subject
    return jsonify(result)


@app.route('/status', methods=['GET'])
def status():
    """Return Dave's current operational status (lightweight, mtime-cached)."""
    if CLOUD_MODE:
        return jsonify({"error": "Not available in cloud deployment"}), 404
    try:
        return jsonify(load_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Activity Log (reads JSONL run files for comprehensive history) ---

_RUNS_DIR = Path("/home/node/.openclaw/cron/runs")
_activity_cache = None
_activity_mtime = 0

_BOT_NAMES = {
    "history": "BeckyBot", "geography": "NicolaBot",
    "science": "SimonBot", "english": "ScottBot",
    "rs": "LaurenBot", "mfl": "JenniBot", "maths": "RachaelKrisBot",
}


def _build_job_id_map():
    """Map JSONL filename stems to job names and types."""
    id_map = {}
    try:
        with open(_JOBS_PATH) as f:
            data = json.load(f)
        jobs_list = data.get("jobs", []) if isinstance(data, dict) else data
    except Exception:
        jobs_list = []

    for j in jobs_list:
        if not isinstance(j, dict):
            continue
        job_id = j.get("id", "")
        name = j.get("name", "")
        is_builder = name.startswith("cb-") or name.startswith("curriculum-builder")
        if job_id:
            if is_builder:
                id_map[job_id] = {"name": name, "display": "Knowledge Builder", "desc": "", "type": "builder"}
            else:
                display = _JOB_DISPLAY.get(name, (name, ""))
                id_map[job_id] = {"name": name, "display": display[0], "desc": display[1], "type": "system"}
        if name.startswith("cb-"):
            id_map[name] = {"name": name, "display": "Knowledge Builder", "desc": "", "type": "builder"}

    id_map["undefined"] = {"name": "curriculum-builder-chain", "display": "Knowledge Builder", "desc": "", "type": "builder"}
    return id_map


_SESSIONS_DIR = Path("/home/node/.openclaw/agents/main/sessions")

def _check_session_for_actual_success(session_id):
    """Check a session log to see if a write ACTUALLY succeeded despite the summary saying 'Write failed'.

    Kimi k2.5 often omits the 'path' parameter on first write attempts, causing errors,
    but eventually retries with the correct parameters and succeeds.  The JSONL summary
    captures the *first* error, so the dashboard shows failure even though the file was written.

    Returns a dict with build info if a successful write is found, else None.
    """
    if not session_id:
        return None
    session_file = _SESSIONS_DIR / f"{session_id}.jsonl"
    if not session_file.is_file():
        return None
    try:
        success_path = None
        success_bytes = 0
        api_task_id = None
        api_task_desc = None
        api_words = None
        api_output_file = None
        api_subject = None
        with open(session_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                msg = d.get("message", {})
                role = msg.get("role", "")
                tool_name = msg.get("toolName", "")
                content = msg.get("content", [])
                # Extract text from content
                if isinstance(content, list) and content:
                    text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                elif isinstance(content, str):
                    text = content
                else:
                    text = ""
                # Check for successful write
                if role == "toolResult" and tool_name == "write":
                    m = re.search(r'Successfully wrote (\d+) bytes to (.+)', text)
                    if m:
                        success_bytes = int(m.group(1))
                        success_path = m.group(2).strip()
                # Check for /next-task API response (contains task_id, subject, description)
                if role == "toolResult" and tool_name == "exec" and "task_id" in text:
                    try:
                        api_data = json.loads(text)
                        api_task_id = api_data.get("task_id")
                        api_task_desc = api_data.get("task_desc")
                        api_output_file = api_data.get("output_file")
                        api_subject = api_data.get("subject")
                    except (json.JSONDecodeError, ValueError):
                        pass
                # Check for mark-done call (contains actual word count)
                # Could be in text, or in a toolCall's command argument
                mark_done_text = text
                if role == "assistant" and isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            # Check toolCall arguments for mark-done
                            args = block.get("arguments", {})
                            cmd = args.get("command", "")
                            if "mark-done" in cmd:
                                mark_done_text = cmd
                                break
                            # Also check text blocks
                            bt = block.get("text", "")
                            if "mark-done" in bt:
                                mark_done_text = bt
                                break
                if "mark-done" in mark_done_text:
                    m = re.search(r'"words"\s*:\s*(\d+)', mark_done_text)
                    if m:
                        api_words = m.group(1)
        if not success_path:
            return None
        # Extract subject from path like "knowledge/history-curriculum/..."
        subject = None
        m = re.search(r'knowledge/(\w+)-curriculum/', success_path)
        if m:
            subject = m.group(1)
        # Fallback: try "knowledge/SUBJ" without "-curriculum/" (handles truncated paths)
        if not subject:
            m = re.search(r'knowledge/(\w+)', success_path)
            if m:
                # Only accept if it looks like one of our known subjects
                candidate = m.group(1).lower().replace("-", "")
                known = {"history", "geography", "english", "maths", "science", "mfl", "rs"}
                if candidate in known:
                    subject = candidate
        # Use API data for accurate info, fall back to estimates
        final_subject = api_subject or subject
        final_words = api_words or str(success_bytes // 6)
        final_output = api_output_file or success_path
        final_task_id = api_task_id
        # Build a readable description
        if api_task_desc:
            desc = f"{final_subject.capitalize() if final_subject else ''} {api_task_id or ''} {api_task_desc}".strip()
        else:
            filename = success_path.split("/")[-1] if "/" in success_path else success_path
            desc = f"{final_subject.capitalize() if final_subject else ''} {filename.replace('.md','').replace('_',' ').title()}".strip()
        return {
            "subject": final_subject,
            "is_build": True,
            "is_dedup": False,
            "output_file": final_output,
            "words": final_words,
            "short_desc": desc,
            "task_id": final_task_id,
            "git_hash": None,
        }
    except Exception:
        return None


def _parse_builder_summary(summary):
    """Extract structured data from a builder run's markdown summary."""
    result = {
        "subject": None, "task_id": None, "is_dedup": False,
        "is_build": False, "words": None, "output_file": None,
        "git_hash": None, "short_desc": "",
    }
    if not summary:
        return result

    # --- Successful build: "Build Complete" ---
    if "Build Complete" in summary:
        result["is_build"] = True
        m = re.search(r'\*\*Subject\*\*\s*\|\s*(\w+)', summary)
        if m:
            result["subject"] = m.group(1).lower()
        # Try "**Task (ID)**  | TASK_ID — Description" (em/en-dash separator)
        m = re.search(r'\*\*Task(?:\s*ID)?\*\*\s*\|\s*([A-Za-z]+-\d+\w*)\s+[—–]\s+(.*?)(?:\s*\||\s*$)', summary, re.MULTILINE)
        if m:
            result["task_id"] = m.group(1).strip()
            result["short_desc"] = m.group(2).strip()
        else:
            # Fallback: just task ID without description
            m = re.search(r'\*\*Task(?:\s*ID)?\*\*\s*\|\s*([A-Za-z]+-\d+\w*)', summary)
            if not m:
                m = re.search(r'\*\*Task(?:\s*ID)?\*\*\s*\|\s*(\S+)', summary)
            if m:
                result["task_id"] = m.group(1).strip()
        # Try to get description from "Build Complete: Subject TASK_ID" header
        if not result["short_desc"]:
            m = re.search(r'Build Complete[:\s]+(.+?)(?:\*\*|\n|$)', summary)
            if m and m.group(1).strip():
                result["short_desc"] = m.group(1).strip()
        m = re.search(r'\*\*Output(?:\s*file)?\*\*\s*\|\s*`([^`]+)`', summary)
        if m:
            result["output_file"] = m.group(1)
        m = re.search(r'\*\*Word [Cc]ount\*\*\s*\|\s*\*{0,2}([\d,]+)\s*words?\*{0,2}', summary)
        if m:
            result["words"] = m.group(1).replace(",", "")
        m = re.search(r'\*\*Git [Cc]ommit\*\*\s*\|\s*.*?`([a-f0-9]{7,})`', summary)
        if m:
            result["git_hash"] = m.group(1)
        return result

    # --- DEDUP skip (multiple formats) ---
    if "DEDUP" in summary or "already exists" in summary or "SKIPPED" in summary:
        result["is_dedup"] = True
        # Extract subject from "Slot N (Subject)" pattern
        m = re.search(r'Slot\s+\d+\s*\((\w+)\)', summary)
        if m:
            result["subject"] = m.group(1).lower()
        # Extract from "knowledge/subject-curriculum/" path
        if not result["subject"]:
            m = re.search(r'knowledge/(\w+)-curriculum/', summary)
            if m:
                result["subject"] = m.group(1)
        # Task ID from "Task NC-02" or "Task S-02"
        m = re.search(r'Task\s+(\S+)', summary)
        if m:
            result["task_id"] = m.group(1)
        # File and words from "`file.md` already exists (N words"
        m = re.search(r'`([^`]+\.md)`\s*already exists\s*\(([^)]+)\)', summary)
        if m:
            result["output_file"] = m.group(1)
            result["short_desc"] = f"Skipped {m.group(1).split('/')[-1]} ({m.group(2)})"
        else:
            m = re.search(r'`([^`]+\.md)`\s*already exists', summary)
            if m:
                result["output_file"] = m.group(1)
                result["short_desc"] = f"Skipped {m.group(1).split('/')[-1]}"
            else:
                result["short_desc"] = "DEDUP skip"
        # Git hash
        m = re.search(r'`([a-f0-9]{7,})`', summary)
        if m:
            result["git_hash"] = m.group(1)
        return result

    # --- Queue complete / rotation advance ---
    if "DONE" in summary and ("build queue complete" in summary or "rotation complete" in summary or "advancing to" in summary.lower()):
        m = re.search(r'Slot\s+\d+\s*\((\w+)\)', summary)
        if m:
            result["subject"] = m.group(1).lower()
        result["is_dedup"] = True  # No file produced — treat as skip
        m = re.search(r'`([a-f0-9]{7,})`', summary)
        if m:
            result["git_hash"] = m.group(1)
        subj_name = (result["subject"] or "").capitalize()
        result["short_desc"] = f"DONE \u2014 {subj_name} build queue complete" if subj_name else "Build queue complete"
        return result

    # --- Edit/write failure ---
    if "Edit:" in summary[:80] or "Write failed" in summary[:80]:
        m = re.search(r'knowledge/(\w+)-curriculum/', summary)
        if m:
            result["subject"] = m.group(1)
        result["short_desc"] = summary.split("\n")[0][:150].lstrip("\u26a0\ufe0f \U0001f4dd ")
        return result

    # --- Fallback ---
    result["short_desc"] = summary.split("\n")[0][:150]
    return result


def _load_all_activity():
    """Scan all JSONL run files and return a sorted list of activity entries."""
    id_map = _build_job_id_map()
    entries = []

    if not _RUNS_DIR.is_dir():
        return entries

    for jsonl_file in _RUNS_DIR.glob("*.jsonl"):
        stem = jsonl_file.stem
        job_info = id_map.get(stem, {"name": stem, "display": stem, "desc": "", "type": "unknown"})

        try:
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue

                    ts = raw.get("ts", 0)
                    status = raw.get("status", "unknown")
                    duration_ms = raw.get("durationMs")
                    summary = raw.get("summary", "")
                    error_msg = raw.get("error")
                    model = raw.get("model")

                    entry = {
                        "ts": ts,
                        "job_name": job_info["display"],
                        "raw_name": job_info["name"],
                        "job_type": job_info["type"],
                        "status": status,
                        "duration_ms": duration_ms,
                        "model": model,
                    }

                    if job_info["type"] == "builder":
                        if summary:
                            parsed = _parse_builder_summary(summary)
                            # If summary says "Write failed" but the session actually succeeded,
                            # override with the real result (Kimi k2.5 retries until it works)
                            if not parsed["is_build"] and "Write failed" in summary:
                                session_id = raw.get("sessionId")
                                actual = _check_session_for_actual_success(session_id)
                                if actual:
                                    parsed = actual
                            entry["subject"] = parsed["subject"]
                            entry["task_id"] = parsed["task_id"]
                            entry["is_dedup"] = parsed["is_dedup"]
                            entry["is_build"] = parsed["is_build"]
                            entry["words"] = parsed["words"]
                            entry["output_file"] = parsed["output_file"]
                            entry["git_hash"] = parsed["git_hash"]
                            entry["bot_name"] = _BOT_NAMES.get(parsed["subject"]) if parsed["subject"] else None
                            entry["summary_short"] = parsed["short_desc"]
                        elif error_msg:
                            entry["summary_short"] = error_msg[:150]
                            entry["is_dedup"] = False
                            entry["is_build"] = False
                        else:
                            entry["summary_short"] = ""
                            entry["is_dedup"] = False
                            entry["is_build"] = False
                    else:
                        # System job — first non-empty line of summary
                        if summary:
                            first_line = summary.split("\n")[0][:200]
                            entry["summary_short"] = first_line
                        elif error_msg:
                            entry["summary_short"] = error_msg[:150]
                        else:
                            entry["summary_short"] = job_info["desc"]

                    entries.append(entry)
        except Exception:
            continue

    entries.sort(key=lambda e: e.get("ts", 0), reverse=True)
    return entries


@app.route('/activity', methods=['GET'])
def activity():
    """Return comprehensive chronological activity log from JSONL run files."""
    if CLOUD_MODE:
        return jsonify({"error": "Not available in cloud deployment"}), 404
    global _activity_cache, _activity_mtime

    limit = request.args.get('limit', 500, type=int)
    limit = min(limit, 500)

    try:
        current_mtime = _safe_mtime(_RUNS_DIR)
    except Exception:
        current_mtime = 0

    if _activity_cache is None or current_mtime != _activity_mtime:
        _activity_cache = _load_all_activity()
        _activity_mtime = current_mtime

    return jsonify({
        "entries": _activity_cache[:limit],
        "total": len(_activity_cache),
    })


@app.route('/usage', methods=['GET'])
def usage():
    """Return aggregated model usage stats from JSONL logs."""
    if CLOUD_MODE:
        return jsonify({"error": "Not available in cloud deployment"}), 404
    try:
        from usage_tracker import aggregate_usage
        return jsonify(aggregate_usage())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Knowledge Status ---

_KNOWLEDGE_DIR = WORKSPACE / "knowledge"
_knowledge_cache = {}
_KNOWLEDGE_TTL = 60  # seconds

_FILE_RE = re.compile(r'`([^`]+\.md)`')
_SECTION_RE = re.compile(r'^#{2,3}\s+(\d{2})\s*[—–-]?\s*(.+)', re.MULTILINE)


def _folder_label(folder_name):
    """Convert '01_national_curriculum' to 'National Curriculum'."""
    return re.sub(r'^\d+_', '', folder_name).replace('_', ' ').title()


def _match_folder(subject_dir, num, heading_name):
    """Match an INDEX heading to an actual folder on disk.
    Uses both the number prefix AND heading name for accuracy when
    multiple folders share the same number prefix.
    """
    candidates = [d for d in sorted(subject_dir.iterdir())
                  if d.is_dir() and d.name.startswith(num)]
    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates[0].name
    # Multiple folders with same number — match by name similarity
    heading_words = set(heading_name.lower().replace('-', ' ').replace('_', ' ').split())
    best = None
    best_score = -1
    for d in candidates:
        folder_words = set(d.name.lower().replace('_', ' ').split())
        score = len(heading_words & folder_words)
        if score > best_score:
            best_score = score
            best = d.name
    return best


def _parse_index_planned(index_path, subject_dir):
    """Parse 00_INDEX.md to get planned files grouped by folder.
    Returns {folder_name: [{file, description}]}.
    """
    try:
        text = index_path.read_text()
    except Exception:
        return {}

    planned = {}
    current_folder = None

    for line in text.splitlines():
        # Detect section headings like "## 01 — National Curriculum" or "### 02 Geographical Knowledge"
        heading_m = _SECTION_RE.match(line)
        if heading_m:
            num = heading_m.group(1)
            heading_name = heading_m.group(2).strip()
            current_folder = _match_folder(subject_dir, num, heading_name)
            if current_folder and current_folder not in planned:
                planned[current_folder] = []
            continue

        # Extract files from table rows
        file_m = _FILE_RE.search(line)
        if file_m:
            raw_path = file_m.group(1)
            # When path includes a folder component, derive folder from it directly.
            # This is more reliable than heading-number matching, which breaks when
            # INDEX section numbers don't match the actual folder number prefix
            # (e.g. "## 08 — Knowledge Richness" pointing to 07_knowledge_richness/).
            if '/' in raw_path:
                folder_from_path = raw_path.split('/')[0]
                filename = raw_path.split('/')[-1]
                # Use path-embedded folder if it exists on disk; fall back to heading-derived
                folder_key = folder_from_path if (subject_dir / folder_from_path).is_dir() else current_folder
            else:
                folder_key = current_folder
                filename = raw_path

            if folder_key is None:
                continue

            if folder_key not in planned:
                planned[folder_key] = []

            # Extract description from table columns
            cols = [c.strip() for c in line.split('|')]
            desc = ''
            for col in cols:
                col_clean = col.strip()
                if col_clean and '`' not in col_clean and col_clean not in (
                    'NOT YET BUILT', 'PENDING', 'TODO', '—', '-', 'COMPLETE', 'Status',
                    'File', 'Description', 'Built'
                ) and not col_clean.startswith('✅') and not col_clean.startswith('BUILT ') \
                  and not col_clean.startswith('**'):
                    desc = col_clean
                    break

            planned[folder_key].append({
                "file": filename,
                "description": desc,
            })

    return planned


def _scan_subject_dir(subject_dir):
    """Scan subject directory for actual .md files on disk.
    Returns {folder_name: [filename, ...]}.
    """
    skip_files = {'00_INDEX.md', 'BUILD_QUEUE.md', 'PROGRESS.md'}
    found = {}
    for d in sorted(subject_dir.iterdir()):
        if not d.is_dir() or d.name.startswith('.') or d.name == 'memory':
            continue
        files = []
        for f in sorted(d.iterdir()):
            if f.suffix == '.md' and f.name not in skip_files:
                files.append(f.name)
        if files:
            found[d.name] = files
    return found


def _parse_progress(progress_path, limit=5):
    """Parse PROGRESS.md for recent builds. Returns list of dicts."""
    try:
        text = progress_path.read_text()
    except Exception:
        return []

    builds = []
    for line in text.splitlines():
        m = re.match(
            r'\|\s*(20\d{2}-\d{2}-\d{2})\s*\|\s*(\S+)\s*\|\s*`?(.*?)`?\s*\|\s*([\d,]+)\s*\|\s*(.*?)\s*\|',
            line
        )
        if m:
            builds.append({
                "date": m.group(1),
                "task": m.group(2).strip(),
                "file": m.group(3).strip(),
                "words": m.group(4).strip(),
                "notes": m.group(5).strip(),
            })

    return builds[-limit:] if builds else []


def _count_subject_chunks(subject_key):
    """Count ChromaDB chunks for a subject."""
    try:
        results = pipeline.collection.get(
            where={"subject": subject_key},
            include=[]
        )
        return len(results.get("ids", []))
    except Exception:
        return -1


def _build_knowledge_status(subject_key, subject_config):
    """Build full knowledge status for a subject by scanning disk and parsing INDEX."""
    subject_dir = _KNOWLEDGE_DIR / f"{subject_key}-curriculum"
    if not subject_dir.is_dir():
        return {
            "subject": subject_key,
            "subject_name": subject_config.get("name", subject_key),
            "summary": {"total_planned": 0, "total_built": 0, "percent_complete": 0},
            "folders": [],
            "indexing": {"chromadb_chunks": 0, "embedding_model": "text-embedding-3-small",
                         "collection_name": "curriculum"},
            "recent_builds": [],
        }

    index_path = subject_dir / "00_INDEX.md"
    progress_path = subject_dir / "PROGRESS.md"

    # Get planned files from INDEX and actual files from disk
    planned = _parse_index_planned(index_path, subject_dir)
    on_disk = _scan_subject_dir(subject_dir)

    # Only show INDEX-planned folders; ignore orphan disk folders Dave may have created
    # by mistake (e.g. 03_curriculum_architecture in history, 03_key_thinkers in geography).
    # Fallback to on_disk if there is no INDEX file.
    if planned:
        all_folders = sorted(planned.keys())
    else:
        all_folders = sorted(on_disk.keys())

    folders = []
    total_planned = 0
    total_built = 0

    for folder in all_folders:
        planned_files = {f["file"]: f["description"] for f in planned.get(folder, [])}
        disk_files = set(on_disk.get(folder, []))

        # Build file list in INDEX order first, then any disk-only extras alphabetically.
        # This ensures the dashboard "next pending" matches the build queue order,
        # not alphabetical order (which would always show british_values_and_smsc first).
        planned_order = [f["file"] for f in planned.get(folder, [])]
        disk_only_extras = sorted(disk_files - set(planned_order))
        ordered_files = planned_order + disk_only_extras

        folder_files = []
        for fname in ordered_files:
            is_built = fname in disk_files
            desc = planned_files.get(fname, "")
            built_date = None

            if is_built:
                try:
                    mtime = (subject_dir / folder / fname).stat().st_mtime
                    built_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
                except OSError:
                    pass

            folder_files.append({
                "file": fname,
                "status": "built" if is_built else "pending",
                "description": desc,
                "built_date": built_date,
            })

        built_count = sum(1 for f in folder_files if f["status"] == "built")
        planned_count = len(folder_files)
        total_planned += planned_count
        total_built += built_count

        folders.append({
            "folder": folder,
            "label": _folder_label(folder),
            "planned": planned_count,
            "built": built_count,
            "files": folder_files,
        })

    pct = round(total_built / total_planned * 100) if total_planned > 0 else 0
    chunks = _count_subject_chunks(subject_key)

    return {
        "subject": subject_key,
        "subject_name": subject_config.get("name", subject_key),
        "summary": {
            "total_planned": total_planned,
            "total_built": total_built,
            "percent_complete": pct,
        },
        "folders": folders,
        "indexing": {
            "chromadb_chunks": chunks,
            "embedding_model": "text-embedding-3-small",
            "collection_name": "curriculum",
        },
        "recent_builds": _parse_progress(progress_path, limit=5),
    }


def _refresh_index_file(subject_key):
    """Update a subject's 00_INDEX.md status markers to match actual files on disk.
    Handles both History-format (folder/file in col, status in col 2) and
    Geography-format (bare file, status in col 3, built date in col 4).
    Also updates the summary status line.
    """
    subject_dir = _KNOWLEDGE_DIR / f"{subject_key}-curriculum"
    index_path = subject_dir / "00_INDEX.md"
    if not index_path.is_file():
        return

    text = index_path.read_text()
    lines = text.splitlines()
    current_folder = None
    total_planned = 0
    total_built = 0
    updated_lines = []

    for line in lines:
        # Track current section folder from headings
        heading_m = _SECTION_RE.match(line)
        if heading_m:
            num = heading_m.group(1)
            heading_name = heading_m.group(2).strip()
            current_folder = _match_folder(subject_dir, num, heading_name)
            updated_lines.append(line)
            continue

        # Check if this is a table row with a .md file reference
        file_m = _FILE_RE.search(line)
        if file_m and '|' in line:
            raw_path = file_m.group(1)
            filename = raw_path.split('/')[-1] if '/' in raw_path else raw_path
            # Use path-embedded folder when available — same fix as _parse_index_planned
            if '/' in raw_path:
                folder_from_path = raw_path.split('/')[0]
                effective_folder = folder_from_path if (subject_dir / folder_from_path).is_dir() else current_folder
            else:
                effective_folder = current_folder
            if effective_folder is None:
                updated_lines.append(line)
                continue
            file_on_disk = subject_dir / effective_folder / filename

            total_planned += 1

            if file_on_disk.is_file():
                total_built += 1
                try:
                    mtime = file_on_disk.stat().st_mtime
                    built_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
                except OSError:
                    built_date = datetime.now().strftime("%Y-%m-%d")

                # Determine format and update the status column
                cols = line.split('|')
                if len(cols) >= 4:
                    # Check which column has the status by looking for known markers
                    # History format: | `file` | STATUS | Description |
                    # Geography format: | `file` | Description | Status | Built |
                    status_updated = False
                    for i, col in enumerate(cols):
                        stripped = col.strip()
                        if stripped in ('NOT YET BUILT', 'PENDING') or stripped.startswith('⬜'):
                            # Found the pending status column — update it
                            if any(k in line for k in ('| Built |', '| — |', '|----')):
                                # Geography format: status col + built col
                                cols[i] = ' **COMPLETE** '
                                # Find the "Built" / "—" column (next one)
                                if i + 1 < len(cols) and cols[i + 1].strip() in ('—', '-', ''):
                                    cols[i + 1] = f' {built_date} '
                            else:
                                # History format: inline status
                                cols[i] = f' BUILT {built_date} '
                            status_updated = True
                            break
                    if status_updated:
                        line = '|'.join(cols)
            updated_lines.append(line)
            continue

        # Update the summary status line
        status_m = re.match(r'(##\s*Status:\s*)\d+\s*(of\s*)\d+(.*)$', line)
        if status_m:
            # Defer — we'll fix this after counting
            updated_lines.append(f"## Status: {{BUILT}} of {{PLANNED}}{status_m.group(3)}")
            continue

        # Geography-style "Total:" line
        total_m = re.match(r'\*\*Total:\s*\d+/\d+\s*items built\*\*', line)
        if total_m:
            updated_lines.append("**Total: {BUILT}/{PLANNED} items built**")
            continue

        updated_lines.append(line)

    # Replace placeholders with actual counts
    new_text = '\n'.join(updated_lines)
    new_text = new_text.replace('{BUILT}', str(total_built)).replace('{PLANNED}', str(total_planned))

    # Update last updated date
    today = datetime.now().strftime("%Y-%m-%d")
    new_text = re.sub(r'_Last updated: \d{4}-\d{2}-\d{2}_', f'_Last updated: {today}_', new_text)

    # Only write if changed
    if new_text != text:
        index_path.write_text(new_text)


@app.route('/knowledge-status/<subject>', methods=['GET'])
def knowledge_status_subject(subject):
    """Return detailed knowledge base build status for a single subject."""
    config = load_config()
    if subject not in config['subjects']:
        return jsonify({"error": f"Unknown subject: {subject}"}), 404

    now = time.time()
    cached = _knowledge_cache.get(subject)
    if cached and (now - cached['_ts']) < _KNOWLEDGE_TTL:
        return jsonify(cached['data'])

    try:
        data = _build_knowledge_status(subject, config['subjects'][subject])
        _knowledge_cache[subject] = {'data': data, '_ts': now}
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/knowledge-status', methods=['GET'])
def knowledge_status_all():
    """Return compact knowledge status summary for all subjects."""
    config = load_config()
    now = time.time()
    subjects_out = {}

    for key, val in config['subjects'].items():
        cached = _knowledge_cache.get(key)
        if cached and (now - cached['_ts']) < _KNOWLEDGE_TTL:
            d = cached['data']
        else:
            try:
                d = _build_knowledge_status(key, val)
                _knowledge_cache[key] = {'data': d, '_ts': now}
            except Exception:
                d = {"summary": {"total_planned": 0, "total_built": 0, "percent_complete": 0},
                     "indexing": {"chromadb_chunks": 0}}

        subjects_out[key] = {
            "name": val.get("name", key),
            "planned": d["summary"]["total_planned"],
            "built": d["summary"]["total_built"],
            "percent": d["summary"]["percent_complete"],
            "chunks": d["indexing"]["chromadb_chunks"],
        }

    total_chunks = sum(s["chunks"] for s in subjects_out.values() if s["chunks"] >= 0)
    return jsonify({
        "subjects": subjects_out,
        "indexing": {
            "total_chunks": total_chunks,
            "embedding_model": "text-embedding-3-small",
            "collection_name": "curriculum",
        },
    })


@app.route('/all-recent-builds', methods=['GET'])
def all_recent_builds():
    """Return all knowledge files built in the last N days, sorted newest first.
    Used by the dashboard to show actual build activity (the /activity log misses many).
    Query params: days=7 (default), limit=200 (default)."""
    if CLOUD_MODE:
        return jsonify({"error": "Not available in cloud deployment"}), 404
    days = int(request.args.get('days', 7))
    limit = int(request.args.get('limit', 200))
    config = load_config()
    cutoff = time.time() - (days * 86400)
    builds = []

    for key, val in config['subjects'].items():
        subject_dir = _KNOWLEDGE_DIR / f"{key}-curriculum"
        if not subject_dir.is_dir():
            continue
        bot_name = {"history": "BeckyBot", "geography": "NicolaBot", "science": "SimonBot",
                    "english": "ScottBot", "rs": "LaurenBot", "mfl": "JenniBot",
                    "maths": "RachaelKrisBot"}.get(key, key + " Agent")
        subject_name = val.get("name", key)

        for md_file in subject_dir.rglob("*.md"):
            # Skip metadata files
            if md_file.name in ("00_INDEX.md", "PROGRESS.md", "BUILD_QUEUE.md", "README.md"):
                continue
            try:
                stat = md_file.stat()
                if stat.st_mtime < cutoff:
                    continue
                content = md_file.read_text(encoding="utf-8", errors="replace")
                words = len(content.split())
                # Get relative path within knowledge dir
                rel = str(md_file.relative_to(_KNOWLEDGE_DIR))
                folder = md_file.parent.name
                builds.append({
                    "ts": int(stat.st_mtime * 1000),
                    "subject": key,
                    "subject_name": subject_name,
                    "bot_name": bot_name,
                    "file": rel,
                    "folder": folder,
                    "filename": md_file.stem.replace("_", " ").title(),
                    "words": words,
                })
            except (OSError, UnicodeDecodeError):
                continue

    builds.sort(key=lambda b: b["ts"], reverse=True)
    return jsonify({"builds": builds[:limit], "total": len(builds)})


@app.route('/subjects', methods=['GET'])
def subjects():
    """List available subjects (legacy endpoint)."""
    return jsonify({"subjects": ["history", "geography"]})


VISUALISE_SYSTEM_PROMPT = """You are a world-class infographic designer and HTML/SVG specialist creating visuals for UK curriculum leaders.

You will be given a question and an expert curriculum answer. Read both carefully, identify the SINGLE most powerful visual structure for this content, then build it in HTML/SVG/Chart.js.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE RULES — NEVER BREAK THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Return ONLY raw HTML. No markdown, no code fences, no explanation. Start immediately with: <div class="infographic"
2. NEVER use <table>, <thead>, <tbody>, <tr>, <td>, or <th>. Tables are banned entirely.
3. The infographic must look stunning at 1100px wide.
4. Colour palette ONLY: navy #1a365d, amber #c05621, slate #4a5568, sage #276749, gold #d69e2e, light bg #f7fafc. No other colours.
5. Font: system-ui, -apple-system, sans-serif throughout.
6. All styles via inline <style> block or inline style="" attributes. No external CSS.
7. Title and heading text MUST use HTML elements (<h1>, <h2>, <p>) — NEVER SVG <text>. SVG text elements lose word spacing and are banned for titles.
8. ALWAYS use UK English spelling and grammar throughout (e.g. colour, organise, programme, behaviour, centre, specialised, analysed). Never use American English.
9. NEVER use position:absolute or position:relative to place content cards or text boxes. Use CSS Grid or flexbox ONLY. Absolute positioning causes overlapping content which ruins the infographic. The only exception is small decorative SVG elements (arrows, connector lines).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CHOOSE YOUR VISUAL TYPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pick EXACTLY ONE based on the content:

A) TIMELINE — for historical sequences, curriculum evolution, policy changes
   Structure: SVG horizontal track with filled circles at each node, year label above, event text below. Minimum 4 nodes. Use navy track line, amber filled circles.

B) COMPARISON CARDS — for contrasting two or more concepts, thinkers, or approaches
   Structure: CSS Grid with one card per item. Each card: coloured top bar (alternate navy/amber), concept name in white on bar, bullet points below. NO tables.

C) CHART.JS VISUAL — for any ranking, quantity, proportion, or spectrum data
   Structure: Load Chart.js from https://cdn.jsdelivr.net/npm/chart.js then render a bar, radar, or doughnut chart in a <canvas>. Must include a clear legend and title.

D) FRAMEWORK DIAGRAM — for models, hierarchies, spectrums, or processes with named stages
   Structure: Use CSS Grid or flexbox (NOT absolute positioning) showing the stages/levels. Use coloured bands, numbered steps, or a vertical/horizontal flow. NEVER use radial/circular layouts with absolute-positioned boxes — these always overlap at 1100px. If showing 4-6 concepts, use a 2×3 or 3×2 CSS Grid with generous gap (20px+). Each card must have a fixed min-width and the layout must not allow overlap.

E) CONCEPT MAP — for key thinkers, ideas, or terms with brief descriptions
   Structure: CSS Grid of labelled icon-cards with explicit grid-template-columns (e.g. repeat(auto-fit, minmax(280px, 1fr))) and gap: 20px. Each card: large initial letter or emoji in a coloured circle, thinker/concept name bold, 2-3 line description. NEVER use absolute positioning or transform to place cards — always use CSS Grid flow. Cards must never overlap.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — BUILD WITH POLISH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Begin with a strong header band: navy (#1a365d) background, generous padding (24px 32px).
  * Title MUST be an HTML <h1> element (NOT SVG <text> — SVG text loses word spacing). White (#fff), 26px, font-weight 700.
  * Subtitle MUST be an HTML <p> element in gold (#d69e2e), 15px, font-weight 400.
  * Title text MUST preserve natural word spacing. NEVER concatenate words. If the topic is "Mathematical Reasoning at KS2", the title must read exactly that with spaces.
- Each section/card must have clear visual separation — padding, subtle shadow (box-shadow: 0 2px 8px rgba(0,0,0,0.08)), rounded corners (border-radius: 10px)
- Use generous whitespace — padding: 20px minimum inside cards
- Include a small footer: "Curriculum Expert · Westcountry Schools Trust" in slate, 11px
- Minimum height: 400px. Aim for something that genuinely looks like a designed asset.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY BAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ask yourself: would a curriculum leader be proud to include this in a presentation to governors? If not, redesign it. A table with borders is never the answer. An SVG timeline, a Chart.js radar, or a grid of rich cards always is."""


def _fix_title_spacing(html: str) -> str:
    """Fix common LLM title spacing issues — camelCase concatenation in heading elements."""
    def fix_header_text(match):
        tag, attrs, text, close = match.group(1), match.group(2), match.group(3), match.group(4)
        # Insert space before uppercase letters that follow lowercase (camelCase split)
        fixed = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # Fix missing space before common prepositions/articles stuck to prior word
        fixed = re.sub(r'([a-zA-Z])(at|in|of|for|and|the|on|to|by|with)\s', r'\1 \2 ', fixed)
        return f'<{tag}{attrs}>{fixed}</{close}>'
    return re.sub(
        r'<(h[1-4])([^>]*)>(.*?)</(h[1-4])>',
        fix_header_text, html,
        flags=re.IGNORECASE | re.DOTALL
    )


def call_visualise(question: str, answer: str) -> str:
    """Generate infographic HTML via GPT-4.1 (fallback: Claude Opus)."""
    user_message = f"""Question asked: {question}

Expert curriculum answer:
{answer}

Now create a rich, beautiful HTML infographic that captures the key information from this answer visually."""

    def _generate_html():
        """Generate the HTML infographic (GPT-4.1 first, Opus fallback)."""
        # --- Try OpenAI GPT-4.1 first ---
        oai_client = _get_openai_llm_client()
        if oai_client:
            try:
                response = oai_client.chat.completions.create(
                    model="gpt-4.1",
                    max_tokens=4096,
                    messages=[
                        {"role": "system", "content": VISUALISE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ]
                )
                html = response.choices[0].message.content.strip()
                html = html.replace("```html", "").replace("```", "").strip()
                if html:
                    print("Infographic generated via OpenAI GPT-4.1")
                    return html
            except Exception as e:
                print(f"OpenAI visualise call failed, falling back to Opus: {e}")

        # --- Fall back to Claude Opus ---
        client = _get_anthropic_client()
        if client:
            try:
                response = client.messages.create(
                    model="claude-opus-4-5-20251101",
                    max_tokens=4096,
                    system=VISUALISE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}]
                )
                html = response.content[0].text.strip()
                html = html.replace("```html", "").replace("```", "").strip()
                print("Infographic generated via Claude Opus (fallback)")
                return html
            except Exception as e:
                print(f"Opus visualise call failed: {e}")

        return None

    # --- Generate HTML infographic ---
    html_result = _generate_html()

    if not html_result:
        return None

    # Apply title spacing fix
    html_result = _fix_title_spacing(html_result)

    return html_result


@app.route('/visualise', methods=['POST'])
def visualise():
    """Generate a rich HTML infographic from a curriculum answer.

    Uses OpenAI GPT-4o as primary model, falls back to Claude Opus.

    POST body: { "question": str, "answer": str }
    Returns: { "html": str } or { "error": str }
    """
    data = request.get_json() or {}
    question = data.get('question', '').strip()
    answer = data.get('answer', '').strip()

    if not question or not answer:
        return jsonify({"error": "Both 'question' and 'answer' are required."}), 400

    if not _get_openai_key() and not _get_anthropic_key():
        return jsonify({
            "error": "No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
            "setup_hint": "export OPENAI_API_KEY=sk-... or ANTHROPIC_API_KEY=sk-ant-..."
        }), 503

    _t0 = time.time()
    html = call_visualise(question, answer)
    _dur = int((time.time() - _t0) * 1000)
    log_infographic(question, subject=None, duration_ms=_dur, success=bool(html))
    if html:
        return jsonify({"html": html})
    else:
        return jsonify({"error": "Visualisation generation failed. Check server logs."}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', os.environ.get('RAG_PORT', 8000)))
    print(f"Curriculum Expert starting on port {port}...")
    print(f"Collection count: {pipeline.collection.count()}")
    print(f"PDF support: {HAS_PDF}")
    print(f"DOCX support: {HAS_DOCX}")
    print(f"Anthropic support: {HAS_ANTHROPIC}")
    app.run(host='0.0.0.0', port=port, debug=False)
