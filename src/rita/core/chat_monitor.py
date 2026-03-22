"""
RITA Core — Chat Monitor

CSV-based monitoring for the RITA chat classifier.
Captures per-query metrics: intent, confidence, handler, latency, response preview.

CSV stored at: {OUTPUT_DIR}/chat_monitor.csv
No database required — plain stdlib csv module only.
"""

import csv
import os
from datetime import datetime, timezone

from rita.config import OUTPUT_DIR as _DEFAULT_OUTPUT_DIR

COLUMNS = [
    "id", "timestamp", "query_text", "intent_name", "handler",
    "confidence", "low_confidence", "latency_ms", "response_preview", "status",
]


def _csv_path() -> str:
    return os.path.join(os.getenv("OUTPUT_DIR", _DEFAULT_OUTPUT_DIR), "chat_monitor.csv")


def _read_rows() -> list[dict]:
    """Read all rows from CSV; return [] if file does not exist."""
    path = _csv_path()
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def log_query(
    query_text: str,
    intent_name: str,
    handler: str,
    confidence: float,
    low_confidence: bool,
    latency_ms: float,
    response_preview: str,
    status: str = "success",
) -> None:
    """Append one chat query row to the CSV."""
    path = _csv_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path)
    rows = _read_rows()
    next_id = len(rows) + 1
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "id":               next_id,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "query_text":       query_text[:500],
            "intent_name":      intent_name,
            "handler":          handler,
            "confidence":       round(confidence, 4),
            "low_confidence":   int(low_confidence),
            "latency_ms":       round(latency_ms, 1),
            "response_preview": response_preview[:200],
            "status":           status,
        })


# ── Read helpers ────────────────────────────────────────────────────────────────

def get_summary() -> dict:
    """Return KPI aggregates across all recorded chat queries."""
    rows = _read_rows()
    if not rows:
        return {"total_queries": 0, "avg_confidence": 0.0,
                "avg_latency_ms": 0.0, "low_conf_pct": 0.0, "queries_today": 0}
    today = datetime.now(timezone.utc).date().isoformat()
    confs = [float(r["confidence"]) for r in rows if r["confidence"]]
    lats  = [float(r["latency_ms"]) for r in rows if r["latency_ms"]]
    low   = sum(1 for r in rows if r["low_confidence"] == "1")
    today_count = sum(1 for r in rows if r["timestamp"].startswith(today))
    return {
        "total_queries":  len(rows),
        "avg_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
        "avg_latency_ms": round(sum(lats)  / len(lats),  1) if lats  else 0.0,
        "low_conf_pct":   round(low / len(rows) * 100, 1),
        "queries_today":  today_count,
    }


def get_recent_queries(limit: int = 20) -> list:
    """Return the most recent chat queries, newest first."""
    rows = _read_rows()
    return list(reversed(rows[-limit:])) if rows else []


def get_intent_distribution() -> list:
    """Return query count grouped by intent_name with handler, descending."""
    rows = _read_rows()
    groups: dict[str, dict] = {}
    for r in rows:
        name = r.get("intent_name") or "unknown"
        if name not in groups:
            groups[name] = {"intent_name": name, "handler": r.get("handler", ""),
                            "count": 0, "conf_sum": 0.0}
        groups[name]["count"] += 1
        try:
            groups[name]["conf_sum"] += float(r["confidence"])
        except (ValueError, KeyError):
            pass
    result = []
    for g in groups.values():
        avg_conf = round(g["conf_sum"] / g["count"], 3) if g["count"] else 0.0
        result.append({
            "intent_name":    g["intent_name"],
            "handler":        g["handler"],
            "count":          g["count"],
            "avg_confidence": avg_conf,
        })
    return sorted(result, key=lambda x: x["count"], reverse=True)


def get_confidence_trend(n: int = 20) -> list:
    """Return rolling confidence + latency for the last n queries (oldest first)."""
    rows = _read_rows()
    recent = rows[-n:] if len(rows) >= n else rows
    return [
        {"confidence": float(r["confidence"]) if r["confidence"] else 0.0,
         "latency_ms": float(r["latency_ms"]) if r["latency_ms"] else 0.0,
         "timestamp":  r["timestamp"],
         "intent_name": r["intent_name"]}
        for r in recent
    ]
