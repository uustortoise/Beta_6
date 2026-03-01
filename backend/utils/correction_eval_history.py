import json
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd


def parse_json_object(raw_value: Any) -> Dict[str, Any]:
    """Parse a JSON-like payload into a dict."""
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def enrich_correction_evaluation_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized evaluation columns derived from training_history.metadata.
    """
    if df.empty:
        return df

    enriched = df.copy()
    enriched["metadata_obj"] = enriched["metadata"].apply(parse_json_object)
    enriched["decision"] = enriched["metadata_obj"].apply(lambda x: x.get("decision", "unknown"))
    enriched["artifact_path"] = enriched["metadata_obj"].apply(lambda x: x.get("artifact_path"))
    enriched["local_gain"] = pd.to_numeric(
        enriched["metadata_obj"].apply(
            lambda x: (x.get("corrected_window_report") or {}).get("local_gain")
        ),
        errors="coerce",
    )
    enriched["global_drop"] = pd.to_numeric(
        enriched["metadata_obj"].apply(
            lambda x: (x.get("corrected_window_report") or {}).get("global_drop")
        ),
        errors="coerce",
    )
    return enriched


def summarize_correction_evaluation_decisions(df: pd.DataFrame) -> Dict[str, int]:
    """Return decision counts for evaluation rows."""
    if df.empty or "decision" not in df.columns:
        return {"total": 0, "PASS": 0, "PASS_WITH_FLAG": 0, "FAIL": 0}

    return {
        "total": int(len(df)),
        "PASS": int((df["decision"] == "PASS").sum()),
        "PASS_WITH_FLAG": int((df["decision"] == "PASS_WITH_FLAG").sum()),
        "FAIL": int((df["decision"] == "FAIL").sum()),
    }


def fetch_and_enrich_correction_evaluations(
    query_fn,
    elder_filter=None,
    days: int = 3650,
    now: datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch correction evaluation rows via an injected query function and enrich fields.

    Args:
        query_fn: Callable `(query: str, params: tuple) -> pd.DataFrame`
        elder_filter: Optional resident filter
        days: Window in days
        now: Optional clock override for deterministic tests
    """
    clock = now or datetime.now()
    query = """
        SELECT id, elder_id, training_date, model_type, epochs, accuracy, status, metadata
        FROM training_history
        WHERE model_type = 'Correction Retrain'
          AND training_date >= ?
    """
    cutoff = (clock - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    params = [cutoff]
    if elder_filter:
        query += " AND elder_id = ?"
        params.append(elder_filter)
    query += " ORDER BY training_date DESC"

    df = query_fn(query, tuple(params))
    return enrich_correction_evaluation_df(df)
