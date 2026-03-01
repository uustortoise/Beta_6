import pandas as pd

from utils.correction_eval_history import (
    enrich_correction_evaluation_df,
    parse_json_object,
    summarize_correction_evaluation_decisions,
)


def test_parse_json_object_handles_invalid_and_valid_payloads():
    assert parse_json_object(None) == {}
    assert parse_json_object("not-json") == {}
    assert parse_json_object('["x"]') == {}
    assert parse_json_object({"k": 1}) == {"k": 1}
    assert parse_json_object('{"k": 2}') == {"k": 2}


def test_enrich_correction_evaluation_df_extracts_metadata_fields():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "metadata": '{"decision":"PASS","artifact_path":"/tmp/a.json","corrected_window_report":{"local_gain":0.12,"global_drop":0.01}}',
            },
            {
                "id": 2,
                "metadata": '{"decision":"FAIL","corrected_window_report":{"local_gain":-0.03,"global_drop":0.05}}',
            },
            {
                "id": 3,
                "metadata": "bad-json",
            },
        ]
    )

    out = enrich_correction_evaluation_df(df)

    assert list(out["decision"]) == ["PASS", "FAIL", "unknown"]
    assert out.loc[0, "artifact_path"] == "/tmp/a.json"
    assert abs(float(out.loc[0, "local_gain"]) - 0.12) < 1e-9
    assert abs(float(out.loc[1, "global_drop"]) - 0.05) < 1e-9
    assert pd.isna(out.loc[2, "local_gain"])


def test_summarize_correction_evaluation_decisions_counts_statuses():
    df = pd.DataFrame(
        {
            "decision": ["PASS", "PASS_WITH_FLAG", "FAIL", "PASS", "unknown"],
        }
    )
    summary = summarize_correction_evaluation_decisions(df)
    assert summary == {
        "total": 5,
        "PASS": 2,
        "PASS_WITH_FLAG": 1,
        "FAIL": 1,
    }
