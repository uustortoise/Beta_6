from utils.elder_id_utils import (
    apply_canonical_alias_map,
    choose_canonical_elder_id,
    elder_id_lineage_matches,
    parse_elder_id_from_filename,
)


def test_parse_elder_id_from_filename_extracts_prefix_and_name():
    assert parse_elder_id_from_filename("HK0011_jessica_train_4dec2025.xlsx") == "HK0011_jessica"
    assert parse_elder_id_from_filename("bad_name.xlsx") == "bad_name"
    assert parse_elder_id_from_filename("invalid.xlsx") == "resident_01"


def test_elder_id_lineage_allows_numeric_suffix_drift():
    assert elder_id_lineage_matches("HK001_jessica", "HK0011_jessica") is True
    assert elder_id_lineage_matches("HK0011_jessica", "HK001_jessica") is True
    assert elder_id_lineage_matches("HK001_jessica", "HK002_jessica") is False


def test_choose_canonical_elder_id_prefers_shorter_numeric_token():
    out = choose_canonical_elder_id(["HK0011_jessica", "HK001_jessica"])
    assert out == "HK001_jessica"


def test_apply_canonical_alias_map_from_env(monkeypatch):
    monkeypatch.setenv("ELDER_ID_CANONICAL_MAP", "HK0011_jessica=HK001_jessica")
    assert apply_canonical_alias_map("HK0011_jessica") == "HK001_jessica"
    assert apply_canonical_alias_map("HK001_jessica") == "HK001_jessica"
