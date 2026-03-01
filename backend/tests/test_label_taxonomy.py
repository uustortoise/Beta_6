from ml import label_taxonomy


def test_critical_labels_are_canonicalized():
    labels = label_taxonomy.get_critical_labels_for_room("Bedroom")
    assert "sleep" in labels
    assert "sleeping" not in labels


def test_alias_equivalents_include_sleep_and_shower_pairs():
    eq = label_taxonomy.get_label_alias_equivalents()
    assert set(eq["sleep"]) >= {"sleep", "sleeping"}
    assert set(eq["sleeping"]) >= {"sleep", "sleeping"}
    assert set(eq["shower"]) >= {"shower", "showering"}
    assert set(eq["showering"]) >= {"shower", "showering"}


def test_valid_prediction_labels_include_alias_and_canonical_tokens():
    labels = label_taxonomy.get_valid_prediction_labels()
    assert "sleep" in labels
    assert "sleeping" in labels
    assert "shower" in labels
    assert "showering" in labels
