import datetime
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "app/pages/1_correction_studio.py"
MODULE_SPEC = importlib.util.spec_from_file_location("correction_studio_page", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
correction_studio_page = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(correction_studio_page)


def test_apply_pending_widget_date_moves_pending_value_into_widget_state():
    state = {
        "correction_selected_date": datetime.date(2026, 3, 9),
        "correction_pending_selected_date": datetime.date(2026, 3, 7),
    }

    selected_date = correction_studio_page._apply_pending_widget_date(
        state,
        widget_key="correction_selected_date",
        pending_key="correction_pending_selected_date",
        default_date=datetime.date(2026, 3, 1),
    )

    assert selected_date == datetime.date(2026, 3, 7)
    assert state["correction_selected_date"] == datetime.date(2026, 3, 7)
    assert "correction_pending_selected_date" not in state


def test_apply_pending_widget_date_sets_default_when_widget_is_missing():
    state = {}

    selected_date = correction_studio_page._apply_pending_widget_date(
        state,
        widget_key="correction_selected_date",
        pending_key="correction_pending_selected_date",
        default_date=datetime.date(2026, 3, 5),
    )

    assert selected_date == datetime.date(2026, 3, 5)
    assert state["correction_selected_date"] == datetime.date(2026, 3, 5)


def test_queue_widget_date_update_stashes_next_rerun_date():
    state = {}

    correction_studio_page._queue_widget_date_update(
        state,
        pending_key="correction_pending_selected_date",
        next_date="2026-03-08",
    )

    assert state["correction_pending_selected_date"] == datetime.date(2026, 3, 8)
