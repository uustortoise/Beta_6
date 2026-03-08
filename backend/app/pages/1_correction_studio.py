import datetime
import sys
from pathlib import Path

_backend = str(Path(__file__).resolve().parent.parent.parent)
if _backend not in sys.path: sys.path.extend([_backend, str(Path(_backend).parent)])

import altair as alt
import pandas as pd
import streamlit as st
from elderlycare_v1_16.config.settings import (
    DEFAULT_DENOISING_THRESHOLD,
    DEFAULT_DENOISING_WINDOW,
)
from elderlycare_v1_16.preprocessing.noise import hampel_filter

try:
    from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode
    AGGRID_AVAILABLE = True
except ModuleNotFoundError:
    AgGrid = None
    DataReturnMode = None
    GridOptionsBuilder = None
    GridUpdateMode = None
    AGGRID_AVAILABLE = False

import services.correction_service as correction_service
from app._sidebar import render_sidebar
from config import get_room_config

def _normalize_selected_rows(selected_rows):
    if selected_rows is None:
        return []
    if isinstance(selected_rows, pd.DataFrame):
        return selected_rows.to_dict("records")
    if isinstance(selected_rows, list):
        return selected_rows
    return []


def _selected_time_bounds(selected_rows: list[dict]):
    timestamps = []
    for row in selected_rows:
        ts = pd.to_datetime(row.get("timestamp"), errors="coerce")
        if pd.notna(ts):
            timestamps.append(ts)
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def _render_review_queue_selector(queue_df: pd.DataFrame) -> list[dict]:
    display_cols = [
        "timestamp",
        "activity_type",
        "confidence",
        "duration_minutes",
        "review_reason",
        "training_activity_type",
        "review_source",
    ]
    display_cols = [col for col in display_cols if col in queue_df.columns]
    queue_display_df = queue_df.copy().reset_index(drop=True)
    queue_display_df["_queue_row_id"] = queue_display_df.index.astype(int)

    if AGGRID_AVAILABLE:
        gb = GridOptionsBuilder.from_dataframe(queue_display_df[["_queue_row_id", *display_cols]])
        gb.configure_selection("multiple", use_checkbox=True, header_checkbox=True)
        gb.configure_column("_queue_row_id", hide=True)
        if "confidence" in queue_display_df.columns:
            gb.configure_column(
                "confidence",
                type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
                valueFormatter="data.confidence.toFixed(2)",
            )

        jscode = """
        function(params) {
            if (params.data.confidence < 0.3) {
                return {'backgroundColor': '#ffebee'};
            } else if (params.data.confidence < 0.5) {
                return {'backgroundColor': '#fff3e0'};
            } else {
                return {'backgroundColor': '#f1f8e9'};
            }
        }
        """
        gb.configure_grid_options(getRowStyle=jscode)
        grid_response = AgGrid(
            queue_display_df,
            gridOptions=gb.build(),
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            fit_columns_on_grid_load=True,
            theme="balham",
            allow_unsafe_jscode=True,
            height=500,
        )
        return _normalize_selected_rows(grid_response.get("selected_rows"))

    st.warning("`st_aggrid` is not installed. Using the built-in fallback review selector.")
    fallback_df = queue_display_df[display_cols].copy()
    if "timestamp" in fallback_df.columns:
        fallback_df["timestamp"] = pd.to_datetime(fallback_df["timestamp"], errors="coerce")
    st.dataframe(fallback_df, use_container_width=True, height=500)

    option_labels = []
    option_map: dict[str, int] = {}
    for _, row in queue_display_df.iterrows():
        ts = pd.to_datetime(row.get("timestamp"), errors="coerce")
        ts_txt = ts.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(ts) else "unknown time"
        label = row.get("activity_type") or "unknown"
        reason = row.get("review_reason") or "clear"
        option = f"{ts_txt} | {label} | {reason}"
        option_labels.append(option)
        option_map[option] = int(row["_queue_row_id"])

    selected_options = st.multiselect(
        "Select review rows",
        options=option_labels,
        key="fallback_review_queue_selector",
    )
    selected_ids = {option_map[option] for option in selected_options}
    selected_df = queue_display_df[queue_display_df["_queue_row_id"].isin(selected_ids)].copy()
    return selected_df.drop(columns=["_queue_row_id"], errors="ignore").to_dict("records")


def _render_sensor_panel(sensor_df: pd.DataFrame):
    st.subheader("Sensor Data Context (Legacy Parity)")
    st.caption("Select one or more sensors to inspect raw/denoised trends alongside activity timeline.")

    if sensor_df.empty:
        st.info("No sensor context rows found for the selected resident / room / date.")
        return

    sensor_cols = [
        c
        for c in correction_service.SENSOR_COLUMNS
        if c in sensor_df.columns and sensor_df[c].notna().any()
    ]
    if not sensor_cols:
        st.info("No usable sensor features were found in `adl_history.sensor_features` for this day.")
        return

    selected_sensors = st.multiselect(
        "Select Sensors to Visualize",
        options=sensor_cols,
        default=sensor_cols[: min(2, len(sensor_cols))],
        key="advanced_sensor_selection",
    )
    show_denoised = st.checkbox(
        "Apply Denoising Preview",
        value=True,
        help="Preview only. No database values are changed.",
        key="advanced_sensor_denoise_toggle",
    )

    preview_window = DEFAULT_DENOISING_WINDOW
    preview_sigma = DEFAULT_DENOISING_THRESHOLD
    if show_denoised:
        col_w, col_s = st.columns(2)
        with col_w:
            preview_window = st.slider(
                "Denoise Window",
                min_value=2,
                max_value=7,
                value=DEFAULT_DENOISING_WINDOW,
                key="advanced_sensor_denoise_window",
            )
        with col_s:
            preview_sigma = st.slider(
                "Denoise Sigma",
                min_value=2.0,
                max_value=5.0,
                value=float(DEFAULT_DENOISING_THRESHOLD),
                step=0.5,
                key="advanced_sensor_denoise_sigma",
            )

    if not selected_sensors:
        st.info("Choose at least one sensor to render charts.")
        return

    plot_df = sensor_df[["timestamp", *selected_sensors]].copy()
    if show_denoised:
        try:
            hampel_filter(
                plot_df,
                selected_sensors,
                window=preview_window,
                n_sigmas=preview_sigma,
                inplace=True,
            )
            st.caption(f"Denoised preview active (window={preview_window}, sigma={preview_sigma}).")
        except Exception as e:
            st.warning(f"Denoising preview failed; showing raw values. ({e})")

    for sensor in selected_sensors:
        sensor_data = plot_df[["timestamp", sensor]].copy()
        sensor_data["time"] = sensor_data["timestamp"].dt.strftime("%H:%M:%S")
        chart = (
            alt.Chart(sensor_data)
            .mark_line(color="steelblue")
            .encode(
                x=alt.X("timestamp:T", axis=alt.Axis(format="%H:%M", title=None)),
                y=alt.Y(f"{sensor}:Q", axis=alt.Axis(title=sensor.capitalize()), scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("time:N", title="Time"),
                    alt.Tooltip(f"{sensor}:Q", title=sensor.capitalize(), format=".2f"),
                ],
            )
            .properties(height=120)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)


def _render_activity_timeline_chart(timeline_df: pd.DataFrame, title: str, y_field: str = "activity_type"):
    timeline_clean = timeline_df[
        timeline_df["activity_type"].notna() & (timeline_df["activity_type"] != "")
    ].copy()
    if timeline_clean.empty:
        return

    if y_field == "room":
        timeline_clean["room"] = timeline_clean["room"].fillna("unknown").astype(str)
        y_encoding = alt.Y("room:N", title="Room", sort="-x")
    else:
        y_encoding = alt.Y("activity_type:N", title="Predicted Activity", sort="-x")

    chart = (
        alt.Chart(timeline_clean)
        .mark_bar()
        .encode(
            x=alt.X("timestamp:T", axis=alt.Axis(format="%H:%M", title="Time")),
            x2="end_time:T",
            y=y_encoding,
            color=alt.Color("activity_type:N", scale=alt.Scale(scheme="tableau20"), legend=None),
            opacity=alt.condition(
                alt.datum.is_corrected == 1,
                alt.value(1.0),
                alt.value(0.7),
            ),
            tooltip=[
                alt.Tooltip("time:N", title="Time"),
                alt.Tooltip("room:N", title="Room"),
                alt.Tooltip("activity_type:N", title="Activity"),
                alt.Tooltip("confidence:Q", title="Confidence", format=".2f"),
                alt.Tooltip("is_corrected:N", title="Corrected"),
            ],
        )
        .properties(
            title=title,
            height=220,
            width="container",
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def _render_prediction_compare_chart(timeline_df: pd.DataFrame, title: str):
    timeline_clean = timeline_df[
        timeline_df["activity_type"].notna() & (timeline_df["activity_type"] != "")
    ].copy()
    if timeline_clean.empty:
        st.info("No prediction timeline blocks found for this day.")
        return

    chart = (
        alt.Chart(timeline_clean)
        .mark_bar()
        .encode(
            x=alt.X("timestamp:T", axis=alt.Axis(format="%H:%M", title="Time")),
            x2="end_time:T",
            y=alt.Y("activity_type:N", title="Prediction", sort="-x"),
            color=alt.condition(
                "datum.review_reason != 'clear'",
                alt.value("#d97706"),
                alt.Color("activity_type:N", scale=alt.Scale(scheme="tableau20"), legend=None),
            ),
            opacity=alt.condition(
                "datum.review_reason != 'clear'",
                alt.value(1.0),
                alt.value(0.7),
            ),
            tooltip=[
                alt.Tooltip("time:N", title="Time"),
                alt.Tooltip("activity_type:N", title="Predicted Label"),
                alt.Tooltip("prediction_state_label:N", title="Prediction State"),
                alt.Tooltip("training_activity_type:N", title="Training / Corrected Label"),
                alt.Tooltip("review_reason:N", title="Review Reason"),
                alt.Tooltip("confidence:Q", title="Confidence", format=".2f"),
            ],
        )
        .properties(
            title=title,
            height=220,
            width="container",
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def _context_queue(elder_id: str, room: str, selected_date: datetime.date) -> list[dict]:
    queue = st.session_state.setdefault("correction_batch_queue", [])
    date_key = selected_date.isoformat()
    return [
        item
        for item in queue
        if item.get("elder_id") == elder_id and item.get("room") == room and item.get("record_date") == date_key
    ]


def _render_batch_queue_panel(elder_id: str, room: str, selected_date: datetime.date):
    st.subheader("Batch Label Queue (Legacy Parity)")
    st.caption("Queue multiple correction ranges, apply once, and optionally trigger one-shot retrain.")

    queue = st.session_state.setdefault("correction_batch_queue", [])
    ctx_queue = _context_queue(elder_id, room, selected_date)
    preview = correction_service.preview_batch_corrections(
        elder_id=elder_id,
        room=room,
        record_date=selected_date,
        corrections=ctx_queue,
    )

    all_labels = correction_service.get_activity_labels()
    default_label_index = min(3, len(all_labels) - 1) if all_labels else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        batch_start = st.time_input(
            "Start Time",
            value=datetime.time(hour=0, minute=0),
            step=60,
            key=f"batch_start_{room}_{selected_date.isoformat()}",
        )
    with c2:
        batch_end = st.time_input(
            "End Time",
            value=datetime.time(hour=0, minute=10),
            step=60,
            key=f"batch_end_{room}_{selected_date.isoformat()}",
        )
    with c3:
        batch_label = st.selectbox(
            "Activity",
            options=all_labels,
            index=default_label_index,
            key=f"batch_label_{room}_{selected_date.isoformat()}",
        )

    add_col, apply_col = st.columns(2)
    with add_col:
        if st.button("Add To Queue", key=f"batch_add_{room}_{selected_date.isoformat()}"):
            if batch_start >= batch_end:
                st.error("Start time must be before end time.")
            else:
                has_overlap = any(
                    batch_start < item.get("end_time") and item.get("start_time") < batch_end
                    for item in ctx_queue
                )
                duplicate = any(
                    item.get("start_time") == batch_start
                    and item.get("end_time") == batch_end
                    and item.get("new_label") == batch_label
                    for item in ctx_queue
                )
                if duplicate:
                    st.warning("This correction range is already queued.")
                elif has_overlap:
                    st.error("Range overlaps with an existing queued correction. Resolve overlaps before adding.")
                else:
                    queue.append(
                        {
                            "elder_id": elder_id,
                            "room": room,
                            "record_date": selected_date.isoformat(),
                            "start_time": batch_start,
                            "end_time": batch_end,
                            "new_label": batch_label,
                        }
                    )
                    st.success("Queued correction range.")
                    st.rerun()

    with apply_col:
        run_retrain = st.checkbox(
            "Retrain Once After Apply",
            value=False,
            key=f"batch_retrain_{room}_{selected_date.isoformat()}",
        )
        train_retro = st.checkbox(
            "Include Archive In Retrain",
            value=False,
            disabled=not run_retrain,
            key=f"batch_retro_{room}_{selected_date.isoformat()}",
        )
        if st.button(
            f"Apply All ({len(ctx_queue)})",
            type="primary",
            disabled=len(ctx_queue) == 0 or bool(preview.get("has_invalid")),
            key=f"batch_apply_{room}_{selected_date.isoformat()}",
        ):
            with st.spinner("Applying queued corrections..."):
                result = correction_service.apply_batch_corrections(
                    elder_id=elder_id,
                    room=room,
                    record_date=selected_date,
                    corrections=ctx_queue,
                    corrected_by="admin_ui",
                )
            if not result.get("ok"):
                st.error(result.get("error", "Batch apply failed"))
            else:
                st.success(
                    f"Applied {result.get('ranges_applied', 0)} ranges "
                    f"({result.get('rows_updated', 0)} rows updated)."
                )
                queue[:] = [
                    item
                    for item in queue
                    if not (
                        item.get("elder_id") == elder_id
                        and item.get("room") == room
                        and item.get("record_date") == selected_date.isoformat()
                    )
                ]
                if run_retrain:
                    with st.spinner("Running one-shot retrain..."):
                        try:
                            retrain = correction_service.retrain_after_batch_corrections(
                                elder_id=elder_id,
                                room=room,
                                record_date=selected_date,
                                train_retro=train_retro,
                            )
                            st.success(
                                f"Retrain completed. files_used={retrain.get('files_used', 0)} "
                                f"manual_file={retrain.get('manual_file', '')}"
                            )
                        except Exception as e:
                            st.error(f"Retrain failed: {e}")
                st.rerun()

    if not ctx_queue:
        st.info("No queued correction ranges for this resident/room/date.")
        return

    p1, p2, p3 = st.columns(3)
    p1.metric("Queued Ranges", len(ctx_queue))
    p2.metric("Estimated Rows", int(preview.get("estimated_rows", 0)))
    p3.metric("Dry-Run Status", "Blocked" if preview.get("has_invalid") else "Ready")
    if preview.get("has_invalid"):
        st.warning("Dry-run found invalid ranges (overlap/time/label). Fix queue items before applying.")

    display_rows = []
    preview_rows = {int(r.get("index", -1)): r for r in preview.get("rows", [])}
    for idx, item in enumerate(ctx_queue):
        p = preview_rows.get(idx, {})
        display_rows.append(
            {
                "queue_index": idx,
                "start": item["start_time"].strftime("%H:%M:%S"),
                "end": item["end_time"].strftime("%H:%M:%S"),
                "new_label": item["new_label"],
                "estimated_rows": int(p.get("estimated_rows", 0)),
                "status": str(p.get("status", "ok")),
                "reason": str(p.get("reason", "")),
            }
        )
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    remove_idx = st.number_input(
        "Remove Queue Index",
        min_value=0,
        max_value=max(0, len(ctx_queue) - 1),
        value=0,
        step=1,
        key=f"batch_remove_idx_{room}_{selected_date.isoformat()}",
    )
    remove_col, clear_col = st.columns(2)
    with remove_col:
        if st.button("Remove Selected Queue Item", key=f"batch_remove_btn_{room}_{selected_date.isoformat()}"):
            target = ctx_queue[int(remove_idx)]
            queue.remove(target)
            st.rerun()
    with clear_col:
        if st.button("Clear Context Queue", key=f"batch_clear_{room}_{selected_date.isoformat()}"):
            queue[:] = [
                item
                for item in queue
                if not (
                    item.get("elder_id") == elder_id
                    and item.get("room") == room
                    and item.get("record_date") == selected_date.isoformat()
                )
            ]
            st.rerun()


def render():
    render_sidebar()
    st.header("🏷️ Label Correction Studio")
    
    elder_id = st.session_state.get("global_resident", "All")
    if elder_id == "All":
        st.warning("Please select a specific Resident Context from the sidebar.")
        return

    if "correction_selected_date" not in st.session_state:
        st.session_state["correction_selected_date"] = datetime.date.today()

    # 1. Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rooms = [r for r in get_room_config().get_all_rooms().keys() if r != "default"]
        selected_room = st.selectbox("Room", rooms)
        
    with col2:
        selected_date = st.date_input("Date", key="correction_selected_date")
        
    with col3:
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=0.99, value=0.60, step=0.05)

    workflow_mode = st.radio(
        "Workflow Mode",
        options=["Quick Ops (Current)", "Advanced Review (Legacy + Sensor Panel)"],
        index=1,
        horizontal=True,
        help="Quick Ops keeps the existing low-confidence flow. Advanced adds legacy-style sensor chart review.",
    )
    is_advanced_mode = workflow_mode.startswith("Advanced")

    # ── Activity Timeline (Gantt Chart) ──────────────────────────────
    st.markdown("---")
    timeline_df = correction_service.get_activity_timeline(elder_id, selected_room, selected_date)

    if not timeline_df.empty:
        st.subheader("Activity Timeline")
        st.caption("Colour-coded by label. Low-confidence rows are highlighted in the table below.")
        _render_activity_timeline_chart(
            timeline_df,
            title=f"Activity Patterns — {selected_room} — {selected_date}",
            y_field="activity_type",
        )
    else:
        fallback_timeline_df = correction_service.get_activity_timeline_any_room(elder_id, selected_date)
        if not fallback_timeline_df.empty:
            st.subheader("Activity Timeline (Legacy Fallback)")
            st.warning(
                f"No activity rows matched room `{selected_room}` on {selected_date}. "
                "Showing all-room timeline for this date."
            )
            _render_activity_timeline_chart(
                fallback_timeline_df,
                title=f"Activity Patterns — All Rooms — {selected_date}",
                y_field="room",
            )
        else:
            st.info(f"No activity data for {selected_room} on {selected_date}.")
            nearest = correction_service.find_nearest_activity_date(
                elder_id=elder_id,
                target_date=selected_date,
                room=selected_room,
            )
            if nearest:
                nearest_day = nearest["date"]
                st.caption(
                    f"Nearest available date for `{selected_room}` is `{nearest_day}` "
                    f"({nearest.get('rows', 0)} rows)."
                )
                if st.button(
                    f"Load Nearest Date ({nearest_day})",
                    key=f"corr_nearest_{elder_id}_{selected_room}_{selected_date.isoformat()}",
                ):
                    st.session_state["correction_selected_date"] = nearest_day
                    st.rerun()
            else:
                nearest_any = correction_service.find_nearest_activity_date(
                    elder_id=elder_id,
                    target_date=selected_date,
                    room=None,
                )
                if nearest_any:
                    nearest_day = nearest_any["date"]
                    st.caption(
                        "No activity rows were found for this room across recorded days. "
                        f"Nearest resident-level date is `{nearest_day}` ({nearest_any.get('rows', 0)} rows)."
                    )
                    if st.button(
                        f"Load Nearest Resident Date ({nearest_day})",
                        key=f"corr_nearest_any_{elder_id}_{selected_room}_{selected_date.isoformat()}",
                    ):
                        st.session_state["correction_selected_date"] = nearest_day
                        st.rerun()
                else:
                    st.warning(
                        "No activity rows were found for this resident in runtime tables "
                        "(`adl_history`/`activity_segments`). This resident context may be model-only."
                    )

    if is_advanced_mode:
        st.markdown("---")
        sensor_df = correction_service.get_sensor_timeseries(elder_id, selected_room, selected_date)
        _render_sensor_panel(sensor_df)

    compare_payload = correction_service.build_compare_timeline_payload(
        elder_id=elder_id,
        room=selected_room,
        date=selected_date,
        confidence_threshold=float(confidence_threshold),
    )
    training_compare_df = compare_payload.get("training_timeline", pd.DataFrame())
    prediction_compare_df = compare_payload.get("prediction_timeline", pd.DataFrame())

    st.markdown("---")
    st.subheader("Compare Timelines")
    st.caption(
        "Top shows training/corrected labels from `adl_history`. "
        "Bottom shows raw predictions from `predictions`. Orange blocks need review."
    )
    if training_compare_df is None or training_compare_df.empty:
        st.info("No training/corrected labels found for this resident, room, and day.")
    else:
        _render_activity_timeline_chart(
            training_compare_df,
            title=f"Training / Corrected Timeline — {selected_room} — {selected_date}",
            y_field="activity_type",
        )
    if prediction_compare_df is None or prediction_compare_df.empty:
        st.info("No raw prediction rows found for this resident, room, and day.")
    else:
        review_blocks = prediction_compare_df[prediction_compare_df["review_reason"] != "clear"].copy()
        c1, c2, c3 = st.columns(3)
        c1.metric("Training Blocks", int(len(training_compare_df.index)) if training_compare_df is not None else 0)
        c2.metric("Prediction Blocks", int(len(prediction_compare_df.index)))
        c3.metric("Review Needed", int(len(review_blocks.index)))
        _render_prediction_compare_chart(
            prediction_compare_df,
            title=f"Prediction / Runtime Timeline — {selected_room} — {selected_date}",
        )

    low_confidence_df = correction_service.get_low_confidence_queue(
        elder_id, selected_room, selected_date, confidence_threshold
    )
    review_queue_parts = []
    if low_confidence_df is not None and not low_confidence_df.empty:
        low_confidence_df = low_confidence_df.copy()
        low_confidence_df["review_reason"] = "low_confidence"
        low_confidence_df["review_source"] = "low_confidence_queue"
        low_confidence_df["training_activity_type"] = None
        low_confidence_df["prediction_state_label"] = "low confidence / review needed"
        review_queue_parts.append(low_confidence_df)
    if prediction_compare_df is not None and not prediction_compare_df.empty:
        compare_review_df = prediction_compare_df[
            prediction_compare_df["review_reason"] != "clear"
        ].copy()
        if not compare_review_df.empty:
            compare_review_df["review_source"] = "compare_timeline"
            review_queue_parts.append(compare_review_df)
    if review_queue_parts:
        queue_df = pd.concat(review_queue_parts, ignore_index=True, sort=False)
        queue_df["timestamp"] = pd.to_datetime(queue_df["timestamp"], errors="coerce")
        queue_df = (
            queue_df.sort_values("timestamp")
            .dropna(subset=["timestamp"])
            .drop_duplicates(subset=["timestamp", "activity_type"], keep="first")
        )
    else:
        queue_df = pd.DataFrame()
    
    st.markdown("---")
    
    if queue_df.empty:
        st.success(
            f"No low-confidence or compare-flagged records found for {selected_room} on {selected_date}."
        )
        if is_advanced_mode:
            st.markdown("---")
            _render_batch_queue_panel(elder_id=elder_id, room=selected_room, selected_date=selected_date)
        return

    col_left, col_right = st.columns([2, 1])
    
    # 3. Left Pane: AgGrid
    with col_left:
        st.subheader("Review Queue")
        st.caption("Combined low-confidence queue plus prediction/training mismatch blocks.")
        selected_rows = _render_review_queue_selector(queue_df)
        
    # 4. Right Pane: Edit Panel
    with col_right:
        st.subheader("Edit Selection")
        selected_rows = _normalize_selected_rows(selected_rows)

        if not selected_rows:
            st.info("Select one or more rows from the table to edit.")
            
            # Show a generic Undo button here even if nothing selected
            st.markdown("---")
            st.caption("Last Action")
            if st.button("Undo Last Correction"):
                st.warning("Undo functionality not yet wired up.")
        else:
            n_selected = len(selected_rows)
            st.write(f"**{n_selected} row(s) selected.**")

            review_reasons = sorted(
                {
                    str(row.get("review_reason", "")).strip()
                    for row in selected_rows
                    if str(row.get("review_reason", "")).strip()
                }
            )
            training_labels = sorted(
                {
                    str(row.get("training_activity_type", "")).strip()
                    for row in selected_rows
                    if str(row.get("training_activity_type", "")).strip()
                }
            )
            if review_reasons:
                st.write(f"Review reason: `{', '.join(review_reasons)}`")
            if training_labels:
                st.write(f"Training label(s): `{', '.join(training_labels)}`")

            selection_start, selection_end = _selected_time_bounds(selected_rows)
            if selection_start is None or selection_end is None:
                st.warning("Selected rows do not contain valid timestamps.")
            else:
                st.write(f"First: `{selection_start}`")
                st.write(f"Last: `{selection_end}`")

                # Fetch Sensor Context
                with st.spinner("Fetching sensors..."):
                    ctx = correction_service.get_sensor_context(
                        elder_id,
                        selected_room,
                        selection_start.strftime("%Y-%m-%d %H:%M:%S"),
                        selection_end.strftime("%Y-%m-%d %H:%M:%S"),
                    )

                st.markdown("**Sensor Context (Max in Window):**")
                st.write(f"- 🚶 Motion Events: {ctx.get('motion', 0)}")
                st.write(f"- 💨 CO2 Max: {ctx.get('co2_max', 0):.0f} ppm")
                st.write(f"- 🔊 Sound Max: {ctx.get('sound_max', 0):.0f} dB")
            
            st.markdown("---")
            all_labels = correction_service.get_activity_labels()
            # Default to the first selected row's label if it's in the list
            current_label = selected_rows[0].get('activity_type')
            default_index = all_labels.index(current_label) if current_label in all_labels else 0
            
            new_label = st.selectbox("Assign New Label", options=all_labels, index=default_index)
            
            if st.button("Save Correction", type="primary"):
                success, msg, rows_affected = correction_service.save_correction(
                    elder_id=elder_id,
                    room=selected_room,
                    df_rows=selected_rows,
                    new_activity=new_label,
                    corrected_by="admin_ui"
                )
                if success:
                    st.success(f"{msg} ({rows_affected} rows)")
                    st.rerun()
                else:
                    st.error(msg)

    if is_advanced_mode:
        st.markdown("---")
        _render_batch_queue_panel(elder_id=elder_id, room=selected_room, selected_date=selected_date)


if __name__ == "__main__":
    if "global_resident" not in st.session_state:
        st.session_state["global_resident"] = "HK999"
    render()
