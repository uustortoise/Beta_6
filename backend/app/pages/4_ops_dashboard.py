import sys
from pathlib import Path

_backend = str(Path(__file__).resolve().parent.parent.parent)
if _backend not in sys.path: sys.path.extend([_backend, str(Path(_backend).parent)])

import altair as alt
import pandas as pd
import streamlit as st

import services.ops_service as ops_service
from app._sidebar import render_sidebar


def _format_metric(value) -> str:
    """Render numeric metrics safely for UI display."""
    try:
        if value is None:
            return "N/A"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def _friendly_gate_reason(raw: str) -> str:
    txt = str(raw or "").strip()
    if not txt:
        return "None"
    txt = txt.split(":")[0].replace("_", " ").replace(".", " ")
    return " ".join(part for part in txt.split() if part)


def _friendly_reasons_join(raw) -> str:
    if not isinstance(raw, list) or not raw:
        return "None"
    parts = []
    for item in raw:
        reason = _friendly_gate_reason(str(item))
        if reason:
            parts.append(reason)
    return "; ".join(parts) if parts else "None"


def render():
    render_sidebar()
    st.header("📊 Operations & Health Dashboard")
    
    elder_id = st.session_state.get("global_resident", "All")
    if elder_id == "All":
        st.warning("Select a specific Resident Context from the sidebar.")
        return
        
    view_mode = st.radio("Display Mode", options=["Ops View (Status)", "ML View (Deep Dive)"], horizontal=True)
    is_ml_mode = "ML View" in view_mode
    
    st.markdown("---")
    
    # 1. Daily Operations (Data Flow)
    st.subheader("Section A: Daily Operations Health")
    with st.spinner("Fetching ops..."):
        ops_sum = ops_service.get_daily_ops_summary(elder_id)
        
    colA1, colA2, colA3 = st.columns(3)
    
    stale = ops_sum.get("is_stale", True)
    hrs = ops_sum.get("hours_since_last_ingestion")
    health = "🔴 Stale/Missing" if stale else "🟢 Healthy"
    
    colA1.metric("Ingestion Status", health)
    colA2.metric("Last Data (Hours Ago)", f"{hrs:.1f}h" if hrs is not None else "N/A", 
                 delta="Action Needed" if stale else "Normal", delta_color="inverse")
    if ops_sum.get("last_ingestion_time") is not None:
        colA2.caption(f"Last timestamp: {pd.to_datetime(ops_sum.get('last_ingestion_time')).strftime('%Y-%m-%d %H:%M:%S')}")
    colA3.metric("Open Alerts", ops_sum.get("open_alerts", 0),
                 delta="Review" if ops_sum.get("open_alerts", 0) > 0 else "Clear", delta_color="inverse")
                 
    # 2. Sample Collection
    st.markdown("---")
    st.subheader("Section B: Sample Collection (Beta 6 Requirements)")
    with st.spinner("Analyzing archives..."):
        samples = ops_service.get_sample_collection_status(elder_id)
        
    if not samples:
        st.info("No archive data found.")
    else:
        import altair as alt

        target = samples.get("target", 14)
        counts = samples.get("counts", {})
        count_label = str(samples.get("count_label") or "Days Recorded")
        target_label = str(samples.get("target_label") or "Labeled Day Target")
        
        # Build simple dataframe for bar chart
        df_samples = pd.DataFrame(list(counts.items()), columns=["Room", count_label])
        
        colB1, colB2 = st.columns([1, 2])
        with colB1:
            ready = len(samples.get("ready_rooms", []))
            total = len(counts)
            st.metric("Rooms Ready for ML", f"{ready} / {total}")
            st.caption(f"{target_label}: {target} days")
            st.caption("Readiness is based on labeled day coverage, not raw file count.")
            
        with colB2:
            # Altair bar chart with threshold line
            bars = alt.Chart(df_samples).mark_bar().encode(
                x=alt.X(f'{count_label}:Q', title=count_label),
                y=alt.Y('Room:N', sort='-x'),
                color=alt.condition(
                    alt.datum[count_label] >= target,
                    alt.value('#1b5e20'), # Green if met
                    alt.value('#f57c00')  # Orange if under
                )
            )
            rule = alt.Chart(pd.DataFrame({target_label: [target]})).mark_rule(color='red').encode(
                x=alt.X(f'{target_label}:Q', title=count_label)
            )
            st.altair_chart((bars + rule).properties(height=250), use_container_width=True)


    # 3. Model Maintenance
    st.markdown("---")
    st.subheader("Section C: Model Performance")
    with st.spinner("Fetching model status..."):
        ml_status = ops_service.get_model_status(elder_id)
        
    overall = ml_status.get("status", {}).get("overall", "unknown")
    
    if is_ml_mode:
        st.write(f"**Overall Status:** `{overall}`")
        st.caption("WF metrics below are walk-forward quality metrics, not raw training-run accuracy.")
        if "rooms" in ml_status:
            for r in ml_status["rooms"]:
                metrics = r.get("metrics") or {}
                st.markdown(f"**{str(r.get('room')).title()}**")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("WF Candidate F1", _format_metric(metrics.get("candidate_macro_f1_mean")))
                champion = metrics.get("champion_macro_f1_mean")
                prev_run = metrics.get("previous_run_macro_f1_mean")
                if champion is None and prev_run is not None:
                    mc2.metric("Previous WF F1", _format_metric(prev_run))
                else:
                    mc2.metric("Champion WF F1", _format_metric(champion))
                mc3.metric("Status", r.get("status"))
    else:
        # Ops view - simpler traffic lights
        st.write("Model performance is evaluated nightly via Walk-Forward validation.")
        
        status_map = {
            "healthy": "🟢 Healthy",
            "watch": "🟡 Needs Monitoring",
            "action_needed": "🔴 Action Needed",
            "not_available": "⚪ Not Available/Pending"
        }
        
        st.metric("Global Health", status_map.get(overall, "⚪ Unknown"))
        
        if "rooms" in ml_status:
            room_texts = []
            for r in ml_status["rooms"]:
                r_status = r.get("status")
                room_texts.append(f"- **{str(r.get('room')).title()}**: {status_map.get(r_status, 'Unknown')}")
            for text in room_texts:
                st.write(text)

    if is_ml_mode:
        st.markdown("---")
        st.subheader("Section C2: Model Update Safety")
        st.caption("Monitor pilot thresholds, gate blockers, and WF F1/accuracy progression over time.")

        monitor_cfg1, monitor_cfg2 = st.columns([1, 1])
        monitor_window_label = monitor_cfg1.selectbox(
            "Time Window",
            options=["Last 7 Days", "Last 30 Days", "Last 90 Days"],
            index=1,
            key="ops_promotion_gate_window",
        )
        monitor_days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}
        monitor_days = monitor_days_map.get(monitor_window_label, 30)
        monitor_limit = int(
            monitor_cfg2.number_input(
                "Recent Runs to Show",
                min_value=10,
                max_value=200,
                value=60,
                step=10,
                key="ops_promotion_gate_max_runs",
            )
        )

        with st.spinner("Fetching promotion safety trends..."):
            monitor = ops_service.get_model_update_monitor(
                elder_id=elder_id,
                days=int(monitor_days),
                limit=int(monitor_limit),
            )

        if monitor.get("total_runs", 0) == 0:
            st.info("No recent training records found for this resident.")
        else:
            pass_rate_pct = monitor.get("wf_pass_rate_pct")
            pass_rate_text = f"{pass_rate_pct:.1f}%" if pass_rate_pct is not None else "N/A"
            top_reason = _friendly_gate_reason(monitor.get("top_failure_reason") or "")
            blocked_runs = int(monitor.get("wf_fail_runs", 0))

            latest_delta = monitor.get("latest_delta")
            latest_delta_text = "No comparison yet"
            if latest_delta and latest_delta.get("delta_vs_previous") is not None:
                delta_value = float(latest_delta["delta_vs_previous"])
                delta_room = latest_delta.get("room", "unknown")
                if delta_value > 0:
                    latest_delta_text = f"Improved ({delta_room})"
                elif delta_value < 0:
                    latest_delta_text = f"Dropped ({delta_room})"
                else:
                    latest_delta_text = f"No change ({delta_room})"

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Safe Update Rate", pass_rate_text)
            p2.metric("Latest Change", latest_delta_text)
            p3.metric("Most Common Blocker", top_reason)
            p4.metric("Blocked Runs", blocked_runs)

            latest_run = monitor.get("latest_run") or {}
            metric_labels = monitor.get("metric_labels", {}) if isinstance(monitor.get("metric_labels"), dict) else {}
            if latest_run:
                latest_accuracy = latest_run.get("accuracy")
                latest_accuracy_txt = f"{float(latest_accuracy):.3f}" if latest_accuracy is not None else "N/A"
                latest_run_id_txt = latest_run.get("run_id")
                st.caption(
                    f"Latest Run ID: {latest_run_id_txt if latest_run_id_txt is not None else 'N/A'} | "
                    f"Latest run: {latest_run.get('training_date', 'N/A')} | "
                    f"Result: {str(latest_run.get('status', 'N/A')).replace('_', ' ').title()} | "
                    f"{metric_labels.get('latest_run_accuracy', 'Raw Training Run Accuracy')}: {latest_accuracy_txt}"
                )

            release_profile = monitor.get("release_gate_profile", {})
            if isinstance(release_profile, dict) and release_profile:
                st.markdown("#### Pilot Gate Profile")
                rp1, rp2, rp3, rp4 = st.columns(4)
                rp1.metric("WF Min Train Days", int(release_profile.get("wf_min_train_days", 7) or 7))
                rp2.metric("WF Validation Days", int(release_profile.get("wf_valid_days", 1) or 1))
                rp3.metric("Bootstrap Phase-1 Max Day", int(release_profile.get("bootstrap_phase1_max_days", 7) or 7))
                rp4.metric("Bootstrap Max Day", int(release_profile.get("bootstrap_max_days", 14) or 14))

                st.caption(
                    f"Evidence profile: {release_profile.get('evidence_profile', 'unknown')} | "
                    f"Strict prior drift max: {float(release_profile.get('strict_prior_drift_max', 0.10)):.2f}"
                )

                tracker_rows = release_profile.get("release_threshold_tracker", [])
                if isinstance(tracker_rows, list) and tracker_rows:
                    with st.expander("View Effective Release Thresholds", expanded=False):
                        st.dataframe(pd.DataFrame(tracker_rows), hide_index=True, use_container_width=True)

            room_trends = monitor.get("room_trends", [])
            if room_trends:
                trend_df = pd.DataFrame(room_trends)
                trend_df["latest_reasons"] = trend_df["latest_reasons"].map(_friendly_reasons_join)
                trend_df = trend_df.rename(
                    columns={
                        "room": "Room",
                        "latest_run_id": "Run ID (Last Seen)",
                        "latest_training_date": "Latest Training Time",
                        "latest_pass": "Passed Safety Check",
                        "latest_training_days": "Latest Training Days",
                        "latest_required_threshold": "Required Threshold",
                        "latest_threshold_day_bucket": "Threshold Day Bucket",
                        "latest_candidate_macro_f1_mean": "Latest WF Candidate F1",
                        "latest_candidate_accuracy_mean": "Latest WF Candidate Accuracy",
                        "previous_candidate_macro_f1_mean": "Previous WF Candidate F1",
                        "previous_candidate_accuracy_mean": "Previous WF Candidate Accuracy",
                        "delta_vs_previous": "Change vs Previous",
                        "latest_reasons": "Pause Reason",
                    }
                )
                with st.expander("View Room-by-Room Details", expanded=False):
                    st.dataframe(trend_df, hide_index=True, use_container_width=True)

            room_history_points = monitor.get("room_history_points", [])
            if isinstance(room_history_points, list) and room_history_points:
                history_df = pd.DataFrame(room_history_points)
                history_df["training_time"] = pd.to_datetime(history_df.get("training_date"), errors="coerce")
                history_df["sensor_event_time"] = pd.to_datetime(
                    history_df.get("sensor_event_time"), errors="coerce"
                )
                history_df = history_df.dropna(subset=["training_time"])
                if not history_df.empty:
                    st.markdown("#### F1/Accuracy Over Time")
                    st.caption("Hover any point to see exact run/time/metric values.")
                    time_axis_mode = st.radio(
                        "X-axis Time Source",
                        options=["Sensor Event Time (Chronology)", "Training Time (Run Chronology)"],
                        index=0,
                        horizontal=True,
                        key="ops_promotion_trend_time_axis",
                    )
                    prefer_sensor_time = time_axis_mode.startswith("Sensor Event Time")
                    if prefer_sensor_time and int(history_df["sensor_event_time"].notna().sum()) == 0:
                        st.info("No sensor event timestamps found in training manifests yet; using training run time.")
                    axis_col = "sensor_event_time" if prefer_sensor_time else "training_time"
                    axis_title = "Sensor Event Time" if prefer_sensor_time else "Training Time"
                    history_df["plot_time"] = history_df[axis_col].where(
                        history_df[axis_col].notna(),
                        history_df["training_time"],
                    )
                    history_df = history_df.dropna(subset=["plot_time"])
                    room_options = sorted(history_df["room"].dropna().astype(str).unique().tolist())
                    selected_rooms = st.multiselect(
                        "Rooms to Plot",
                        options=room_options,
                        default=room_options[: min(len(room_options), 6)],
                        key="ops_promotion_trend_rooms",
                    )
                    if selected_rooms:
                        history_df = history_df[history_df["room"].isin(selected_rooms)].copy()
                    history_df = history_df.sort_values(["plot_time", "run_id", "room"])

                    f1_df = history_df[history_df["candidate_macro_f1_mean"].notna()].copy()
                    acc_df = history_df[history_df["candidate_accuracy_mean"].notna()].copy()
                    threshold_df = history_df[history_df["required_threshold"].notna()].copy()

                    if not f1_df.empty:
                        f1_tooltips = [
                            alt.Tooltip("room:N", title="Room"),
                            alt.Tooltip("run_id:Q", title="Run ID"),
                            alt.Tooltip("sensor_event_time:T", title="Sensor Event Time"),
                            alt.Tooltip("training_time:T", title="Training Time"),
                            alt.Tooltip("candidate_macro_f1_mean:Q", title="WF Candidate F1", format=".6f"),
                            alt.Tooltip("required_threshold:Q", title="Required Threshold", format=".6f"),
                            alt.Tooltip("training_days:Q", title="Training Days", format=".1f"),
                        ]
                        f1_base = alt.Chart(f1_df).encode(
                            x=alt.X("plot_time:T", title=axis_title),
                            y=alt.Y("candidate_macro_f1_mean:Q", title="WF Candidate F1"),
                            color=alt.Color("room:N", title="Room"),
                        )
                        f1_chart = (
                            f1_base.mark_line()
                            + f1_base.mark_circle(size=70, filled=True)
                            + f1_base.mark_circle(size=220, opacity=0).encode(tooltip=f1_tooltips)
                        )
                        f1_chart = f1_chart.properties(height=260)
                        if not threshold_df.empty:
                            threshold_tooltips = [
                                alt.Tooltip("room:N", title="Room"),
                                alt.Tooltip("run_id:Q", title="Run ID"),
                                alt.Tooltip("sensor_event_time:T", title="Sensor Event Time"),
                                alt.Tooltip("training_time:T", title="Training Time"),
                                alt.Tooltip("required_threshold:Q", title="Required Threshold", format=".6f"),
                                alt.Tooltip("training_days:Q", title="Training Days", format=".1f"),
                            ]
                            threshold_base = alt.Chart(threshold_df).encode(
                                x=alt.X("plot_time:T", title=axis_title),
                                y=alt.Y("required_threshold:Q", title="WF Candidate F1"),
                                color=alt.Color("room:N", title="Room"),
                            )
                            threshold_chart = (
                                threshold_base.mark_line(point=False, strokeDash=[4, 4], opacity=0.55)
                                + threshold_base.mark_circle(size=64, filled=True, opacity=0.35)
                                + threshold_base.mark_circle(size=220, opacity=0).encode(tooltip=threshold_tooltips)
                            .properties(height=260)
                            )
                            st.altair_chart((f1_chart + threshold_chart).interactive(), use_container_width=True)
                        else:
                            st.altair_chart(f1_chart.interactive(), use_container_width=True)

                    if not acc_df.empty:
                        acc_tooltips = [
                            alt.Tooltip("room:N", title="Room"),
                            alt.Tooltip("run_id:Q", title="Run ID"),
                            alt.Tooltip("sensor_event_time:T", title="Sensor Event Time"),
                            alt.Tooltip("training_time:T", title="Training Time"),
                            alt.Tooltip("candidate_accuracy_mean:Q", title="WF Candidate Accuracy", format=".6f"),
                            alt.Tooltip("training_days:Q", title="Training Days", format=".1f"),
                        ]
                        acc_base = alt.Chart(acc_df).encode(
                            x=alt.X("plot_time:T", title=axis_title),
                            y=alt.Y("candidate_accuracy_mean:Q", title="WF Candidate Accuracy"),
                            color=alt.Color("room:N", title="Room"),
                        )
                        acc_chart = (
                            acc_base.mark_line()
                            + acc_base.mark_circle(size=70, filled=True)
                            + acc_base.mark_circle(size=220, opacity=0).encode(tooltip=acc_tooltips)
                        )
                        acc_chart = acc_chart.properties(height=260)
                        st.altair_chart(acc_chart.interactive(), use_container_width=True)

    st.markdown("---")
    st.subheader("Section D: Timeline Reliability & Correction Load")
    with st.spinner("Computing timeline reliability scorecard..."):
        scorecard = ops_service.get_timeline_reliability_scorecard(
            elder_id=elder_id,
            days=30,
            confidence_threshold=0.60,
        )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Corrections (30d)", int(scorecard.get("correction_volume", 0) or 0))
    d2.metric("Review Backlog", int(scorecard.get("review_backlog", 0) or 0))
    contradiction_rate = scorecard.get("contradiction_rate")
    d3.metric(
        "Contradiction Rate",
        f"{float(contradiction_rate) * 100:.1f}%" if contradiction_rate is not None else "N/A",
    )
    fragmentation_rate = scorecard.get("fragmentation_rate")
    d4.metric(
        "Fragmentation Rate",
        f"{float(fragmentation_rate) * 100:.1f}%" if fragmentation_rate is not None else "N/A",
    )

    e1, e2, e3 = st.columns(3)
    manual_review_rate = scorecard.get("manual_review_rate")
    unknown_rate = scorecard.get("unknown_abstain_rate")
    e1.metric(
        "Manual Review Rate",
        f"{float(manual_review_rate) * 100:.1f}%" if manual_review_rate is not None else "N/A",
    )
    e2.metric(
        "Unknown / Abstain Rate",
        f"{float(unknown_rate) * 100:.1f}%" if unknown_rate is not None else "N/A",
    )
    e3.metric("Authority State", str(scorecard.get("authority_state", "unknown")).replace("_", " ").title())
    st.caption(
        f"Active system: {scorecard.get('active_system', 'beta6')} | "
        f"Policy-sensitive rooms: {', '.join(scorecard.get('policy_sensitive_rooms', [])) or 'none'}"
    )

    trend_rows = scorecard.get("unknown_abstain_trend", [])
    if isinstance(trend_rows, list) and trend_rows:
        trend_df = pd.DataFrame(trend_rows)
        trend_df["day"] = pd.to_datetime(trend_df["day"], errors="coerce")
        trend_df = trend_df.dropna(subset=["day"])
        if not trend_df.empty:
            trend_chart = (
                alt.Chart(trend_df.sort_values("day"))
                .mark_line(point=True)
                .encode(
                    x=alt.X("day:T", title="Date"),
                    y=alt.Y("unknown_abstain_rate:Q", title="Unknown / Abstain Rate"),
                    tooltip=[
                        alt.Tooltip("day:T", title="Date"),
                        alt.Tooltip("prediction_blocks:Q", title="Prediction Blocks"),
                        alt.Tooltip("unknown_abstain_blocks:Q", title="Unknown / Abstain Blocks"),
                        alt.Tooltip("unknown_abstain_rate:Q", title="Rate", format=".2%"),
                    ],
                )
                .properties(height=220)
            )
            st.altair_chart(trend_chart, use_container_width=True)

    # 5. Hard Negatives
    st.markdown("---")
    st.subheader("Section E: Active Learning Queue")
    with st.spinner("Fetching queue..."):
        hn_sum = ops_service.get_hard_negative_summary(elder_id)
        
    hc1, hc2, hc3 = st.columns(3)
    hc1.metric("Open Hard Negatives", hn_sum.get("open_count", 0))
    hc2.metric("Recently Resolved", hn_sum.get("recent_resolved", 0))
    hc3.metric("Last Miner Run", str(hn_sum.get("last_run"))[:16] if hn_sum.get("last_run") else "Never")

if __name__ == "__main__":
    if "global_resident" not in st.session_state:
        st.session_state["global_resident"] = "All"
    render()
