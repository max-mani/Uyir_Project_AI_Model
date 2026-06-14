import config


def fuse_scores(
    trajectory_stop=0.0,
    ttc_critical=0.0,
    emergency_stop=0.0,
    cnn_lstm=0.0,
    optical_flow=0.0,
    flow_dispersion=0.0,
    scene_density=0,
    avg_scene_speed=10.0,
    stopped_ratio=0.0,
    threshold=None,
    **legacy_kwargs,
):
    """
    Research-backed 6-signal fusion. CNN-LSTM weight is 0 until trained on IITH data.
    Accepts legacy kwargs for backward compatibility (proximity, trajectory, flow, etc.).
    """
    threshold = threshold if threshold is not None else config.FUSION_THRESHOLD
    weights = config.FUSION_WEIGHTS

    if legacy_kwargs:
        trajectory_stop = max(
            trajectory_stop,
            legacy_kwargs.get("post_intersect_static", False) and 1.0 or 0.0,
            legacy_kwargs.get("trajectory", 0.0),
        )
        ttc_critical = max(ttc_critical, legacy_kwargs.get("proximity", 0.0))
        emergency_stop = max(emergency_stop, legacy_kwargs.get("energy_drop", 0.0))
        optical_flow = max(optical_flow, legacy_kwargs.get("flow", 0.0))
        flow_dispersion = max(flow_dispersion, legacy_kwargs.get("flow_dispersion", 0.0))
        cnn_lstm = legacy_kwargs.get("lstm_peak", 0.0)

    traffic_density = min(scene_density / 15.0, 1.0)
    is_congested = traffic_density > 0.40 and avg_scene_speed < 5.0
    is_stopped_traffic = scene_density >= 4 and stopped_ratio > 0.60

    # Only apply congestion suppression when there are NO active collision signals.
    # If emergency_stop or trajectory_stop is elevated, vehicles likely stopped
    # because of a crash — not because of normal traffic congestion.
    collision_signal_present = (
        trajectory_stop > 0.3
        or emergency_stop > 0.3
        or cnn_lstm > 0.4
    )

    if (is_congested or is_stopped_traffic) and not collision_signal_present:
        ttc_critical *= 0.10
        trajectory_stop *= 0.10
        emergency_stop *= 0.10
        optical_flow *= 0.30

    scores = {
        "trajectory_stop": float(trajectory_stop),
        "ttc_critical": float(ttc_critical),
        "emergency_stop": float(emergency_stop),
        "cnn_lstm": float(cnn_lstm),
        "optical_flow": float(optical_flow),
        "flow_dispersion": float(flow_dispersion),
    }

    final_score = sum(weights[k] * scores.get(k, 0.0) for k in weights)
    final_score = max(0.0, min(1.0, final_score))
    is_accident = final_score >= threshold

    triggers = []
    if (is_congested or is_stopped_traffic) and not collision_signal_present:
        triggers.append("Congested - Suppressed")
    if scores["trajectory_stop"] > 0.5:
        triggers.append("Trajectory Stop (IITH)")
    if scores["ttc_critical"] > 0.5:
        triggers.append("TTC Critical")
    if scores["emergency_stop"] > 0.5:
        triggers.append("Emergency Stop")
    if scores["optical_flow"] > 0.5:
        triggers.append("Optical Flow Spike")
    if scores["flow_dispersion"] > 0.5:
        triggers.append("Flow Dispersion")
    if not triggers:
        triggers.append("Weighted Fusion")

    return {
        "is_accident": is_accident,
        "score": float(final_score),
        "trigger_phase": " & ".join(triggers),
        "details": {
            "trajectory_stop_score": scores["trajectory_stop"],
            "ttc_score": scores["ttc_critical"],
            "emergency_stop_score": scores["emergency_stop"],
            "flow_score": scores["optical_flow"],
            "flow_dispersion": scores["flow_dispersion"],
            "lstm_peak": scores["cnn_lstm"],
            "proximity_score": scores["ttc_critical"],
            "trajectory_score": scores["trajectory_stop"],
            "energy_drop": scores["emergency_stop"],
            "traffic_density": float(traffic_density),
            "avg_speed": float(avg_scene_speed),
            "stopped_ratio": float(stopped_ratio),
            "post_intersect_static": bool(scores["trajectory_stop"] > 0.5),
        },
    }