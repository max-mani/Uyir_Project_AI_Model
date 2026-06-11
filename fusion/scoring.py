def fuse_scores(proximity, trajectory, flow, lstm_peak, occlusion, merge, energy_drop, spin, scene_interruption, diff_burst, flow_dispersion,
                scene_density=0, avg_scene_speed=10.0, stopped_ratio=0.0, post_intersect_static=False,
                w1=0.10, w2=0.15, w3=0.30, w4=0.25, w5=0.05, w6=0.05, w7=0.10, threshold=0.55):
    """
    Fuses multi-stage spatio-temporal signals with:
    1. Stage 2: Congestion Gate hard suppression
    2. Stage 3: Phase C + CNN threshold gate
    3. Stage 4: Weighted voting using exact requested weights:
       - Proximity: 10% (w1)
       - Trajectory: 15% (w2)
       - Anomaly (Phase C flow): 30% (w3)
       - CNN-LSTM (DL peak): 25% (w4)
       - Occlusion: 5% (w5)
       - Merge: 5% (w6)
       - Kinetic Energy Drop: 10% (w7)
    """
    # ================= Stage 3: Phase-C + CNN Gate =================
    # Reject immediately if physical anomaly (Phase C flow) and deep learning confidence are both low
    if flow < 0.20 and lstm_peak < 0.40:
        return {
            "is_accident": False,
            "score": 0.0,
            "trigger_phase": "Suppressed (Phase C & CNN low)",
            "details": {
                "proximity_score": float(proximity),
                "trajectory_score": float(trajectory),
                "flow_score": float(flow),
                "lstm_peak": float(lstm_peak),
                "occlusion_score": float(occlusion),
                "merge_score": float(merge),
                "energy_drop": float(energy_drop),
                "spin_score": float(spin),
                "scene_interruption": float(scene_interruption),
                "diff_burst": float(diff_burst),
                "flow_dispersion": float(flow_dispersion),
                "post_intersect_static": bool(post_intersect_static),
                "traffic_density": float(min(scene_density / 20.0, 1.0)),
                "avg_speed": float(avg_scene_speed),
                "stopped_ratio": float(stopped_ratio)
            }
        }

    # ================= Stage 2: Congestion Gate =================
    traffic_density = min(scene_density / 15.0, 1.0)
    
    # Traffic jam or slow moving queue
    is_congested = (
        traffic_density > 0.40 and
        avg_scene_speed < 5.0
    )
    
    # Parking lot / completely stopped
    is_stopped_traffic = (
        scene_density >= 4 and
        stopped_ratio > 0.60
    )

    if is_congested or is_stopped_traffic:
        proximity = proximity * 0.10
        occlusion = occlusion * 0.10
        merge = merge * 0.10
        trajectory = trajectory * 0.10  # suppress trajectory conflicts in a traffic jam
        flow = flow * 0.30

    # ================= Stage 4: Weighted Voting =================
    # Use exact weights: 0.10, 0.15, 0.30, 0.25, 0.05, 0.05, 0.10
    final_score = (w1 * proximity) + \
                  (w2 * trajectory) + \
                  (w3 * flow) + \
                  (w4 * lstm_peak) + \
                  (w5 * occlusion) + \
                  (w6 * merge) + \
                  (w7 * energy_drop)

    # ================= Stage 5: Zero-Metric Confidence Penalty =================
    # If key physical metrics are exactly 0.0, subtract a penalty to heavily suppress false positives
    penalty = 0.0
    if trajectory == 0.0:
        penalty += 0.10
    if flow == 0.0:
        penalty += 0.15
    if energy_drop == 0.0:
        penalty += 0.05
    if occlusion == 0.0 and merge == 0.0:
        penalty += 0.05
        
    final_score -= penalty

    # Clip final score to [0.0, 1.0]
    final_score = max(0.0, min(1.0, final_score))
    is_accident = final_score >= threshold

    # Identify active triggering phases for explanation
    triggers = []
    if is_congested or is_stopped_traffic:
        triggers.append("Congested")
    if proximity > 0.7:
        triggers.append("Proximity")
    if post_intersect_static:
        triggers.append("Post-Intersection Static (Collision Confirmed)")
    elif trajectory > 0.6 or spin > 0.5:
        triggers.append("Trajectory/Spin")
    if flow > 0.6:
        triggers.append("Flow Spike")
    if lstm_peak > 0.6:
        triggers.append("CNN-LSTM Peak")
    if max(occlusion, merge) > 0.7:
        triggers.append("Spatial Entanglement")
    if energy_drop > 0.7:
        triggers.append("Kinetic Energy Drop")

    if not triggers:
        triggers.append("Weighted Fusion")

    trigger_phase = " & ".join(triggers)

    return {
        "is_accident": is_accident,
        "score": float(final_score),
        "trigger_phase": trigger_phase,
        "details": {
            "proximity_score": float(proximity),
            "trajectory_score": float(trajectory),
            "flow_score": float(flow),
            "lstm_peak": float(lstm_peak),
            "occlusion_score": float(occlusion),
            "merge_score": float(merge),
            "energy_drop": float(energy_drop),
            "spin_score": float(spin),
            "scene_interruption": float(scene_interruption),
            "diff_burst": float(diff_burst),
            "flow_dispersion": float(flow_dispersion),
            "post_intersect_static": bool(post_intersect_static),
            "traffic_density": float(traffic_density),
            "avg_speed": float(avg_scene_speed),
            "stopped_ratio": float(stopped_ratio)
        }
    }
