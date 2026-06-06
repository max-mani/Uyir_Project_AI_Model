def fuse_scores(proximity, trajectory, flow, lstm_peak, occlusion, merge, energy_drop, spin, scene_interruption, diff_burst, flow_dispersion,
                w1=0.10, w2=0.15, w3=0.10, w4=0.25, w5=0.10, w6=0.10, w7=0.10, w8=0.05, w9=0.05, threshold=0.55):
    """
    Fuses 11 multi-stage spatio-temporal signals using a 9-weight formula.
    Weights sum to 1.0:
    w1: Proximity (0.10)
    w2: Trajectory conflict + Spin (0.15)
    w3: Optical flow magnitude (0.10)
    w4: CNN-LSTM DL peak (0.25)
    w5: Containment Occlusion (0.10)
    w6: BBox Merge (0.10)
    w7: Kinetic Energy drop (0.10)
    w8: Scene-level traffic interruption (0.05)
    w9: Visual burst / Flow angular dispersion (0.05)
    """
    # Blend trajectory conflict and spin instability
    traj_blended = 0.6 * trajectory + 0.4 * spin
    # Blend visual bursts
    burst_blended = max(diff_burst, flow_dispersion)

    final_score = (w1 * proximity) + \
                  (w2 * traj_blended) + \
                  (w3 * flow) + \
                  (w4 * lstm_peak) + \
                  (w5 * occlusion) + \
                  (w6 * merge) + \
                  (w7 * energy_drop) + \
                  (w8 * scene_interruption) + \
                  (w9 * burst_blended)

    # Clip final score to [0.0, 1.0]
    final_score = max(0.0, min(1.0, final_score))
    is_accident = final_score >= threshold

    # Identify active triggering phases for explanation
    triggers = []
    if proximity > 0.7:
        triggers.append("Proximity")
    if trajectory > 0.6 or spin > 0.5:
        triggers.append("Trajectory/Spin")
    if flow > 0.6:
        triggers.append("Flow Spike")
    if lstm_peak > 0.6:
        triggers.append("CNN-LSTM Peak")
    if occlusion > 0.7:
        triggers.append("Occlusion-Containment")
    if merge > 0.7:
        triggers.append("BBox Merge")
    if energy_drop > 0.7:
        triggers.append("Kinetic Energy Drop")
    if scene_interruption > 0.7:
        triggers.append("Traffic Interruption")
    if burst_blended > 0.6:
        triggers.append("Visual Burst/Scatter")

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
            "flow_dispersion": float(flow_dispersion)
        }
    }
