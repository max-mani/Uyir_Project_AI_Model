def fuse_scores(proximity, trajectory, flow, lstm_peak, occlusion, merge, energy_drop, spin, scene_interruption, diff_burst, flow_dispersion,
                scene_density=0, avg_scene_speed=10.0,
                w1=0.10, w2=0.15, w3=0.10, w4=0.25, w5=0.20, w7=0.10, w8=0.05, w9=0.05, threshold=0.55):
    """
    Fuses 11 multi-stage spatio-temporal signals with Context-Aware Suppression and a Physical Anomaly Guard.
    
    Weights:
    w1: Proximity (0.10)
    w2: Trajectory conflict + Spin (0.15)
    w3: Optical flow magnitude (0.10)
    w4: CNN-LSTM DL peak (0.25)
    w5: Spatial Entanglement (occlusion + merge) (0.20)
    w7: Kinetic Energy drop (0.10)
    w8: Scene-level traffic interruption (0.05)
    w9: Visual burst / Flow angular dispersion (0.05)
    """
    # Blend trajectory conflict and spin instability
    traj_blended = 0.6 * trajectory + 0.4 * spin
    # Blend visual bursts
    burst_blended = max(diff_burst, flow_dispersion)
    # Combine occlusion and merge
    entangle_score = max(occlusion, merge)

    # ================= 1. DYNAMIC IMPACT CHECK =================
    # A real physical collision always produces a dynamic impact signature (flow spike, visual burst, or kinetic energy collapse).
    is_impact_event = flow >= 0.35 or burst_blended >= 0.35 or energy_drop >= 0.60

    # ================= 2. TRAFFIC CONTEXT SUPPRESSION =================
    # In slow-moving or parked traffic, if there is NO active physical impact event,
    # we heavily suppress geometric features to avoid false positives.
    if not is_impact_event:
        if avg_scene_speed < 4.0 or scene_density > 4:
            # Scale suppression based on how slow the traffic is
            speed_factor = max(0.15, min(1.0, avg_scene_speed / 4.0)) if avg_scene_speed < 4.0 else 1.0
            
            # Scale suppression based on how dense the traffic is
            density_factor = 0.4 if scene_density > 5 else (0.6 if scene_density >= 3 else 1.0)
            
            suppress_coef = speed_factor * density_factor
            
            proximity = proximity * suppress_coef
            occlusion = occlusion * suppress_coef
            merge = merge * suppress_coef
            entangle_score = entangle_score * suppress_coef
            trajectory = trajectory * suppress_coef
            traj_blended = traj_blended * suppress_coef
            energy_drop = energy_drop * speed_factor

    # ================= 3. WEIGHTED FUSION FUSION =================
    final_score = (w1 * proximity) + \
                  (w2 * traj_blended) + \
                  (w3 * flow) + \
                  (w4 * lstm_peak) + \
                  (w5 * entangle_score) + \
                  (w7 * energy_drop) + \
                  (w8 * scene_interruption) + \
                  (w9 * burst_blended)

    # ================= 4. PHYSICAL ANOMALY GUARD =================
    # Heuristic geometry overlaps should never trigger an accident alert on their own
    # if there is no physical anomaly (optical flow spike or visual burst) AND the DL model
    # is not extremely confident.
    physical_anomaly_present = flow >= 0.20 or burst_blended >= 0.20 or lstm_peak >= 0.75
    
    if not physical_anomaly_present:
        final_score = final_score * 0.30

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
    if entangle_score > 0.7:
        triggers.append("Spatial Entanglement")
    if energy_drop > 0.7:
        triggers.append("Kinetic Energy Drop")
    if scene_interruption > 0.7:
        triggers.append("Traffic Interruption")
    if burst_blended > 0.6:
        triggers.append("Visual Burst/Scatter")

    if not physical_anomaly_present:
        triggers.append("Suppressed (No Physics/DL)")

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
