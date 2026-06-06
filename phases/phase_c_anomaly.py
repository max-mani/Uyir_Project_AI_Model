from utils.optical_flow import get_mean_flow_magnitude, calculate_flow_angular_dispersion

def analyze_anomaly(track, flow):
    """
    Performs anomaly confirmation on a tracked vehicle:
    1. Optical Flow magnitude spike
    2. Bounding box deformation (sudden area or aspect ratio change)
    3. Flow Angular Dispersion (chaotic radial flow vs parallel)
    4. Multi-frame consistency check (at least 3 consecutive frames)
    
    Returns:
        {
            "anomaly_confirmed": bool,
            "anomaly_score": float, (0.0 to 1.0)
            "flow_spike": bool,
            "deformation": bool,
            "dispersion": bool,
            "dispersion_val": float,
            "streak": int
        }
    """
    # Initialize histories on the Track object if not present
    if not hasattr(track, "flow_history"):
        track.flow_history = []
    if not hasattr(track, "anomaly_streak"):
        track.anomaly_streak = 0

    # 1. Optical Flow Magnitude Spike
    curr_flow = get_mean_flow_magnitude(flow, track.bbox)
    flow_spike = False
    
    if len(track.flow_history) >= 4:
        avg_past_flow = sum(track.flow_history[-5:]) / len(track.flow_history[-5:])
        if curr_flow > 2.5 * avg_past_flow and curr_flow > 4.0:
            flow_spike = True
    elif curr_flow > 8.0:
        flow_spike = True
        
    track.flow_history.append(curr_flow)
    if len(track.flow_history) > 30:
        track.flow_history.pop(0)

    # 2. Bounding Box Deformation (size/aspect ratio change)
    deformation = False
    deformation_score = 0.0
    
    x1, y1, x2, y2 = track.bbox
    curr_w = x2 - x1
    curr_h = y2 - y1
    curr_area = curr_w * curr_h
    curr_ratio = curr_w / max(1.0, curr_h)

    if len(track.bbox_history) >= 4:
        past_areas = []
        past_ratios = []
        for bbox in track.bbox_history[:-1]:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            past_areas.append(w * h)
            past_ratios.append(w / max(1.0, h))
            
        avg_area = sum(past_areas) / len(past_areas)
        avg_ratio = sum(past_ratios) / len(past_ratios)
        
        area_diff = abs(curr_area - avg_area) / max(1.0, avg_area)
        ratio_diff = abs(curr_ratio - avg_ratio) / max(0.1, avg_ratio)
        
        if area_diff > 0.40 or ratio_diff > 0.30:
            deformation = True
            
        deformation_score = min(1.0, max(area_diff, ratio_diff))

    # 3. Flow Angular Dispersion (radial chaos)
    dispersion_val = calculate_flow_angular_dispersion(flow, track.bbox)
    # Circular variance > 0.50 signifies chaotic flow angles inside the bbox (impact scatter)
    dispersion = dispersion_val > 0.50

    # Calculate individual anomaly score
    # Blend flow magnitude, deformation, and angular dispersion
    flow_score = min(1.0, curr_flow / 12.0)
    anomaly_raw_score = 0.4 * flow_score + 0.3 * deformation_score + 0.3 * dispersion_val

    # 4. Multi-Frame Consistency Check
    # Anomaly is flagged if there is a flow spike, deformation, or dispersion
    if flow_spike or deformation or dispersion:
        track.anomaly_streak += 1
    else:
        track.anomaly_streak = max(0, track.anomaly_streak - 1)
        
    anomaly_confirmed = track.anomaly_streak >= 3
    
    final_anomaly_score = anomaly_raw_score
    if anomaly_confirmed:
        final_anomaly_score = max(0.70, final_anomaly_score)
    else:
        final_anomaly_score = final_anomaly_score * 0.5

    return {
        "anomaly_confirmed": anomaly_confirmed,
        "anomaly_score": float(final_anomaly_score),
        "flow_spike": flow_spike,
        "deformation": deformation,
        "dispersion": dispersion,
        "dispersion_val": float(dispersion_val),
        "streak": track.anomaly_streak
    }
