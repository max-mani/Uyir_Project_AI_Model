import math
from utils.geometry import calculate_angle_diff, line_intersection, euclidean_distance, directional_variance, calculate_bbox_containment_ratio

def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0.0, x2_inter - x1_inter) * max(0.0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0.0
        
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)

def check_ke_drop(track):
    """
    Checks if there is a sudden Kinetic Energy drop (area * speed^2) > 80% in 3-4 frames.
    Returns: (is_dropped, drop_ratio)
    """
    if len(track.bbox_history) < 4 or len(track.velocities) < 4:
        return False, 0.0
        
    # Past stats (4 frames ago)
    box_past = track.bbox_history[-4]
    v_past = track.velocities[-4]
    area_past = (box_past[2] - box_past[0]) * (box_past[3] - box_past[1])
    speed_past_sq = v_past[0]**2 + v_past[1]**2
    ke_past = area_past * speed_past_sq

    # Current stats
    box_curr = track.bbox
    v_curr = track.velocities[-1]
    area_curr = (box_curr[2] - box_curr[0]) * (box_curr[3] - box_curr[1])
    speed_curr_sq = v_curr[0]**2 + v_curr[1]**2
    ke_curr = area_curr * speed_curr_sq
    
    # We only count drops from active movement (ke_past > 200 pixels^3 / frame^2)
    if ke_past > 200.0:
        drop = (ke_past - ke_curr) / ke_past
        if drop > 0.80:
            return True, float(drop)
            
    return False, 0.0

def check_spin(track):
    """
    Calculates circular variance of heading angles over the last 5 frames.
    Returns: (is_spinning, variance_value)
    """
    if len(track.velocities) < 5:
        return False, 0.0
        
    angles = []
    for vx, vy in track.velocities[-5:]:
        # Filter static frames
        if (vx**2 + vy**2) > 0.5:
            angle = math.degrees(math.atan2(vy, vx))
            angles.append(angle)
            
    if len(angles) < 3:
        return False, 0.0
        
    c_var = directional_variance(angles)
    # Circular variance > 0.15 indicates skidding or spinning (dispersed vectors)
    if c_var > 0.15:
        return True, float(c_var)
        
    return False, 0.0

def analyze_trajectory_conflict(track1, track2):
    """
    Performs full multi-stage kinematic conflict checks for a close pair of vehicles.
    """
    # 1. Trajectory line intersection
    hist1 = track1.history[-15:]
    hist2 = track2.history[-15:]
    intersected = False
    
    if len(hist1) >= 2 and len(hist2) >= 2:
        for i in range(len(hist1) - 1):
            seg1 = (hist1[i], hist1[i+1])
            for j in range(len(hist2) - 1):
                seg2 = (hist2[j], hist2[j+1])
                if line_intersection(seg1, seg2):
                    intersected = True
                    break
            if intersected:
                break

    # 2. Kinetic Energy Drops
    ke_drop1, val_ke1 = check_ke_drop(track1)
    ke_drop2, val_ke2 = check_ke_drop(track2)
    energy_dropped = ke_drop1 or ke_drop2

    # 3. Spin / Skid Heading Instability
    spin1, val_spin1 = check_spin(track1)
    spin2, val_spin2 = check_spin(track2)
    spinning = spin1 or spin2

    # 4. BBox Merge Detection (Heavy overlap)
    iou = compute_iou(track1.bbox, track2.bbox)
    merged = iou > 0.60

    # 5. Occlusion / Containment
    # Calculate ratio of smaller bbox contained in larger
    area1 = (track1.bbox[2] - track1.bbox[0]) * (track1.bbox[3] - track1.bbox[1])
    area2 = (track2.bbox[2] - track2.bbox[0]) * (track2.bbox[3] - track2.bbox[1])
    
    if area1 < area2:
        containment = calculate_bbox_containment_ratio(track1.bbox, track2.bbox)
    else:
        containment = calculate_bbox_containment_ratio(track2.bbox, track1.bbox)
        
    occluded = containment > 0.60

    # Base Classification & Score Assignment
    conflict_class = "Normal"
    score = 0.0
    
    # Calculate weighted heuristics
    if intersected or merged or occluded:
        if energy_dropped or spinning:
            conflict_class = "Collision"
            score = 0.95
        elif occluded and not energy_dropped:
            conflict_class = "Occlusion"
            score = 0.40
        else:
            conflict_class = "Occlusion"
            score = 0.30
    else:
        # High proximity and kinetic energy drop
        dist = euclidean_distance(track1.get_centroid(), track2.get_centroid())
        if dist < 50.0:
            if energy_dropped or spinning:
                conflict_class = "Collision"
                score = 0.80
            else:
                conflict_class = "Normal"
                score = 0.20

    return {
        "class": conflict_class,
        "score": float(score),
        "intersected": intersected,
        "energy_dropped": energy_dropped,
        "spinning": spinning,
        "merged": merged,
        "occluded": occluded,
        "containment": float(containment),
        "max_ke_drop": float(max(val_ke1, val_ke2)),
        "max_spin_var": float(max(val_spin1, val_spin2))
    }
