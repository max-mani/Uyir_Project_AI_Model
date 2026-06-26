import math

import numpy as np

import config
from utils.geometry import (
    calculate_angle_diff,
    line_intersection,
    euclidean_distance,
    directional_variance,
    calculate_bbox_containment_ratio,
)


def compute_iou(box1, box2):
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


def is_stationary(track, frames=10, speed_thresh=1.2, displacement_thresh=8.0):
    if not track.velocities:
        return True

    check_len = min(len(track.velocities), frames)
    speeds = [math.sqrt(vx ** 2 + vy ** 2) for vx, vy in track.velocities[-check_len:]]
    mean_speed = sum(speeds) / len(speeds)

    if mean_speed < speed_thresh:
        return True

    if len(track.history) >= 2:
        hist_len = min(len(track.history), frames)
        p_start = track.history[-hist_len]
        p_end = track.history[-1]
        disp = math.sqrt((p_end[0] - p_start[0]) ** 2 + (p_end[1] - p_start[1]) ** 2)
        if disp < displacement_thresh:
            return True

    return False


def was_recently_moving(track,
                        frames=None,
                        min_peak_speed=None):
    """
    Returns True if the track had significant speed at any point in its
    recent history — distinguishing a post-crash stationary vehicle from
    a permanently parked / background object.

    Uses config.RECENTLY_MOVING_FRAMES and config.RECENTLY_MOVING_MIN_SPEED
    as defaults so the threshold is tunable without touching this file.
    """
    if frames is None:
        frames = config.RECENTLY_MOVING_FRAMES      # default 15
    if min_peak_speed is None:
        min_peak_speed = config.RECENTLY_MOVING_MIN_SPEED  # default 3.0 px/frame

    if not track.speed_history:
        return False

    check = track.speed_history[-min(frames, len(track.speed_history)):]
    return max(check) > min_peak_speed


def is_emergency_stop(track, baseline_frames=None, recent_frames=None):
    """
    Emergency stop = 75%+ drop from long baseline in under 3 frames.
    Normal braking = gradual drop over 10+ frames.
    """
    baseline_frames = baseline_frames or config.EMERGENCY_BASELINE_FRAMES
    recent_frames = recent_frames or config.EMERGENCY_RECENT_FRAMES

    if len(track.speed_history) < baseline_frames + recent_frames:
        return False

    baseline_speed = float(np.mean(
        track.speed_history[-(baseline_frames + recent_frames):-recent_frames]
    ))
    recent_speed = float(np.mean(track.speed_history[-recent_frames:]))

    if baseline_speed < 2.0:
        return False

    drop_percent = (baseline_speed - recent_speed) / baseline_speed * 100.0
    mid_speed = float(np.mean(track.speed_history[-8:-5])) if len(track.speed_history) >= 8 else baseline_speed
    sudden = (mid_speed - recent_speed) / max(mid_speed, 0.1) > config.EMERGENCY_SUDDEN_RATIO

    return drop_percent > config.EMERGENCY_DROP_PERCENT and sudden


def check_trajectory_stop_after_intersection(track_a, track_b, n_frames=None):
    """
    IITH 2018: after path intersection, collision only if one vehicle stops.
    """
    n_frames = n_frames or config.TRAJECTORY_STOP_FRAMES
    need = n_frames + 5

    if len(track_a.speed_history) < need or len(track_b.speed_history) < need:
        return False

    def _stopped(track):
        recent = float(np.mean(track.speed_history[-n_frames:]))
        prev = float(np.mean(track.speed_history[-(n_frames + 5):-5]))
        return (
            prev > config.TRAJECTORY_STOP_PREV_SPEED
            and recent < config.TRAJECTORY_STOP_RECENT_SPEED
        )

    return _stopped(track_a) or _stopped(track_b)


def relative_velocity_anomaly(track_a, track_b):
    """
    Rear-end signal: speeds were different, then converged after impact.
    """
    if len(track_a.speed_history) < 15 or len(track_b.speed_history) < 15:
        return False

    prev_diff = abs(
        float(np.mean(track_a.speed_history[-15:-5]))
        - float(np.mean(track_b.speed_history[-15:-5]))
    )
    curr_diff = abs(
        float(np.mean(track_a.speed_history[-3:]))
        - float(np.mean(track_b.speed_history[-3:]))
    )

    return prev_diff > config.REL_VEL_PREV_DIFF_MIN and curr_diff < config.REL_VEL_CURR_DIFF_MAX


def is_smooth_stopping(track):
    if len(track.velocities) < 8:
        return False

    speeds = [math.sqrt(vx ** 2 + vy ** 2) for vx, vy in track.velocities[-8:]]
    decreases = 0
    for i in range(len(speeds) - 1):
        if speeds[i] > speeds[i + 1] or abs(speeds[i] - speeds[i + 1]) < 0.2:
            decreases += 1

    return decreases >= 5


def check_ke_drop(track):
    if len(track.bbox_history) < 4 or len(track.velocities) < 4:
        return False, 0.0

    if is_emergency_stop(track):
        recent = float(np.mean(track.speed_history[-3:])) if track.speed_history else 0.0
        baseline = float(np.mean(track.speed_history[-18:-3])) if len(track.speed_history) >= 18 else recent
        drop = (baseline - recent) / max(baseline, 0.1)
        return True, float(min(1.0, drop))

    box_past = track.bbox_history[-4]
    v_past = track.velocities[-4]
    area_past = (box_past[2] - box_past[0]) * (box_past[3] - box_past[1])
    speed_past_sq = v_past[0] ** 2 + v_past[1] ** 2
    speed_past = math.sqrt(speed_past_sq)
    ke_past = area_past * speed_past_sq

    box_curr = track.bbox
    v_curr = track.velocities[-1]
    area_curr = (box_curr[2] - box_curr[0]) * (box_curr[3] - box_curr[1])
    speed_curr_sq = v_curr[0] ** 2 + v_curr[1] ** 2
    ke_curr = area_curr * speed_curr_sq

    if speed_past < 2.5:
        return False, 0.0

    if is_smooth_stopping(track):
        return False, 0.0

    if ke_past > 300.0:
        drop = (ke_past - ke_curr) / ke_past
        if drop > 0.80:
            return True, float(drop)

    return False, 0.0


def check_spin(track):
    if len(track.velocities) < 5:
        return False, 0.0

    angles = []
    for vx, vy in track.velocities[-5:]:
        if (vx ** 2 + vy ** 2) > 0.5:
            angles.append(math.degrees(math.atan2(vy, vx)))

    if len(angles) < 3:
        return False, 0.0

    c_var = directional_variance(angles)
    if c_var > 0.15:
        return True, float(c_var)

    return False, 0.0


def analyze_trajectory_conflict(track1, track2):
    """
    Evaluate kinematic evidence of a collision between two tracks.

    FIX: The original code returned _empty_result("Normal") immediately when
    both vehicles were stationary or slow.  This silently killed Phase B for
    every post-crash frame where both vehicles have already stopped.

    Now we only skip the pair when NEITHER track was recently moving —
    meaning they are permanently parked objects, not crash-stopped vehicles.
    A post-crash vehicle will have high peak speed in its recent history even
    though its current speed is 0.
    """
    both_stationary = is_stationary(track1) and is_stationary(track2)
    recently_moving_1 = was_recently_moving(track1)
    recently_moving_2 = was_recently_moving(track2)
    either_recently_moving = recently_moving_1 or recently_moving_2

    # Both are permanently static objects (parked cars, roadside objects) — skip.
    if both_stationary and not either_recently_moving:
        return _empty_result("Normal")

    s1 = math.sqrt(track1.velocities[-1][0] ** 2 + track1.velocities[-1][1] ** 2) if track1.velocities else 0.0
    s2 = math.sqrt(track2.velocities[-1][0] ** 2 + track2.velocities[-1][1] ** 2) if track2.velocities else 0.0

    # Both currently slow AND neither was recently moving → background objects.
    # If at least one was recently moving (now stopped), fall through to evaluate
    # emergency stop / trajectory stop which are the key post-crash signals.
    if max(s1, s2) < 2.5 and not either_recently_moving:
        return _empty_result("Normal", score=0.05)

    hist1 = track1.history[-15:]
    hist2 = track2.history[-15:]
    intersected = False

    if len(hist1) >= 2 and len(hist2) >= 2:
        for i in range(len(hist1) - 1):
            seg1 = (hist1[i], hist1[i + 1])
            for j in range(len(hist2) - 1):
                seg2 = (hist2[j], hist2[j + 1])
                if line_intersection(seg1, seg2):
                    intersected = True
                    break
            if intersected:
                break

    # trajectory_stop only makes sense if paths have crossed
    trajectory_stop = False
    if intersected:
        trajectory_stop = check_trajectory_stop_after_intersection(track1, track2)

    # Emergency stop is evaluated INDEPENDENTLY of intersection —
    # a rear-end or side-swipe collision never produces path intersection
    # from a fixed CCTV angle, but the struck vehicle still brakes hard.
    emergency_a = is_emergency_stop(track1)
    emergency_b = is_emergency_stop(track2)
    emergency_stop = emergency_a or emergency_b

    rel_vel_converged = relative_velocity_anomaly(track1, track2)

    ke_drop1, val_ke1 = check_ke_drop(track1)
    ke_drop2, val_ke2 = check_ke_drop(track2)
    energy_dropped = ke_drop1 or ke_drop2 or emergency_stop

    spin1, val_spin1 = check_spin(track1)
    spin2, val_spin2 = check_spin(track2)
    spinning = spin1 or spin2

    iou = compute_iou(track1.bbox, track2.bbox)
    merged = iou > 0.60

    area1 = (track1.bbox[2] - track1.bbox[0]) * (track1.bbox[3] - track1.bbox[1])
    area2 = (track2.bbox[2] - track2.bbox[0]) * (track2.bbox[3] - track2.bbox[1])
    if area1 < area2:
        containment = calculate_bbox_containment_ratio(track1.bbox, track2.bbox)
    else:
        containment = calculate_bbox_containment_ratio(track2.bbox, track1.bbox)

    occluded = containment > 0.60

    collision_signal = (
        (intersected and trajectory_stop)
        or emergency_stop
        or rel_vel_converged
        or merged
        or spinning
    )

    conflict_class = "Normal"
    score = 0.0
    trajectory_stop_score = 1.0 if trajectory_stop else 0.0
    emergency_stop_score = 1.0 if emergency_stop else 0.0
    relative_velocity_score = 1.0 if rel_vel_converged else 0.0

    if collision_signal:
        conflict_class = "Collision"
        score = 0.95
        if trajectory_stop:
            score = max(score, 0.90)
        if emergency_stop:
            score = max(score, 0.85)
        if rel_vel_converged:
            score = max(score, 0.80)
    elif intersected or occluded:
        conflict_class = "Occlusion"
        score = 0.30 if occluded else 0.20

    dist = euclidean_distance(track1.get_centroid(), track2.get_centroid())
    if dist < 50.0 and (energy_dropped or spinning) and not collision_signal:
        conflict_class = "Collision"
        score = 0.75

    return {
        "class": conflict_class,
        "score": float(score),
        "intersected": intersected,
        "post_intersect_static": trajectory_stop,
        "trajectory_stop": trajectory_stop,
        "trajectory_stop_score": float(trajectory_stop_score),
        "emergency_stop": emergency_stop,
        "emergency_stop_score": float(emergency_stop_score),
        "relative_velocity_converged": rel_vel_converged,
        "relative_velocity_score": float(relative_velocity_score),
        "energy_dropped": energy_dropped,
        "spinning": spinning,
        "merged": merged,
        "occluded": occluded,
        "containment": float(containment),
        "max_ke_drop": float(max(val_ke1, val_ke2)),
        "max_spin_var": float(max(val_spin1, val_spin2)),
    }


def _empty_result(conflict_class, score=0.0):
    return {
        "class": conflict_class,
        "score": float(score),
        "intersected": False,
        "post_intersect_static": False,
        "trajectory_stop": False,
        "trajectory_stop_score": 0.0,
        "emergency_stop": False,
        "emergency_stop_score": 0.0,
        "relative_velocity_converged": False,
        "relative_velocity_score": 0.0,
        "energy_dropped": False,
        "spinning": False,
        "merged": False,
        "occluded": False,
        "containment": 0.0,
        "max_ke_drop": 0.0,
        "max_spin_var": 0.0,
    }
