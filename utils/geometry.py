import math

import config


def calculate_centroid(bbox):
    """
    Computes the centroid of a bounding box.
    bbox: List or tuple of (x1, y1, x2, y2)
    Returns: (cx, cy)
    """
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

def euclidean_distance(pt1, pt2):
    """
    Computes the Euclidean distance between two 2D points.
    pt1, pt2: Tuple or list of (x, y)
    """
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def _get_velocity(track):
    if track.velocities:
        return track.velocities[-1]
    return (0.0, 0.0)


def time_to_collision(track_a, track_b):
    """
    Frames until contact if both continue at current velocity.
    Returns float('inf') when not converging.
    """
    c1 = track_a.get_centroid()
    c2 = track_b.get_centroid()
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    distance = math.sqrt(dx * dx + dy * dy)

    vax, vay = _get_velocity(track_a)
    vbx, vby = _get_velocity(track_b)
    rel_vx = vax - vbx
    rel_vy = vay - vby
    closing_speed = math.sqrt(rel_vx * rel_vx + rel_vy * rel_vy)

    if closing_speed < config.TTC_MIN_CLOSING_SPEED:
        return float("inf")

    return distance / closing_speed


def ttc_score(ttc):
    if ttc == float("inf"):
        return 0.0
    return max(0.0, 1.0 - (ttc / config.TTC_MAX_FRAMES))


def calculate_angle_diff(v1, v2):
    """
    Computes the absolute angle difference in degrees between two vectors v1 and v2.
    """
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
        
    cos_angle = dot_product / (mag1 * mag2)
    # Clamp to avoid floating point issues out of range [-1, 1]
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

def line_intersection(line1, line2):
    """
    Checks if two line segments line1 (p1 to p2) and line2 (q1 to q2) intersect.
    line1: ((x1, y1), (x2, y2))
    line2: ((x3, y3), (x4, y4))
    Returns: True if they intersect, False otherwise.
    """
    p1, p2 = line1
    q1, q2 = line2
    
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # Return true if line segments AB and CD intersect
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def calculate_bbox_containment_ratio(box_small, box_large):
    """
    Calculates the ratio of the smaller bounding box contained inside the larger bounding box.
    Returns value between 0.0 (no overlap) and 1.0 (fully contained).
    """
    # Areas
    w_small = box_small[2] - box_small[0]
    h_small = box_small[3] - box_small[1]
    area_small = w_small * h_small
    if area_small <= 0:
        return 0.0

    # Intersection coordinates
    x1_inter = max(box_small[0], box_large[0])
    y1_inter = max(box_small[1], box_large[1])
    x2_inter = min(box_small[2], box_large[2])
    y2_inter = min(box_small[3], box_large[3])

    inter_w = max(0.0, x2_inter - x1_inter)
    inter_h = max(0.0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h

    return float(inter_area / area_small)

def directional_variance(angles_deg):
    """
    Computes circular (directional) variance on a list of angles in degrees.
    Handles angle wrapping (e.g. 359 to 0) correctly.
    Returns: Value from 0.0 (all angles pointing in same direction) to 1.0 (totally dispersed).
    """
    if not angles_deg:
        return 0.0
        
    rads = [math.radians(a) for a in angles_deg]
    x_coords = [math.cos(r) for r in rads]
    y_coords = [math.sin(r) for r in rads]
    
    x_mean = sum(x_coords) / len(angles_deg)
    y_mean = sum(y_coords) / len(angles_deg)
    
    R = math.sqrt(x_mean**2 + y_mean**2)
    # Circular variance
    return float(1.0 - R)
