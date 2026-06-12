from utils.geometry import euclidean_distance

def proximity_filter(tracks, threshold=121.0):
    """
    Filters vehicle pairs that are in close proximity.
    tracks: list of Track objects
    threshold: Euclidean distance threshold (in pixels)
    Returns: List of tuples (track1, track2, distance)
    """
    candidate_pairs = []
    num_tracks = len(tracks)
    
    for i in range(num_tracks):
        for j in range(i + 1, num_tracks):
            t1 = tracks[i]
            t2 = tracks[j]
            
            p1 = t1.get_centroid()
            p2 = t2.get_centroid()
            
            dist = euclidean_distance(p1, p2)
            if dist <= threshold:
                candidate_pairs.append((t1, t2, dist))
                
    return candidate_pairs
