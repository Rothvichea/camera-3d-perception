"""
ByteTrack Main Tracker.

Manages all active tracks and performs frame-by-frame association.
"""

import numpy as np
from src.tracking.byte_tracker import Track, compute_iou_matrix, linear_assignment


class ByteTracker:
    """
    ByteTrack multi-object tracker.
    
    Key parameters:
        high_threshold: detections above this are "high confidence"
        low_threshold: detections above this but below high are "low confidence"
        max_age: remove track if not seen for this many frames
        min_hits: track must be matched this many times before displayed
        iou_threshold: IoU threshold for matching
    """
    
    def __init__(self, high_threshold=0.5, low_threshold=0.1,
                 max_age=30, min_hits=3, iou_threshold=0.3):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = []
        self.frame_count = 0
    
    def update(self, detections):
        """
        Update tracker with new frame detections.
        
        Args:
            detections: list of dicts with keys:
                bbox: [x1, y1, x2, y2]
                cls_id: int
                conf: float
                det_3d: dict with x3d, y3d, z3d, distance (optional)
        
        Returns:
            active_tracks: list of Track objects to display
        """
        self.frame_count += 1
        
        # ---- Step 1: Predict all existing tracks ----
        for track in self.tracks:
            track.predict()
        
        if len(detections) == 0:
            # Remove dead tracks
            self.tracks = [t for t in self.tracks 
                          if t.time_since_update <= self.max_age]
            return self._get_active_tracks()
        
        # ---- Step 2: Split into high and low confidence ----
        det_bboxes = np.array([d['bbox'] for d in detections])
        det_confs = np.array([d['conf'] for d in detections])
        
        high_mask = det_confs >= self.high_threshold
        low_mask = (det_confs >= self.low_threshold) & (~high_mask)
        
        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]
        
        # ---- Step 3: Match HIGH detections to tracks ----
        unmatched_tracks = list(range(len(self.tracks)))
        
        if len(high_indices) > 0 and len(self.tracks) > 0:
            track_bboxes = np.array([t.bbox for t in self.tracks])
            high_bboxes = det_bboxes[high_indices]
            
            iou_matrix = compute_iou_matrix(track_bboxes, high_bboxes)
            cost_matrix = 1 - iou_matrix
            
            matches, unmatched_tracks, unmatched_high = linear_assignment(
                cost_matrix, 1 - self.iou_threshold
            )
            
            # Apply matches
            for track_idx, det_idx in matches:
                d = detections[high_indices[det_idx]]
                self.tracks[track_idx].update(
                    np.array(d['bbox']), d['cls_id'], d['conf'],
                    d.get('det_3d', None)
                )
        else:
            unmatched_high = list(range(len(high_indices)))
        
        # ---- Step 4: Match remaining tracks to LOW detections ----
        if len(low_indices) > 0 and len(unmatched_tracks) > 0:
            remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
            track_bboxes = np.array([t.bbox for t in remaining_tracks])
            low_bboxes = det_bboxes[low_indices]
            
            iou_matrix = compute_iou_matrix(track_bboxes, low_bboxes)
            cost_matrix = 1 - iou_matrix
            
            matches_low, still_unmatched, _ = linear_assignment(
                cost_matrix, 1 - self.iou_threshold
            )
            
            for track_idx, det_idx in matches_low:
                real_track_idx = unmatched_tracks[track_idx]
                d = detections[low_indices[det_idx]]
                self.tracks[real_track_idx].update(
                    np.array(d['bbox']), d['cls_id'], d['conf'],
                    d.get('det_3d', None)
                )
        
        # ---- Step 5: Create new tracks from unmatched high detections ----
        for det_idx in unmatched_high:
            d = detections[high_indices[det_idx]]
            new_track = Track(
                np.array(d['bbox']), d['cls_id'], d['conf'],
                d.get('det_3d', None)
            )
            self.tracks.append(new_track)
        
        # ---- Step 6: Remove dead tracks ----
        self.tracks = [t for t in self.tracks
                      if t.time_since_update <= self.max_age]
        
        return self._get_active_tracks()
    
    def _get_active_tracks(self):
        """Return tracks that have been confirmed (enough hits)."""
        return [t for t in self.tracks
                if t.hits >= self.min_hits and t.time_since_update == 0]


if __name__ == '__main__':
    tracker = ByteTracker(high_threshold=0.5, min_hits=1)
    
    # Frame 1: two cars
    dets1 = [
        {'bbox': [100, 200, 200, 300], 'cls_id': 2, 'conf': 0.9},
        {'bbox': [400, 200, 500, 300], 'cls_id': 2, 'conf': 0.8},
    ]
    tracks = tracker.update(dets1)
    print(f'Frame 1: {len(tracks)} tracks')
    for t in tracks:
        print(f'  Track #{t.track_id} cls={t.cls_id} bbox={t.bbox.astype(int)}')
    
    # Frame 2: same cars moved slightly
    dets2 = [
        {'bbox': [105, 200, 205, 300], 'cls_id': 2, 'conf': 0.85},
        {'bbox': [410, 200, 510, 300], 'cls_id': 2, 'conf': 0.75},
    ]
    tracks = tracker.update(dets2)
    print(f'Frame 2: {len(tracks)} tracks')
    for t in tracks:
        print(f'  Track #{t.track_id} cls={t.cls_id} hits={t.hits}')
    
    # Frame 3: one car disappears
    dets3 = [
        {'bbox': [110, 200, 210, 300], 'cls_id': 2, 'conf': 0.9},
    ]
    tracks = tracker.update(dets3)
    print(f'Frame 3: {len(tracks)} tracks (one car gone)')
    for t in tracks:
        print(f'  Track #{t.track_id} hits={t.hits}')
    
    print('✅ ByteTracker working!')
