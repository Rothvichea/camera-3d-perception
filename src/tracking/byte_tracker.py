"""
ByteTrack Multi-Object Tracker.

Based on: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
           Zhang et al., ECCV 2022

How it works:
  1. Split detections into HIGH confidence and LOW confidence groups
  2. Match HIGH detections to existing tracks using IoU
  3. Match remaining tracks to LOW detections (recover lost objects)
  4. Create new tracks for unmatched high detections
  5. Remove tracks that haven't been seen for too long

Why ByteTrack:
  - Simple and fast (no deep features needed)
  - Handles occlusion well (low-confidence recovery)
  - State-of-the-art on MOT benchmarks
  - Perfect for real-time robotics applications
"""

import numpy as np
from collections import defaultdict

try:
    import lap
    USE_LAP = True
except ImportError:
    USE_LAP = False
    print("Warning: lap not installed, using greedy matching")


class KalmanFilter2D:
    """
    Simple 2D Kalman Filter for bounding box tracking.
    
    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    
    Predicts where the box will be in the next frame
    based on its current velocity.
    """
    
    def __init__(self, bbox):
        """Initialize with first detection [x1, y1, x2, y2]."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        # Uncertainty
        self.P = np.eye(8, dtype=np.float32) * 10
        self.P[4:, 4:] *= 100  # high uncertainty on velocity
        
        # Process noise
        self.Q = np.eye(8, dtype=np.float32) * 0.1
        self.Q[4:, 4:] *= 1.0
        
        # Measurement noise
        self.R = np.eye(4, dtype=np.float32) * 1.0
    
    def predict(self):
        """Predict next state based on constant velocity model."""
        # State transition: position += velocity
        F = np.eye(8, dtype=np.float32)
        F[0, 4] = 1  # cx += vx
        F[1, 5] = 1  # cy += vy
        F[2, 6] = 1  # w += vw
        F[3, 7] = 1  # h += vh
        
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        
        return self.get_bbox()
    
    def update(self, bbox):
        """Update state with new measurement [x1, y1, x2, y2]."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w, h], dtype=np.float32)
        
        # Measurement matrix
        H = np.zeros((4, 8), dtype=np.float32)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update
        y = z - H @ self.state
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P
        
        return self.get_bbox()
    
    def get_bbox(self):
        """Get current bbox as [x1, y1, x2, y2]."""
        cx, cy, w, h = self.state[:4]
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])


class Track:
    """Single object track with ID, state, and history."""
    
    _next_id = 1
    
    def __init__(self, bbox, cls_id, conf, det_3d=None):
        self.track_id = Track._next_id
        Track._next_id += 1
        
        self.kf = KalmanFilter2D(bbox)
        self.cls_id = cls_id
        self.conf = conf
        self.det_3d = det_3d  # 3D info (x3d, y3d, z3d, distance)
        
        self.hits = 1          # total successful matches
        self.age = 0           # frames since creation
        self.time_since_update = 0  # frames since last match
        self.history = [bbox.copy()]
    
    def predict(self):
        """Predict next position."""
        self.age += 1
        self.time_since_update += 1
        return self.kf.predict()
    
    def update(self, bbox, cls_id, conf, det_3d=None):
        """Update with new detection."""
        self.kf.update(bbox)
        self.cls_id = cls_id
        self.conf = conf
        self.det_3d = det_3d
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox.copy())
        if len(self.history) > 30:
            self.history.pop(0)
    
    @property
    def bbox(self):
        return self.kf.get_bbox()


def compute_iou_matrix(boxes_a, boxes_b):
    """Compute IoU between two sets of [x1,y1,x2,y2] boxes."""
    N, M = len(boxes_a), len(boxes_b)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    
    iou = np.zeros((N, M), dtype=np.float32)
    
    for i in range(N):
        x1 = np.maximum(boxes_a[i, 0], boxes_b[:, 0])
        y1 = np.maximum(boxes_a[i, 1], boxes_b[:, 1])
        x2 = np.minimum(boxes_a[i, 2], boxes_b[:, 2])
        y2 = np.minimum(boxes_a[i, 3], boxes_b[:, 3])
        
        inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
        area_a = (boxes_a[i,2]-boxes_a[i,0]) * (boxes_a[i,3]-boxes_a[i,1])
        area_b = (boxes_b[:,2]-boxes_b[:,0]) * (boxes_b[:,3]-boxes_b[:,1])
        
        iou[i] = inter / np.maximum(area_a + area_b - inter, 1e-6)
    
    return iou


def linear_assignment(cost_matrix, threshold):
    """
    Solve assignment problem using LAP (Linear Assignment Problem).
    Returns matched pairs and unmatched indices.
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    
    if USE_LAP:
        # Use lap library for optimal assignment
        cost = cost_matrix.copy()
        cost[cost > threshold] = threshold + 1e-4
        _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=threshold)
        
        matches = []
        unmatched_a = []
        unmatched_b = list(range(cost_matrix.shape[1]))
        
        for i, j in enumerate(x):
            if j >= 0 and cost_matrix[i, j] <= threshold:
                matches.append((i, j))
                if j in unmatched_b:
                    unmatched_b.remove(j)
            else:
                unmatched_a.append(i)
        
        return matches, unmatched_a, unmatched_b
    else:
        # Greedy matching fallback
        matches = []
        unmatched_a = list(range(cost_matrix.shape[0]))
        unmatched_b = list(range(cost_matrix.shape[1]))
        
        while True:
            idx = np.unravel_index(cost_matrix.argmin(), cost_matrix.shape)
            if cost_matrix[idx] > threshold:
                break
            
            i, j = idx
            matches.append((i, j))
            unmatched_a.remove(i)
            unmatched_b.remove(j)
            cost_matrix[i, :] = threshold + 1
            cost_matrix[:, j] = threshold + 1
        
        return matches, unmatched_a, unmatched_b
