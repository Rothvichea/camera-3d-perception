"""
Bird's Eye View (BEV) Renderer.

Draws detected objects from a top-down perspective.
Similar to the BEV visualization in our PointPillars project,
but generated from camera-based 3D estimates instead of LiDAR.
"""

import numpy as np
import cv2


CLASS_COLORS = {
    0: ((0, 0, 255), 'person'),
    1: ((255, 165, 0), 'bicycle'),
    2: ((0, 255, 0), 'car'),
    3: ((0, 200, 200), 'motorcycle'),
    5: ((255, 0, 255), 'bus'),
    7: ((0, 150, 255), 'truck'),
}


class BEVRenderer:
    """Render a Bird's Eye View map from 3D detections."""
    
    def __init__(self, size=400, range_x=40, range_y=20):
        self.size = size
        self.range_x = range_x
        self.range_y = range_y
        self.scale_x = size / range_x
        self.scale_y = size / (2 * range_y)
    
    def render(self, detections_3d):
        """
        Draw BEV from list of 3D detections.
        Each detection: dict with cls_id, x3d, z3d, distance, label, conf
        """
        bev = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # Grid lines every 5 meters
        for d in range(5, self.range_x, 5):
            y_px = int(self.size - d * self.scale_x)
            if 0 <= y_px < self.size:
                cv2.line(bev, (0, y_px), (self.size, y_px), (30, 30, 30), 1)
                cv2.putText(bev, f'{d}m', (5, y_px - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
        
        # Center line (ego forward direction)
        cx = self.size // 2
        cv2.line(bev, (cx, 0), (cx, self.size), (30, 30, 30), 1)
        
        # Ego vehicle
        ego_y = self.size - 15
        cv2.rectangle(bev, (cx-8, ego_y-15), (cx+8, ego_y), (0, 255, 255), -1)
        cv2.putText(bev, 'EGO', (cx-12, ego_y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        
        # Draw detections
        for det in detections_3d:
            px = int(cx + det['x3d'] * self.scale_y)
            py = int(self.size - det['z3d'] * self.scale_x)
            
            if not (0 <= px < self.size and 0 <= py < self.size):
                continue
            
            color = CLASS_COLORS.get(det['cls_id'], ((200, 200, 200), 'unk'))[0]
            
            # Size based on class
            if det['cls_id'] in [2, 5, 7]:
                hw, hh = 6, 10
            else:
                hw, hh = 3, 3
            
            cv2.rectangle(bev, (px-hw, py-hh), (px+hw, py+hh), color, 2)
            label = f'{det["label"]} {det["distance"]:.0f}m'
            cv2.putText(bev, label, (px+hw+2, py+3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Title
        cv2.putText(bev, f'BEV - {len(detections_3d)} objects',
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return bev


if __name__ == '__main__':
    renderer = BEVRenderer()
    
    test_dets = [
        {'cls_id': 2, 'x3d': 3.0, 'z3d': 15.0, 'distance': 15.3, 'label': 'car', 'conf': 0.9},
        {'cls_id': 0, 'x3d': -2.0, 'z3d': 8.0, 'distance': 8.2, 'label': 'person', 'conf': 0.8},
        {'cls_id': 7, 'x3d': 5.0, 'z3d': 25.0, 'distance': 25.5, 'label': 'truck', 'conf': 0.7},
    ]
    
    bev = renderer.render(test_dets)
    cv2.imwrite('/tmp/test_bev.png', bev)
    print(f'BEV shape: {bev.shape}')
    print('✅ BEVRenderer working!')
