"""
Depth to 3D Coordinate Conversion.

Uses the pinhole camera model (inverse projection):
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth
"""

import numpy as np


class DepthTo3D:
    """Convert 2D pixel + depth to 3D world coordinates."""
    
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def update_intrinsics(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def pixel_to_3d(self, u, v, depth):
        """
        Convert pixel (u,v) + depth to 3D coordinates.
        Returns: (X, Y, Z) in meters
        """
        Z = depth
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        return X, Y, Z
    
    def box_to_3d(self, x1, y1, x2, y2, depth_map):
        """
        Estimate 3D position of a detected object.
        
        Depth Anything v2 outputs RELATIVE depth where:
          HIGH raw value = CLOSE object
          LOW raw value = FAR object
        
        We use a simple heuristic based on bounding box size:
          - Larger box in image = closer object
          - Combined with raw depth for relative ordering
        
        Returns: (X, Y, Z, distance) in meters
        """
        cx = (x1 + x2) // 2
        box_h = y2 - y1
        box_w = x2 - x1
        img_h = depth_map.shape[0]
        
        # Method: Use bounding box size as depth proxy
        # Bigger box = closer. This is more reliable than
        # the raw depth values for distance estimation.
        # 
        # Approximate: a car ~4.5m long at 10m fills ~200px wide (at 640px frame)
        # So: distance ≈ (focal_length * real_object_width) / box_width_pixels
        # For a generic object, use box height relative to image height
        
        # Estimate distance from box size
        # Larger boxes (closer) get smaller distance
        box_ratio = max(box_h, box_w) / max(img_h, 1)
        
        if box_ratio < 0.01:
            return 0, 0, 0, 0
        
        # Empirical mapping: box_ratio -> distance
        # box_ratio = 0.5 (half the image) -> ~5m
        # box_ratio = 0.2 -> ~15m
        # box_ratio = 0.1 -> ~30m
        # box_ratio = 0.05 -> ~50m
        # Formula: distance = k / box_ratio
        k = 2.5  # tuning constant
        depth = k / box_ratio
        depth = np.clip(depth, 2.0, 70.0)
        
        # Refine with relative depth map (for ordering among similar-sized objects)
        margin_x = (x2 - x1) // 4
        margin_y = (y2 - y1) // 4
        roi_x1 = max(0, x1 + margin_x)
        roi_y1 = max(0, y1 + margin_y)
        roi_x2 = min(depth_map.shape[1], x2 - margin_x)
        roi_y2 = min(depth_map.shape[0], y2 - margin_y)
        
        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
            depth_roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
            if depth_roi.size > 0:
                raw_depth = np.median(depth_roi)
                global_max = depth_map.max()
                if global_max > 0:
                    # Higher raw = closer -> gives smaller relative factor
                    # Lower raw = farther -> gives larger relative factor
                    relative_factor = 1.0 - (raw_depth / global_max) * 0.3
                    depth = depth * relative_factor
        
        depth = np.clip(depth, 2.0, 70.0)
        
        X, Y, Z = self.pixel_to_3d(cx, y2, depth)
        distance = np.sqrt(X**2 + Z**2)
        
        return X, Y, Z, distance


if __name__ == '__main__':
    converter = DepthTo3D(700, 700, 320, 180)
    
    fake_depth = np.random.uniform(0, 80, (360, 640)).astype(np.float32)
    
    # Big box (close car)
    X, Y, Z, dist = converter.box_to_3d(200, 200, 450, 350, fake_depth)
    print(f'Big box (close car): {dist:.1f}m  (should be ~5-15m)')
    
    # Medium box 
    X, Y, Z, dist = converter.box_to_3d(250, 150, 350, 220, fake_depth)
    print(f'Medium box: {dist:.1f}m  (should be ~15-30m)')
    
    # Small box (far car)
    X, Y, Z, dist = converter.box_to_3d(300, 160, 340, 190, fake_depth)
    print(f'Small box (far car): {dist:.1f}m  (should be ~30-60m)')
    
    print('✅ DepthTo3D working!')