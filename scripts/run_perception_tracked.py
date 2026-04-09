"""
Camera-Based 3D Perception Pipeline with Multi-Object Tracking.

RGB Frame -> YOLOv8 + Depth Anything v2 -> 3D Fusion -> ByteTrack
-> Output: tracked objects with IDs, trajectories, BEV map
"""

import os
import sys
import time
from shapely import distance
import yaml
import numpy as np
import cv2
import torch
from PIL import Image

from ultralytics import YOLO
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.depth.depth_to_3d import DepthTo3D
from src.visualization.bev_renderer import BEVRenderer, CLASS_COLORS
from src.tracking.tracker import ByteTracker
# C++ accelerated operations
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'cpp'))
try:
    import perception_cpp
    USE_CPP = True
    print('[C++] perception_cpp loaded - accelerated mode')
except ImportError:
    USE_CPP = False
    print('[C++] perception_cpp not found - Python fallback')


def load_config(path='configs/perception.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def estimate_depth(frame_rgb, depth_processor, depth_model, h, w, max_depth, device):
    """Run Depth Anything v2 and return depth map in meters."""
    pil_img = Image.fromarray(frame_rgb)
    inputs = depth_processor(images=pil_img, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = depth_model(**inputs)
        depth_pred = outputs.predicted_depth
    
    depth_map = torch.nn.functional.interpolate(
        depth_pred.unsqueeze(1), size=(h, w),
        mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()
    
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min > 0:
        depth_map = (depth_map - d_min) / (d_max - d_min) * max_depth * 0.5   
    depth_map = np.clip(depth_map, 0.5, max_depth)
    
    return depth_map


def detect_and_fuse(frame, depth_map, yolo, depth_to_3d, cfg):
    """Run YOLO + fuse with depth -> 3D detections for tracker."""
    results = yolo(frame, verbose=False, conf=0.35,
                   classes=cfg['detection']['classes'])
    
    detections = []
    if results[0].boxes is None:
        return detections, 0
    
    boxes_list = []
    cls_ids = []
    confs = []
    
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        boxes_list.append([x1, y1, x2, y2])
        cls_ids.append(cls_id)
        confs.append(conf)
    
    if len(boxes_list) == 0:
        return detections, 0
    
    boxes_arr = np.array(boxes_list, dtype=np.int32)
    confs_arr = np.array(confs, dtype=np.float32)
    
    import time
    t_cpp = time.time()
    
    if USE_CPP:
        # C++ accelerated: batch depth sampling + 3D conversion + NMS
        depths = perception_cpp.sample_depth_batch(depth_map, boxes_arr, 0.25)
        
        cam_cfg = cfg['camera']
        PROC_W = cfg['process_width']
        PROC_H = cfg['process_height']
        
        pos_3d = perception_cpp.boxes_to_3d(
            boxes_arr, depths,
            float(cam_cfg['focal_length']), float(cam_cfg['focal_length']),
            float(PROC_W // 2), float(PROC_H // 2),
            PROC_H
        )
        
        # NMS
        boxes_f = boxes_arr.astype(np.float32)
        keep = perception_cpp.nms_2d(boxes_f, confs_arr, 0.5)
        
        cpp_ms = (time.time() - t_cpp) * 1000
        
        for idx in keep:
            x3d, y3d, z3d, distance = pos_3d[idx]
            
            if distance < 1.0 or distance > 70:
                continue
            
            label = CLASS_COLORS.get(cls_ids[idx], ((200,200,200), 'unknown'))[1]
            
            detections.append({
                'bbox': boxes_list[idx],
                'cls_id': cls_ids[idx],
                'conf': confs[idx],
                'det_3d': {
                    'x3d': float(x3d), 'y3d': float(y3d), 'z3d': float(z3d),
                    'distance': float(distance), 'label': label
                }
            })
    else:
        # Python fallback
        cpp_ms = 0
        for i in range(len(boxes_list)):
            x1, y1, x2, y2 = boxes_list[i]
            x3d, y3d, z3d, distance = depth_to_3d.box_to_3d(x1, y1, x2, y2, depth_map)
            
            if distance < 1.0 or distance > 70:
                continue
            
            label = CLASS_COLORS.get(cls_ids[i], ((200,200,200), 'unknown'))[1]
            
            detections.append({
                'bbox': boxes_list[i],
                'cls_id': cls_ids[i],
                'conf': confs[i],
                'det_3d': {
                    'x3d': x3d, 'y3d': y3d, 'z3d': z3d,
                    'distance': distance, 'label': label
                }
            })
    
    return detections, cpp_ms


# Unique colors for track IDs
TRACK_COLORS = [
    (255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 50),
    (255, 50, 255), (50, 255, 255), (255, 150, 50), (150, 50, 255),
    (50, 255, 150), (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (200, 200, 50), (200, 50, 200), (50, 200, 200), (255, 200, 100),
]

def get_track_color(track_id):
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def draw_tracked_frame(frame, tracks, lat_det, lat_depth, total_ms, cpp_ms,
                       frame_num, max_frames):
    """Draw tracked objects with IDs and trajectories."""
    display = frame.copy()
    
    for track in tracks:
        bbox = track.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        color = get_track_color(track.track_id)
        
        # Class label
        cls_name = CLASS_COLORS.get(track.cls_id, ((200,200,200), 'unk'))[1]
        
        # Distance from 3D info
        dist_str = ''
        if track.det_3d and 'distance' in track.det_3d:
            dist_str = f' {track.det_3d["distance"]:.1f}m'
        
        # Draw box
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID label
        label = f'#{track.track_id} {cls_name}{dist_str}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(display, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(display, label, (x1+3, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw trajectory (last N positions)
        if len(track.history) > 1:
            pts = []
            for hist_bbox in track.history[-15:]:
                hcx = int((hist_bbox[0] + hist_bbox[2]) / 2)
                hcy = int((hist_bbox[1] + hist_bbox[3]) / 2)
                pts.append((hcx, hcy))
            
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thickness = max(1, int(alpha * 3))
                cv2.line(display, pts[i-1], pts[i], color, thickness)
    
    # HUD
    cv2.rectangle(display, (0, 0), (500, 70), (0, 0, 0), -1)
    cv2.putText(display,
                f'Det:{lat_det:.0f}ms Depth:{lat_depth:.0f}ms '
                f'Total:{total_ms:.0f}ms ({1000/max(total_ms,1):.0f}FPS)',
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 120), 1)
    cv2.putText(display,
                f'Tracked: {len(tracks)} objects | Frame {frame_num}/{max_frames}',
                (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cpp_label = f'C++ NMS+Depth+3D: {cpp_ms:.2f}ms' if cpp_ms > 0 else 'Python mode'
    cpp_color = (0, 255, 255) if cpp_ms > 0 else (100, 100, 100)
    cv2.putText(display, cpp_label,
                (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cpp_color, 1)
    
    return display


def draw_tracked_bev(tracks, bev_size=400, range_x=40, range_y=20):
    """Draw BEV with tracked objects and trajectories."""
    bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    
    scale_x = bev_size / range_x
    scale_y = bev_size / (2 * range_y)
    cx = bev_size // 2
    
    # Grid
    for d in range(5, range_x, 5):
        y_px = int(bev_size - d * scale_x)
        if 0 <= y_px < bev_size:
            cv2.line(bev, (0, y_px), (bev_size, y_px), (30, 30, 30), 1)
            cv2.putText(bev, f'{d}m', (5, y_px-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
    
    cv2.line(bev, (cx, 0), (cx, bev_size), (30, 30, 30), 1)
    
    # Ego
    ego_y = bev_size - 15
    cv2.rectangle(bev, (cx-8, ego_y-15), (cx+8, ego_y), (0, 255, 255), -1)
    cv2.putText(bev, 'EGO', (cx-12, ego_y+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    
    # Draw tracks
    for track in tracks:
        if not track.det_3d:
            continue
        
        x3d = track.det_3d.get('x3d', 0)
        z3d = track.det_3d.get('z3d', 0)
        dist = track.det_3d.get('distance', 0)
        label = track.det_3d.get('label', 'unk')
        
        px = int(cx + x3d * scale_y)
        py = int(bev_size - z3d * scale_x)
        
        if not (0 <= px < bev_size and 0 <= py < bev_size):
            continue
        
        color = get_track_color(track.track_id)
        
        # Size based on class
        if track.cls_id in [2, 5, 7]:
            hw, hh = 6, 10
        else:
            hw, hh = 3, 3
        
        cv2.rectangle(bev, (px-hw, py-hh), (px+hw, py+hh), color, 2)
        
        txt = f'#{track.track_id} {label} {dist:.0f}m'
        cv2.putText(bev, txt, (px+hw+2, py+3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    cv2.putText(bev, f'BEV - {len(tracks)} tracked',
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return bev


def main():
    cfg = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Load models
    print('Loading YOLOv8m...')
    yolo = YOLO(cfg['detection']['model'])
    yolo.to(device)
    print('  Done')
    
    print('Loading Depth Anything v2...')
    depth_processor = AutoImageProcessor.from_pretrained(cfg['depth']['model'])
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        cfg['depth']['model']).to(device).eval()
    print('  Done')
    
    # Setup
    PROC_W = cfg['process_width']
    PROC_H = cfg['process_height']
    cam_cfg = cfg['camera']
    
    depth_to_3d = DepthTo3D(cam_cfg['focal_length'], cam_cfg['focal_length'],
                             PROC_W // 2, PROC_H // 2)
    tracker = ByteTracker(high_threshold=0.35, low_threshold=0.2,
                          max_age=30, min_hits=1, iou_threshold=0.3)
    
    bev_size = cfg['bev']['size']
    
    # Open video
    cap = cv2.VideoCapture(cfg['video_path'])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    max_frames = min(total_frames, cfg['max_frames'])
    print(f'Video: {fps_in:.0f}FPS, processing {max_frames} frames')
    
    # Output
    out_path = 'outputs/videos/perception_tracked.mp4'
    os.makedirs('outputs/videos', exist_ok=True)
    os.makedirs('outputs/inference', exist_ok=True)
    
    combined_h = max(PROC_H, bev_size)
    combined_w = PROC_W + bev_size
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, cfg['output']['fps'],
                              (combined_w, combined_h))
    
    print(f'\nProcessing with tracking...\n')
    
    frame_count = 0
    total_latency = 0
    
    while frame_count < max_frames:
        ret, frame_raw = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame_raw, (PROC_W, PROC_H))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        t_start = time.time()
        
        # Depth
        t_depth = time.time()
        depth_map = estimate_depth(frame_rgb, depth_processor, depth_model,
                                    PROC_H, PROC_W, cfg['depth']['max_depth'], device)
        lat_depth = (time.time() - t_depth) * 1000
        
        # Detection + 3D + C++ NMS
        t_det = time.time()
        detections, cpp_ms = detect_and_fuse(frame, depth_map, yolo, depth_to_3d, cfg)
        lat_det = (time.time() - t_det) * 1000
        
        # Tracking
        active_tracks = tracker.update(detections)
        
        total_ms = (time.time() - t_start) * 1000
        total_latency += total_ms
        
        # Visualize
        display = draw_tracked_frame(frame, active_tracks, lat_det, lat_depth,
                                      total_ms, cpp_ms, frame_count+1, max_frames)
        bev = draw_tracked_bev(active_tracks, bev_size,
                                cfg['bev']['range_x'], cfg['bev']['range_y'])
        
        # Combine
        if PROC_H < bev_size:
            pad = np.zeros((bev_size - PROC_H, PROC_W, 3), dtype=np.uint8)
            display = np.vstack([display, pad])
        else:
            display = display[:bev_size]
        
        combined = np.hstack([display, bev])
        writer.write(combined)
        
        if frame_count % 50 == 0:
            cv2.imwrite(f'outputs/inference/tracked_{frame_count:04d}.png', combined)
        
        frame_count += 1
        
        if frame_count % 20 == 0 or frame_count == 1:
            avg_ms = total_latency / frame_count
            print(f'  Frame {frame_count}/{max_frames} | '
                  f'Tracked: {len(active_tracks)} | '
                  f'C++: {cpp_ms:.2f}ms | '
                  f'{total_ms:.0f}ms | Avg: {avg_ms:.0f}ms')
    
    writer.release()
    cap.release()
    
    avg = total_latency / max(frame_count, 1)
    print(f'\n{"="*55}')
    print(f'Done! {frame_count} frames | Avg: {avg:.0f}ms ({1000/avg:.1f} FPS)')
    print(f'Video: {out_path}')
    print(f'Play:  xdg-open {out_path}')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
