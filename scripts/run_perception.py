"""
Camera-Based 3D Perception Pipeline.

RGB Frame -> YOLOv8 (detection) + Depth Anything v2 (depth)
-> Fuse into 3D positions -> BEV map -> Video output

No LiDAR needed. Works with any camera or video file.
"""

import os
import sys
import time
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
    
    # Normalize to approximate meters
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min > 0:
        depth_map = (depth_map - d_min) / (d_max - d_min) * max_depth
    depth_map = np.clip(depth_map, 0.5, max_depth)
    
    return depth_map


def detect_and_fuse(frame, depth_map, yolo, depth_to_3d, cfg):
    """Run YOLO + fuse with depth to get 3D detections."""
    results = yolo(frame, verbose=False, conf=cfg['detection']['confidence'],
                   classes=cfg['detection']['classes'])
    
    detections_3d = []
    if results[0].boxes is None:
        return detections_3d
    
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        x3d, y3d, z3d, distance = depth_to_3d.box_to_3d(x1, y1, x2, y2, depth_map)
        
        if distance < 0.5 or distance > 60:
            continue
        
        label = CLASS_COLORS.get(cls_id, ((200, 200, 200), 'unknown'))[1]
        
        detections_3d.append({
            'cls_id': cls_id, 'conf': conf,
            'bbox': (x1, y1, x2, y2),
            'x3d': x3d, 'y3d': y3d, 'z3d': z3d,
            'distance': distance, 'label': label
        })
    
    return detections_3d


def draw_detections(frame, detections_3d, lat_yolo, lat_depth, total_ms, frame_num, max_frames):
    """Draw 2D boxes with 3D distance on camera frame."""
    display = frame.copy()
    
    for det in detections_3d:
        x1, y1, x2, y2 = det['bbox']
        color = CLASS_COLORS.get(det['cls_id'], ((200, 200, 200), 'unk'))[0]
        
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        
        label = f'{det["label"]} {det["distance"]:.1f}m ({det["conf"]:.2f})'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(display, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(display, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Depth at center
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.putText(display, f'{det["z3d"]:.1f}m', (cx-15, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # HUD overlay
    cv2.rectangle(display, (0, 0), (420, 50), (0, 0, 0), -1)
    cv2.putText(display,
                f'YOLO:{lat_yolo:.0f}ms Depth:{lat_depth:.0f}ms '
                f'Total:{total_ms:.0f}ms ({1000/max(total_ms,1):.0f}FPS)',
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 120), 1)
    cv2.putText(display, f'Objects: {len(detections_3d)} | Frame {frame_num}/{max_frames}',
                (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return display


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
    bev_renderer = BEVRenderer(cfg['bev']['size'], cfg['bev']['range_x'], cfg['bev']['range_y'])
    
    # Open video
    cap = cv2.VideoCapture(cfg['video_path'])
    if not cap.isOpened():
        print(f'ERROR: Cannot open {cfg["video_path"]}')
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = min(total_frames, cfg['max_frames'])
    print(f'Video: {w_in}x{h_in} @ {fps_in:.0f}FPS, processing {max_frames} frames')
    
    # Video writer
    os.makedirs(os.path.dirname(cfg['output']['video']), exist_ok=True)
    os.makedirs(cfg['output']['frames_dir'], exist_ok=True)
    
    bev_size = cfg['bev']['size']
    combined_h = max(PROC_H, bev_size)
    combined_w = PROC_W + bev_size
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(cfg['output']['video'], fourcc,
                              cfg['output']['fps'], (combined_w, combined_h))
    
    print(f'\nProcessing...\n')
    
    frame_count = 0
    total_latency = 0
    
    while frame_count < max_frames:
        ret, frame_raw = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame_raw, (PROC_W, PROC_H))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        t_start = time.time()
        
        # YOLOv8
        t_yolo = time.time()
        lat_yolo = 0
        
        # Depth estimation
        t_depth = time.time()
        depth_map = estimate_depth(frame_rgb, depth_processor, depth_model,
                                    PROC_H, PROC_W, cfg['depth']['max_depth'], device)
        lat_depth = (time.time() - t_depth) * 1000
        
        # Detection + 3D fusion
        detections_3d = detect_and_fuse(frame, depth_map, yolo, depth_to_3d, cfg)
        lat_yolo = (time.time() - t_start) * 1000 - lat_depth
        
        total_ms = (time.time() - t_start) * 1000
        total_latency += total_ms
        
        # Visualize
        display = draw_detections(frame, detections_3d, lat_yolo, lat_depth,
                                   total_ms, frame_count+1, max_frames)
        bev = bev_renderer.render(detections_3d)
        
        # Combine
        if PROC_H < bev_size:
            pad = np.zeros((bev_size - PROC_H, PROC_W, 3), dtype=np.uint8)
            display = np.vstack([display, pad])
        else:
            display = display[:bev_size]
        
        combined = np.hstack([display, bev])
        writer.write(combined)
        
        if frame_count % 50 == 0:
            cv2.imwrite(f'{cfg["output"]["frames_dir"]}/frame_{frame_count:04d}.png', combined)
        
        frame_count += 1
        
        if frame_count % 20 == 0 or frame_count == 1:
            avg_ms = total_latency / frame_count
            print(f'  Frame {frame_count}/{max_frames} | '
                  f'Obj: {len(detections_3d)} | '
                  f'{total_ms:.0f}ms | Avg: {avg_ms:.0f}ms')
    
    writer.release()
    cap.release()
    
    avg = total_latency / max(frame_count, 1)
    print(f'\n{"="*55}')
    print(f'Done! {frame_count} frames | Avg: {avg:.0f}ms ({1000/avg:.1f} FPS)')
    print(f'Video: {cfg["output"]["video"]}')
    print(f'Play:  xdg-open {cfg["output"]["video"]}')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()