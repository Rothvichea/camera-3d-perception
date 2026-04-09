"""
ROS2 3D Perception Node.

Subscribes to: camera image (or reads from video file)
Publishes:
  - /perception/detections (MarkerArray) - 3D boxes for RViz
  - /perception/image (Image) - annotated camera feed
  - /perception/tracked (String) - JSON tracked objects
  - /perception/bev (Image) - Bird's Eye View

Run: ros2 run -- python3 src/ros2_node/perception_node.py
"""

import os
import sys
import time
import json
import yaml
import numpy as np
import cv2
import torch
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration

from ultralytics import YOLO
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'cpp'))

from src.depth.depth_to_3d import DepthTo3D
from src.tracking.tracker import ByteTracker

try:
    import perception_cpp
    USE_CPP = True
except ImportError:
    USE_CPP = False

CLASS_COLORS = {
    0: ((1.0, 0.0, 0.0), 'person'),
    1: ((1.0, 0.6, 0.0), 'bicycle'),
    2: ((0.0, 1.0, 0.0), 'car'),
    3: ((0.0, 0.8, 0.8), 'motorcycle'),
    5: ((1.0, 0.0, 1.0), 'bus'),
    7: ((0.0, 0.6, 1.0), 'truck'),
}


class PerceptionNode(Node):
    """ROS2 node for real-time 3D perception."""
    
    def __init__(self):
        super().__init__('perception_3d_node')
        
        # Parameters
        self.declare_parameter('video_path', 'data/videos/test_algo.webm')
        self.declare_parameter('config_path', 'configs/perception.yaml')
        self.declare_parameter('use_camera_topic', False)
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('publish_rate', 15.0)
        
        video_path = self.get_parameter('video_path').value
        config_path = self.get_parameter('config_path').value
        self.use_camera = self.get_parameter('use_camera_topic').value
        pub_rate = self.get_parameter('publish_rate').value
        
        # Load config
        with open(os.path.join(PROJECT_ROOT, config_path), 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.PROC_W = self.cfg['process_width']
        self.PROC_H = self.cfg['process_height']
        
        self.get_logger().info(f'C++ acceleration: {"ON" if USE_CPP else "OFF"}')
        
        # ---- Publishers ----
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        self.pub_markers = self.create_publisher(MarkerArray, '/perception/markers', 10)
        self.pub_image = self.create_publisher(Image, '/perception/image', qos)
        self.pub_bev = self.create_publisher(Image, '/perception/bev', qos)
        self.pub_detections = self.create_publisher(String, '/perception/detections', 10)
        
        # ---- Load models ----
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Device: {self.device}')
        
        self.get_logger().info('Loading YOLOv8m...')
        self.yolo = YOLO(self.cfg['detection']['model'])
        self.yolo.to(self.device)
        
        self.get_logger().info('Loading Depth Anything v2...')
        self.depth_processor = AutoImageProcessor.from_pretrained(
            self.cfg['depth']['model'])
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            self.cfg['depth']['model']).to(self.device).eval()
        
        # ---- Setup ----
        cam = self.cfg['camera']
        self.depth_to_3d = DepthTo3D(cam['focal_length'], cam['focal_length'],
                                      self.PROC_W // 2, self.PROC_H // 2)
        self.tracker = ByteTracker(high_threshold=0.35, low_threshold=0.2,
                                    max_age=30, min_hits=1, iou_threshold=0.3)
        
        # ---- Video or camera input ----
        if self.use_camera:
            self.subscription = self.create_subscription(
                Image, self.get_parameter('camera_topic').value,
                self.camera_callback, qos)
            self.get_logger().info(f'Subscribing to camera topic')
        else:
            full_video = os.path.join(PROJECT_ROOT, video_path)
            self.cap = cv2.VideoCapture(full_video)
            if not self.cap.isOpened():
                self.get_logger().error(f'Cannot open {full_video}')
                return
            self.get_logger().info(f'Reading from video: {video_path}')
            
            # Timer for video playback
            self.timer = self.create_timer(1.0 / pub_rate, self.timer_callback)
        
        self.frame_count = 0
        self.get_logger().info('Perception node ready!')
    
    def process_frame(self, frame):
        """Run full perception pipeline on one frame."""
        frame = cv2.resize(frame, (self.PROC_W, self.PROC_H))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        t_start = time.time()
        
        # Depth estimation
        pil_img = PILImage.fromarray(frame_rgb)
        inputs = self.depth_processor(images=pil_img, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            depth_pred = outputs.predicted_depth
        
        depth_map = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1), size=(self.PROC_H, self.PROC_W),
            mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()
        
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 0:
            depth_map = (depth_map - d_min) / (d_max - d_min) * self.cfg['depth']['max_depth']
        depth_map = np.clip(depth_map, 0.5, self.cfg['depth']['max_depth'])
        
        # Detection
        results = self.yolo(frame, verbose=False, conf=0.35,
                           classes=self.cfg['detection']['classes'])
        
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                x3d, y3d, z3d, dist = self.depth_to_3d.box_to_3d(
                    x1, y1, x2, y2, depth_map)
                
                if dist < 1.0 or dist > 70:
                    continue
                
                label = CLASS_COLORS.get(cls_id, ((0.5, 0.5, 0.5), 'unk'))[1]
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'cls_id': cls_id, 'conf': conf,
                    'det_3d': {'x3d': x3d, 'y3d': y3d, 'z3d': z3d,
                              'distance': dist, 'label': label}
                })
        
        # Tracking
        tracks = self.tracker.update(detections)
        
        latency = (time.time() - t_start) * 1000
        
        return frame, tracks, latency
    
    def publish_results(self, frame, tracks, latency):
        """Publish all ROS2 messages."""
        stamp = self.get_clock().now().to_msg()
        
        # 1. Publish MarkerArray (3D boxes for RViz)
        marker_array = MarkerArray()
        
        # Clear old markers
        clear_marker = Marker()
        clear_marker.header.frame_id = 'base_link'
        clear_marker.header.stamp = stamp
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        for i, track in enumerate(tracks):
            if not track.det_3d:
                continue
            
            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.header.stamp = stamp
            marker.ns = 'detections'
            marker.id = track.track_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Position (convert: X=right, Z=forward -> ROS: X=forward, Y=left)
            marker.pose.position.x = float(track.det_3d['z3d'])
            marker.pose.position.y = float(-track.det_3d['x3d'])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Size based on class
            if track.cls_id in [2, 5, 7]:  # car, bus, truck
                marker.scale = Vector3(x=4.5, y=1.8, z=1.5)
            else:
                marker.scale = Vector3(x=0.5, y=0.5, z=1.7)
            
            # Color by class
            rgb = CLASS_COLORS.get(track.cls_id, ((0.5, 0.5, 0.5), 'unk'))[0]
            marker.color = ColorRGBA(r=float(rgb[0]), g=float(rgb[1]),
                                      b=float(rgb[2]), a=0.7)
            
            marker.lifetime = Duration(sec=0, nanosec=500000000)
            marker_array.markers.append(marker)
            
            # Text label
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = 'labels'
            text_marker.id = track.track_id + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = marker.pose.position.x
            text_marker.pose.position.y = marker.pose.position.y
            text_marker.pose.position.z = 2.0
            text_marker.scale.z = 0.8
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text_marker.text = f'#{track.track_id} {track.det_3d["label"]} {track.det_3d["distance"]:.1f}m'
            text_marker.lifetime = Duration(sec=0, nanosec=500000000)
            marker_array.markers.append(text_marker)
        
        self.pub_markers.publish(marker_array)
        
        # 2. Publish annotated image
        img_msg = Image()
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = 'camera'
        img_msg.height = frame.shape[0]
        img_msg.width = frame.shape[1]
        img_msg.encoding = 'bgr8'
        img_msg.step = frame.shape[1] * 3
        img_msg.data = frame.tobytes()
        self.pub_image.publish(img_msg)
        
        # 3. Publish JSON detections
        det_list = []
        for track in tracks:
            if not track.det_3d:
                continue
            det_list.append({
                'id': track.track_id,
                'class': track.det_3d['label'],
                'distance': round(track.det_3d['distance'], 2),
                'x': round(track.det_3d['x3d'], 2),
                'z': round(track.det_3d['z3d'], 2),
                'confidence': round(track.conf, 2)
            })
        
        json_msg = String()
        json_msg.data = json.dumps({
            'frame': self.frame_count,
            'latency_ms': round(latency, 1),
            'num_tracked': len(tracks),
            'objects': det_list
        })
        self.pub_detections.publish(json_msg)
    
    def timer_callback(self):
        """Called by timer when reading from video file."""
        ret, frame_raw = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        
        frame, tracks, latency = self.process_frame(frame_raw)
        self.publish_results(frame, tracks, latency)
        
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f'Frame {self.frame_count} | Tracked: {len(tracks)} | '
                f'{latency:.0f}ms ({1000/max(latency,1):.0f}FPS)')
    
    def camera_callback(self, msg):
        """Called when receiving camera image from ROS2 topic."""
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        
        frame, tracks, latency = self.process_frame(frame)
        self.publish_results(frame, tracks, latency)
        
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f'Frame {self.frame_count} | Tracked: {len(tracks)} | '
                f'{latency:.0f}ms')


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
