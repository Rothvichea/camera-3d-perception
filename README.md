# Camera-Based 3D Perception Stack

Real-time 3D object detection, depth estimation, multi-object tracking, and bird's-eye-view projection from a single RGB camera — with ROS2 integration for robotic deployment.

![Pipeline Demo](docs/demo_preview.png)

## Key Features

- **3D Object Detection** — YOLOv8 + Depth Anything v2 for camera-only 3D perception
- **Multi-Object Tracking** — ByteTrack with Kalman filter for persistent object IDs
- **Bird's Eye View** — Real-time top-down projection of detected objects
- **C++ Acceleration** — pybind11 module with up to 670x speedup on post-processing
- **ROS2 Integration** — Publishes MarkerArray, Image, and JSON detections for robotic systems
- **Real-Time** — 22+ FPS on RTX 3060 laptop GPU (46ms per frame)

## Architecture

```
RGB Camera / Video
    |
    +-- YOLOv8m ----------- 2D Detection (boxes, classes, confidence)
    |                           |
    +-- Depth Anything v2 -- Monocular Depth Map
    |                           |
    +-- 3D Fusion <------------- 2D Boxes + Depth -> 3D Positions
            |
            +-- ByteTrack ----- Multi-Object Tracking (persistent IDs)
            |
            +-- C++ NMS ------- Fast post-processing (670x speedup)
            |
            +-- BEV Renderer -- Bird's Eye View visualization
            |
            +-- ROS2 Node ----- /perception/markers (RViz)
                                /perception/detections (JSON)
                                /perception/image (annotated feed)
```

## Results

| Metric | Value |
|--------|-------|
| FPS | **22+ FPS** (RTX 3060 6GB) |
| Objects per frame | 8-17 |
| Detection classes | car, person, truck, bus, bicycle, motorcycle |
| Depth estimation | Monocular, 2-70m range |
| Tracking | ByteTrack with Kalman filter |
| C++ NMS speedup | **670x** over Python |
| C++ depth sampling speedup | **5.7x** over Python |
| Total C++ overhead | **0.38ms** per frame |
| GPU memory | ~2GB |

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Detection | YOLOv8m (Ultralytics) |
| Depth | Depth Anything v2 (HuggingFace) |
| Tracking | ByteTrack + Kalman Filter |
| Deep Learning | PyTorch, CUDA |
| C++ Acceleration | pybind11, g++ -O3 |
| Visualization | OpenCV, RViz2 |
| Robotics | ROS2 Humble |
| Experiment Tracking | MLflow |
| Language | Python 3.10, C++17 |

## Project Structure

```
camera-3d-perception/
|-- configs/
|   |-- perception.yaml          # Pipeline configuration
|   +-- perception.rviz          # RViz2 display config
|-- src/
|   |-- detection/               # YOLOv8 wrapper
|   |-- depth/
|   |   +-- depth_to_3d.py       # Pinhole camera model, 2D+depth -> 3D
|   |-- tracking/
|   |   |-- byte_tracker.py      # Kalman filter + IoU matching
|   |   +-- tracker.py           # ByteTrack main tracker
|   |-- visualization/
|   |   +-- bev_renderer.py      # Bird's Eye View rendering
|   |-- cpp/
|   |   |-- perception_cpp.cpp   # C++ NMS, depth sampling, IoU, 3D conversion
|   |   +-- build.sh             # Compilation script
|   +-- ros2_node/
|       +-- perception_node.py   # ROS2 publisher node
|-- scripts/
|   |-- run_perception.py        # Standalone pipeline (no tracking)
|   +-- run_perception_tracked.py # Full pipeline with tracking + C++
|-- launch/
|   +-- perception_launch.py     # ROS2 launch file (starts everything)
|-- models/                      # Downloaded model weights (gitignored)
|-- data/                        # Input videos (gitignored)
|-- outputs/                     # Generated videos and frames
|-- docs/                        # Documentation and diagrams
+-- README.md
```

## Quick Start

### Prerequisites

- Ubuntu 22.04
- NVIDIA GPU with CUDA support
- Conda (Miniconda or Anaconda)
- ROS2 Humble (for ROS2 features)

### Installation

```bash
# Clone
git clone https://github.com/Rothvichea/camera-3d-perception.git
cd camera-3d-perception

# Create conda environment
conda create -n perception python=3.10 -y
conda activate perception

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python numpy matplotlib pyyaml
pip install transformers huggingface_hub
pip install lap  # for ByteTrack

# Build C++ module
bash src/cpp/build.sh

# Download a test video (or use your own)
# Place .mp4 files in data/videos/
```

### Run Standalone (no ROS2)

```bash
# Basic pipeline
python3 scripts/run_perception.py

# Full pipeline with tracking + C++ acceleration
python3 scripts/run_perception_tracked.py

# Output video saved to outputs/videos/
```

### Run with ROS2 + RViz

```bash
source /opt/ros/humble/setup.bash
conda activate perception

# Launch everything (perception + TF + RViz)
ros2 launch launch/perception_launch.py

# Or run manually:
# Terminal 1: ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map base_link
# Terminal 2: python3 src/ros2_node/perception_node.py
# Terminal 3: rviz2 -d configs/perception.rviz
```

### ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/perception/markers` | MarkerArray | 3D boxes for RViz visualization |
| `/perception/image` | Image | Annotated camera feed |
| `/perception/detections` | String | JSON with tracked objects and 3D positions |
| `/perception/bev` | Image | Bird's eye view |

### Example Detection JSON

```json
{
  "frame": 150,
  "latency_ms": 43.2,
  "num_tracked": 8,
  "objects": [
    {"id": 3, "class": "car", "distance": 12.5, "x": 2.1, "z": 12.3, "confidence": 0.87},
    {"id": 7, "class": "person", "distance": 8.2, "x": -1.5, "z": 8.1, "confidence": 0.72}
  ]
}
```

## Methods

### Depth-to-3D Projection

Uses the inverse pinhole camera model to project 2D detections into 3D space:

```
X = (u - cx) * Z / fx      (lateral position)
Y = (v - cy) * Z / fy      (vertical position)
Z = estimated depth         (forward distance)
distance = sqrt(X^2 + Z^2) (ground distance)
```

Depth is estimated using a combination of bounding box size heuristic (larger box = closer object) refined by the Depth Anything v2 relative depth map.

### ByteTrack Multi-Object Tracking

Based on ByteTrack (Zhang et al., ECCV 2022):

1. **Split** detections into high-confidence and low-confidence groups
2. **Match** high-confidence detections to existing tracks using IoU
3. **Recover** lost tracks by matching with low-confidence detections
4. **Predict** motion between frames using Kalman filter
5. **Create** new tracks for unmatched high-confidence detections
6. **Remove** tracks not seen for 30 frames

### C++ Acceleration

Performance-critical operations implemented in C++17 with pybind11 bindings:

| Operation | Python | C++ | Speedup |
|-----------|--------|-----|---------|
| IoU matrix (20x20) | 629 us | 0.9 us | **670x** |
| Depth sampling (20 boxes) | 2,150 us | 374 us | **5.7x** |
| 2D NMS | -- | 2.3 us | native |
| 3D conversion (batch) | -- | 0.6 us | native |
| **Total post-processing** | -- | **0.38 ms** | near-zero overhead |

### Bird's Eye View (BEV)

Objects are projected onto a top-down grid centered on the ego vehicle:
- X-axis: lateral position (left/right)
- Y-axis: forward distance (depth)
- Color-coded by class (green=car, red=person, orange=truck)
- Grid lines every 5 meters for scale reference

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1050 Ti (4GB) | RTX 3060+ (6GB+) |
| RAM | 8GB | 16GB |
| CPU | Intel i5 8th gen | Intel i7/i9 |
| Storage | 5GB free | 10GB free |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |
| CUDA | 11.8+ | 12.1+ |

## References

- [YOLOv8](https://github.com/ultralytics/ultralytics) - Ultralytics, 2023
- [Depth Anything v2](https://arxiv.org/abs/2406.09414) - Yang et al., 2024
- [ByteTrack](https://arxiv.org/abs/2110.06864) - Zhang et al., ECCV 2022
- [ECA-Net](https://arxiv.org/abs/1910.03151) - Wang et al., CVPR 2020
- [PointPillars](https://arxiv.org/abs/1812.05784) - Lang et al., CVPR 2019
- [ROS2 Humble](https://docs.ros.org/en/humble/)

## Related Projects

- [AV 3D Perception Stack (LiDAR)](https://github.com/Rothvichea/av-perception-stack) - PointPillars-based 3D detection from LiDAR point clouds
- [Inspection Robot Nav2](https://github.com/Rothvichea/inspection-robot-nav2) - 4WS autonomous navigation with SLAM

## License

MIT License

## Author

**Rothvichea CHEA** - Mechatronics Engineer, IMT Mines Ales, France

- [Portfolio](https://rothvicheachea.netlify.app)
- [LinkedIn](https://www.linkedin.com/in/chea-rothvichea-a96154227/)
- [GitHub](https://github.com/Rothvichea)