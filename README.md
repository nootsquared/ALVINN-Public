# ALVINN — Assistive Vision System for the Visually Impaired

ALVINN combines real-time saliency detection and monocular depth estimation neural networks to identify visually important objects and calculate their distances, enabling assistive glasses to alert visually impaired users about nearby obstacles and points of interest.

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch, TorchVision, timm |
| **Vision Models** | DINOv2 (Vision Transformer), MobileNetV2, DPT (Dense Prediction Transformer) |
| **Computer Vision** | OpenCV, Open3D |
| **Visualization** | Matplotlib, Gradio |
| **3D Processing** | Open3D (point cloud generation & visualization) |
| **Hardware Acceleration** | CUDA 11.8 |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ALVINN Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐                                                          │
│   │  Camera/     │                                                          │
│   │  Video Input │                                                          │
│   └──────┬───────┘                                                          │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                    Parallel Processing                            │     │
│   │  ┌────────────────────┐      ┌────────────────────┐              │     │
│   │  │  Saliency Module   │      │   Depth Module     │              │     │
│   │  │  (FastSal)         │      │   (ALVINNDepth)    │              │     │
│   │  │                    │      │                    │              │     │
│   │  │  MobileNetV2       │      │  DINOv2 ViT-S/14   │              │     │
│   │  │  Encoder +         │      │  + DPT Head        │              │     │
│   │  │  Adaptation Layers │      │                    │              │     │
│   │  └─────────┬──────────┘      └─────────┬──────────┘              │     │
│   │            │                           │                          │     │
│   │            ▼                           ▼                          │     │
│   │     Saliency Map               Metric Depth Map                   │     │
│   │     (attention regions)        (distance in meters)               │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                    │                       │                                │
│                    └───────────┬───────────┘                                │
│                                ▼                                            │
│                    ┌───────────────────────┐                                │
│                    │    Fusion & Output    │                                │
│                    │  • Salient regions    │                                │
│                    │  • Distance labels    │                                │
│                    │  • Motion tracking    │                                │
│                    │  • 3D point clouds    │                                │
│                    └───────────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Breakdown

| Module | Description |
|--------|-------------|
| **Saliency_Predictions/** | FastSal model — lightweight saliency detection using MobileNetV2 encoder with adaptation layers. Predicts where humans would naturally look in a scene. |
| **Metric_Depth_Estimation/** | ALVINNDepth model — monocular depth estimation using DINOv2 Vision Transformer backbone with DPT (Dense Prediction Transformer) head for pixel-wise depth prediction. |
| **Main/** | Combined inference scripts for real-time webcam processing, video processing, and 3D point cloud generation. |

---

## Features

- **Real-Time Object Detection**: Identifies salient (visually important) objects in the scene
- **Metric Depth Estimation**: Calculates absolute distance to objects in meters
- **Temporal Stabilization**: Reduces depth flickering using motion-aware smoothing
- **Optical Flow Motion Tracking**: Displays camera movement direction via minimap overlay
- **3D Point Cloud Generation**: Converts depth maps to navigable 3D point clouds
- **Multiple Input Modes**: Supports live webcam feed and video file processing

---

## Model Specifications

### Saliency Model (FastSal)
| Specification | Value |
|--------------|-------|
| Backbone | MobileNetV2 (pretrained on ImageNet) |
| Architecture | Encoder-decoder with pixel shuffle upsampling |
| Input Resolution | 192 × 256 |
| Training Dataset | SALICON (10,000 images with eye-tracking data) |
| Output | Single-channel attention heatmap |

### Depth Model (ALVINNDepth)
| Specification | Value |
|--------------|-------|
| Backbone | DINOv2 ViT-S/14 (Vision Transformer Small, 14×14 patches) |
| Head | DPT (Dense Prediction Transformer) with feature fusion blocks |
| Input Resolution | 518 × 518 |
| Intermediate Layers | Layers [2, 5, 8, 11] extracted for multi-scale features |
| Output Channels | [48, 96, 192, 384] |
| Max Depth | 80 meters (indoor) |
| Checkpoints | Indoor and outdoor variants available |

### DINOv2 ViT-S Architecture
| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 384 |
| Transformer Depth | 12 blocks |
| Attention Heads | 6 |
| MLP Ratio | 4× |
| Patch Size | 14 × 14 |

---

## Test Results

### Video Processing Demo
Processed test videos with the following configuration:
- **Input**: 1080p video @ 30 FPS
- **Saliency Threshold**: 0.5
- **Temporal Stabilization**: Enabled

| Metric | Result |
|--------|--------|
| Processing Speed | ~8-12 FPS (RTX GPU) |
| Depth Range Accuracy | Objects detected from 0.5m to 15m reliably |
| Salient Object Detection | Successfully identifies people, furniture, doors |

### Point Cloud Generation
Generated 100-frame point cloud sequences with:
- **Per-frame points**: ~300,000+ points
- **Depth scale**: 50× for enhanced visualization
- **Output format**: PLY (with RGB color data preserved)

### Sample Output
The system overlays:
1. **Green saliency heatmap** blended with original frame
2. **Green dots** at saliency peaks with distance labels (e.g., "2.45m")
3. **Motion minimap** (top-right) showing camera movement direction
4. **Real-time FPS counter** and stabilization status

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ALVINN-Public.git
cd ALVINN-Public

# Install dependencies
pip install -r requirements.txt

# Download model weights (place in respective directories)
# - Saliency_Predictions/weights/SALICON_A.pth
# - Metric_Depth_Estimation/checkpoints/depth_indoor.pth
```

---

## Usage

### Real-Time Webcam Processing
```bash
cd Main
python combRunDOT.py
```
**Controls:**
- `+/-` — Adjust saliency threshold
- `s` — Toggle depth stabilization
- `q` — Quit

### Video File Processing
```bash
python Main/combDOTMov.py --input Media/Input_Videos/input_video.mp4 \
                          --output Media/Output_Videos/output.mp4 \
                          --threshold 0.5
```

### 3D Point Cloud Playback
```bash
python Main/PMLoader.py --path point_cloud_data/20250316_000219
```
**Controls:**
- `Space` — Play/pause
- `,.` — Frame step
- `[]` — Jump 10 frames
- `po` — Adjust point size
- `r` — Reset view

---

## Project Structure

```
ALVINN-Public/
├── Main/                          # Combined inference scripts
│   ├── combRun.py                 # Basic webcam demo
│   ├── combRunDOT.py              # Webcam with DOT overlay
│   ├── combDOTMov.py              # Video processing
│   └── PMLoader.py                # Point cloud video player
├── Saliency_Predictions/          # Saliency detection module
│   ├── model/                     # FastSal architecture
│   ├── weights/                   # Pretrained weights
│   └── dataset/                   # Dataset loaders
├── Metric_Depth_Estimation/       # Depth estimation module
│   ├── metric_depth/
│   │   └── alvinn_depth/          # ALVINNDepth model
│   │       ├── dpt.py             # DPT head implementation
│   │       ├── dinov2.py          # DINOv2 backbone
│   │       └── dinov2_layers/     # Transformer components
│   └── checkpoints/               # Pretrained depth weights
├── Media/                         # Input/output videos
└── point_cloud_data/              # Generated 3D point clouds
```

---

## Future Work

- [ ] Audio feedback system for obstacle alerts
- [ ] Edge device optimization (TensorRT/ONNX)
- [ ] Object classification for salient regions
- [ ] Haptic feedback integration
- [ ] Multi-camera SLAM integration

---

## Acknowledgments

- DINOv2 architecture from Meta AI Research
- FastSal saliency model adapted from knowledge distillation research
- DPT head design inspired by Intel's MiDaS depth estimation
