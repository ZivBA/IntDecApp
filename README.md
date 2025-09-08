# 3D Scene Reconstruction Application

Transform your photos and videos into immersive, navigable 3D environments.

## 🎯 Project Vision

This application processes user-provided images and videos to create interactive 3D scenes that can be explored as if you were physically present in the original location. Whether it's a panoramic video, multiple photos of a room, or a walkthrough recording, our system reconstructs the space in 3D for free navigation.

## 🚀 Key Features

- **Multi-source Input**: Process single images, image sequences, panoramic videos, or standard videos
- **3D Scene Generation**: Create navigable 3D meshes and point clouds from 2D inputs
- **Cross-platform Deployment**: Run as a web application or standalone Android app
- **Real-time Navigation**: Walk through, rotate, and zoom within reconstructed scenes
- **View Interpolation**: Generate novel viewpoints not present in original captures

## 📋 Technical Overview

### Core Processing Pipeline

```
Input (Images/Video) → Feature Detection → Camera Pose Estimation → 
Dense Reconstruction → Mesh Generation → Texture Mapping → 3D Scene
```

### Technology Stack

#### Backend Processing (Python)
- **OpenCV**: Image/video processing and feature detection
- **COLMAP** or **OpenMVG**: Structure from Motion (SfM) pipeline
- **Open3D**: Point cloud processing and mesh generation
- **NumPy/SciPy**: Mathematical computations
- **Pillow**: Image manipulation
- **FFmpeg-python**: Video frame extraction

#### Modern ML Approaches (Optional)
- **NeRF** (Neural Radiance Fields): Novel view synthesis
- **Gaussian Splatting**: Real-time rendering of 3D scenes
- **MiDaS/DPT**: Monocular depth estimation

#### Web Application
- **FastAPI** or **Flask**: REST API server
- **Three.js**: WebGL-based 3D visualization
- **WebAssembly**: High-performance client-side processing

#### Android Deployment
- **Kivy** or **BeeWare**: Python-to-APK packaging
- **Buildozer**: Android build automation

#### Infrastructure
- **Docker**: Containerized deployment
- **Redis**: Task queue for async processing
- **MinIO**: Object storage for 3D assets

## 🏗️ Project Structure

```
3d-scene-reconstruction/
│
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── feature_extraction.py    # SIFT/SURF/ORB detection
│   │   ├── camera_calibration.py    # Camera pose estimation
│   │   ├── reconstruction.py        # SfM/MVS pipeline
│   │   ├── mesh_generation.py       # Point cloud to mesh
│   │   └── texture_mapping.py       # UV mapping and texturing
│   │
│   ├── preprocessors/
│   │   ├── __init__.py
│   │   ├── video_processor.py       # Frame extraction, stabilization
│   │   ├── image_processor.py       # Enhancement, normalization
│   │   └── panorama_processor.py    # Spherical projection handling
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── depth_estimation.py      # Monocular depth networks
│   │   ├── nerf_model.py           # NeRF implementation (optional)
│   │   └── gaussian_splatting.py    # 3DGS implementation (optional)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py                # FastAPI application
│   │   ├── routes.py                # API endpoints
│   │   └── websocket.py             # Real-time progress updates
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── scene_exporter.py        # GLTF/OBJ export
│       └── renderer.py              # Preview generation
│
├── web/
│   ├── index.html
│   ├── app.js                       # Three.js viewer
│   └── styles.css
│
├── android/
│   ├── buildozer.spec              # Android build config
│   └── main.py                     # Kivy application entry
│
├── tests/
│   ├── unit/
│   │   ├── test_feature_extraction.py
│   │   ├── test_reconstruction.py
│   │   └── test_mesh_generation.py
│   │
│   ├── integration/
│   │   └── test_pipeline.py
│   │
│   └── fixtures/
│       ├── sample_images/
│       └── sample_videos/
│
├── docker/
│   ├── Dockerfile.processor        # Processing service
│   ├── Dockerfile.web             # Web server
│   └── docker-compose.yml
│
├── scripts/
│   ├── setup_colmap.sh            # Install COLMAP binaries
│   ├── process_batch.py           # Batch processing utility
│   └── benchmark.py               # Performance testing
│
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── CLAUDE.md
└── README.md
```

## 🔧 Installation

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/3d-scene-reconstruction.git
cd 3d-scene-reconstruction

# Build containers
docker-compose build

# Start services
docker-compose up
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install COLMAP (Ubuntu/Debian)
sudo apt-get install colmap

# Install COLMAP (macOS)
brew install colmap

# Run setup script
bash scripts/setup_colmap.sh
```

## 🎮 Usage Examples

### Basic Pipeline

```python
from src.core import SceneReconstructor
from src.preprocessors import VideoProcessor

# Process a panoramic video
video_processor = VideoProcessor()
frames = video_processor.extract_frames("panorama.mp4", fps=2)

# Reconstruct 3D scene
reconstructor = SceneReconstructor()
scene = reconstructor.process(frames, mode="panoramic")

# Export for web viewing
scene.export("output/scene.gltf")
```

### Web API

```bash
# Upload and process images
curl -X POST http://localhost:8000/api/reconstruct \
  -F "files=@room_photo1.jpg" \
  -F "files=@room_photo2.jpg" \
  -F "files=@room_photo3.jpg"

# Check processing status
curl http://localhost:8000/api/status/{job_id}

# Download 3D scene
curl http://localhost:8000/api/download/{job_id} -o scene.gltf
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit -v

# Run integration tests
pytest tests/integration -v

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/unit/test_reconstruction.py::test_camera_pose_estimation
```

## 📊 Processing Pipeline Details

### 1. Feature Detection & Matching
- Extract keypoints using SIFT/SURF/ORB
- Match features across multiple views
- Filter matches using RANSAC

### 2. Structure from Motion (SfM)
- Estimate camera poses
- Triangulate 3D points
- Bundle adjustment optimization

### 3. Dense Reconstruction
- Multi-View Stereo (MVS) for dense point clouds
- Depth map fusion
- Point cloud filtering and cleaning

### 4. Surface Reconstruction
- Poisson surface reconstruction
- Mesh simplification
- Texture atlas generation

### 5. Rendering & Export
- GLTF/GLB export for web
- OBJ/MTL for compatibility
- Point cloud formats (PLY, PCD)

## 🎯 Roadmap

### Phase 1: Core Pipeline (Current)
- [x] Project structure setup
- [ ] Basic SfM implementation
- [ ] Point cloud generation
- [ ] Simple mesh reconstruction

### Phase 2: Enhancement
- [ ] Improved texture mapping
- [ ] Panorama-specific optimizations
- [ ] Real-time preview during processing
- [ ] GPU acceleration (CUDA/OpenCL)

### Phase 3: Advanced Features
- [ ] NeRF integration for view synthesis
- [ ] Gaussian Splatting for real-time rendering
- [ ] AI-powered scene completion
- [ ] Semantic segmentation of scenes

### Phase 4: Deployment
- [ ] Web application optimization
- [ ] Android APK packaging
- [ ] Cloud deployment (GCP/Kubernetes)
- [ ] Processing queue with RabbitMQ

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References & Resources

- [COLMAP Documentation](https://colmap.github.io/)
- [Open3D Documentation](http://www.open3d.org/docs/)
- [NeRF: Neural Radiance Fields](https://www.matthewtancik.com/nerf)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Structure from Motion Tutorial](https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html)

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- COLMAP team for the excellent SfM framework
- Open3D community for 3D processing tools
- Three.js for WebGL visualization
