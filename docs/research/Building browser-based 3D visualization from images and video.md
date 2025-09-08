# Building browser-based 3D visualization from images and video

Creating a fully browser-based interactive 3D scene from uploaded media requires combining several cutting-edge technologies. Based on comprehensive research of current JavaScript frameworks, GPU acceleration techniques, and emerging technologies, here's a detailed technical implementation guide for building this type of application in 2025.

## Core technical stack and architecture decisions

The optimal technical stack combines **Three.js as the primary 3D framework** with **TensorFlow.js for depth estimation**, utilizing **WebGPU where available** (falling back to WebGL 2.0), and implementing **Gaussian Splatting for advanced visualization**. This combination provides the best balance of browser compatibility, performance, and cutting-edge capabilities.

For depth estimation from single images, **MiDaS models ported to TensorFlow.js** offer the most mature solution, running at **22 FPS on iPhone 13 Pro and 51 FPS on MacBook M1 Pro**. The quantized Depth Anything V2 models through ONNX.js provide an alternative with smaller model sizes (18MB for 4-bit quantization versus 97MB full model), crucial for mobile deployment.

## Implementation strategy for image-to-3D conversion

### Phase 1: File upload and preprocessing

```javascript
class MediaProcessor {
  constructor() {
    this.supportedFormats = ['image/jpeg', 'image/png', 'video/mp4'];
    this.maxFileSize = 50 * 1024 * 1024; // 50MB limit for mobile
  }
  
  async processUpload(files) {
    const processed = [];
    for (const file of files) {
      if (this.validateFile(file)) {
        const data = await this.loadFile(file);
        const optimized = await this.optimizeForDevice(data);
        processed.push(optimized);
      }
    }
    return processed;
  }
  
  async optimizeForDevice(data) {
    const isMobile = /Mobile|Android|iPhone/i.test(navigator.userAgent);
    const maxDimension = isMobile ? 1024 : 2048;
    // Resize if needed, compress textures
    return this.resizeImage(data, maxDimension);
  }
}
```

### Phase 2: Depth estimation pipeline

The depth estimation pipeline should adaptively select models based on device capabilities. For mobile devices, use the **ARPortraitDepth model** from TensorFlow.js for human subjects or the quantized Depth Anything V2 for general scenes. Desktop systems can leverage the full MiDaS implementation or larger Depth Anything models.

```javascript
class DepthEstimator {
  async initialize() {
    const isMobile = this.detectMobile();
    
    if (isMobile) {
      // Use lightweight model for mobile
      this.model = await depthEstimation.createEstimator(
        depthEstimation.SupportedModels.ARPortraitDepth
      );
    } else {
      // Use ONNX model with WebGPU acceleration
      this.session = await ort.InferenceSession.create(
        'models/depth-anything-v2-large.onnx',
        { executionProviders: ['webgpu', 'webgl', 'wasm'] }
      );
    }
  }
  
  async estimateDepth(image) {
    if (this.model) {
      // TensorFlow.js path
      return await this.model.estimateDepth(image, {
        minDepth: 0,
        maxDepth: 1
      });
    } else {
      // ONNX path with preprocessing
      const tensor = await this.preprocessImage(image);
      const results = await this.session.run({ input: tensor });
      return this.postprocessDepth(results);
    }
  }
}
```

### Phase 3: 3D scene generation with Three.js

The 3D scene generation uses displacement mapping with the depth data to create interactive geometry. This approach provides real-time performance while maintaining quality.

```javascript
class Scene3DGenerator {
  constructor() {
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.setupRenderer();
  }
  
  setupRenderer() {
    // Prefer WebGPU when available
    if (navigator.gpu) {
      this.renderer = new THREE.WebGPURenderer({ antialias: true });
    } else {
      this.renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        powerPreference: "high-performance"
      });
    }
  }
  
  createDepthMesh(imageTexture, depthTexture) {
    const geometry = new THREE.PlaneGeometry(5, 5, 256, 256);
    
    const material = new THREE.ShaderMaterial({
      uniforms: {
        imageTexture: { value: imageTexture },
        depthTexture: { value: depthTexture },
        displacementScale: { value: 0.5 },
        mousePosition: { value: new THREE.Vector2() }
      },
      vertexShader: `
        uniform sampler2D depthTexture;
        uniform float displacementScale;
        varying vec2 vUv;
        
        void main() {
          vUv = uv;
          vec4 depth = texture2D(depthTexture, uv);
          vec3 newPosition = position;
          newPosition.z += depth.r * displacementScale;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D imageTexture;
        varying vec2 vUv;
        
        void main() {
          gl_FragColor = texture2D(imageTexture, vUv);
        }
      `
    });
    
    return new THREE.Mesh(geometry, material);
  }
}
```

## Advanced visualization with Gaussian Splatting

For cutting-edge visualization, implement **Gaussian Splatting** which provides photorealistic 3D reconstruction. The **antimatter15/splat** implementation offers the most mature WebGL solution, handling approximately **1 million splats efficiently** with progressive loading support.

```javascript
class GaussianSplattingViewer {
  async loadSplatScene(url) {
    // Use antimatter15's approach for WebGL 1.0 compatibility
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    
    // Parse PLY format or custom .splat format
    const splats = this.parseSplatData(buffer);
    
    // Sort by size/opacity for progressive loading
    splats.sort((a, b) => b.opacity * b.size - a.opacity * a.size);
    
    // Create vertex buffer with positions and attributes
    this.createSplatGeometry(splats);
    
    // Implement CPU-based sorting in WebWorker
    this.sortWorker = new Worker('sort-worker.js');
    this.sortWorker.postMessage({ splats, camera: this.camera.position });
  }
  
  render() {
    // Render at 60fps while sorting happens at ~4fps in worker
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(() => this.render());
  }
}
```

## GPU acceleration strategies

Maximizing GPU utilization requires careful optimization across different platforms. **WebGPU provides 3-7% JavaScript performance improvement** over WebGL and enables compute shaders for 10x performance gains in parallel computations.

### Mobile GPU optimization

Mobile devices require aggressive optimization due to memory constraints and thermal throttling. Key strategies include:

```javascript
class MobileOptimizer {
  optimizeForMobile(scene) {
    const isMobile = this.detectMobile();
    
    if (isMobile) {
      // Reduce texture resolution
      scene.traverse((child) => {
        if (child.material && child.material.map) {
          child.material.map.minFilter = THREE.LinearFilter;
          // Skip mipmaps to save memory
        }
      });
      
      // Use compressed textures
      const loader = new THREE.KTX2Loader();
      loader.setTranscoderPath('basis/');
      loader.detectSupport(this.renderer);
      
      // Implement LOD system
      const lod = new THREE.LOD();
      lod.addLevel(highDetailMesh, 0);
      lod.addLevel(mediumDetailMesh, 50);
      lod.addLevel(lowDetailMesh, 100);
      
      // Use lower precision in shaders
      material.precision = 'mediump';
    }
  }
}
```

### WebGPU compute shaders for depth processing

When WebGPU is available, utilize compute shaders for efficient depth map processing:

```wgsl
@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputBuffer: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let coords = vec2<i32>(id.xy);
  let depth = textureLoad(inputTexture, coords, 0).r;
  
  // Apply bilateral filter for smooth depth
  var filtered = 0.0;
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let sample = textureLoad(inputTexture, coords + vec2<i32>(dx, dy), 0).r;
      let weight = exp(-0.5 * f32(dx*dx + dy*dy) / 4.0);
      filtered += sample * weight;
    }
  }
  
  outputBuffer[id.y * textureDimensions.x + id.x] = filtered;
}
```

## Video processing implementation

For video input, implement frame extraction and temporal coherence for smooth 3D visualization:

```javascript
class VideoTo3D {
  constructor() {
    this.video = document.createElement('video');
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.depthCache = new Map();
  }
  
  async processVideo(videoFile) {
    this.video.src = URL.createObjectURL(videoFile);
    this.video.play();
    
    // Extract frames at regular intervals
    const frameInterval = 1000 / 30; // 30 FPS
    
    const processFrame = async () => {
      if (!this.video.paused && !this.video.ended) {
        // Draw current frame to canvas
        this.ctx.drawImage(this.video, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        
        // Estimate depth with temporal smoothing
        const depth = await this.estimateDepthWithTemporal(imageData);
        
        // Update 3D scene
        this.update3DScene(imageData, depth);
        
        setTimeout(processFrame, frameInterval);
      }
    };
    
    processFrame();
  }
  
  async estimateDepthWithTemporal(frame) {
    const depth = await this.depthEstimator.estimate(frame);
    
    // Apply temporal smoothing
    if (this.previousDepth) {
      // Blend with previous frame for stability
      return this.blendDepthMaps(depth, this.previousDepth, 0.7);
    }
    
    this.previousDepth = depth;
    return depth;
  }
}
```

## Performance optimization techniques

Achieving smooth performance across devices requires adaptive quality settings and efficient resource management:

```javascript
class PerformanceManager {
  constructor(renderer) {
    this.renderer = renderer;
    this.targetFPS = 60;
    this.frameHistory = [];
    this.qualityLevels = ['low', 'medium', 'high', 'ultra'];
    this.currentQuality = 'high';
  }
  
  adaptiveQuality() {
    const avgFrameTime = this.calculateAverageFrameTime();
    
    if (avgFrameTime > 16.67) { // Below 60 FPS
      this.reduceQuality();
    } else if (avgFrameTime < 10 && this.currentQuality !== 'ultra') {
      this.increaseQuality();
    }
  }
  
  applyQualitySettings(quality) {
    switch(quality) {
      case 'low':
        this.renderer.setPixelRatio(1);
        this.renderer.shadowMap.enabled = false;
        this.maxTextureSize = 512;
        break;
      case 'medium':
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
        this.renderer.shadowMap.type = THREE.BasicShadowMap;
        this.maxTextureSize = 1024;
        break;
      case 'high':
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.maxTextureSize = 2048;
        break;
      case 'ultra':
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.type = THREE.VSMShadowMap;
        this.maxTextureSize = 4096;
        break;
    }
  }
}
```

## Complete application architecture

The final application combines all components into a cohesive system with progressive enhancement:

```javascript
class Interactive3DVisualizationApp {
  async initialize() {
    // Feature detection and setup
    this.features = {
      webgpu: await this.checkWebGPUSupport(),
      webgl2: this.checkWebGL2Support(),
      webworkers: typeof Worker !== 'undefined'
    };
    
    // Initialize components based on capabilities
    this.mediaProcessor = new MediaProcessor();
    this.depthEstimator = new DepthEstimator();
    this.sceneGenerator = new Scene3DGenerator();
    this.performanceManager = new PerformanceManager(this.sceneGenerator.renderer);
    
    // Setup progressive loading
    if (this.features.webworkers) {
      this.setupBackgroundProcessing();
    }
    
    // Initialize UI
    this.setupUserInterface();
    
    // Start render loop
    this.animate();
  }
  
  async handleFileUpload(files) {
    // Show loading indicator
    this.showLoader();
    
    try {
      // Process uploaded media
      const processed = await this.mediaProcessor.processUpload(files);
      
      // Generate depth maps
      const depths = await Promise.all(
        processed.map(img => this.depthEstimator.estimateDepth(img))
      );
      
      // Create 3D scene
      for (let i = 0; i < processed.length; i++) {
        const mesh = this.sceneGenerator.createDepthMesh(
          processed[i], 
          depths[i]
        );
        this.sceneGenerator.scene.add(mesh);
      }
      
      // Optimize for current device
      if (this.isMobile()) {
        new MobileOptimizer().optimizeForMobile(this.sceneGenerator.scene);
      }
      
    } catch (error) {
      console.error('Processing failed:', error);
      this.showError('Failed to process media');
    } finally {
      this.hideLoader();
    }
  }
  
  animate() {
    requestAnimationFrame(() => this.animate());
    
    // Adaptive quality based on performance
    this.performanceManager.adaptiveQuality();
    
    // Update scene
    this.sceneGenerator.renderer.render(
      this.sceneGenerator.scene, 
      this.sceneGenerator.camera
    );
  }
}

// Initialize application
const app = new Interactive3DVisualizationApp();
app.initialize();
```

## Deployment and optimization considerations

For production deployment, implement **progressive web app features** with service workers for offline caching of models and textures. Use **CDN delivery with Brotli compression** for assets, reducing initial load times by up to 30%. The application should target initial bundle sizes under **3MB for mobile** and **10MB for desktop**, with lazy loading for additional features.

Memory management proves critical on mobile devices, particularly iOS Safari with its aggressive tab reloading. Implement a **texture pooling system** that maintains a maximum of **100MB GPU memory usage** on mobile devices, with automatic quality reduction when approaching limits.

The combination of these technologies and optimization strategies enables creation of sophisticated browser-based 3D visualization tools that work across all modern devices, from high-end desktops achieving **60+ FPS with million-point Gaussian splats** to mobile devices maintaining smooth **30 FPS with optimized mesh representations**.