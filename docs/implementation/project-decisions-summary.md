# Project Decisions and Technical Pivot

## Executive Summary

After comprehensive architectural planning, the project has pivoted to a **minimal proof-of-concept approach** based on clarified requirements and technical constraints.

## Key Decisions Made

### 1. Target User and Use Case (CRITICAL PIVOT)
- **Target:** Interior designers using smartphones to model scenes for client presentations
- **Primary workflow:** Capture room → Navigate 3D space → Generate AI-enhanced renderings
- **Business model:** Solve AI's viewpoint consistency problem by providing accurate 3D geometry

### 2. Technical Requirements (REVISED)
- **Platform:** Mobile-first (smartphones with mobile GPUs/NPUs)
- **Performance:** Initial processing up to several minutes on-device, 2-5 FPS navigation
- **Input:** Multiple room images from various angles, optional video for large spaces
- **Output:** Navigable 3D space + view synthesis for AI enhancement pipeline
- **Fallback:** Cloud processing when mobile processing fails

### 3. Architecture Philosophy (SIMPLIFIED)
- **Previous approach:** Complex multi-backend GPU acceleration with full COLMAP integration
- **Current approach:** Minimal viable pipeline with incremental complexity
- **Core insight:** Prove depth estimation → point cloud fusion → view synthesis works before adding features

### 4. Implementation Strategy (FOCUSED)

#### Phase 1: Minimal POC (Current Priority)
```python
# Core pipeline test
Room Photos → Depth Maps → Point Cloud → Virtual View → Success/Failure
```

**Success criteria:**
- Input: 5 test room photos
- Output: Recognizable rendered view from different angle  
- Performance: <60 seconds on development machine
- No UI, no mobile app, no advanced features

#### Phase 2: Mobile Integration (Future)
- Android app with camera integration
- Real-time depth estimation
- Basic navigation UI

#### Phase 3: AI Enhancement Pipeline (Future)
- View synthesis optimization
- AI API integration
- Interior design modifications

### 5. Technical Approach Decisions

#### Reconstruction Method
- **Selected:** Multi-view depth estimation + point cloud fusion
- **Rationale:** Balance between accuracy and mobile performance
- **Alternative considered:** Full SfM (too heavy), single-view depth (insufficient accuracy)

#### Key Technology Stack (Minimal)
- **Depth estimation:** MiDaS (lightweight model)
- **Point cloud processing:** Open3D
- **View rendering:** Custom point-based renderer
- **Development:** Python for POC, mobile integration later

### 6. Critical Questions Under Investigation

#### Multi-view vs Single-view Approach
**Question:** Given guaranteed multiple viewpoints, should we use multi-view stereo instead of single-image depth estimation?

**Considerations:**
- **Multi-view stereo:** More accurate, handles textureless regions better, requires camera pose estimation
- **Single-image depth:** Faster, works with any images, less accurate, easier to implement
- **Hybrid approach:** Single-image for initial estimates, multi-view refinement

**Decision pending:** Requires empirical testing with actual room photos

## Rejected Approaches and Why

### 1. Full COLMAP Integration
- **Reason:** Too heavy for mobile deployment
- **Dependencies:** ~500MB, requires significant processing power
- **Alternative:** Lightweight depth estimation with point cloud fusion

### 2. Complex GPU Abstraction Layer
- **Reason:** Premature optimization without proven core functionality
- **Complexity:** Multiple backends (CUDA/OpenCL/WebGPU) before basic pipeline works
- **Alternative:** CPU-first implementation, GPU acceleration later

### 3. Comprehensive Platform Support (Web/Android/iOS/Desktop)
- **Reason:** Feature creep without validated core technology
- **Risk:** Building deployment infrastructure before knowing if reconstruction works
- **Alternative:** Python POC first, mobile after validation

### 4. Advanced Features (Interactive Photo Guide, Coverage Analysis)
- **Reason:** UI features before core functionality is proven
- **Risk:** Weeks of development on features that may not be needed
- **Alternative:** Manual photo taking for POC, automated guidance later

## Risk Mitigation Strategies

### Technical Risks
1. **Depth estimation quality on phone photos**
   - Mitigation: Test with actual device photos early
   - Fallback: Cloud-based processing

2. **Point cloud fusion accuracy**
   - Mitigation: Compare multiple fusion algorithms
   - Fallback: Higher-quality single-image depth maps

3. **Mobile performance constraints**
   - Mitigation: Profile early and often
   - Fallback: Progressive quality settings

### Project Risks
1. **Feature creep**
   - Mitigation: Strict POC scope, feature additions only after validation
   
2. **Over-engineering**
   - Mitigation: Minimal implementation first, refactoring when needed

## Next Steps (Immediate)

### Week 1: Core Pipeline POC
1. Implement basic depth estimation on test room photos
2. Create simple point cloud fusion
3. Build minimal view renderer
4. Test end-to-end pipeline

### Week 2: Validation and Iteration
1. Evaluate reconstruction quality
2. Test with different room types
3. Measure performance bottlenecks
4. Decide on multi-view vs single-view approach

### Decision Points
- **Go/No-go:** Does basic depth estimation produce usable room geometry?
- **Technical choice:** Multi-view stereo vs refined single-image depth?
- **Architecture:** When to add mobile deployment vs cloud processing?

## Key Learnings

1. **Requirements clarity drives architecture:** Understanding the interior design use case completely changed technical priorities
2. **Proof before polish:** Complex architectural planning is worthless without validated core functionality
3. **Mobile constraints:** Smartphone deployment fundamentally changes algorithm selection and optimization priorities
4. **User workflow matters:** The sparse-to-dense reconstruction serves a specific business need (AI viewpoint consistency)

---

*Document updated: 2025-01-07*  
*Next review: After POC completion*