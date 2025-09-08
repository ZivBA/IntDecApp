#!/usr/bin/env python3
"""
Single-image depth estimation using MiDaS for dense reconstruction
"""

import numpy as np
import cv2
import torch
from pathlib import Path


def setup_midas_model():
    """Setup MiDaS model for depth estimation"""

    print("🧠 SETTING UP MIDAS DEPTH ESTIMATION")
    print("=" * 35)

    try:
        # Try to load MiDaS model (will download on first use)
        print("📥 Loading MiDaS model...")
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        model.eval()

        # Setup transforms
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print("✅ MiDaS loaded successfully")
        print(f"🔧 Using device: {device}")

        return model, transform, device

    except Exception as e:
        print(f"❌ MiDaS setup failed: {e}")
        print("💡 Trying alternative: downloading manually...")
        return setup_midas_fallback()


def setup_midas_fallback():
    """Fallback MiDaS setup with explicit model download"""

    try:
        # Use smaller, faster MiDaS model for CPU
        print("📥 Loading MiDaS_small (CPU-optimized)...")
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        model.eval()

        transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        device = torch.device("cpu")  # Force CPU for compatibility
        model.to(device)

        print("✅ MiDaS_small loaded successfully (CPU mode)")
        return model, transform, device

    except Exception as e:
        print(f"❌ MiDaS fallback failed: {e}")
        return None, None, None


def estimate_depth_single_image(image_path, model, transform, device):
    """Estimate depth for a single image using MiDaS"""

    # Load and preprocess image
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply MiDaS transform
    input_batch = transform(img_rgb).to(device)

    # Predict depth
    with torch.no_grad():
        prediction = model(input_batch)

        # Convert to numpy
        depth_map = prediction.squeeze().cpu().numpy()

        # MiDaS outputs inverse depth, convert to actual depth
        # Normalize and invert
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = 1.0 / (depth_map + 1e-6)  # Invert to get depth

    return img_rgb, depth_map


def create_dense_point_cloud_from_depth(
    image, depth_map, camera_matrix, max_points=10000
):
    """Create dense point cloud from image and depth map"""

    h, w = depth_map.shape
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # Create coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Convert to 3D points
    z = depth_map
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack coordinates
    points_3d = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Get corresponding colors
    colors = image.reshape(-1, 3) / 255.0  # Normalize to [0,1]

    # Filter out invalid depths
    valid_mask = (z.flatten() > 0) & (z.flatten() < np.inf) & (z.flatten() < 100)
    points_3d = points_3d[valid_mask]
    colors = colors[valid_mask]

    # Subsample if too many points
    if len(points_3d) > max_points:
        indices = np.random.choice(len(points_3d), max_points, replace=False)
        points_3d = points_3d[indices]
        colors = colors[indices]
        print(f"   📉 Subsampled to {max_points:,} points")
    else:
        print(f"   📊 Using all {len(points_3d):,} valid points")

    return points_3d, colors


def process_all_images_depth(image_dir="test_data2", max_points_per_image=5000):
    """Process all images to create dense point clouds using depth estimation"""

    print("🎯 PROCESSING ALL IMAGES WITH DEPTH ESTIMATION")
    print("=" * 50)

    # Setup MiDaS
    model, transform, device = setup_midas_model()
    if model is None:
        print("❌ Cannot proceed without depth estimation model")
        return None

    # Load images
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG"))

    print(f"📸 Processing {len(image_files)} images...")

    # Camera matrix (estimated from COLMAP results)
    # Using focal length ~1279 from COLMAP, scaled for 800x1066 images
    K = np.array(
        [
            [1279.2, 0, 400],  # fx, 0, cx
            [0, 1279.2, 533],  # 0, fy, cy
            [0, 0, 1],  # 0, 0, 1
        ]
    )

    all_dense_points = []
    all_dense_colors = []
    image_point_counts = {}

    for i, img_path in enumerate(sorted(image_files)):
        print(f"\n🔍 Processing {img_path.name} ({i + 1}/{len(image_files)})...")

        try:
            # Estimate depth
            img_rgb, depth_map = estimate_depth_single_image(
                img_path, model, transform, device
            )

            if depth_map is None:
                print("   ❌ Depth estimation failed")
                continue

            # Create point cloud from depth
            points_3d, colors = create_dense_point_cloud_from_depth(
                img_rgb, depth_map, K, max_points=max_points_per_image
            )

            if len(points_3d) > 0:
                all_dense_points.append(points_3d)
                all_dense_colors.append(colors)
                image_point_counts[img_path.name] = len(points_3d)
                print(f"   ✅ Generated {len(points_3d):,} 3D points")

                # Save depth map visualization
                depth_vis = (depth_map - depth_map.min()) / (
                    depth_map.max() - depth_map.min()
                )
                depth_vis = (depth_vis * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
                cv2.imwrite(f"depth_{img_path.stem}.jpg", depth_colored)

            else:
                print("   ⚠️  No valid 3D points generated")

        except Exception as e:
            print(f"   ❌ Processing failed: {e}")

    if not all_dense_points:
        print("❌ No point clouds generated")
        return None

    # Combine all point clouds
    combined_points = np.vstack(all_dense_points)
    combined_colors = np.vstack(all_dense_colors)

    print("\n📊 DENSE POINT CLOUD SUMMARY:")
    print(f"   Images processed: {len(all_dense_points)}")
    print(f"   Total points: {len(combined_points):,}")
    print(f"   Average per image: {len(combined_points) // len(all_dense_points):,}")

    for img_name, count in image_point_counts.items():
        print(f"   {img_name}: {count:,} points")

    # Save combined point cloud
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)

    dense_path = "dense_points_midas.ply"
    o3d.io.write_point_cloud(dense_path, pcd)
    print(f"💾 Saved combined dense cloud: {dense_path}")

    return {
        "points": combined_points,
        "colors": combined_colors,
        "total_points": len(combined_points),
        "images_processed": len(all_dense_points),
        "ply_path": dense_path,
        "point_counts": image_point_counts,
    }


def test_single_image_depth():
    """Test single-image depth estimation pipeline"""

    print("🧪 TESTING SINGLE-IMAGE DEPTH ESTIMATION")
    print("=" * 40)

    # Process all images with depth estimation
    result = process_all_images_depth(max_points_per_image=5000)

    if result:
        print("\n🎉 SINGLE-IMAGE DEPTH SUCCESS!")
        print(f"✅ Generated {result['total_points']:,} dense points")
        print(
            f"📈 vs 515 sparse COLMAP points = {result['total_points'] / 515:.0f}x improvement!"
        )

        if result["total_points"] >= 20000:
            print("🚀 Excellent density for view synthesis!")
        elif result["total_points"] >= 5000:
            print("✅ Good density - should enable better view synthesis")
        else:
            print("⚠️  Moderate density - may need more images or higher point count")

        return True, result
    else:
        print("❌ Single-image depth estimation failed")
        return False, None


if __name__ == "__main__":
    success, result = test_single_image_depth()

    if success:
        print("\n🎯 NEXT STEPS:")
        print("1. Apply coordinate transformation to MiDaS points")
        print(f"2. Test enhanced view synthesis with {result['total_points']:,} points")
        print(
            f"3. Compare quality vs 515 sparse points (should be {result['total_points'] / 515:.0f}x better)"
        )
        print("4. Validate identifiable content in synthesized views")
    else:
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check PyTorch installation")
        print("2. Ensure internet connection for model download")
        print("3. Try different MiDaS model variants")
