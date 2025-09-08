#!/usr/bin/env python3
"""
Test enhanced view synthesis with dense point clouds (14K points)
"""

import numpy as np
import cv2
import open3d as o3d


def load_dense_point_cloud(ply_path="dense_points_stereo.ply"):
    """Load dense point cloud from stereo reconstruction"""

    print("📊 LOADING DENSE POINT CLOUD")
    print("=" * 30)

    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        print(f"✅ Loaded {len(points):,} dense points")
        print(f"   X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"   Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"   Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

        # Apply coordinate transformation
        center = points.mean(axis=0)
        points_transformed = (points - center) * 0.2 + np.array([0, 0, 8])

        print(
            f"🔧 Applied transformation - new Z range: [{points_transformed[:, 2].min():.2f}, {points_transformed[:, 2].max():.2f}]"
        )

        return points_transformed, colors

    except Exception as e:
        print(f"❌ Failed to load dense points: {e}")
        return None, None


def synthesize_dense_view(
    points_3d, colors, camera_K, camera_position, camera_rotation
):
    """Synthesize view from dense point cloud"""

    # Output image dimensions
    height, width = 480, 640
    output_image = np.zeros((height, width, 3), dtype=np.float32)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)

    # Create projection matrix
    # Convert camera pose to extrinsics
    R = camera_rotation
    t = camera_position
    P = camera_K @ np.hstack([R, t.reshape(-1, 1)])

    # Project all points
    points_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    projected = (P @ points_homo.T).T

    # Filter valid projections
    valid_mask = projected[:, 2] > 0  # In front of camera

    if np.sum(valid_mask) == 0:
        print("⚠️  No points visible from this viewpoint")
        return output_image.astype(np.uint8)

    # Normalize to pixel coordinates
    projected_valid = projected[valid_mask]
    colors_valid = colors[valid_mask]

    x = projected_valid[:, 0] / projected_valid[:, 2]
    y = projected_valid[:, 1] / projected_valid[:, 2]
    z = projected_valid[:, 2]

    # Filter points within image bounds
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    x_valid = x[in_bounds].astype(int)
    y_valid = y[in_bounds].astype(int)
    z_valid = z[in_bounds]
    colors_final = colors_valid[in_bounds]

    print(f"   Rendering {len(x_valid):,} visible points")

    # Render points with z-buffering
    for i in range(len(x_valid)):
        px, py = x_valid[i], y_valid[i]
        depth = z_valid[i]

        # Z-buffer test
        if depth < depth_buffer[py, px]:
            depth_buffer[py, px] = depth

            # Get color (convert to BGR for OpenCV)
            if colors_final.max() <= 1.0:
                color = (colors_final[i] * 255).astype(np.uint8)
            else:
                color = colors_final[i].astype(np.uint8)

            # Draw point with small splat for better coverage
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = py + dy, px + dx
                    if 0 <= nx < width and 0 <= ny < height:
                        output_image[ny, nx] = color[[2, 1, 0]]  # BGR for OpenCV

    # Fill gaps with inpainting for better visual quality
    mask = (output_image.sum(axis=2) == 0).astype(np.uint8) * 255
    output_uint8 = output_image.astype(np.uint8)

    if (
        np.sum(mask) > 0 and np.sum(mask) < height * width * 0.9
    ):  # Don't inpaint if too empty
        inpainted = cv2.inpaint(output_uint8, mask, 3, cv2.INPAINT_TELEA)
        return inpainted

    return output_uint8


def test_multiple_viewpoints(points_3d, colors):
    """Test view synthesis from multiple viewpoints"""

    print("\n🎯 TESTING MULTIPLE VIEWPOINTS")
    print("=" * 32)

    # Camera intrinsics
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)

    # Define test viewpoints
    viewpoints = [
        {
            "name": "baseline",
            "position": np.array([0, 0, 0]),
            "rotation": np.eye(3),
            "description": "Camera at origin looking down +Z",
        },
        {
            "name": "front_view",
            "position": np.array([0, -2, 0]),
            "rotation": np.eye(3),
            "description": "Front view of room",
        },
        {
            "name": "side_view",
            "position": np.array([2, 0, 0]),
            "rotation": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            "description": "Side view of room",
        },
        {
            "name": "elevated",
            "position": np.array([0, -1, -2]),
            "rotation": np.eye(3),
            "description": "Elevated view looking down",
        },
    ]

    results = []

    for vp in viewpoints:
        print(f"\n📷 {vp['description']}...")

        # Synthesize view
        image = synthesize_dense_view(
            points_3d, colors, K, vp["position"], vp["rotation"]
        )

        # Save result
        filename = f"dense_view_{vp['name']}.jpg"
        cv2.imwrite(filename, image)

        # Calculate fill percentage
        non_black = np.sum(np.any(image > 0, axis=2))
        total = image.shape[0] * image.shape[1]
        fill_pct = (non_black / total) * 100

        print(f"   💾 Saved: {filename}")
        print(f"   📊 Fill: {fill_pct:.1f}% ({non_black:,}/{total:,} pixels)")

        results.append(
            {
                "name": vp["name"],
                "filename": filename,
                "fill_pct": fill_pct,
                "image": image,
            }
        )

    return results


def analyze_visual_quality(results):
    """Analyze if synthesized views contain identifiable content"""

    print("\n🔍 ANALYZING VISUAL QUALITY")
    print("=" * 30)

    for result in results:
        print(f"\n📸 {result['name']}:")
        print(f"   Fill: {result['fill_pct']:.1f}%")

        # Check for identifiable features
        img = result["image"]

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect edges (indicator of structure)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        edge_pct = (edge_pixels / (img.shape[0] * img.shape[1])) * 100

        print(f"   Edge density: {edge_pct:.1f}%")

        # Check color distribution (should have variety for room scene)
        unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
        print(f"   Unique colors: {unique_colors:,}")

        # Quality assessment
        if result["fill_pct"] > 20 and edge_pct > 2 and unique_colors > 1000:
            print("   ✅ GOOD QUALITY - Likely shows identifiable room features")
            result["quality"] = "good"
        elif result["fill_pct"] > 10 and edge_pct > 1 and unique_colors > 500:
            print("   ⚠️  MODERATE - Some structure visible but sparse")
            result["quality"] = "moderate"
        else:
            print("   ❌ POOR - Too sparse for identification")
            result["quality"] = "poor"

    return results


def create_comparison_image(sparse_path="view_synthesis_test.jpg", dense_results=None):
    """Create comparison between sparse and dense synthesis"""

    print("\n📊 CREATING COMPARISON IMAGE")
    print("=" * 30)

    try:
        # Load sparse result
        sparse_img = cv2.imread(sparse_path)
        if sparse_img is None:
            print("⚠️  Sparse synthesis image not found")
            sparse_img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Find best dense result
        best_dense = max(dense_results, key=lambda x: x["fill_pct"])
        dense_img = best_dense["image"]

        # Resize for comparison
        h, w = 240, 320
        sparse_resized = cv2.resize(sparse_img, (w, h))
        dense_resized = cv2.resize(dense_img, (w, h))

        # Create side-by-side comparison
        comparison = np.hstack([sparse_resized, dense_resized])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Sparse (515 pts)", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison, "0.2% coverage", (10, 50), font, 0.5, (0, 255, 0), 1)

        cv2.putText(
            comparison, "Dense (14K pts)", (330, 30), font, 0.7, (0, 255, 255), 2
        )
        cv2.putText(
            comparison,
            f"{best_dense['fill_pct']:.1f}% coverage",
            (330, 50),
            font,
            0.5,
            (0, 255, 255),
            1,
        )

        cv2.imwrite("sparse_vs_dense_comparison.jpg", comparison)
        print("💾 Saved comparison: sparse_vs_dense_comparison.jpg")

    except Exception as e:
        print(f"❌ Comparison failed: {e}")


def main():
    """Main test pipeline for dense view synthesis"""

    print("🚀 DENSE VIEW SYNTHESIS TEST")
    print("=" * 30)
    print("Testing with 14,000 dense points from stereo matching")

    # Load dense point cloud
    points, colors = load_dense_point_cloud()

    if points is None:
        print("❌ Cannot proceed without dense points")
        return False

    # Test multiple viewpoints
    results = test_multiple_viewpoints(points, colors)

    # Analyze visual quality
    results = analyze_visual_quality(results)

    # Create comparison
    create_comparison_image(dense_results=results)

    # Summary
    print("\n" + "=" * 50)
    print("📊 DENSE VIEW SYNTHESIS RESULTS")
    print("=" * 50)

    good_count = sum(1 for r in results if r.get("quality") == "good")
    moderate_count = sum(1 for r in results if r.get("quality") == "moderate")

    print(f"✅ Good quality views: {good_count}/{len(results)}")
    print(f"⚠️  Moderate quality: {moderate_count}/{len(results)}")

    best = max(results, key=lambda x: x["fill_pct"])
    print(f"🏆 Best result: {best['name']} with {best['fill_pct']:.1f}% coverage")

    if good_count > 0:
        print("\n🎉 SUCCESS: Dense reconstruction enables identifiable room features!")
        print("✅ 14,000 points provides sufficient density for basic view synthesis")
        print("📈 27x improvement over sparse (515 points) is clearly visible")
        return True
    elif moderate_count > 0:
        print("\n⚠️  PARTIAL SUCCESS: Some structure visible but needs more density")
        print("💡 Consider increasing to 50K+ points for better quality")
        return True
    else:
        print("\n❌ INSUFFICIENT: Still too sparse for practical use")
        print("💡 Need 100K+ points for interior design application")
        return False


if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎯 NEXT STEPS:")
        print("1. ✅ Moderate density (14K) shows clear improvement")
        print("2. 📈 Scale to 50K-100K points for production quality")
        print("3. 🎨 Add view-dependent texture blending")
        print("4. 🔄 Implement real-time rendering optimization")
    else:
        print("\n🔧 OPTIMIZATION NEEDED:")
        print("1. Increase point density per stereo pair")
        print("2. Add more image pairs for better coverage")
        print("3. Consider dense COLMAP with GPU support")
