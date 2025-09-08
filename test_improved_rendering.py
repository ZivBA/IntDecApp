#!/usr/bin/env python3
"""
Test rendering with improved point cloud
"""

import numpy as np
import cv2
from test_improved_matching import test_improved_reconstruction
from src.core.renderer import VirtualCamera, PointRenderer
from src.core.types import RenderConfig, Point3D


def filter_outliers(points_3d, max_distance=3.0):
    """Remove outlier points that are too far from the median"""
    positions = np.array([p.position for p in points_3d])

    # Calculate median position
    median_pos = np.median(positions, axis=0)

    # Calculate distances from median
    distances = np.linalg.norm(positions - median_pos, axis=1)

    # Filter outliers
    filtered_points = []
    for i, point in enumerate(points_3d):
        if distances[i] < max_distance:
            filtered_points.append(point)

    print(f"Filtered {len(points_3d) - len(filtered_points)} outliers")
    return filtered_points


def test_improved_rendering():
    """Test rendering with more points"""
    print("🎨 TESTING IMPROVED RENDERING")
    print("=" * 50)

    # Get improved reconstruction
    points_3d, poses = test_improved_reconstruction()

    print(f"\nOriginal points: {len(points_3d)}")

    # Filter outliers
    points_3d = filter_outliers(points_3d, max_distance=5.0)
    print(f"After filtering: {len(points_3d)} points")

    if len(points_3d) < 50:
        print("❌ Too few points after filtering")
        return

    # Analyze filtered points
    positions = np.array([p.position for p in points_3d])
    print(f"\nFiltered spatial extent: {positions.max(axis=0) - positions.min(axis=0)}")

    # Transform to camera-friendly coordinates
    centroid = np.mean(positions, axis=0)
    positions_centered = positions - centroid

    # Scale to reasonable room size (max dimension = 5 meters)
    max_extent = np.max(np.abs(positions_centered))
    if max_extent > 5.0:
        scale_factor = 5.0 / max_extent
        positions_scaled = positions_centered * scale_factor
    else:
        positions_scaled = positions_centered
        scale_factor = 1.0

    # Move to positive Z
    positions_final = positions_scaled + np.array([0, 0, 8])

    # Create transformed points
    transformed_points = []
    for i, point in enumerate(points_3d):
        transformed_points.append(
            Point3D(
                position=positions_final[i],
                color=point.color,
                confidence=point.confidence,
            )
        )

    print(f"Scale factor: {scale_factor}")
    print(
        f"Transformed extent: {positions_final.max(axis=0) - positions_final.min(axis=0)}"
    )

    # Setup rendering
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    camera = VirtualCamera(K)

    config = RenderConfig(
        image_width=640, image_height=480, background_color=(30, 30, 30)
    )
    renderer = PointRenderer(config)
    renderer.point_size = 2  # Smaller points for denser cloud

    # Test multiple viewpoints
    test_views = [
        ("front", np.array([0, 0, 0]), np.array([0, 0, 8])),
        ("side", np.array([3, 0, 1]), np.array([0, 0, 8])),
        ("above", np.array([0, 0, 3]), np.array([0, 0, 8])),
        ("corner", np.array([2, 2, 1]), np.array([0, 0, 8])),
    ]

    successful_renders = 0

    for name, cam_pos, look_at in test_views:
        print(f"\nRendering {name} view...")

        # Set camera
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)

        up = np.array([0, 1, 0])  # Y up
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([1, 0, 0])
            right = np.cross(forward, up)

        right = right / np.linalg.norm(right)
        up_corrected = np.cross(right, forward)

        camera.position = cam_pos
        camera.rotation = np.column_stack([right, up_corrected, -forward])

        # Render
        image, depth = renderer.render(transformed_points, camera)

        # Check quality
        non_bg = np.sum(np.any(image != config.background_color, axis=2))
        print(f"  Rendered pixels: {non_bg}")

        if non_bg > 100:  # At least some visible content
            filename = f"improved_{name}.jpg"
            cv2.imwrite(filename, image)
            print(f"  ✅ Saved as {filename}")
            successful_renders += 1
        else:
            print("  ❌ No visible content")

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {successful_renders}/4 successful renders")

    if successful_renders > 0:
        print("✅ Improved reconstruction is working!")
        print("Check the improved_*.jpg files")
    else:
        print("❌ Still having rendering issues")

    return transformed_points


if __name__ == "__main__":
    points = test_improved_rendering()

    if points and len(points) > 100:
        print("\n💡 NEXT STEPS:")
        print("1. Add dense stereo matching for wall/floor surfaces")
        print("2. Implement mesh generation from point cloud")
        print("3. Add texture mapping from original images")
        print("4. Test with AI enhancement pipeline")
