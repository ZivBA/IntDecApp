#!/usr/bin/env python3
"""
Minimal rendering test with known coordinates
"""

import numpy as np
import cv2
from src.core.renderer import VirtualCamera, PointRenderer
from src.core.types import Point3D, RenderConfig


def test_minimal_rendering():
    """Test rendering with simple known coordinates"""
    print("🔬 MINIMAL RENDERING TEST")

    # Create simple test points
    test_points = [
        Point3D(
            position=np.array([0.0, 0.0, 5.0]), color=np.array([255, 0, 0])
        ),  # Red point at Z=5
        Point3D(
            position=np.array([1.0, 0.0, 5.0]), color=np.array([0, 255, 0])
        ),  # Green point
        Point3D(
            position=np.array([0.0, 1.0, 5.0]), color=np.array([0, 0, 255])
        ),  # Blue point
        Point3D(
            position=np.array([-1.0, 0.0, 5.0]), color=np.array([255, 255, 0])
        ),  # Yellow point
        Point3D(
            position=np.array([0.0, -1.0, 5.0]), color=np.array([255, 0, 255])
        ),  # Magenta point
    ]

    print(f"Created {len(test_points)} test points at Z=5")

    # Create camera at origin looking down positive Z axis
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    camera = VirtualCamera(K)

    # Set camera at origin looking forward (this should work)
    camera.position = np.array([0.0, 0.0, 0.0])
    camera.rotation = np.eye(3)  # Identity = looking down +Z

    print(f"Camera position: {camera.position}")
    print(f"Camera rotation:\n{camera.rotation}")

    # Test manual projection
    P = camera.get_projection_matrix()
    print(f"Projection matrix:\n{P}")

    print("\nManual projection test:")
    for i, point in enumerate(test_points):
        point_4d = np.append(point.position, 1.0)
        projected = P @ point_4d
        print(f"Point {i}: {point.position} -> {projected}")

        if projected[2] > 0:
            x = projected[0] / projected[2]
            y = projected[1] / projected[2]
            print(
                f"  Screen: ({x:.1f}, {y:.1f}) - {'VISIBLE' if 0 <= x < 640 and 0 <= y < 480 else 'OUT OF BOUNDS'}"
            )
        else:
            print("  BEHIND CAMERA")

    # Try rendering
    config = RenderConfig(image_width=640, image_height=480, background_color=(0, 0, 0))
    renderer = PointRenderer(config)

    rendered_image, depth_map = renderer.render(test_points, camera)

    # Check result
    non_black_pixels = np.sum(np.any(rendered_image != [0, 0, 0], axis=2))
    print(f"\nRendered {non_black_pixels} non-black pixels")

    if non_black_pixels > 0:
        cv2.imwrite("minimal_test_success.jpg", rendered_image)
        print("✅ SUCCESS! Saved as minimal_test_success.jpg")
        return True
    else:
        cv2.imwrite("minimal_test_failed.jpg", rendered_image)
        print("❌ FAILED - saved as minimal_test_failed.jpg")
        return False


def test_with_real_data_fixed_coords():
    """Test with real data but fixed coordinate system"""
    print("\n🔧 TESTING WITH REAL DATA - FIXED COORDS")

    # Load reconstruction
    from src.core.reconstruction import RoomReconstructor
    import os

    image_files = sorted([f for f in os.listdir("test_data") if f.endswith(".jpg")])
    images = []
    for filename in image_files:
        image_path = os.path.join("test_data", filename)
        image = cv2.imread(image_path)
        if image is not None:
            height, width = image.shape[:2]
            if width > 1024:
                scale = 1024 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            images.append(image)

    reconstructor = RoomReconstructor()
    result = reconstructor.reconstruct(images)

    if not result.success:
        return False

    # Analyze the coordinate system
    positions = np.array([p.position for p in result.points_3d])
    print("Real points range:")
    print(f"  X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
    print(f"  Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    print(f"  Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")

    scene_center = np.mean(positions, axis=0)
    scene_size = np.max(positions, axis=0) - np.min(positions, axis=0)

    print(f"Scene center: {scene_center}")
    print(f"Scene size: {scene_size}")

    # Position camera MUCH further back
    camera_distance = np.max(scene_size) * 2  # 2x the scene size
    camera_pos = scene_center + np.array([0, -camera_distance, 0])  # Back along Y

    print(f"Camera position: {camera_pos}")
    print(f"Camera distance: {camera_distance}")

    # Create camera
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    camera = VirtualCamera(K)

    # Use simple look-at (manual calculation to avoid the division by zero issue)
    forward = scene_center - camera_pos
    forward = forward / np.linalg.norm(forward)

    # Use X as up vector to avoid parallel vectors
    up = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up)

    if np.linalg.norm(right) < 1e-6:  # Vectors are parallel, use different up
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)

    right = right / np.linalg.norm(right)
    up_corrected = np.cross(right, forward)

    # Set camera manually
    camera.position = camera_pos
    camera.rotation = np.column_stack([right, up_corrected, -forward])

    print("Testing projection of first few points:")
    P = camera.get_projection_matrix()

    visible_count = 0
    for i in range(min(10, len(result.points_3d))):
        point = result.points_3d[i]
        point_4d = np.append(point.position, 1.0)
        projected = P @ point_4d

        if projected[2] > 0:
            x = projected[0] / projected[2]
            y = projected[1] / projected[2]
            if 0 <= x < 640 and 0 <= y < 480:
                visible_count += 1
                print(f"  Point {i}: VISIBLE at ({x:.1f}, {y:.1f})")

    print(f"Visible points: {visible_count}/10")

    if visible_count > 0:
        # Try rendering
        config = RenderConfig(image_width=640, image_height=480)
        renderer = PointRenderer(config)
        rendered_image, _ = renderer.render(result.points_3d, camera)

        non_bg_pixels = np.sum(
            np.any(rendered_image != config.background_color, axis=2)
        )
        print(f"Rendered pixels: {non_bg_pixels}")

        if non_bg_pixels > 0:
            cv2.imwrite("real_data_success.jpg", rendered_image)
            print("✅ SUCCESS with real data! Saved as real_data_success.jpg")
            return True

    return False


if __name__ == "__main__":
    # Test 1: Simple known coordinates
    success1 = test_minimal_rendering()

    if success1:
        # Test 2: Real data with fixed positioning
        success2 = test_with_real_data_fixed_coords()

        if success2:
            print("\n🎉 BOTH TESTS SUCCESSFUL!")
            print("The coordinate system issues are resolved")
        else:
            print("\n⚠️  Minimal test worked, but real data still has issues")
    else:
        print("\n❌ Even minimal test failed - deeper rendering issue")
