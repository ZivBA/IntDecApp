#!/usr/bin/env python3
"""
Debug the rendering system to find why points aren't visible
"""

import numpy as np
from src.core.reconstruction import RoomReconstructor
from src.core.renderer import VirtualCamera, PointRenderer
from src.core.types import RenderConfig
import os
import cv2


def load_test_images():
    """Load test images"""
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
    return images


def debug_reconstruction_and_rendering():
    """Debug step by step"""
    print("🔍 DEBUGGING RENDERING SYSTEM")
    print("=" * 50)

    # 1. Reconstruction
    images = load_test_images()
    reconstructor = RoomReconstructor()
    result = reconstructor.reconstruct(images)

    if not result.success:
        print("❌ Reconstruction failed")
        return

    print(f"✅ Reconstruction successful: {len(result.points_3d)} points")

    # 2. Analyze 3D points
    print("\n📊 3D POINT ANALYSIS")
    positions = np.array([p.position for p in result.points_3d])
    print(f"Point positions shape: {positions.shape}")
    print(f"X range: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"Y range: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"Z range: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

    scene_center = np.mean(positions, axis=0)
    print(
        f"Scene center: [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]"
    )

    # 3. Analyze camera poses
    print("\n📷 CAMERA POSE ANALYSIS")
    for i, pose in enumerate(result.camera_poses):
        print(
            f"Camera {i}: pos=[{pose.position[0]:.3f}, {pose.position[1]:.3f}, {pose.position[2]:.3f}]"
        )
        print(
            f"  Intrinsics shape: {pose.intrinsics.shape if pose.intrinsics is not None else None}"
        )

    # 4. Test simple projection
    print("\n🔬 PROJECTION TEST")

    # Create virtual camera at a reasonable position
    if result.camera_poses[0].intrinsics is not None:
        K = result.camera_poses[0].intrinsics
    else:
        # Default intrinsics
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

    virtual_camera = VirtualCamera(K)

    # Position camera looking at scene center
    camera_pos = scene_center + np.array([1.0, 0.0, 0.0])  # 1 meter to the right
    virtual_camera.set_pose(camera_pos, scene_center)

    print(f"Virtual camera position: {virtual_camera.position}")
    print(f"Virtual camera rotation shape: {virtual_camera.rotation.shape}")

    # Get projection matrix
    P = virtual_camera.get_projection_matrix()
    print(f"Projection matrix shape: {P.shape}")
    print("Projection matrix:")
    print(P)

    # Test projecting a few points manually
    print("\n🎯 MANUAL PROJECTION TEST")
    for i in range(min(5, len(result.points_3d))):
        point = result.points_3d[i]
        point_4d = np.append(point.position, 1.0)

        # Project
        projected = P @ point_4d
        print(f"Point {i}: 3D={point.position} -> 4D={point_4d}")
        print(f"  Projected: {projected}")

        if projected[2] > 0:
            x = projected[0] / projected[2]
            y = projected[1] / projected[2]
            print(f"  Screen coords: ({x:.1f}, {y:.1f})")

            if 0 <= x < 640 and 0 <= y < 480:
                print("  ✅ Point is visible!")
            else:
                print("  ❌ Point outside screen bounds")
        else:
            print("  ❌ Point behind camera")

    # 5. Try rendering with debug info
    print("\n🎨 RENDER TEST")
    config = RenderConfig(image_width=640, image_height=480)
    renderer = PointRenderer(config)

    rendered_image, depth_map = renderer.render(result.points_3d, virtual_camera)

    # Count non-background pixels
    background_color = np.array(config.background_color)
    non_bg_pixels = np.sum(np.any(rendered_image != background_color, axis=2))
    print(f"Non-background pixels: {non_bg_pixels} / {640 * 480}")

    if non_bg_pixels > 0:
        print("✅ Some points were rendered!")
    else:
        print("❌ No points rendered - all background")

    # Save debug render
    cv2.imwrite("debug_render.jpg", rendered_image)
    print("Debug render saved as debug_render.jpg")


if __name__ == "__main__":
    debug_reconstruction_and_rendering()
