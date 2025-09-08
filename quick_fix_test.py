#!/usr/bin/env python3
"""
Quick fix test - adjust camera positioning
"""

import numpy as np
import cv2
from src.core.reconstruction import RoomReconstructor
from src.core.renderer import VirtualCamera, PointRenderer
from src.core.types import RenderConfig
import os


def load_test_images():
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


def test_fixed_camera_positioning():
    print("🔧 TESTING FIXED CAMERA POSITIONING")

    # Get reconstruction
    images = load_test_images()
    reconstructor = RoomReconstructor()
    result = reconstructor.reconstruct(images)

    if not result.success:
        print("❌ Reconstruction failed")
        return

    positions = np.array([p.position for p in result.points_3d])
    scene_center = np.mean(positions, axis=0)

    print(f"Scene center: {scene_center}")
    print(f"Scene bounds: {positions.min(axis=0)} to {positions.max(axis=0)}")

    # Create camera with default intrinsics
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    virtual_camera = VirtualCamera(K)

    # Try different camera positions
    test_positions = [
        # Further back along each axis
        scene_center + np.array([3.0, 0.0, 0.0]),  # Right
        scene_center + np.array([-3.0, 0.0, 0.0]),  # Left
        scene_center + np.array([0.0, 3.0, 0.0]),  # Forward
        scene_center + np.array([0.0, -3.0, 0.0]),  # Back
        scene_center + np.array([0.0, 0.0, 2.0]),  # Up
        scene_center + np.array([2.0, 2.0, 1.0]),  # Corner
    ]

    config = RenderConfig(image_width=640, image_height=480)
    renderer = PointRenderer(config)

    for i, cam_pos in enumerate(test_positions):
        print(f"\n🎯 Testing position {i}: {cam_pos}")

        # Set camera pose
        virtual_camera.set_pose(cam_pos, scene_center)

        # Test manual projection of first few points
        P = virtual_camera.get_projection_matrix()
        visible_count = 0

        for j in range(min(5, len(result.points_3d))):
            point = result.points_3d[j]
            point_4d = np.append(point.position, 1.0)
            projected = P @ point_4d

            if projected[2] > 0:  # In front of camera
                x = projected[0] / projected[2]
                y = projected[1] / projected[2]
                if 0 <= x < 640 and 0 <= y < 480:
                    visible_count += 1

        print(f"  Visible points (sample): {visible_count}/5")

        # Try rendering
        rendered_image, _ = renderer.render(result.points_3d, virtual_camera)

        # Count non-background pixels
        background_color = np.array(config.background_color)
        non_bg_pixels = np.sum(np.any(rendered_image != background_color, axis=2))

        print(f"  Rendered pixels: {non_bg_pixels}")

        if non_bg_pixels > 0:
            # Save successful render
            filename = f"fixed_render_{i}.jpg"
            cv2.imwrite(filename, rendered_image)
            print(f"  ✅ SUCCESS! Saved as {filename}")
            break
        else:
            print("  ❌ No visible points")


if __name__ == "__main__":
    test_fixed_camera_positioning()
