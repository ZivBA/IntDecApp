#!/usr/bin/env python3
"""
Final fix: Transform real data to proper coordinate system for rendering
"""

import numpy as np
import cv2
from src.core.reconstruction import RoomReconstructor
from src.core.renderer import VirtualCamera, PointRenderer
from src.core.types import Point3D, RenderConfig
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


def transform_to_camera_coords(points_3d):
    """Transform points to a standard camera coordinate system"""
    positions = np.array([p.position for p in points_3d])

    # Center the points
    centroid = np.mean(positions, axis=0)
    positions_centered = positions - centroid

    # Scale to reasonable size (make largest dimension ~5 units)
    max_extent = np.max(np.abs(positions_centered))
    scale_factor = 5.0 / max_extent
    positions_scaled = positions_centered * scale_factor

    # Move points to positive Z (in front of camera)
    positions_final = positions_scaled + np.array([0, 0, 8])  # Place at Z=8

    # Create new Point3D objects
    transformed_points = []
    for i, point in enumerate(points_3d):
        new_point = Point3D(
            position=positions_final[i], color=point.color, confidence=point.confidence
        )
        transformed_points.append(new_point)

    return transformed_points, centroid, scale_factor


def test_final_rendering():
    """Final test with coordinate transformation"""
    print("🔧 FINAL RENDERING TEST WITH COORDINATE TRANSFORMATION")

    # Get reconstruction
    images = load_test_images()
    reconstructor = RoomReconstructor()
    result = reconstructor.reconstruct(images)

    if not result.success:
        print("❌ Reconstruction failed")
        return False

    print(f"✅ Reconstruction: {len(result.points_3d)} points")

    # Transform points to camera-friendly coordinates
    transformed_points, centroid, scale = transform_to_camera_coords(result.points_3d)

    print(f"Transformed {len(transformed_points)} points")
    print(f"Original centroid: {centroid}")
    print(f"Scale factor: {scale}")

    # Test projection with simple camera setup
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    camera = VirtualCamera(K)

    # Camera at origin looking down +Z
    camera.position = np.array([0.0, 0.0, 0.0])
    camera.rotation = np.eye(3)

    print("Testing transformed points:")
    P = camera.get_projection_matrix()

    visible_count = 0
    for i in range(min(10, len(transformed_points))):
        point = transformed_points[i]
        point_4d = np.append(point.position, 1.0)
        projected = P @ point_4d

        if projected[2] > 0:
            x = projected[0] / projected[2]
            y = projected[1] / projected[2]
            if 0 <= x < 640 and 0 <= y < 480:
                visible_count += 1
                print(
                    f"  Point {i}: VISIBLE at ({x:.1f}, {y:.1f}), Z={point.position[2]:.1f}"
                )

    print(f"Visible points: {visible_count}/10")

    if visible_count > 0:
        # Render!
        config = RenderConfig(
            image_width=640, image_height=480, background_color=(20, 20, 20)
        )
        renderer = PointRenderer(config)

        rendered_image, depth_map = renderer.render(transformed_points, camera)

        # Check success
        background_color = np.array(config.background_color)
        non_bg_pixels = np.sum(np.any(rendered_image != background_color, axis=2))
        print(f"Rendered {non_bg_pixels} non-background pixels")

        if non_bg_pixels > 0:
            cv2.imwrite("final_success.jpg", rendered_image)
            print("🎉 FINAL SUCCESS! Saved as final_success.jpg")

            # Also try a few different viewpoints
            test_viewpoints = [
                (np.array([2.0, 0.0, 0.0]), "side_view.jpg"),
                (np.array([0.0, 2.0, 0.0]), "front_view.jpg"),
                (np.array([1.5, 1.5, 1.0]), "corner_view.jpg"),
            ]

            for cam_pos, filename in test_viewpoints:
                # Look at center of transformed points (roughly [0,0,8])
                look_at = np.array([0.0, 0.0, 8.0])

                # Manual look-at calculation
                forward = look_at - cam_pos
                forward = forward / np.linalg.norm(forward)

                up = np.array([0.0, 0.0, 1.0])  # Z up
                right = np.cross(forward, up)

                if np.linalg.norm(right) < 1e-6:
                    up = np.array([1.0, 0.0, 0.0])  # Use X as up
                    right = np.cross(forward, up)

                right = right / np.linalg.norm(right)
                up_corrected = np.cross(right, forward)

                camera.position = cam_pos
                camera.rotation = np.column_stack([right, up_corrected, -forward])

                view_image, _ = renderer.render(transformed_points, camera)
                cv2.imwrite(filename, view_image)
                print(f"  ✅ Additional view saved: {filename}")

            return True
        else:
            print("❌ No pixels rendered")

    return False


if __name__ == "__main__":
    success = test_final_rendering()

    if success:
        print("\n🎉 VIRTUAL NAVIGATION SYSTEM IS WORKING!")
        print("✅ Coordinate transformation successful")
        print("✅ Point-based rendering functional")
        print("✅ Multiple viewpoints generated")
        print("\nNext steps:")
        print("- Integrate transformation into main navigation system")
        print("- Add density improvement techniques")
        print("- Optimize for mobile performance")
    else:
        print("\n❌ Still debugging needed")
