#!/usr/bin/env python3
"""
Test coordinate system fix by matching working debug render setup exactly
"""

import numpy as np
import cv2
import open3d as o3d
from src.core.renderer import VirtualCamera, PointRenderer
from src.core.types import RenderConfig, Point3D


def test_coordinate_fix():
    """Test with exact same setup as working debug render"""

    print("🔧 COORDINATE SYSTEM FIX TEST")
    print("=" * 35)

    # Load COLMAP points
    pcd = o3d.io.read_point_cloud("colmap_points.ply")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    print(f"✅ Loaded {len(points)} COLMAP points")

    # Apply the WORKING transformation (identical to debug)
    center = points.mean(axis=0)
    points_transformed = (points - center) * 0.2 + np.array([0, 0, 8])

    print("🔧 Applied working transformation")
    print(
        f"   Z range: {points_transformed[:, 2].min():.2f} to {points_transformed[:, 2].max():.2f}"
    )

    # Convert to Point3D format
    colmap_points = []
    for i in range(len(points_transformed)):
        color = (
            (colors[i] * 255).astype(np.uint8)
            if colors.max() <= 1.0
            else colors[i].astype(np.uint8)
        )
        colmap_points.append(
            Point3D(position=points_transformed[i], color=color, confidence=1.0)
        )

    # Setup camera EXACTLY like working debug render
    # Debug used: camera at origin, looking down +Z, focal=500, center=(320,240)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    camera = VirtualCamera(K)

    # Camera at origin, no rotation (identity matrix = looking down +Z)
    camera.position = np.array([0.0, 0.0, 0.0])
    camera.rotation = np.eye(3)  # Identity = no rotation

    print("📷 Camera setup:")
    print(f"   Position: {camera.position}")
    print("   Looking direction: +Z (same as debug)")

    config = RenderConfig(
        image_width=640,
        image_height=480,
        background_color=(50, 50, 50),  # Same as debug
    )
    renderer = PointRenderer(config)
    renderer.point_size = 2

    # Test single viewpoint (same as working debug)
    print("\n🎯 Testing with debug camera setup...")

    try:
        image, depth = renderer.render(colmap_points, camera)

        # Check result
        non_bg = np.sum(np.any(image != config.background_color, axis=2))
        print(f"   Non-background pixels: {non_bg}")

        if non_bg > 100:
            cv2.imwrite("coordinate_fix_test.jpg", image)
            print("   ✅ SUCCESS! Saved as coordinate_fix_test.jpg")

            # Quick verification - count points that should be visible
            points_in_front = points_transformed[points_transformed[:, 2] > 0]
            print(f"   Expected visible points: {len(points_in_front)}")

            return True
        else:
            print(f"   ❌ Still not working - only {non_bg} pixels")

            # Debug: check what the renderer sees
            print("   Debugging renderer projection...")

            # Manual projection check
            test_points = points_transformed[:5]  # First 5 points
            for i, pt in enumerate(test_points):
                # Same projection as debug
                if pt[2] > 0:  # In front of camera
                    x_proj = pt[0] * 500 / pt[2] + 320
                    y_proj = pt[1] * 500 / pt[2] + 240
                    print(f"     Point {i}: {pt} -> ({x_proj:.1f}, {y_proj:.1f})")

            return False

    except Exception as e:
        print(f"   ❌ Render failed: {e}")
        return False


def verify_debug_render_still_works():
    """Verify the original debug render still works"""

    print("\n🔍 Verifying debug render still works...")

    # Load and transform points (same as debug)
    pcd = o3d.io.read_point_cloud("colmap_points.ply")
    points = np.asarray(pcd.points)
    np.asarray(pcd.colors)

    center = points.mean(axis=0)
    transformed_points = (points - center) * 0.2 + np.array([0, 0, 8])

    # Simple projection (same as debug)
    np.array([0, 0, 0])
    visible_points = transformed_points[transformed_points[:, 2] > 0]

    focal = 500
    cx, cy = 320, 240

    projected_x = visible_points[:, 0] * focal / visible_points[:, 2] + cx
    projected_y = visible_points[:, 1] * focal / visible_points[:, 2] + cy

    in_bounds = (
        (projected_x >= 0)
        & (projected_x < 640)
        & (projected_y >= 0)
        & (projected_y < 480)
    )

    visible_in_image = np.sum(in_bounds)

    print(f"   Debug projection: {visible_in_image} points visible")

    if visible_in_image > 500:
        print("   ✅ Debug render should still work")
        return True
    else:
        print("   ❌ Debug render broken too!")
        return False


if __name__ == "__main__":
    # First verify debug still works
    debug_works = verify_debug_render_still_works()

    if debug_works:
        # Test if we can make renderer work with same setup
        success = test_coordinate_fix()

        if success:
            print("\n🎉 COORDINATE SYSTEM FIXED!")
            print("✅ Renderer now working with COLMAP points")
            print("✅ Using transformation: (points - center) * 0.2 + [0,0,8]")
            print("✅ Camera at origin looking down +Z")
        else:
            print("\n🔧 Still debugging renderer vs debug projection difference...")
    else:
        print("\n❌ Debug render broken - coordinate system changed")
