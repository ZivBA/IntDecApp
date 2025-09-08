#!/usr/bin/env python3
"""
Debug COLMAP coordinate system issues
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d


def analyze_colmap_coordinates():
    """Analyze COLMAP point cloud coordinates"""

    print("🔍 DEBUGGING COLMAP COORDINATES")
    print("=" * 35)

    # Load point cloud
    pcd = o3d.io.read_point_cloud("colmap_points.ply")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    print("📊 Point cloud statistics:")
    print(f"   Total points: {len(points)}")
    print(f"   X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"   Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"   Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    center = points.mean(axis=0)
    print(f"   Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

    # Create visualization
    fig = plt.figure(figsize=(15, 5))

    # 1. 3D scatter plot
    ax1 = fig.add_subplot(131, projection="3d")

    # Color points by their original colors if available
    if colors.max() <= 1.0:
        point_colors = colors
    else:
        point_colors = colors / 255.0

    ax1.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=point_colors, s=1, alpha=0.6
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("COLMAP Points (Original)")

    # 2. Top-down view (XY)
    ax2 = fig.add_subplot(132)
    ax2.scatter(points[:, 0], points[:, 1], c=point_colors, s=2, alpha=0.6)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Top View (XY)")
    ax2.set_aspect("equal")
    ax2.grid(True)

    # 3. Side view (XZ)
    ax3 = fig.add_subplot(133)
    ax3.scatter(points[:, 0], points[:, 2], c=point_colors, s=2, alpha=0.6)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.set_title("Side View (XZ)")
    ax3.set_aspect("equal")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("colmap_coords_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("💾 Saved coordinate analysis: colmap_coords_analysis.png")

    return points, colors


def test_simple_rendering():
    """Test with simplified coordinate transformation"""

    print("\n🧪 TESTING SIMPLE RENDERING")
    print("=" * 30)

    points, colors = analyze_colmap_coordinates()

    # Try different transformation approaches
    approaches = [
        ("no_transform", lambda p: p),
        ("center_only", lambda p: p - p.mean(axis=0)),
        ("center_and_flip_z", lambda p: (p - p.mean(axis=0)) * np.array([1, 1, -1])),
        ("center_and_move_z", lambda p: (p - p.mean(axis=0)) + np.array([0, 0, 10])),
        ("full_transform", lambda p: (p - p.mean(axis=0)) * 0.2 + np.array([0, 0, 8])),
    ]

    for name, transform_func in approaches:
        print(f"\n🎯 Testing {name}...")

        # Apply transformation
        transformed_points = transform_func(points)

        print("   Transformed range:")
        print(
            f"     X: [{transformed_points[:, 0].min():.2f}, {transformed_points[:, 0].max():.2f}]"
        )
        print(
            f"     Y: [{transformed_points[:, 1].min():.2f}, {transformed_points[:, 1].max():.2f}]"
        )
        print(
            f"     Z: [{transformed_points[:, 2].min():.2f}, {transformed_points[:, 2].max():.2f}]"
        )

        # Quick visibility test
        # Simple camera at origin looking down +Z
        np.array([0, 0, 0])

        # Count points in front of camera (positive Z)
        visible_points = transformed_points[transformed_points[:, 2] > 0]

        if len(visible_points) > 0:
            # Project to image plane (simple perspective)
            # Assuming focal length = 500, image center = (320, 240)
            focal = 500
            cx, cy = 320, 240

            projected_x = visible_points[:, 0] * focal / visible_points[:, 2] + cx
            projected_y = visible_points[:, 1] * focal / visible_points[:, 2] + cy

            # Count points within image bounds
            in_bounds = (
                (projected_x >= 0)
                & (projected_x < 640)
                & (projected_y >= 0)
                & (projected_y < 480)
            )

            visible_in_image = np.sum(in_bounds)

            print(f"   Points in front of camera: {len(visible_points)}")
            print(f"   Points visible in image: {visible_in_image}")

            if visible_in_image > 50:
                print("   ✅ This transformation should work!")

                # Save this transformation for testing

                # Create simple test render
                img = np.full((480, 640, 3), 50, dtype=np.uint8)  # Dark gray background

                for i in range(len(visible_points)):
                    if in_bounds[i]:
                        x = int(projected_x[i])
                        y = int(projected_y[i])

                        if 0 <= x < 640 and 0 <= y < 480:
                            # Get color
                            if colors.max() <= 1.0:
                                color = (colors[i] * 255).astype(int)
                            else:
                                color = colors[i].astype(int)

                            # Draw point (small circle)
                            cv2.circle(
                                img,
                                (x, y),
                                2,
                                (int(color[2]), int(color[1]), int(color[0])),
                                -1,
                            )

                cv2.imwrite(f"simple_render_{name}.jpg", img)
                print(f"   💾 Saved test render: simple_render_{name}.jpg")

            else:
                print("   ❌ Not enough visible points")
        else:
            print("   ❌ No points in front of camera")


if __name__ == "__main__":
    test_simple_rendering()
