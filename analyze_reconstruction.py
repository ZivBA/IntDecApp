#!/usr/bin/env python3
"""
Analyze what's actually in our reconstruction
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.core.reconstruction import RoomReconstructor
import os


def analyze_reconstruction():
    """Deep dive into what we're actually reconstructing"""

    # Load and reconstruct
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

    print("=" * 50)
    print("RECONSTRUCTION ANALYSIS")
    print("=" * 50)

    if not result.success:
        print("❌ Reconstruction failed!")
        return

    print(f"✅ Points reconstructed: {len(result.points_3d)}")

    # Extract data
    positions = np.array([p.position for p in result.points_3d])
    colors = np.array([p.color for p in result.points_3d])

    # Spatial analysis
    print("\n📊 SPATIAL DISTRIBUTION:")
    print(
        f"X range: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f} ({positions[:, 0].max() - positions[:, 0].min():.2f}m)"
    )
    print(
        f"Y range: {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f} ({positions[:, 1].max() - positions[:, 1].min():.2f}m)"
    )
    print(
        f"Z range: {positions[:, 2].min():.2f} to {positions[:, 2].max():.2f} ({positions[:, 2].max() - positions[:, 2].min():.2f}m)"
    )

    # Color analysis
    print("\n🎨 COLOR DISTRIBUTION:")
    print(
        f"Color mean: R={colors[:, 0].mean():.0f}, G={colors[:, 1].mean():.0f}, B={colors[:, 2].mean():.0f}"
    )
    print(f"Unique colors: {len(np.unique(colors, axis=0))}")

    # Point clustering analysis
    print("\n📍 POINT CLUSTERING:")
    centroid = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    print(f"Average distance from centroid: {distances.mean():.2f}m")
    print(f"Max distance from centroid: {distances.max():.2f}m")

    # Identify point locations
    print("\n🎯 LIKELY POINT LOCATIONS:")
    for i in range(min(10, len(result.points_3d))):
        p = result.points_3d[i]
        print(
            f"Point {i}: pos=[{p.position[0]:.2f}, {p.position[1]:.2f}, {p.position[2]:.2f}], "
            f"color=[{p.color[0]}, {p.color[1]}, {p.color[2]}]"
        )

    # Create better 3D visualization
    fig = plt.figure(figsize=(15, 5))

    # View 1: Top-down (XY plane)
    ax1 = fig.add_subplot(131)
    ax1.scatter(positions[:, 0], positions[:, 1], c=colors / 255, s=50)
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.set_title("Top-Down View (XY)")
    ax1.set_aspect("equal")
    ax1.grid(True)

    # View 2: Side view (XZ plane)
    ax2 = fig.add_subplot(132)
    ax2.scatter(positions[:, 0], positions[:, 2], c=colors / 255, s=50)
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Z (meters)")
    ax2.set_title("Side View (XZ)")
    ax2.set_aspect("equal")
    ax2.grid(True)

    # View 3: 3D view
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=colors / 255,
        s=50,
        alpha=0.6,
    )
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("3D View")

    plt.tight_layout()
    plt.savefig("reconstruction_analysis.png", dpi=150)
    print("\n📊 Saved detailed analysis plot: reconstruction_analysis.png")

    # Problem diagnosis
    print("\n⚠️ PROBLEMS IDENTIFIED:")
    problems = []

    if len(result.points_3d) < 100:
        problems.append(
            f"Too few points ({len(result.points_3d)}). Need 1000+ for good visual quality"
        )

    if positions[:, 2].max() - positions[:, 2].min() > 5:
        problems.append("Z-range too large - likely erroneous triangulations")

    avg_neighbor_dist = []
    for i in range(len(positions)):
        dists = np.linalg.norm(positions - positions[i], axis=1)
        dists = dists[dists > 0]  # Exclude self
        if len(dists) > 0:
            avg_neighbor_dist.append(dists.min())

    if np.mean(avg_neighbor_dist) > 0.5:
        problems.append(
            f"Points too sparse (avg nearest neighbor: {np.mean(avg_neighbor_dist):.2f}m)"
        )

    for problem in problems:
        print(f"  ❌ {problem}")

    print("\n💡 RECOMMENDATIONS:")
    print("1. Need dense stereo matching, not just sparse features")
    print("2. Add more intermediate camera positions for better coverage")
    print("3. Use depth estimation to fill gaps between sparse points")
    print("4. Consider COLMAP's dense reconstruction instead of sparse")

    return result


if __name__ == "__main__":
    result = analyze_reconstruction()
