#!/usr/bin/env python3
"""
POC Test Script: Multi-view Room Reconstruction

This script tests our minimal multi-view stereo implementation
using the room photos in test_data/
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.core.reconstruction import RoomReconstructor


def load_test_images(test_data_dir: str = "test_data") -> list:
    """Load all images from test data directory"""
    image_files = sorted([f for f in os.listdir(test_data_dir) if f.endswith(".jpg")])
    images = []

    for filename in image_files:
        image_path = os.path.join(test_data_dir, filename)
        image = cv2.imread(image_path)
        if image is not None:
            # Resize to reasonable size for processing
            height, width = image.shape[:2]
            if width > 1024:
                scale = 1024 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))

            images.append(image)
            print(f"Loaded {filename}: {image.shape}")
        else:
            print(f"Failed to load {filename}")

    return images


def visualize_3d_points(points_3d: list, title: str = "3D Reconstruction"):
    """Visualize reconstructed 3D points"""
    if not points_3d:
        print("No 3D points to visualize")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Extract positions and colors
    positions = np.array([p.position for p in points_3d])
    colors = np.array([p.color / 255.0 for p in points_3d])  # Normalize to [0,1]

    # Plot points
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=1, alpha=0.6
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Set equal aspect ratio
    max_range = np.max(np.ptp(positions, axis=0)) / 2
    mid_x = np.mean(positions[:, 0])
    mid_y = np.mean(positions[:, 1])
    mid_z = np.mean(positions[:, 2])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig("reconstruction_result.png", dpi=150, bbox_inches="tight")
    plt.show()


def analyze_reconstruction(result):
    """Analyze and print reconstruction statistics"""
    print("\n" + "=" * 50)
    print("RECONSTRUCTION ANALYSIS")
    print("=" * 50)

    if not result.success:
        print(f"❌ FAILED: {result.error_message}")
        return

    print(f"✅ SUCCESS in {result.processing_time:.2f} seconds")
    print(f"📊 Generated {len(result.points_3d)} 3D points")
    print(f"📷 Estimated {len(result.camera_poses)} camera poses")

    if result.points_3d:
        # Analyze point cloud
        positions = np.array([p.position for p in result.points_3d])

        print("\n3D Point Cloud Statistics:")
        print(f"  X range: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f}")
        print(f"  Y range: {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f}")
        print(f"  Z range: {positions[:, 2].min():.2f} to {positions[:, 2].max():.2f}")

        # Check for reasonable room dimensions
        x_span = positions[:, 0].max() - positions[:, 0].min()
        y_span = positions[:, 1].max() - positions[:, 1].min()
        z_span = positions[:, 2].max() - positions[:, 2].min()

        print(f"  Room dimensions: {x_span:.2f} x {y_span:.2f} x {z_span:.2f}")

        if 1.0 < x_span < 10.0 and 1.0 < y_span < 10.0:
            print("  ✅ Room dimensions look reasonable")
        else:
            print("  ⚠️  Room dimensions may be incorrect")

    # Analyze camera poses
    if result.camera_poses:
        positions = np.array([pose.position for pose in result.camera_poses])
        print("\nCamera Positions:")
        for i, pos in enumerate(positions):
            print(f"  Camera {i}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")


def main():
    """Main test function"""
    print("🏠 Room Reconstruction POC Test")
    print("=" * 50)

    # Check if test data exists
    if not os.path.exists("test_data"):
        print("❌ test_data directory not found!")
        print("Please create test_data/ and add your room photos")
        return

    # Load test images
    print("Loading test images...")
    images = load_test_images()

    if len(images) < 2:
        print("❌ Need at least 2 images for reconstruction!")
        return

    print(f"Loaded {len(images)} images")

    # Initialize reconstructor
    reconstructor = RoomReconstructor()

    # Run reconstruction
    print("\nStarting reconstruction...")
    result = reconstructor.reconstruct(images)

    # Analyze results
    analyze_reconstruction(result)

    # Visualize if successful
    if result.success and result.points_3d:
        print("\nGenerating 3D visualization...")
        visualize_3d_points(result.points_3d, "Room Reconstruction POC")
        print("Visualization saved as 'reconstruction_result.png'")

    print("\n✅ POC Test Complete!")


if __name__ == "__main__":
    main()
