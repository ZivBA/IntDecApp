#!/usr/bin/env python3
"""
Virtual Navigation Test Script

This script tests the view rendering system by creating virtual walkthroughs
of the reconstructed room from different viewpoints.
"""

import os
import cv2
import numpy as np

from src.core.reconstruction import RoomReconstructor
from src.core.renderer import PointRenderer, NavigationSystem
from src.core.types import RenderConfig


def load_test_images(test_data_dir: str = "test_data") -> list:
    """Load all images from test data directory"""
    image_files = sorted([f for f in os.listdir(test_data_dir) if f.endswith(".jpg")])
    images = []

    for filename in image_files:
        image_path = os.path.join(test_data_dir, filename)
        image = cv2.imread(image_path)
        if image is not None:
            # Resize to reasonable size
            height, width = image.shape[:2]
            if width > 1024:
                scale = 1024 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))

            images.append(image)

    return images


def create_virtual_walkthrough(
    reconstruction_result, output_dir: str = "virtual_views"
):
    """Create a series of virtual views walking through the room"""

    if not reconstruction_result.success:
        print("❌ Cannot create walkthrough - reconstruction failed")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize rendering system
    render_config = RenderConfig(
        image_width=640,
        image_height=480,
        background_color=(50, 50, 50),  # Dark gray background
    )

    renderer = PointRenderer(render_config)
    navigation = NavigationSystem(
        reconstruction_result.points_3d, reconstruction_result.camera_poses
    )

    print(
        f"🎬 Creating virtual walkthrough with {len(reconstruction_result.points_3d)} points"
    )
    print(f"Scene center: {navigation.scene_center}")
    print(
        f"Scene bounds: {navigation.scene_bounds['min']} to {navigation.scene_bounds['max']}"
    )

    # Create virtual camera
    virtual_camera = navigation.create_virtual_camera(
        (render_config.image_height, render_config.image_width)
    )

    # Test different viewpoints
    viewpoints = []

    # 1. Walking positions (eye level)
    walking_positions = navigation.get_walking_positions(n_positions=8)
    for i, pos in enumerate(walking_positions):
        viewpoints.append(
            {
                "name": f"walk_{i:02d}",
                "description": f"Walking view {i + 1}",
                "position": pos,
                "look_at": navigation.scene_center,
            }
        )

    # 2. Overview positions (elevated)
    overview_positions = navigation.get_overview_positions()
    for i, pos in enumerate(overview_positions):
        viewpoints.append(
            {
                "name": f"overview_{i:02d}",
                "description": f"Overview {i + 1}",
                "position": pos,
                "look_at": navigation.scene_center,
            }
        )

    # 3. Original camera positions for comparison
    for i, pose in enumerate(reconstruction_result.camera_poses):
        viewpoints.append(
            {
                "name": f"original_{i:02d}",
                "description": f"Original camera {i + 1}",
                "position": pose.position,
                "look_at": navigation.scene_center,
            }
        )

    # Render all viewpoints
    successful_renders = 0

    for vp in viewpoints:
        try:
            print(f"Rendering {vp['name']}: {vp['description']}")

            # Set virtual camera pose
            virtual_camera.set_pose(vp["position"], vp["look_at"])

            # Render view
            rendered_image, depth_map = renderer.render(
                reconstruction_result.points_3d, virtual_camera, depth_smoothing=True
            )

            # Save rendered image
            output_path = os.path.join(output_dir, f"{vp['name']}.jpg")

            # Convert BGR to RGB for saving
            rendered_rgb = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_path, rendered_rgb)

            # Save depth map
            depth_path = os.path.join(output_dir, f"{vp['name']}_depth.png")
            depth_normalized = (depth_map / depth_map.max() * 255).astype(np.uint8)
            cv2.imwrite(depth_path, depth_normalized)

            successful_renders += 1

        except Exception as e:
            print(f"Failed to render {vp['name']}: {e}")

    print(f"✅ Successfully rendered {successful_renders}/{len(viewpoints)} views")
    print(f"📁 Virtual views saved to: {output_dir}/")

    return successful_renders


def create_comparison_grid(
    reconstruction_result, output_path: str = "view_comparison.jpg"
):
    """Create a grid comparing original views with virtual views"""

    if not reconstruction_result.success:
        return

    # Load original images for comparison
    original_images = load_test_images()

    # Setup rendering
    render_config = RenderConfig(image_width=320, image_height=240)
    renderer = PointRenderer(render_config)
    navigation = NavigationSystem(
        reconstruction_result.points_3d, reconstruction_result.camera_poses
    )

    virtual_camera = navigation.create_virtual_camera(
        (render_config.image_height, render_config.image_width)
    )

    # Create comparison grid
    n_views = min(len(original_images), 4)  # Show up to 4 comparisons
    grid_height = n_views * 240
    grid_width = 640  # Original + Virtual side by side

    comparison_grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for i in range(n_views):
        # Original image (resized)
        orig_img = cv2.resize(original_images[i], (320, 240))
        comparison_grid[i * 240 : (i + 1) * 240, 0:320] = orig_img

        # Virtual view from similar position
        if i < len(reconstruction_result.camera_poses):
            try:
                pose = reconstruction_result.camera_poses[i]
                virtual_camera.set_pose(pose.position, navigation.scene_center)

                virtual_img, _ = renderer.render(
                    reconstruction_result.points_3d,
                    virtual_camera,
                    depth_smoothing=True,
                )

                comparison_grid[i * 240 : (i + 1) * 240, 320:640] = virtual_img

            except Exception as e:
                print(f"Failed to render comparison {i}: {e}")

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison_grid, "ORIGINAL", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison_grid, "VIRTUAL", (330, 30), font, 1, (0, 0, 255), 2)

    # Save comparison
    cv2.imwrite(output_path, comparison_grid)
    print(f"📊 Comparison grid saved to: {output_path}")


def main():
    """Main test function"""
    print("🎮 Virtual Navigation Test")
    print("=" * 50)

    # Check for existing reconstruction or run new one
    if not os.path.exists("test_data"):
        print("❌ test_data directory not found!")
        return

    print("Loading images and running reconstruction...")
    images = load_test_images()

    if len(images) < 2:
        print("❌ Need at least 2 images for reconstruction!")
        return

    # Run reconstruction
    reconstructor = RoomReconstructor()
    result = reconstructor.reconstruct(images)

    if not result.success:
        print(f"❌ Reconstruction failed: {result.error_message}")
        return

    print(f"✅ Reconstruction successful: {len(result.points_3d)} points")

    # Create virtual walkthrough
    print("\n🚶 Creating virtual walkthrough...")
    create_virtual_walkthrough(result)

    # Create comparison grid
    print("\n📊 Creating comparison grid...")
    create_comparison_grid(result)

    print("\n" + "=" * 50)
    print("VIRTUAL NAVIGATION TEST COMPLETE")
    print("=" * 50)
    print("Check the following outputs:")
    print("📁 virtual_views/ - Individual virtual camera views")
    print("📊 view_comparison.jpg - Side-by-side original vs virtual")
    print("\nNext steps:")
    print("- Examine rendered views for quality")
    print("- Test different camera positions")
    print("- Integrate with AI enhancement pipeline")


if __name__ == "__main__":
    main()
