#!/usr/bin/env python3
"""
Test view synthesis using COLMAP camera poses and source images
"""

import numpy as np
import cv2
import pycolmap
from pathlib import Path


def load_colmap_reconstruction():
    """Load COLMAP reconstruction with camera poses"""

    print("📷 LOADING COLMAP RECONSTRUCTION")
    print("=" * 35)

    # Load reconstruction from COLMAP output
    reconstruction_path = Path(
        "colmap_output/sparse/0"
    )  # Correct path to first reconstruction

    if not reconstruction_path.exists():
        print("❌ COLMAP reconstruction not found at colmap_output/sparse/0")
        return None, None, None

    try:
        # Load reconstruction
        reconstruction = pycolmap.Reconstruction(str(reconstruction_path))

        print("✅ Loaded reconstruction:")
        print(f"   Cameras: {len(reconstruction.cameras)}")
        print(f"   Images: {len(reconstruction.images)}")
        print(f"   Points: {len(reconstruction.points3D)}")

        # Extract camera poses and points
        camera_poses = {}
        for image_id, image in reconstruction.images.items():
            # Get camera pose from cam_from_world matrix
            cam_from_world_rigid = (
                image.cam_from_world()
            )  # Call the method to get Rigid3d
            cam_from_world = cam_from_world_rigid.matrix()  # Get 4x4 matrix

            # Extract rotation (3x3) and translation (3x1)
            R = cam_from_world[:3, :3]  # Rotation matrix
            t = cam_from_world[:3, 3]  # Translation vector

            # Camera position in world coordinates (camera center)
            cam_center = image.projection_center()

            camera_poses[image.name] = {
                "rotation": R,
                "translation": t,
                "center": cam_center,
                "camera_id": image.camera_id,
                "cam_from_world": cam_from_world,
            }

            print(
                f"   📸 {image.name}: center at [{cam_center[0]:.2f}, {cam_center[1]:.2f}, {cam_center[2]:.2f}]"
            )

        # Extract 3D points
        points_3d = []
        point_colors = []

        for point_id, point in reconstruction.points3D.items():
            points_3d.append(point.xyz)
            point_colors.append(point.color)

        points_3d = np.array(points_3d)
        point_colors = np.array(point_colors)

        return camera_poses, points_3d, point_colors

    except Exception as e:
        print(f"❌ Failed to load COLMAP reconstruction: {e}")
        return None, None, None


def load_source_images():
    """Load source images used in reconstruction"""

    print("\n📁 LOADING SOURCE IMAGES")
    print("=" * 25)

    image_dir = Path("test_data2")
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return {}

    source_images = {}
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG"))

    for img_path in sorted(image_files):
        img = cv2.imread(str(img_path))
        if img is not None:
            source_images[img_path.name] = img
            print(f"   ✅ Loaded {img_path.name}: {img.shape}")

    print(f"   Total images: {len(source_images)}")
    return source_images


def synthesize_view(
    target_position, target_rotation, camera_poses, source_images, points_3d
):
    """
    Synthesize new view using image-based rendering

    Args:
        target_position: 3D position for new camera
        target_rotation: 3x3 rotation matrix for new camera
        camera_poses: Dict of source camera poses
        source_images: Dict of source images
        points_3d: Array of 3D points
    """

    print("\n🎨 SYNTHESIZING VIEW")
    print(f"   Target position: {target_position}")
    print(f"   Available source images: {len(source_images)}")

    # Output image dimensions
    height, width = 480, 640
    output_image = np.zeros((height, width, 3), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)

    # Camera intrinsics (assume same for all cameras for now)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)

    # Process each source image
    for img_name, pose in camera_poses.items():
        if img_name not in source_images:
            continue

        print(f"   Processing {img_name}...")
        source_img = source_images[img_name]

        # Calculate relative pose from source to target
        # This determines how to warp the source image

        # For simplicity, let's start with a basic approach:
        # 1. Project 3D points to source image
        # 2. Project same 3D points to target view
        # 3. Create correspondences for warping

        # Source camera projection matrix
        R_src = pose["rotation"]
        t_src = pose["translation"]
        P_src = K @ np.hstack([R_src, t_src.reshape(-1, 1)])

        # Target camera projection matrix
        P_target = K @ np.hstack([target_rotation, target_position.reshape(-1, 1)])

        # Project 3D points to both views
        points_3d_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])

        # Project to source image
        src_proj = (P_src @ points_3d_homo.T).T
        src_proj = src_proj[:, :2] / src_proj[:, 2:3]  # Normalize

        # Project to target view
        target_proj = (P_target @ points_3d_homo.T).T
        target_proj = target_proj[:, :2] / target_proj[:, 2:3]  # Normalize

        # Filter valid projections (within image bounds)
        h_src, w_src = source_img.shape[:2]
        valid_src = (
            (src_proj[:, 0] >= 0)
            & (src_proj[:, 0] < w_src)
            & (src_proj[:, 1] >= 0)
            & (src_proj[:, 1] < h_src)
        )

        valid_target = (
            (target_proj[:, 0] >= 0)
            & (target_proj[:, 0] < width)
            & (target_proj[:, 1] >= 0)
            & (target_proj[:, 1] < height)
        )

        valid_both = valid_src & valid_target

        if np.sum(valid_both) < 10:
            print(f"     ⚠️  Only {np.sum(valid_both)} valid correspondences")
            continue

        # Get valid correspondences
        src_points = src_proj[valid_both]
        target_points = target_proj[valid_both]

        print(f"     ✅ {len(src_points)} valid correspondences")

        # For each valid target pixel, sample from source image
        for i in range(len(target_points)):
            tx, ty = target_points[i]
            sx, sy = src_points[i]

            # Convert to integer pixel coordinates
            tx, ty = int(round(tx)), int(round(ty))
            sx, sy = int(round(sx)), int(round(sy))

            if 0 <= tx < width and 0 <= ty < height:
                if 0 <= sx < w_src and 0 <= sy < h_src:
                    # Sample source pixel and add to output
                    pixel_value = source_img[sy, sx].astype(np.float32)
                    output_image[ty, tx] += pixel_value
                    weight_map[ty, tx] += 1.0

    # Normalize by weights
    valid_mask = weight_map > 0
    output_image[valid_mask] /= weight_map[valid_mask, np.newaxis]

    # Fill empty regions with background color
    empty_mask = weight_map == 0
    output_image[empty_mask] = [50, 50, 50]  # Dark gray background

    # Convert to uint8
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    # Show statistics
    filled_pixels = np.sum(valid_mask)
    total_pixels = height * width
    fill_percentage = (filled_pixels / total_pixels) * 100

    print("   📊 Synthesis result:")
    print(
        f"      Filled pixels: {filled_pixels}/{total_pixels} ({fill_percentage:.1f}%)"
    )

    return output_image


def test_view_synthesis():
    """Test view synthesis with COLMAP data"""

    print("🎬 VIEW SYNTHESIS TEST")
    print("=" * 22)

    # Load COLMAP reconstruction
    camera_poses, points_3d, point_colors = load_colmap_reconstruction()
    if camera_poses is None:
        return False

    # Load source images
    source_images = load_source_images()
    if not source_images:
        return False

    # Apply our working coordinate transformation to points
    center = points_3d.mean(axis=0)
    points_transformed = (points_3d - center) * 0.2 + np.array([0, 0, 8])

    print("\n🔧 Applied coordinate transformation")
    print(
        f"   Point range: {points_transformed.min(axis=0)} to {points_transformed.max(axis=0)}"
    )

    # Test synthesis from our working baseline view
    target_position = np.array([0, 0, 0])  # Camera at origin
    target_rotation = np.eye(3)  # Looking down +Z

    print("\n🎯 Testing view synthesis from baseline position...")

    synthesized_image = synthesize_view(
        target_position,
        target_rotation,
        camera_poses,
        source_images,
        points_transformed,
    )

    # Save result
    cv2.imwrite("view_synthesis_test.jpg", synthesized_image)
    print("💾 Saved result: view_synthesis_test.jpg")

    return True


if __name__ == "__main__":
    success = test_view_synthesis()

    if success:
        print("\n🎉 VIEW SYNTHESIS IMPLEMENTED")
        print("✅ Using sparse 3D points for correspondences")
        print("✅ Projecting source image pixels to new view")
        print("✅ Blending multiple source contributions")

        print("\n📋 Next steps if results are insufficient:")
        print("1. Generate dense depth maps from COLMAP")
        print("2. Implement proper multi-view stereo blending")
        print("3. Add view-dependent texture filtering")
    else:
        print("\n❌ Setup issues - check COLMAP output and source images")
