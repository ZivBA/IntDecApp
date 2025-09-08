#!/usr/bin/env python3
"""
Dense depth estimation using OpenCV stereo matching instead of MiDaS
"""

import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import pycolmap
from itertools import combinations


def load_colmap_camera_poses():
    """Load COLMAP camera poses for stereo matching"""

    print("📷 LOADING CAMERA POSES FROM COLMAP")
    print("=" * 35)

    try:
        reconstruction_path = Path("colmap_output/sparse/0")
        reconstruction = pycolmap.Reconstruction(str(reconstruction_path))

        camera_poses = {}
        for image_id, image in reconstruction.images.items():
            cam_from_world = image.cam_from_world().matrix()
            camera_poses[image.name] = {
                "R": cam_from_world[:3, :3],
                "t": cam_from_world[:3, 3],
                "center": image.projection_center(),
                "image_id": image_id,
            }

        print(f"✅ Loaded {len(camera_poses)} camera poses")
        return camera_poses

    except Exception as e:
        print(f"❌ Failed to load camera poses: {e}")
        return {}


def compute_stereo_depth_opencv(img1, img2, baseline_distance, focal_length):
    """Compute depth using OpenCV stereo matching"""

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create stereo matcher
    # Use StereoBM for speed, StereoSGBM for quality
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,  # Must be divisible by 16
        blockSize=11,
        P1=8 * 3 * 11**2,  # Controls smoothness
        P2=32 * 3 * 11**2,  # Controls smoothness
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )

    # Compute disparity
    disparity = stereo.compute(gray1, gray2)

    # Convert to depth
    # depth = (baseline * focal_length) / disparity
    depth_map = np.zeros_like(disparity, dtype=np.float32)
    mask = disparity > 0
    depth_map[mask] = (baseline_distance * focal_length) / (
        disparity[mask] / 16.0
    )  # Disparity is in fixed point

    # Filter unrealistic depths
    depth_map[depth_map > 50] = 0  # Max 50 meters
    depth_map[depth_map < 0.1] = 0  # Min 10 cm

    return depth_map, disparity


def create_dense_points_from_stereo_pairs(
    image_dir="test_data2", max_points_total=20000
):
    """Create dense point cloud using stereo matching between image pairs"""

    print("🔍 CREATING DENSE POINTS FROM STEREO PAIRS")
    print("=" * 45)

    # Load camera poses
    camera_poses = load_colmap_camera_poses()
    if not camera_poses:
        return None

    # Load images
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG"))

    # Load images into memory
    images = {}
    for img_path in sorted(image_files):
        if img_path.name in camera_poses:
            img = cv2.imread(str(img_path))
            if img is not None:
                # Resize to reasonable size for processing
                h, w = img.shape[:2]
                if w > 800:
                    scale = 800 / w
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img = cv2.resize(img, (new_w, new_h))
                images[img_path.name] = img

    print(f"📸 Loaded {len(images)} images with poses")

    # Camera matrix (from COLMAP, scaled for 800px width)
    K = np.array(
        [
            [1279.2 * 800 / 4032, 0, 400],  # fx scaled, cx
            [0, 1279.2 * 800 / 4032, 533],  # fy scaled, cy
            [0, 0, 1],
        ]
    )

    all_dense_points = []
    all_dense_colors = []
    successful_pairs = 0

    # Process stereo pairs
    image_names = list(images.keys())
    total_pairs = len(list(combinations(image_names, 2)))
    print(f"🔄 Processing {total_pairs} stereo pairs...")

    for i, (name1, name2) in enumerate(combinations(image_names, 2)):
        if (
            len(all_dense_points) * 1000 > max_points_total
        ):  # Early stop if we have enough
            print(f"✋ Stopping early - reached {max_points_total:,} point target")
            break

        print(f"\n🔍 Pair {i + 1}/{total_pairs}: {name1} ↔ {name2}")

        try:
            # Get camera poses
            pose1 = camera_poses[name1]
            pose2 = camera_poses[name2]

            # Calculate baseline (distance between cameras)
            baseline = np.linalg.norm(pose1["center"] - pose2["center"])

            if baseline < 0.1:
                print(f"   ⏩ Skipping - baseline too small: {baseline:.3f}m")
                continue

            if baseline > 5.0:
                print(f"   ⏩ Skipping - baseline too large: {baseline:.3f}m")
                continue

            print(f"   📏 Baseline: {baseline:.2f}m")

            # Get images
            img1 = images[name1]
            img2 = images[name2]

            # Compute stereo depth
            focal_length = K[0, 0]  # fx
            depth_map, disparity = compute_stereo_depth_opencv(
                img1, img2, baseline, focal_length
            )

            # Count valid depth pixels
            valid_pixels = np.sum(depth_map > 0)

            if valid_pixels < 1000:
                print(f"   ⚠️  Too few valid pixels: {valid_pixels}")
                continue

            print(f"   ✅ Valid depth pixels: {valid_pixels:,}")

            # Create 3D points from depth map
            h, w = depth_map.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))

            # Valid depth mask
            valid_mask = depth_map > 0

            if np.sum(valid_mask) == 0:
                continue

            # Get valid coordinates
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]
            depth_valid = depth_map[valid_mask]

            # Backproject to 3D (in camera 1 coordinate system)
            x = (u_valid - K[0, 2]) * depth_valid / K[0, 0]
            y = (v_valid - K[1, 2]) * depth_valid / K[1, 1]
            z = depth_valid

            # Transform to world coordinates using camera 1 pose
            points_cam = np.stack([x, y, z], axis=1)

            # Transform from camera to world coordinates
            R_inv = pose1["R"].T  # Inverse rotation
            t = pose1["t"]
            points_world = (R_inv @ points_cam.T).T - (R_inv @ t)

            # Get corresponding colors from image 1
            colors = img1[v_valid, u_valid] / 255.0  # Normalize to [0,1]

            # Subsample if too many points
            max_points_per_pair = min(
                5000, max_points_total // 10
            )  # Reasonable limit per pair
            if len(points_world) > max_points_per_pair:
                indices = np.random.choice(
                    len(points_world), max_points_per_pair, replace=False
                )
                points_world = points_world[indices]
                colors = colors[indices]
                print(f"   📉 Subsampled to {max_points_per_pair:,} points")

            all_dense_points.append(points_world)
            all_dense_colors.append(colors)
            successful_pairs += 1

            # Save disparity visualization for this pair
            disp_vis = cv2.normalize(
                disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            cv2.imwrite(f"disparity_{name1}_{name2}.jpg", disp_vis)

            print(f"   💾 Added {len(points_world):,} points to collection")

        except Exception as e:
            print(f"   ❌ Failed: {e}")

    if not all_dense_points:
        print("❌ No stereo pairs produced valid depth maps")
        return None

    # Combine all point clouds
    combined_points = np.vstack(all_dense_points)
    combined_colors = np.vstack(all_dense_colors)

    print("\n📊 STEREO DEPTH RESULTS:")
    print(f"   Successful pairs: {successful_pairs}/{total_pairs}")
    print(f"   Total points: {len(combined_points):,}")
    print(f"   vs COLMAP sparse: {len(combined_points) / 515:.0f}x improvement")

    # Save combined point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)

    dense_path = "dense_points_stereo.ply"
    o3d.io.write_point_cloud(dense_path, pcd)
    print(f"💾 Saved dense stereo cloud: {dense_path}")

    return {
        "points": combined_points,
        "colors": combined_colors,
        "total_points": len(combined_points),
        "successful_pairs": successful_pairs,
        "ply_path": dense_path,
    }


def test_opencv_stereo_depth():
    """Test OpenCV stereo depth estimation"""

    print("🧪 TESTING OPENCV STEREO DEPTH")
    print("=" * 30)

    result = create_dense_points_from_stereo_pairs(max_points_total=20000)

    if result:
        points_count = result["total_points"]
        print("\n🎉 OPENCV STEREO SUCCESS!")
        print(f"✅ Generated {points_count:,} dense points")
        print(f"📈 Improvement: {points_count / 515:.0f}x vs sparse COLMAP")

        if points_count >= 15000:
            print("🚀 Excellent density for view synthesis!")
            return True, result
        elif points_count >= 5000:
            print("✅ Good density - should improve view synthesis significantly")
            return True, result
        else:
            print("⚠️  Moderate density - may need parameter tuning")
            return True, result
    else:
        print("❌ OpenCV stereo depth failed")
        return False, None


if __name__ == "__main__":
    success, result = test_opencv_stereo_depth()

    if success:
        print("\n🎯 READY FOR ENHANCED VIEW SYNTHESIS:")
        print(f"📊 Dense points: {result['total_points']:,}")
        print("🔄 Next: Apply coordinate transformation and test view synthesis")
        print(
            f"🎨 Expected: Much better visual quality with {result['total_points'] / 515:.0f}x more geometry"
        )
    else:
        print("\n🔧 NEXT STEPS:")
        print("1. Check stereo pair quality and baseline distances")
        print("2. Tune OpenCV stereo parameters for room scenes")
        print("3. Consider alternative dense reconstruction approaches")
