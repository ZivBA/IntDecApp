#!/usr/bin/env python3
"""
COLMAP-based 3D reconstruction - Professional quality
"""

import os
import shutil
import cv2
import numpy as np
import pycolmap
import open3d as o3d
import tempfile


def prepare_images_for_colmap(input_dir="test_data", max_size=1024):
    """Prepare images in COLMAP-friendly format"""
    # Create temp directory for COLMAP
    temp_dir = tempfile.mkdtemp(prefix="colmap_")
    image_dir = os.path.join(temp_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    print(f"📁 Working directory: {temp_dir}")

    # Copy and resize images
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])

    for i, filename in enumerate(image_files):
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(image_dir, f"image_{i:03d}.jpg")

        # Load and resize if needed
        img = cv2.imread(src_path)
        h, w = img.shape[:2]

        if w > max_size:
            scale = max_size / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  Resized {filename}: {w}x{h} -> {new_w}x{new_h}")

        cv2.imwrite(dst_path, img)
        print(f"  Copied {filename} -> {dst_path}")

    return temp_dir, image_dir, len(image_files)


def run_colmap_reconstruction(image_dir, output_dir):
    """Run COLMAP sparse reconstruction using pycolmap"""

    print("\n🔧 Running COLMAP reconstruction...")

    # Setup paths
    database_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    try:
        # Step 1: Feature extraction
        print("1️⃣ Extracting features...")
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_dir,
            camera_model="SIMPLE_PINHOLE",
            sift_options={
                "max_num_features": 8000,  # More features than default
                "num_threads": 4,
                "gpu_index": -1,  # CPU only for WSL compatibility
            },
        )

        # Step 2: Feature matching
        print("2️⃣ Matching features...")
        pycolmap.match_exhaustive(
            database_path=database_path,
            sift_options={
                "num_threads": 4,
                "gpu_index": -1,
                "max_ratio": 0.8,
                "max_distance": 0.7,
                "cross_check": True,
            },
        )

        # Step 3: Incremental reconstruction
        print("3️⃣ Running incremental reconstruction...")
        maps = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=image_dir,
            output_path=sparse_dir,
            options={
                "min_num_matches": 15,
                "init_min_num_inliers": 100,
                "abs_pose_min_num_inliers": 30,
                "abs_pose_min_inlier_ratio": 0.25,
                "ba_refine_focal_length": True,
                "ba_refine_principal_point": False,
                "ba_refine_extra_params": True,
            },
        )

        if not maps:
            print("❌ COLMAP reconstruction failed - no valid reconstructions")
            return None

        # Get the first (usually best) reconstruction
        reconstruction = maps[0]

        print("✅ Reconstruction successful!")
        print(f"   Cameras: {len(reconstruction.cameras)}")
        print(f"   Images: {len(reconstruction.images)}")
        print(f"   3D points: {len(reconstruction.points3D)}")

        return reconstruction

    except Exception as e:
        print(f"❌ COLMAP failed: {e}")
        return None


def convert_colmap_to_pointcloud(reconstruction):
    """Convert COLMAP reconstruction to Open3D point cloud"""

    print("\n🎨 Converting to point cloud...")

    # Extract points and colors
    points = []
    colors = []

    for point3D_id, point3D in reconstruction.points3D.items():
        points.append(point3D.xyz)
        colors.append(point3D.color / 255.0)  # Normalize to [0,1]

    points = np.array(points)
    colors = np.array(colors)

    print("📊 Point cloud statistics:")
    print(f"   Total points: {len(points)}")
    print(f"   Spatial extent: {points.max(axis=0) - points.min(axis=0)}")
    print(f"   Center: {points.mean(axis=0)}")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Remove statistical outliers
    print("🧹 Removing outliers...")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"   Removed {len(points) - len(pcd.points)} outliers")

    # Estimate normals
    print("📐 Estimating normals...")
    pcd.estimate_normals()

    return pcd


def save_results(pcd, reconstruction, output_base="colmap_output"):
    """Save reconstruction results"""

    print("\n💾 Saving results...")

    # Save point cloud
    ply_path = f"{output_base}.ply"
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"   Point cloud saved: {ply_path}")

    # Save camera poses
    poses_path = f"{output_base}_poses.txt"
    with open(poses_path, "w") as f:
        f.write("# Camera poses from COLMAP reconstruction\n")
        f.write("# Format: image_id, qw, qx, qy, qz, tx, ty, tz\n")

        for image_id, image in reconstruction.images.items():
            q = image.qvec  # Quaternion
            t = image.tvec  # Translation
            f.write(
                f"{image_id}, {q[0]}, {q[1]}, {q[2]}, {q[3]}, {t[0]}, {t[1]}, {t[2]}\n"
            )

    print(f"   Camera poses saved: {poses_path}")

    # Create visualization
    print("📊 Creating visualization...")

    # Downsample for visualization
    if len(pcd.points) > 10000:
        pcd_viz = pcd.uniform_down_sample(
            every_k_points=max(1, len(pcd.points) // 10000)
        )
    else:
        pcd_viz = pcd

    # Render point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd_viz)
    vis.poll_events()
    vis.update_renderer()

    # Capture image
    image = vis.capture_screen_float_buffer()
    vis.destroy_window()

    # Save as image
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.imshow(np.asarray(image))
    plt.axis("off")
    plt.title(f"COLMAP Reconstruction: {len(pcd.points)} points")
    plt.savefig(f"{output_base}_preview.png", dpi=150, bbox_inches="tight")
    print(f"   Preview saved: {output_base}_preview.png")

    return ply_path, poses_path


def main():
    """Main reconstruction pipeline using COLMAP"""

    print("🎯 COLMAP 3D RECONSTRUCTION")
    print("=" * 50)

    # Prepare images
    temp_dir, image_dir, n_images = prepare_images_for_colmap()
    print(f"✅ Prepared {n_images} images")

    # Run COLMAP
    reconstruction = run_colmap_reconstruction(image_dir, temp_dir)

    if reconstruction:
        # Convert to point cloud
        pcd = convert_colmap_to_pointcloud(reconstruction)

        # Save results
        ply_path, poses_path = save_results(pcd, reconstruction)

        print("\n" + "=" * 50)
        print("🎉 RECONSTRUCTION COMPLETE!")
        print("=" * 50)
        print(f"✅ Point cloud: {len(pcd.points)} points")
        print("✅ Files saved:")
        print(f"   - {ply_path}")
        print(f"   - {poses_path}")
        print("   - colmap_output_preview.png")

        # Cleanup temp directory
        shutil.rmtree(temp_dir)

        return pcd, reconstruction
    else:
        print("\n❌ Reconstruction failed")
        shutil.rmtree(temp_dir)
        return None, None


if __name__ == "__main__":
    pcd, reconstruction = main()

    if pcd:
        print("\n💡 Next steps:")
        print("1. View the point cloud: open3d colmap_output.ply")
        print("2. Test virtual navigation with the dense point cloud")
        print("3. Export for AI enhancement pipeline")
        print("4. Consider running COLMAP's dense reconstruction for even more detail")
