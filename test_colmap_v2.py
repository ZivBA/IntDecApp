#!/usr/bin/env python3
"""
COLMAP reconstruction with correct API usage
"""

import os
import shutil
import cv2
import numpy as np
import pycolmap
import open3d as o3d
import tempfile


def run_colmap_simple(image_dir="test_data"):
    """Simplified COLMAP reconstruction using pycolmap"""

    print("🎯 COLMAP RECONSTRUCTION V2")
    print("=" * 50)

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="colmap_")
    work_dir = os.path.join(temp_dir, "images")
    os.makedirs(work_dir, exist_ok=True)

    print(f"📁 Working directory: {temp_dir}")

    # Copy images (COLMAP needs them in a specific folder)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

    for filename in image_files:
        src = os.path.join(image_dir, filename)
        dst = os.path.join(work_dir, filename)

        # Resize if needed
        img = cv2.imread(src)
        h, w = img.shape[:2]
        if w > 1024:
            scale = 1024 / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            print(f"  Resized {filename}: {w}x{h} -> {new_w}x{new_h}")

        cv2.imwrite(dst, img)

    print(f"✅ Prepared {len(image_files)} images")

    # Setup paths
    database_path = os.path.join(temp_dir, "database.db")
    output_path = os.path.join(temp_dir, "sparse")
    os.makedirs(output_path, exist_ok=True)

    try:
        # Step 1: Extract features (simplified API)
        print("\n1️⃣ Extracting features...")
        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.max_num_features = 8000
        sift_options.num_threads = 4

        device = pycolmap.Device.auto  # Auto-detect best device

        pycolmap.extract_features(
            database_path=database_path,
            image_path=work_dir,
            sift_options=sift_options,
            device=device,
        )

        # Step 2: Match features
        print("2️⃣ Matching features...")
        matching_options = pycolmap.SiftMatchingOptions()
        matching_options.num_threads = 4

        pycolmap.match_exhaustive(
            database_path=database_path, sift_options=matching_options, device=device
        )

        # Step 3: Reconstruction
        print("3️⃣ Running reconstruction...")
        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.min_num_matches = 15

        maps = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=work_dir,
            output_path=output_path,
            options=mapper_options,
        )

        if not maps:
            print("❌ No valid reconstruction found")
            shutil.rmtree(temp_dir)
            return None

        # Get the best reconstruction
        reconstruction = maps[0]

        print("\n✅ RECONSTRUCTION SUCCESSFUL!")
        print(f"   Cameras: {len(reconstruction.cameras)}")
        print(f"   Images: {len(reconstruction.images)}")
        print(f"   3D points: {len(reconstruction.points3D)}")

        # Convert to point cloud
        points = []
        colors = []

        for point3D_id, point3D in reconstruction.points3D.items():
            points.append(point3D.xyz)
            colors.append(point3D.color / 255.0)

        points = np.array(points)
        colors = np.array(colors)

        print("\n📊 Point cloud statistics:")
        print(f"   Total points: {len(points)}")
        if len(points) > 0:
            print(f"   Spatial extent: {points.max(axis=0) - points.min(axis=0)}")
            print(f"   Center: {points.mean(axis=0)}")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save point cloud
        o3d.io.write_point_cloud("colmap_result.ply", pcd)
        print("\n💾 Point cloud saved: colmap_result.ply")

        # Save camera information
        with open("colmap_cameras.txt", "w") as f:
            f.write("# COLMAP camera poses\n")
            for img_id, img in reconstruction.images.items():
                f.write(f"Image {img_id}: {img.name}\n")
                f.write(f"  Position: {img.tvec}\n")
                f.write(f"  Rotation: {img.qvec}\n")

        print("📷 Camera poses saved: colmap_cameras.txt")

        # Cleanup
        shutil.rmtree(temp_dir)

        return pcd, reconstruction

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        shutil.rmtree(temp_dir)
        return None, None


def compare_with_our_implementation(colmap_pcd):
    """Compare COLMAP results with our implementation"""

    print("\n" + "=" * 50)
    print("📊 COMPARISON WITH OUR IMPLEMENTATION")
    print("=" * 50)

    print("Our implementation: 43-174 points")

    if colmap_pcd:
        colmap_points = len(colmap_pcd.points)
        print(f"COLMAP: {colmap_points} points")
        print(f"Improvement: {colmap_points / 174:.1f}x over our best attempt")

        if colmap_points > 1000:
            print("✅ COLMAP provides visual quality needed for AI enhancement")
        else:
            print("⚠️  Still sparse, but much better than our implementation")
    else:
        print("❌ COLMAP reconstruction failed")


if __name__ == "__main__":
    pcd, reconstruction = run_colmap_simple()

    if pcd and reconstruction:
        compare_with_our_implementation(pcd)

        print("\n💡 NEXT STEPS:")
        print("1. Load colmap_result.ply in our renderer")
        print("2. Test virtual navigation with COLMAP points")
        print("3. If still sparse, try COLMAP's dense reconstruction")
        print("4. Integrate with AI enhancement pipeline")
    else:
        print("\n❌ COLMAP failed - considering alternatives:")
        print("1. Try different COLMAP parameters")
        print("2. Use more images from different angles")
        print("3. Consider OpenMVG as alternative")
        print("4. Fall back to depth estimation approach")
