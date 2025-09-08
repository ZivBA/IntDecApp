#!/usr/bin/env python3
"""
Dense reconstruction pipeline combining COLMAP stereo with single-image depth
"""

import numpy as np
import cv2
import open3d as o3d
import pycolmap
import os
from pathlib import Path
import shutil


def run_colmap_dense_stereo(
    image_dir="test_data2", output_dir="colmap_dense", max_image_size=800
):
    """Run COLMAP dense stereo reconstruction"""

    print("🏗️ COLMAP DENSE STEREO RECONSTRUCTION")
    print("=" * 40)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Use existing sparse reconstruction if available
    sparse_path = Path("colmap_output/sparse/0")
    if not sparse_path.exists():
        print("❌ Sparse reconstruction not found. Run sparse reconstruction first.")
        return False, None

    # Setup dense reconstruction paths
    work_dir = output_path / "images"
    work_dir.mkdir(exist_ok=True)

    # Copy and resize images for dense reconstruction
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith(".jpg") and not f.endswith(".Zone.Identifier")
    ]

    print(f"📸 Preparing {len(image_files)} images for dense reconstruction...")

    for filename in image_files:
        src = os.path.join(image_dir, filename)
        dst = work_dir / filename

        img = cv2.imread(src)
        h, w = img.shape[:2]

        # Resize for dense reconstruction (balance quality vs speed)
        if w > max_image_size:
            scale = max_image_size / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            print(f"   📏 {filename}: {w}x{h} -> {new_w}x{new_h}")
        else:
            print(f"   ✅ {filename}: {w}x{h} (no resize needed)")

        cv2.imwrite(str(dst), img)

    try:
        # Copy sparse reconstruction
        dense_sparse_path = output_path / "sparse"
        if dense_sparse_path.exists():
            shutil.rmtree(dense_sparse_path)
        shutil.copytree(sparse_path.parent, dense_sparse_path)

        # Run dense reconstruction stages
        dense_output = output_path / "dense"
        dense_output.mkdir(exist_ok=True)

        print("\n1️⃣ Undistorting images...")
        pycolmap.undistort_images(
            image_path=str(work_dir),
            input_path=str(dense_sparse_path / "0"),
            output_path=str(dense_output),
            output_type="COLMAP",
        )

        print("2️⃣ Computing stereo depth maps...")
        # Use default options for initial testing
        pycolmap.patch_match_stereo(
            workspace_path=str(dense_output),
            workspace_format="COLMAP",
            pmvs_option_name="option-all",
        )

        print("3️⃣ Fusing depth maps into point cloud...")
        # Use default options for fusion as well
        pycolmap.stereo_fusion(
            workspace_path=str(dense_output),
            workspace_format="COLMAP",
            input_type="geometric",
            output_path=str(dense_output / "fused.ply"),
        )

        # Load and analyze dense point cloud
        dense_ply_path = dense_output / "fused.ply"
        if dense_ply_path.exists():
            pcd = o3d.io.read_point_cloud(str(dense_ply_path))
            dense_points = len(pcd.points)

            print("\n🎉 DENSE RECONSTRUCTION SUCCESS!")
            print("   📊 Point count comparison:")
            print("      Sparse COLMAP: 515 points")
            print(f"      Dense stereo:  {dense_points:,} points")
            print(f"      Improvement:   {dense_points / 515:.1f}x denser!")

            if dense_points >= 5000:
                print("   ✅ Excellent density for view synthesis")
                density_level = "excellent"
            elif dense_points >= 1000:
                print("   ✅ Good density - should improve view synthesis")
                density_level = "good"
            else:
                print("   ⚠️  Low density - may need parameter tuning")
                density_level = "low"

            return True, {
                "dense_points": dense_points,
                "density_level": density_level,
                "ply_path": str(dense_ply_path),
                "workspace_path": str(dense_output),
            }
        else:
            print("❌ Dense reconstruction failed - no output point cloud")
            return False, None

    except Exception as e:
        print(f"❌ Dense reconstruction error: {e}")
        return False, None


def analyze_dense_point_cloud(ply_path, coordinate_transform=True):
    """Analyze dense point cloud and prepare for view synthesis"""

    print("\n🔍 ANALYZING DENSE POINT CLOUD")
    print("=" * 32)

    try:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        print("📊 Dense point cloud statistics:")
        print(f"   Total points: {len(points):,}")
        print(f"   X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"   Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"   Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

        if coordinate_transform:
            # Apply our working coordinate transformation
            center = points.mean(axis=0)
            points_transformed = (points - center) * 0.2 + np.array([0, 0, 8])

            print("\n🔧 Applied working transformation:")
            print(
                f"   New Z range: [{points_transformed[:, 2].min():.2f}, {points_transformed[:, 2].max():.2f}]"
            )

            # Save transformed dense cloud
            pcd_transformed = o3d.geometry.PointCloud()
            pcd_transformed.points = o3d.utility.Vector3dVector(points_transformed)
            pcd_transformed.colors = pcd.colors

            dense_transformed_path = "dense_points_transformed.ply"
            o3d.io.write_point_cloud(dense_transformed_path, pcd_transformed)
            print(f"💾 Saved transformed dense cloud: {dense_transformed_path}")

            return points_transformed, colors, dense_transformed_path
        else:
            return points, colors, ply_path

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None, None, None


def test_dense_reconstruction():
    """Test complete dense reconstruction pipeline"""

    print("🚀 DENSE RECONSTRUCTION TEST")
    print("=" * 30)

    # Run dense reconstruction
    success, dense_info = run_colmap_dense_stereo()

    if not success:
        print("❌ Dense reconstruction failed")
        return False

    # Analyze results
    points, colors, transformed_path = analyze_dense_point_cloud(dense_info["ply_path"])

    if points is None:
        print("❌ Point cloud analysis failed")
        return False

    print("\n✅ DENSE RECONSTRUCTION COMPLETE")
    print(
        f"📈 Density improvement: {dense_info['dense_points']:,} points vs 515 sparse"
    )
    print("🎯 Ready for enhanced view synthesis testing")

    return True, dense_info


if __name__ == "__main__":
    success = test_dense_reconstruction()

    if success:
        print("\n🎯 NEXT STEPS:")
        print("1. Test view synthesis with dense point cloud")
        print("2. Compare visual quality vs sparse reconstruction")
        print("3. Validate identifiable content in synthesized views")
        print("4. Add single-image depth estimation for hybrid approach")
    else:
        print("\n🔧 TROUBLESHOOTING NEEDED:")
        print("1. Check COLMAP installation and dependencies")
        print("2. Verify sparse reconstruction quality")
        print("3. Review image quality and overlap")
