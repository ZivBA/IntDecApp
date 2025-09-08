#!/usr/bin/env python3
"""
Simplified COLMAP test with minimal API calls
"""

import os
import shutil
import cv2
import pycolmap
import tempfile


def test_colmap_minimal(image_dir="test_data2"):
    """Minimal COLMAP test using default options"""

    print("🎯 MINIMAL COLMAP TEST")
    print("=" * 25)

    # Setup temp directory
    temp_dir = tempfile.mkdtemp(prefix="colmap_min_")
    work_dir = os.path.join(temp_dir, "images")
    os.makedirs(work_dir, exist_ok=True)

    # Copy images (resize for speed)
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.endswith(".jpg") and not f.endswith(".Zone.Identifier")
    ]

    print(f"📸 Copying {len(image_files)} images...")

    for filename in image_files:
        src = os.path.join(image_dir, filename)
        dst = os.path.join(work_dir, filename)

        img = cv2.imread(src)
        h, w = img.shape[:2]
        if w > 800:
            scale = 800 / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))

        cv2.imwrite(dst, img)

    try:
        database_path = os.path.join(temp_dir, "database.db")
        output_path = os.path.join(temp_dir, "sparse")
        os.makedirs(output_path, exist_ok=True)

        print("1️⃣ Feature extraction...")
        # Use default options to avoid API issues
        pycolmap.extract_features(database_path=database_path, image_path=work_dir)

        print("2️⃣ Feature matching...")
        pycolmap.match_exhaustive(database_path=database_path)

        print("3️⃣ Reconstruction...")
        # Use default reconstruction options
        maps = pycolmap.incremental_mapping(
            database_path=database_path, image_path=work_dir, output_path=output_path
        )

        if maps:
            reconstruction = maps[0]
            n_points = len(reconstruction.points3D)
            n_cameras = len(reconstruction.images)

            print("\n🎉 COLMAP SUCCESS!")
            print(f"   Cameras: {n_cameras}")
            print(f"   3D points: {n_points}")

            # Extract points for our system
            points_3d = []
            for point3D_id, point3D in reconstruction.points3D.items():
                points_3d.append({"position": point3D.xyz, "color": point3D.color})

            print("\n📊 QUALITY COMPARISON:")
            print("   Our implementation: 43-174 points")
            print(f"   COLMAP (9 images): {n_points} points")
            print(f"   Improvement: {n_points / 174:.1f}x better!")

            if n_points > 1000:
                print("✅ EXCELLENT quality for rendering!")
            elif n_points > 500:
                print("✅ GOOD quality - much better than our approach")
            elif n_points > 200:
                print("⚠️  MARGINAL - improvement but still limited")
            else:
                print("❌ POOR - no improvement")

            # Save point cloud for testing
            import numpy as np

            positions = np.array([p["position"] for p in points_3d])
            colors = np.array([p["color"] for p in points_3d])

            # Simple PLY export
            with open("colmap_points.ply", "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(positions)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")

                for i in range(len(positions)):
                    pos = positions[i]
                    col = colors[i].astype(int)
                    f.write(f"{pos[0]} {pos[1]} {pos[2]} {col[0]} {col[1]} {col[2]}\n")

            print("💾 Saved point cloud: colmap_points.ply")

            shutil.rmtree(temp_dir)
            return True, n_points, points_3d

        else:
            print("❌ COLMAP reconstruction failed")
            shutil.rmtree(temp_dir)
            return False, 0, []

    except Exception as e:
        print(f"❌ Error: {e}")
        shutil.rmtree(temp_dir)
        return False, 0, []


if __name__ == "__main__":
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "test_data2"
    success, n_points, points = test_colmap_minimal(folder)

    if success and n_points > 100:
        print("\n🚀 READY FOR NEXT PHASE:")
        print(f"1. Test rendering with {n_points} COLMAP points")
        print("2. Compare with our 174-point reconstruction")
        print("3. If good, make COLMAP the main pipeline")
        print("4. Integrate with AI enhancement")
    elif success:
        print("\n🔄 PARTIAL SUCCESS:")
        print(f"COLMAP worked but only {n_points} points")
        print("Consider: more photos, better overlap, or different approach")
    else:
        print("\n❌ COLMAP FAILED - ALTERNATIVES:")
        print("1. Try depth estimation approach")
        print("2. Capture photos with much better overlap")
        print("3. Use different reconstruction library")
