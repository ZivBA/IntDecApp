#!/usr/bin/env python3
"""
Run COLMAP and save both points and camera poses for view synthesis
"""

import os
import cv2
import pycolmap
import numpy as np
from pathlib import Path


def run_colmap_full_reconstruction(image_dir="test_data2", output_dir="colmap_output"):
    """Run COLMAP and save complete reconstruction including camera poses"""

    print("📷 COLMAP FULL RECONSTRUCTION")
    print("=" * 30)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Setup working directory
    work_dir = output_path / "images"
    work_dir.mkdir(exist_ok=True)

    # Copy and resize images
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith(".jpg") and not f.endswith(".Zone.Identifier")
    ]

    print(f"📸 Processing {len(image_files)} images...")

    for filename in image_files:
        src = os.path.join(image_dir, filename)
        dst = work_dir / filename

        img = cv2.imread(src)
        h, w = img.shape[:2]
        if w > 800:
            scale = 800 / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))

        cv2.imwrite(str(dst), img)
        print(f"   ✅ {filename}: {new_w}x{new_h}")

    try:
        database_path = str(output_path / "database.db")
        sparse_output = str(output_path / "sparse")
        os.makedirs(sparse_output, exist_ok=True)

        print("\n1️⃣ Feature extraction...")
        pycolmap.extract_features(database_path=database_path, image_path=str(work_dir))

        print("2️⃣ Feature matching...")
        pycolmap.match_exhaustive(database_path=database_path)

        print("3️⃣ Incremental mapping...")
        maps = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=str(work_dir),
            output_path=sparse_output,
        )

        if maps:
            reconstruction = maps[0]
            n_points = len(reconstruction.points3D)
            n_images = len(reconstruction.images)

            print("\n🎉 COLMAP SUCCESS!")
            print(f"   Reconstructed images: {n_images}")
            print(f"   3D points: {n_points}")

            # Save reconstruction in binary format (pycolmap can read this)
            reconstruction_output = Path(sparse_output) / "0"
            reconstruction_output.mkdir(exist_ok=True)
            reconstruction.write(str(reconstruction_output))

            print(f"💾 Saved reconstruction: {reconstruction_output}")

            # Also save as text files for inspection
            reconstruction.write_text(str(reconstruction_output))

            print("📄 Saved text files: cameras.txt, images.txt, points3D.txt")

            # Extract and save point cloud separately
            points_3d = []
            colors = []

            for point3D_id, point3D in reconstruction.points3D.items():
                points_3d.append(point3D.xyz)
                colors.append(point3D.color)

            points_3d = np.array(points_3d)
            colors = np.array(colors)

            # Save PLY file for our renderer
            with open("colmap_points.ply", "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points_3d)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")

                for i in range(len(points_3d)):
                    pos = points_3d[i]
                    col = colors[i].astype(int)
                    f.write(f"{pos[0]} {pos[1]} {pos[2]} {col[0]} {col[1]} {col[2]}\n")

            print("💾 Updated point cloud: colmap_points.ply")

            # Print camera pose summary
            print("\n📷 CAMERA POSES:")
            for image_id, image in reconstruction.images.items():
                R = image.rotation_matrix()
                t = image.translation
                cam_center = -R.T @ t
                print(
                    f"   {image.name}: center at [{cam_center[0]:.2f}, {cam_center[1]:.2f}, {cam_center[2]:.2f}]"
                )

            return True, reconstruction_output

        else:
            print("❌ COLMAP reconstruction failed")
            return False, None

    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None


if __name__ == "__main__":
    success, reconstruction_path = run_colmap_full_reconstruction()

    if success:
        print("\n✅ READY FOR VIEW SYNTHESIS!")
        print(f"📂 Reconstruction saved at: {reconstruction_path}")
        print("📄 Camera poses available in images.txt")
        print("🎯 3D points available in colmap_points.ply")
        print("\n🚀 Next: Run view synthesis test with camera poses")
    else:
        print("\n❌ COLMAP failed - cannot proceed with view synthesis")
