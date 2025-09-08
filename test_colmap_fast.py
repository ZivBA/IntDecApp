#!/usr/bin/env python3
"""
Fast COLMAP test with new images
"""

import os
import shutil
import cv2
import pycolmap
import tempfile
import time


def test_colmap_fast(image_dir="test_data2"):
    """Quick COLMAP test with minimal options"""

    print("🎯 FAST COLMAP TEST")
    print("=" * 30)

    start_time = time.time()

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="colmap_fast_")
    work_dir = os.path.join(temp_dir, "images")
    os.makedirs(work_dir, exist_ok=True)

    # Copy and resize images
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.endswith(".jpg") and not f.endswith(".Zone.Identifier")
    ]

    print(f"📸 Processing {len(image_files)} images...")

    for filename in image_files:
        src = os.path.join(image_dir, filename)
        dst = os.path.join(work_dir, filename)

        # Resize for faster processing
        img = cv2.imread(src)
        h, w = img.shape[:2]
        if w > 800:  # Smaller for speed
            scale = 800 / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))

        cv2.imwrite(dst, img)

    try:
        # Setup
        database_path = os.path.join(temp_dir, "database.db")
        output_path = os.path.join(temp_dir, "sparse")
        os.makedirs(output_path, exist_ok=True)

        print("1️⃣ Extracting features...")
        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.max_num_features = 4000  # Reduced for speed
        sift_options.num_threads = 4

        pycolmap.extract_features(
            database_path=database_path,
            image_path=work_dir,
            sift_options=sift_options,
            device=pycolmap.Device.auto,
        )

        print("2️⃣ Matching features...")
        matching_options = pycolmap.SiftMatchingOptions()
        matching_options.num_threads = 4
        matching_options.max_ratio = 0.8  # More lenient

        pycolmap.match_exhaustive(
            database_path=database_path,
            sift_options=matching_options,
            device=pycolmap.Device.auto,
        )

        print("3️⃣ Reconstructing...")
        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.min_num_matches = 10  # Reduced threshold
        mapper_options.init_min_num_inliers = 50  # Reduced threshold

        maps = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=work_dir,
            output_path=output_path,
            options=mapper_options,
        )

        processing_time = time.time() - start_time

        if maps:
            reconstruction = maps[0]
            n_points = len(reconstruction.points3D)
            n_cameras = len(reconstruction.images)

            print("\n✅ SUCCESS!")
            print(f"   Time: {processing_time:.1f}s")
            print(f"   Cameras: {n_cameras}")
            print(f"   3D points: {n_points}")

            # Quick comparison
            print("\n📊 COMPARISON:")
            print("   Our method: 43-174 points")
            print(f"   COLMAP: {n_points} points")
            print(f"   Improvement: {n_points / 174:.1f}x")

            if n_points > 500:
                print("🎉 EXCELLENT - This should give great results!")
            elif n_points > 100:
                print("✅ GOOD - Much better than our implementation")
            else:
                print("⚠️  Still limited, but progress!")

            # Cleanup
            shutil.rmtree(temp_dir)
            return True, n_points

        else:
            print(f"❌ COLMAP failed after {processing_time:.1f}s")
            shutil.rmtree(temp_dir)
            return False, 0

    except Exception as e:
        print(f"❌ Error: {e}")
        shutil.rmtree(temp_dir)
        return False, 0


if __name__ == "__main__":
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "test_data2"
    success, points = test_colmap_fast(folder)

    if success:
        print("\n💡 NEXT STEPS:")
        print(f"1. Test rendering with {points} COLMAP points")
        print("2. Compare visual quality with our approach")
        print("3. If good, integrate COLMAP as main pipeline")
    else:
        print("\n🔄 ALTERNATIVES:")
        print("1. Try with even more relaxed COLMAP settings")
        print("2. Use depth estimation approach instead")
        print("3. Capture more photos with better overlap")
