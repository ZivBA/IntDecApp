#!/usr/bin/env python3
"""
Fast multi-processed photo validation
"""

import cv2
import numpy as np
import os
from typing import Dict, Tuple
from multiprocessing import Pool, cpu_count
import time


def analyze_single_photo(args: Tuple[str, str, int]) -> Dict:
    """Analyze individual photo quality - optimized for multiprocessing"""
    image_dir, filename, index = args

    image_path = os.path.join(image_dir, filename)
    img = cv2.imread(image_path)

    if img is None:
        return {"filename": filename, "index": index, "error": "Could not load image"}

    # Resize for faster processing
    h, w = img.shape[:2]
    if w > 1024:
        scale = 1024 / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Fast blur detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_sharp = laplacian_var > 100

    # Fast feature detection (reduced features for speed)
    sift = cv2.SIFT_create(nfeatures=500)  # Reduced from 2000
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    n_features = len(keypoints)

    # Basic exposure metrics
    mean_brightness = gray.mean()

    # Quick quality scoring
    quality_score = 0

    # Sharpness (0-3)
    if laplacian_var > 200:
        quality_score += 3
    elif laplacian_var > 100:
        quality_score += 2
    elif laplacian_var > 50:
        quality_score += 1

    # Features (0-3)
    if n_features > 300:
        quality_score += 3
    elif n_features > 150:
        quality_score += 2
    elif n_features > 50:
        quality_score += 1

    # Exposure (0-2)
    if 50 < mean_brightness < 200:
        quality_score += 2
    elif 30 < mean_brightness < 220:
        quality_score += 1

    # Resolution (0-2)
    if w >= 1024 and h >= 768:
        quality_score += 2
    elif w >= 640 and h >= 480:
        quality_score += 1

    return {
        "filename": filename,
        "index": index,
        "quality_score": quality_score,
        "sharp": is_sharp,
        "blur_score": laplacian_var,
        "features": n_features,
        "brightness": mean_brightness,
        "resolution": (img.shape[1], img.shape[0]),
        "descriptors": descriptors,  # Keep for overlap checking
    }


def check_overlap_fast(photo1: Dict, photo2: Dict) -> float:
    """Fast overlap check between two photos"""
    desc1 = photo1.get("descriptors")
    desc2 = photo2.get("descriptors")

    if desc1 is None or desc2 is None or len(desc1) < 10 or len(desc2) < 10:
        return 0.0

    # Fast matching
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc1, desc2, k=2)

    # Quick ratio test
    good_matches = []
    for match_pair in matches[: min(len(matches), 100)]:  # Limit for speed
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:  # Slightly relaxed
                good_matches.append(m)

    # Overlap ratio
    return len(good_matches) / min(len(desc1), len(desc2))


def validate_photos_fast(image_dir: str) -> Dict:
    """Fast multi-processed photo validation"""

    print("🚀 FAST PHOTO VALIDATION")
    print("=" * 40)

    start_time = time.time()

    # Find image files
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.endswith((".jpg", ".jpeg", ".png")) and not f.endswith(".Zone.Identifier")
    ]

    if not image_files:
        return {"valid": False, "error": "No images found"}

    print(f"📸 Processing {len(image_files)} images...")

    # Prepare arguments for multiprocessing
    args = [(image_dir, filename, i) for i, filename in enumerate(image_files)]

    # Process images in parallel
    n_processes = min(cpu_count(), len(image_files))
    print(f"⚡ Using {n_processes} processes")

    with Pool(processes=n_processes) as pool:
        photo_results = pool.map(analyze_single_photo, args)

    # Filter out errors
    valid_results = [r for r in photo_results if "error" not in r]
    error_results = [r for r in photo_results if "error" in r]

    if error_results:
        print(f"⚠️  Could not process {len(error_results)} images")

    analysis_time = time.time() - start_time
    print(f"⏱️  Analysis completed in {analysis_time:.1f}s")

    # Quick scoring
    results = {
        "valid": False,
        "total_photos": len(image_files),
        "processed_photos": len(valid_results),
        "issues": [],
        "recommendations": [],
        "photos": valid_results,
        "processing_time": analysis_time,
    }

    if not valid_results:
        results["issues"].append("❌ No valid images to process")
        return results

    # Quick assessments
    scores = []

    # 1. Quantity (0-25 points)
    n_photos = len(valid_results)
    if n_photos >= 15:
        quantity_score = 25
    elif n_photos >= 10:
        quantity_score = 20
        results["issues"].append("⚠️  More photos recommended (15+ for best results)")
    elif n_photos >= 5:
        quantity_score = 15
        results["issues"].append("⚠️  Minimum photos met, but more would help")
    else:
        quantity_score = 5
        results["issues"].append("❌ Too few photos for good reconstruction")
    scores.append(quantity_score)

    # 2. Individual quality (0-25 points)
    avg_quality = np.mean([p["quality_score"] for p in valid_results])
    blurry_count = len([p for p in valid_results if not p["sharp"]])

    if avg_quality >= 8:
        quality_score = 25
    elif avg_quality >= 6:
        quality_score = 20
    elif avg_quality >= 4:
        quality_score = 15
    else:
        quality_score = 10

    if blurry_count > 0:
        results["issues"].append(f"⚠️  {blurry_count} blurry photos detected")
        quality_score = max(quality_score - 5, 0)

    scores.append(quality_score)

    # 3. Fast overlap check (0-25 points) - sample only
    overlap_score = 25  # Assume good overlap unless proven otherwise
    if len(valid_results) >= 2:
        # Sample a few pairs for speed
        sample_pairs = min(3, len(valid_results) - 1)
        overlaps = []

        for i in range(sample_pairs):
            overlap = check_overlap_fast(valid_results[i], valid_results[i + 1])
            overlaps.append(overlap)

        if overlaps:
            avg_overlap = np.mean(overlaps)
            if avg_overlap < 0.2:
                overlap_score = 10
                results["issues"].append("❌ Very low overlap detected")
            elif avg_overlap < 0.4:
                overlap_score = 15
                results["issues"].append(
                    "⚠️  Low overlap - consider smaller steps between photos"
                )

    scores.append(overlap_score)

    # 4. Coverage diversity (0-25 points)
    feature_counts = [p["features"] for p in valid_results]
    brightness_values = [p["brightness"] for p in valid_results]

    feature_variety = np.std(feature_counts)
    brightness_variety = np.std(brightness_values)

    if feature_variety > 50 and brightness_variety > 20:
        coverage_score = 25
    elif feature_variety > 30 and brightness_variety > 15:
        coverage_score = 20
    else:
        coverage_score = 15
        results["issues"].append(
            "⚠️  Limited variety in photos - consider different angles"
        )

    scores.append(coverage_score)

    # Final scoring
    total_score = sum(scores)

    print("\n📊 QUICK ASSESSMENT:")
    print(f"   Quantity: {scores[0]}/25")
    print(f"   Quality:  {scores[1]}/25")
    print(f"   Overlap:  {scores[2]}/25")
    print(f"   Coverage: {scores[3]}/25")
    print(f"   TOTAL:    {total_score}/100")

    # Verdict
    if total_score >= 70:
        results["valid"] = True
        print("✅ EXCELLENT - Ready for high-quality reconstruction!")
    elif total_score >= 50:
        results["valid"] = True
        print("✅ GOOD - Should produce decent reconstruction")
    elif total_score >= 30:
        print("⚠️  MARGINAL - May work but results will be limited")
    else:
        print("❌ POOR - Consider retaking photos")

    # Quick recommendations
    if n_photos < 15:
        results["recommendations"].append("📸 Take more photos in a systematic circle")
    if blurry_count > 1:
        results["recommendations"].append("📱 Hold phone steadier, tap to focus")
    if scores[2] < 20:
        results["recommendations"].append(
            "🔄 Smaller steps between photos for better overlap"
        )

    return results


def main():
    """Main function with command line support"""
    import sys

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "test_data"

    if not os.path.exists(folder):
        print(f"❌ Folder '{folder}' not found")
        return

    results = validate_photos_fast(folder)

    if results.get("issues"):
        print("\n📋 ISSUES:")
        for issue in results["issues"]:
            print(f"  {issue}")

    if results.get("recommendations"):
        print("\n💡 RECOMMENDATIONS:")
        for rec in results["recommendations"]:
            print(f"  {rec}")

    print(f"\n⏱️  Total processing time: {results.get('processing_time', 0):.1f}s")

    return results


if __name__ == "__main__":
    main()
