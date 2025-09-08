#!/usr/bin/env python3
"""
Photo validation tool - Check if images are suitable for 3D reconstruction
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple


class PhotoValidator:
    """Validate photo quality for 3D reconstruction"""

    def __init__(self):
        self.min_photos = 10
        self.min_overlap = 0.4  # 40% minimum overlap
        self.min_features = 100  # Minimum features per image
        self.blur_threshold = 100  # Laplacian variance threshold

    def validate_photo_set(self, image_dir: str) -> Dict:
        """Validate entire photo set for reconstruction"""

        print("📸 PHOTO VALIDATION FOR 3D RECONSTRUCTION")
        print("=" * 50)

        results = {
            "valid": False,
            "total_score": 0,
            "issues": [],
            "recommendations": [],
            "photos": [],
        }

        # Load images
        image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
        )

        if not image_files:
            results["issues"].append("❌ No images found in directory")
            return results

        print(f"Found {len(image_files)} images")

        # Test 1: Quantity check
        quantity_score, quantity_issues = self._check_quantity(len(image_files))
        results["total_score"] += quantity_score
        results["issues"].extend(quantity_issues)

        # Load and analyze images
        images = []
        photo_results = []

        for i, filename in enumerate(image_files):
            image_path = os.path.join(image_dir, filename)
            img = cv2.imread(image_path)

            if img is None:
                results["issues"].append(f"❌ Could not load {filename}")
                continue

            images.append(img)

            # Individual photo analysis
            photo_result = self._analyze_single_photo(img, filename, i)
            photo_results.append(photo_result)

            print(
                f"  {filename}: Quality={photo_result['quality_score']:.1f}/10, "
                f"Features={photo_result['features']}, "
                f"Blur={'OK' if photo_result['sharp'] else 'BLURRY'}"
            )

        results["photos"] = photo_results

        # Test 2: Individual quality
        quality_score, quality_issues = self._check_individual_quality(photo_results)
        results["total_score"] += quality_score
        results["issues"].extend(quality_issues)

        # Test 3: Feature matching between adjacent photos
        if len(images) >= 2:
            overlap_score, overlap_issues = self._check_overlap(images, image_files)
            results["total_score"] += overlap_score
            results["issues"].extend(overlap_issues)

        # Test 4: Overall coverage
        coverage_score, coverage_issues = self._check_coverage(images)
        results["total_score"] += coverage_score
        results["issues"].extend(coverage_issues)

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        # Final verdict
        max_possible = 40  # 4 tests × 10 points each
        final_score = results["total_score"] / max_possible * 100

        print("\n📊 OVERALL ASSESSMENT:")
        print(f"Score: {final_score:.1f}/100")

        if final_score >= 70:
            results["valid"] = True
            print("✅ EXCELLENT - Ready for high-quality reconstruction!")
        elif final_score >= 50:
            results["valid"] = True
            print("✅ GOOD - Should produce decent reconstruction")
        elif final_score >= 30:
            print("⚠️  MARGINAL - May work but results will be poor")
        else:
            print("❌ POOR - Unlikely to produce usable reconstruction")

        return results

    def _analyze_single_photo(self, img: np.ndarray, filename: str, index: int) -> Dict:
        """Analyze individual photo quality"""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_sharp = laplacian_var > self.blur_threshold

        # Feature detection
        sift = cv2.SIFT_create(nfeatures=2000)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        n_features = len(keypoints)

        # Exposure analysis
        mean_brightness = gray.mean()
        brightness_std = gray.std()

        # Quality scoring
        quality_score = 0

        # Sharpness (0-3 points)
        if laplacian_var > 200:
            quality_score += 3
        elif laplacian_var > self.blur_threshold:
            quality_score += 2
        elif laplacian_var > 50:
            quality_score += 1

        # Features (0-3 points)
        if n_features > 1000:
            quality_score += 3
        elif n_features > 500:
            quality_score += 2
        elif n_features > self.min_features:
            quality_score += 1

        # Exposure (0-2 points)
        if 50 < mean_brightness < 200 and brightness_std > 30:
            quality_score += 2
        elif 30 < mean_brightness < 220:
            quality_score += 1

        # Resolution (0-2 points)
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
            "resolution": (w, h),
        }

    def _check_quantity(self, n_photos: int) -> Tuple[int, List[str]]:
        """Check if enough photos provided"""
        issues = []

        if n_photos >= 20:
            score = 10
        elif n_photos >= 15:
            score = 8
            issues.append(
                "⚠️  Could use more photos for better coverage (20+ recommended)"
            )
        elif n_photos >= self.min_photos:
            score = 6
            issues.append(
                "⚠️  Minimum photo count met, but more would help (15+ recommended)"
            )
        else:
            score = 2
            issues.append(
                f"❌ Too few photos ({n_photos}). Need at least {self.min_photos}, recommend 20+"
            )

        return score, issues

    def _check_individual_quality(
        self, photo_results: List[Dict]
    ) -> Tuple[int, List[str]]:
        """Check quality of individual photos"""
        issues = []

        # Average quality
        avg_quality = np.mean([p["quality_score"] for p in photo_results])

        # Count problems
        blurry_photos = [p["filename"] for p in photo_results if not p["sharp"]]
        low_feature_photos = [
            p["filename"] for p in photo_results if p["features"] < self.min_features
        ]

        # Scoring
        if avg_quality >= 8:
            score = 10
        elif avg_quality >= 6:
            score = 8
        elif avg_quality >= 4:
            score = 6
        else:
            score = 3

        # Report issues
        if blurry_photos:
            issues.append(f"❌ Blurry photos detected: {', '.join(blurry_photos[:3])}")

        if low_feature_photos:
            issues.append(f"⚠️  Low-feature photos: {', '.join(low_feature_photos[:3])}")

        return score, issues

    def _check_overlap(
        self, images: List[np.ndarray], filenames: List[str]
    ) -> Tuple[int, List[str]]:
        """Check overlap between adjacent images"""
        issues = []

        sift = cv2.SIFT_create(nfeatures=1000)
        matcher = cv2.BFMatcher()

        overlaps = []

        # Check overlap between adjacent photos
        for i in range(len(images) - 1):
            img1, img2 = images[i], images[i + 1]

            # Extract features
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            kp1, desc1 = sift.detectAndCompute(gray1, None)
            kp2, desc2 = sift.detectAndCompute(gray2, None)

            if (
                desc1 is not None
                and desc2 is not None
                and len(desc1) > 10
                and len(desc2) > 10
            ):
                # Match features
                matches = matcher.knnMatch(desc1, desc2, k=2)

                # Apply ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                # Estimate overlap
                overlap_ratio = len(good_matches) / min(len(desc1), len(desc2))
                overlaps.append(overlap_ratio)

                if overlap_ratio < 0.2:
                    issues.append(
                        f"❌ Low overlap between {filenames[i]} and {filenames[i + 1]}"
                    )

        if overlaps:
            avg_overlap = np.mean(overlaps)

            if avg_overlap >= 0.6:
                score = 10
            elif avg_overlap >= 0.4:
                score = 8
            elif avg_overlap >= 0.2:
                score = 5
            else:
                score = 2
                issues.append(
                    "❌ Overall overlap too low - photos may be too different"
                )
        else:
            score = 0
            issues.append("❌ Could not compute overlap - feature matching failed")

        return score, issues

    def _check_coverage(self, images: List[np.ndarray]) -> Tuple[int, List[str]]:
        """Check if photos provide good room coverage"""
        issues = []

        # Simple heuristic: analyze feature distribution
        all_features = []

        sift = cv2.SIFT_create(nfeatures=1000)

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = sift.detectAndCompute(gray, None)

            # Get feature positions
            if kp:
                positions = np.array([k.pt for k in kp])
                all_features.extend(positions)

        if all_features:
            all_features = np.array(all_features)

            # Check distribution spread
            std_x = all_features[:, 0].std()
            std_y = all_features[:, 1].std()

            # Higher spread = better coverage
            if std_x > 200 and std_y > 150:
                score = 10
            elif std_x > 150 and std_y > 100:
                score = 8
            elif std_x > 100 and std_y > 80:
                score = 6
            else:
                score = 4
                issues.append("⚠️  Limited coverage - photos may be too similar")
        else:
            score = 0
            issues.append("❌ No features detected for coverage analysis")

        return score, issues

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate specific recommendations"""
        recs = []

        # Based on issues found
        issues = " ".join(results["issues"])

        if "Too few photos" in issues:
            recs.append(
                "📸 Take more photos: Aim for 20+ images in a systematic circle"
            )

        if "Blurry" in issues:
            recs.append("📱 Improve photo sharpness: Hold phone steady, tap to focus")

        if "Low overlap" in issues:
            recs.append(
                "🔄 Reduce steps between photos: Turn 15-18° instead of larger jumps"
            )

        if "Low-feature" in issues:
            recs.append("💡 Better lighting: Turn on room lights, avoid harsh shadows")

        if "coverage" in issues.lower():
            recs.append(
                "📐 Vary positions: Take photos from different distances/heights"
            )

        if not recs:  # No major issues
            recs.append("✅ Photos look good! Proceed with reconstruction")

        return recs


def main():
    """Main validation function"""
    import sys

    # Get folder from command line argument or use default
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "test_data"

    print(f"Validating photos in: {folder}")

    validator = PhotoValidator()
    results = validator.validate_photo_set(folder)

    print("\n" + "=" * 50)
    print("📋 ISSUES FOUND:")
    for issue in results["issues"]:
        print(f"  {issue}")

    print("\n💡 RECOMMENDATIONS:")
    for rec in results["recommendations"]:
        print(f"  {rec}")

    if results["valid"]:
        print("\n🚀 Ready to run reconstruction!")
    else:
        print("\n📝 Please address issues before reconstruction")

    return results


if __name__ == "__main__":
    main()
