#!/usr/bin/env python3
"""
Test improved feature extraction and matching for more points
"""

import cv2
import numpy as np
import os
from typing import List
from src.core.types import ImageFeatures, FeatureMatch
from src.core.triangulation import Triangulator
from src.core.camera_estimation import SimpleCameraPoseEstimator


class ImprovedFeatureExtractor:
    """Enhanced feature extraction with more aggressive parameters"""

    def __init__(self, max_features: int = 10000):  # 5x more features!
        self.max_features = max_features
        # Try multiple feature detectors
        self.sift = cv2.SIFT_create(
            nfeatures=max_features, contrastThreshold=0.02
        )  # Lower threshold
        self.orb = cv2.ORB_create(nfeatures=max_features)

        # Use FLANN matcher for better performance
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def extract_features_multi(
        self, image: np.ndarray, image_idx: int
    ) -> ImageFeatures:
        """Extract features using multiple methods and combine"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Method 1: SIFT with low threshold
        kp1, desc1 = self.sift.detectAndCompute(gray, None)

        # Method 2: Dense grid sampling
        grid_kp = []
        h, w = gray.shape
        step = 20  # Sample every 20 pixels
        for y in range(step, h - step, step):
            for x in range(step, w - step, step):
                grid_kp.append(cv2.KeyPoint(x, y, step))

        # Compute SIFT descriptors for grid points
        grid_kp, grid_desc = self.sift.compute(gray, grid_kp)

        # Combine all keypoints
        all_kp = list(kp1) + list(grid_kp)
        all_desc = (
            np.vstack([desc1, grid_desc])
            if desc1 is not None and grid_desc is not None
            else desc1
        )

        print(
            f"  Image {image_idx}: {len(kp1)} SIFT + {len(grid_kp)} grid = {len(all_kp)} total features"
        )

        # Convert to our format
        kp_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in all_kp])

        return ImageFeatures(
            keypoints=kp_coords, descriptors=all_desc, image_idx=image_idx
        )

    def match_features_relaxed(
        self, features1: ImageFeatures, features2: ImageFeatures
    ) -> FeatureMatch:
        """More relaxed matching to get more points"""

        # Match with relaxed parameters
        matches = self.matcher.knnMatch(
            features1.descriptors, features2.descriptors, k=2
        )

        good_matches = []
        distances = []

        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # More relaxed ratio test (0.8 instead of 0.7)
                if m.distance < 0.8 * n.distance:
                    good_matches.append([m.queryIdx, m.trainIdx])
                    distances.append(m.distance)
            elif len(match_pair) == 1:
                # If only one match found, include it if distance is low
                m = match_pair[0]
                if m.distance < 300:  # Absolute threshold
                    good_matches.append([m.queryIdx, m.trainIdx])
                    distances.append(m.distance)

        # Lower minimum threshold (5 instead of 20)
        if len(good_matches) < 5:
            raise ValueError(f"Not enough matches: {len(good_matches)}")

        return FeatureMatch(
            image1_idx=features1.image_idx,
            image2_idx=features2.image_idx,
            matches=np.array(good_matches),
            distances=np.array(distances),
        )

    def match_with_spatial_consistency(
        self, all_features: List[ImageFeatures]
    ) -> List[FeatureMatch]:
        """Smart matching considering spatial relationships"""
        matches = []
        n_images = len(all_features)

        # Create match matrix to track what matched
        match_matrix = np.zeros((n_images, n_images), dtype=int)

        # Try all pairs
        for i in range(n_images):
            for j in range(i + 1, n_images):
                try:
                    match = self.match_features_relaxed(
                        all_features[i], all_features[j]
                    )
                    matches.append(match)
                    match_matrix[i, j] = len(match.matches)
                    print(f"✅ Images {i}-{j}: {len(match.matches)} matches")
                except ValueError as e:
                    # For adjacent images, try even more relaxed matching
                    if abs(i - j) == 1:  # Adjacent in capture sequence
                        print(
                            f"⚠️  Adjacent images {i}-{j} failed, trying rescue match..."
                        )
                        try:
                            # Just find ANY matches
                            raw_matches = self.matcher.match(
                                all_features[i].descriptors, all_features[j].descriptors
                            )
                            if len(raw_matches) >= 3:
                                good_matches = [
                                    [m.queryIdx, m.trainIdx] for m in raw_matches[:50]
                                ]
                                distances = [m.distance for m in raw_matches[:50]]

                                match = FeatureMatch(
                                    image1_idx=i,
                                    image2_idx=j,
                                    matches=np.array(good_matches),
                                    distances=np.array(distances),
                                )
                                matches.append(match)
                                match_matrix[i, j] = len(good_matches)
                                print(f"  🔧 Rescued with {len(good_matches)} matches")
                        except Exception:
                            print("  ❌ Rescue failed")
                    else:
                        print(f"❌ Images {i}-{j}: {e}")

        print(f"\n📊 Match matrix:\n{match_matrix}")
        return matches


def test_improved_reconstruction():
    """Test with improved parameters"""
    print("🔧 TESTING IMPROVED FEATURE EXTRACTION")
    print("=" * 50)

    # Load images
    image_files = sorted([f for f in os.listdir("test_data") if f.endswith(".jpg")])
    images = []
    for filename in image_files:
        image_path = os.path.join("test_data", filename)
        image = cv2.imread(image_path)
        if image is not None:
            height, width = image.shape[:2]
            if width > 1024:
                scale = 1024 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            images.append(image)

    print(f"Loaded {len(images)} images")

    # Extract features with improved method
    print("\n📸 Extracting enhanced features...")
    extractor = ImprovedFeatureExtractor(max_features=10000)
    all_features = []

    for i, img in enumerate(images):
        features = extractor.extract_features_multi(img, i)
        all_features.append(features)

    # Match with improved strategy
    print("\n🔗 Matching with relaxed parameters...")
    matches = extractor.match_with_spatial_consistency(all_features)

    print(f"\n✅ Total successful matches: {len(matches)}")
    total_point_matches = sum(len(m.matches) for m in matches)
    print(f"📊 Total point correspondences: {total_point_matches}")

    # Estimate poses
    print("\n📷 Estimating camera poses...")
    pose_estimator = SimpleCameraPoseEstimator()
    poses = pose_estimator.estimate_poses_from_features(all_features, matches, images)

    # Triangulate with all matches
    print("\n🎯 Triangulating 3D points...")
    triangulator = Triangulator(reprojection_threshold=8.0)  # More relaxed threshold
    points_3d = triangulator.triangulate_points(all_features, matches, poses, images)

    print("\n🎉 RESULTS:")
    print("  Original approach: 43 points")
    print(f"  Improved approach: {len(points_3d)} points")
    print(f"  Improvement: {len(points_3d) / 43:.1f}x")

    if points_3d:
        # Analyze point distribution
        positions = np.array([p.position for p in points_3d])
        colors = np.array([p.color for p in points_3d])

        print("\n📊 Point statistics:")
        print(f"  Spatial extent: {positions.max(axis=0) - positions.min(axis=0)}")
        print(f"  Color variance: {colors.std(axis=0).mean():.1f}")
        print(f"  Unique colors: {len(np.unique(colors, axis=0))}")

    return points_3d, poses


if __name__ == "__main__":
    points, poses = test_improved_reconstruction()

    if len(points) > 100:
        print("\n✅ SUCCESS! We have enough points for visualization")
        print("Next: Test rendering with denser point cloud")
    else:
        print("\n⚠️  Still need more points. Consider:")
        print("  - Using ORB + SIFT combined")
        print("  - Implementing dense stereo matching")
        print("  - Adding depth estimation to fill gaps")
