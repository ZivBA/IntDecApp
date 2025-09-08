"""
Feature extraction and matching for multi-view reconstruction
"""

import cv2
import numpy as np
from typing import List
from .types import ImageFeatures, FeatureMatch


class FeatureExtractor:
    """Extract and match features between images"""

    def __init__(self, max_features: int = 2000):
        self.max_features = max_features
        self.detector = cv2.SIFT_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher()

    def extract_features(self, image: np.ndarray, image_idx: int) -> ImageFeatures:
        """Extract SIFT features from an image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if descriptors is None:
            raise ValueError(f"No features found in image {image_idx}")

        # Convert keypoints to numpy array
        kp_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])

        return ImageFeatures(
            keypoints=kp_coords, descriptors=descriptors, image_idx=image_idx
        )

    def match_features(
        self,
        features1: ImageFeatures,
        features2: ImageFeatures,
        distance_threshold: float = 0.7,
    ) -> FeatureMatch:
        """Match features between two images using Lowe's ratio test"""

        # Find matches using k-nearest neighbors
        matches = self.matcher.knnMatch(
            features1.descriptors, features2.descriptors, k=2
        )

        # Apply Lowe's ratio test
        good_matches = []
        distances = []

        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < distance_threshold * n.distance:
                    good_matches.append([m.queryIdx, m.trainIdx])
                    distances.append(m.distance)

        if len(good_matches) < 20:
            raise ValueError(f"Not enough good matches: {len(good_matches)}")

        return FeatureMatch(
            image1_idx=features1.image_idx,
            image2_idx=features2.image_idx,
            matches=np.array(good_matches),
            distances=np.array(distances),
        )

    def match_all_pairs(self, all_features: List[ImageFeatures]) -> List[FeatureMatch]:
        """Match features between all image pairs"""
        matches = []

        for i in range(len(all_features)):
            for j in range(i + 1, len(all_features)):
                try:
                    match = self.match_features(all_features[i], all_features[j])
                    matches.append(match)
                    print(f"Matched images {i}-{j}: {len(match.matches)} matches")
                except ValueError as e:
                    print(f"Failed to match images {i}-{j}: {e}")

        return matches
