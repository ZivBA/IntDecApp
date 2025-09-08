"""
Main reconstruction pipeline
"""

import time
import numpy as np
from typing import List
from .types import ReconstructionResult
from .feature_extraction import FeatureExtractor
from .camera_estimation import SimpleCameraPoseEstimator
from .triangulation import Triangulator


class RoomReconstructor:
    """
    Main class for room reconstruction from multiple images
    """

    def __init__(
        self,
        max_features: int = 2000,
        room_height: float = 2.5,
        camera_height: float = 1.6,
    ):
        self.feature_extractor = FeatureExtractor(max_features)
        self.pose_estimator = SimpleCameraPoseEstimator(room_height, camera_height)
        self.triangulator = Triangulator()

    def reconstruct(self, images: List[np.ndarray]) -> ReconstructionResult:
        """
        Reconstruct 3D scene from multiple room images

        Args:
            images: List of room images (BGR format)

        Returns:
            ReconstructionResult with 3D points and camera poses
        """
        start_time = time.time()

        try:
            print(f"Starting reconstruction with {len(images)} images...")

            # Step 1: Extract features from all images
            print("1. Extracting features...")
            all_features = []
            for i, image in enumerate(images):
                features = self.feature_extractor.extract_features(image, i)
                all_features.append(features)
                print(f"   Image {i}: {len(features.keypoints)} features")

            # Step 2: Match features between images
            print("2. Matching features...")
            matches = self.feature_extractor.match_all_pairs(all_features)

            if not matches:
                return ReconstructionResult(
                    success=False,
                    points_3d=[],
                    camera_poses=[],
                    processing_time=time.time() - start_time,
                    error_message="No feature matches found between images",
                )

            # Step 3: Estimate camera poses
            print("3. Estimating camera poses...")
            try:
                poses = self.pose_estimator.estimate_poses_from_features(
                    all_features, matches, images
                )
            except Exception as e:
                print(f"   Pose estimation failed: {e}")
                print("   Falling back to circular assumption...")
                poses = self.pose_estimator.estimate_circular_poses(
                    len(images), images[0].shape
                )

            # Step 4: Triangulate 3D points
            print("4. Triangulating 3D points...")
            points_3d = self.triangulator.triangulate_points(
                all_features, matches, poses, images
            )

            processing_time = time.time() - start_time

            print(f"Reconstruction completed in {processing_time:.2f} seconds")
            print(f"Generated {len(points_3d)} 3D points")

            return ReconstructionResult(
                success=True,
                points_3d=points_3d,
                camera_poses=poses,
                processing_time=processing_time,
            )

        except Exception as e:
            return ReconstructionResult(
                success=False,
                points_3d=[],
                camera_poses=[],
                processing_time=time.time() - start_time,
                error_message=str(e),
            )
