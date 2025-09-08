"""
Simple camera pose estimation for room reconstruction
"""

import cv2
import numpy as np
from typing import List
from .types import CameraPose, ImageFeatures, FeatureMatch


class SimpleCameraPoseEstimator:
    """
    Simplified camera pose estimation assuming:
    - Photos taken from roughly same height
    - Camera looking roughly toward room center
    - Circular path around room
    """

    def __init__(self, room_height: float = 2.5, camera_height: float = 1.6):
        self.room_height = room_height
        self.camera_height = camera_height

    def estimate_intrinsics(self, image_shape: tuple) -> np.ndarray:
        """
        Estimate camera intrinsics for typical smartphone camera
        Assumes ~70-degree horizontal FOV
        """
        height, width = image_shape[:2]
        focal_length = width / (2 * np.tan(np.radians(35)))  # 70 deg / 2

        return np.array(
            [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]]
        )

    def estimate_poses_from_features(
        self,
        all_features: List[ImageFeatures],
        matches: List[FeatureMatch],
        images: List[np.ndarray],
    ) -> List[CameraPose]:
        """
        Estimate camera poses using feature matches
        """
        n_cameras = len(all_features)
        poses = []

        # Estimate intrinsics from first image
        K = self.estimate_intrinsics(images[0].shape)

        # First camera at origin
        poses.append(
            CameraPose(
                position=np.array([0.0, 0.0, 0.0]), rotation=np.eye(3), intrinsics=K
            )
        )

        # Estimate subsequent poses relative to previous
        for i in range(1, n_cameras):
            try:
                pose = self._estimate_relative_pose(
                    all_features[i - 1], all_features[i], matches, K
                )
                poses.append(pose)
            except Exception as e:
                print(f"Failed to estimate pose for camera {i}: {e}")
                # Fallback to circular assumption
                angle = (i / n_cameras) * 2 * np.pi
                pos = np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0])
                rot = self._look_at_rotation(pos, np.zeros(3))
                poses.append(CameraPose(position=pos, rotation=rot, intrinsics=K))

        return poses

    def _estimate_relative_pose(
        self,
        features1: ImageFeatures,
        features2: ImageFeatures,
        all_matches: List[FeatureMatch],
        K: np.ndarray,
    ) -> CameraPose:
        """Estimate pose of camera2 relative to camera1"""

        # Find matching pair
        match = None
        for m in all_matches:
            if (
                m.image1_idx == features1.image_idx
                and m.image2_idx == features2.image_idx
            ) or (
                m.image1_idx == features2.image_idx
                and m.image2_idx == features1.image_idx
            ):
                match = m
                break

        if match is None:
            raise ValueError("No matches found between cameras")

        # Get matched point coordinates
        if match.image1_idx == features1.image_idx:
            pts1 = features1.keypoints[match.matches[:, 0]]
            pts2 = features2.keypoints[match.matches[:, 1]]
        else:
            pts1 = features1.keypoints[match.matches[:, 1]]
            pts2 = features2.keypoints[match.matches[:, 0]]

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999)

        if E is None:
            raise ValueError("Could not estimate essential matrix")

        # Recover pose from essential matrix
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

        return CameraPose(position=t.flatten(), rotation=R, intrinsics=K)

    def _look_at_rotation(
        self, camera_pos: np.ndarray, target_pos: np.ndarray
    ) -> np.ndarray:
        """Create rotation matrix for camera looking at target"""
        forward = target_pos - camera_pos
        forward = forward / np.linalg.norm(forward)

        up = np.array([0, 0, 1])  # Z is up
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        R = np.column_stack([right, up, -forward])
        return R

    def estimate_circular_poses(
        self, n_cameras: int, image_shape: tuple, radius: float = 2.0
    ) -> List[CameraPose]:
        """
        Fallback: assume cameras arranged in circle looking inward
        """
        K = self.estimate_intrinsics(image_shape)
        poses = []

        for i in range(n_cameras):
            angle = (i / n_cameras) * 2 * np.pi

            # Position on circle at camera height
            position = np.array(
                [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    self.camera_height
                    - self.room_height / 2,  # Relative to room center
                ]
            )

            # Look toward room center
            rotation = self._look_at_rotation(position, np.array([0, 0, 0]))

            poses.append(CameraPose(position=position, rotation=rotation, intrinsics=K))

        return poses
