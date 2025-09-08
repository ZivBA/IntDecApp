"""
3D point triangulation from multiple views
"""

import cv2
import numpy as np
from typing import List
from .types import CameraPose, ImageFeatures, FeatureMatch, Point3D


class Triangulator:
    """Triangulate 3D points from multiple camera views"""

    def __init__(self, reprojection_threshold: float = 4.0):
        self.reprojection_threshold = reprojection_threshold

    def triangulate_points(
        self,
        all_features: List[ImageFeatures],
        matches: List[FeatureMatch],
        poses: List[CameraPose],
        images: List[np.ndarray],
    ) -> List[Point3D]:
        """
        Triangulate 3D points from feature matches across multiple views
        """
        points_3d = []

        # Process each pair of views with matches
        for match in matches:
            try:
                pair_points = self._triangulate_pair(
                    all_features[match.image1_idx],
                    all_features[match.image2_idx],
                    match,
                    poses[match.image1_idx],
                    poses[match.image2_idx],
                    images[match.image1_idx],
                    images[match.image2_idx],
                )
                points_3d.extend(pair_points)
            except Exception as e:
                print(
                    f"Failed to triangulate pair {match.image1_idx}-{match.image2_idx}: {e}"
                )

        print(f"Triangulated {len(points_3d)} points from {len(matches)} image pairs")
        return self._remove_duplicates(points_3d)

    def _triangulate_pair(
        self,
        features1: ImageFeatures,
        features2: ImageFeatures,
        match: FeatureMatch,
        pose1: CameraPose,
        pose2: CameraPose,
        image1: np.ndarray,
        image2: np.ndarray,
    ) -> List[Point3D]:
        """Triangulate points between a pair of views"""

        # Get matched points
        pts1 = features1.keypoints[match.matches[:, 0]]
        pts2 = features2.keypoints[match.matches[:, 1]]

        # Create projection matrices
        P1 = self._create_projection_matrix(pose1)
        P2 = self._create_projection_matrix(pose2)

        # Triangulate points using DLT (Direct Linear Transform)
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        # Convert to 3D (homogeneous to Cartesian)
        points_3d_homogeneous = points_4d / points_4d[3, :]
        points_3d_cart = points_3d_homogeneous[:3, :].T

        # Filter out bad triangulations
        valid_points = []

        for i, point_3d in enumerate(points_3d_cart):
            # Check reprojection error
            if self._validate_triangulation(point_3d, pts1[i], pts2[i], P1, P2):
                # Get color from first image
                x, y = int(pts1[i, 0]), int(pts1[i, 1])
                if 0 <= x < image1.shape[1] and 0 <= y < image1.shape[0]:
                    color = image1[y, x]
                    if len(color.shape) == 1:  # Grayscale
                        color = np.array([color[0], color[0], color[0]])
                    else:  # BGR to RGB
                        color = color[::-1]

                    valid_points.append(
                        Point3D(
                            position=point_3d,
                            color=color.astype(np.uint8),
                            confidence=1.0,
                        )
                    )

        return valid_points

    def _create_projection_matrix(self, pose: CameraPose) -> np.ndarray:
        """Create 3x4 projection matrix from camera pose"""
        R = pose.rotation
        t = pose.position.reshape(-1, 1)
        K = pose.intrinsics

        # P = K * [R | t]
        Rt = np.hstack([R, t])
        return K @ Rt

    def _validate_triangulation(
        self,
        point_3d: np.ndarray,
        pt1: np.ndarray,
        pt2: np.ndarray,
        P1: np.ndarray,
        P2: np.ndarray,
    ) -> bool:
        """Validate triangulated point using reprojection error"""

        # Check if point is in front of both cameras
        if point_3d[2] < 0:  # Assuming Z-forward camera convention
            return False

        # Reproject to both images
        point_4d = np.append(point_3d, 1.0)

        proj1 = P1 @ point_4d
        proj1 = proj1[:2] / proj1[2]

        proj2 = P2 @ point_4d
        proj2 = proj2[:2] / proj2[2]

        # Calculate reprojection errors
        error1 = np.linalg.norm(proj1 - pt1)
        error2 = np.linalg.norm(proj2 - pt2)

        return (
            error1 < self.reprojection_threshold
            and error2 < self.reprojection_threshold
        )

    def _remove_duplicates(
        self, points: List[Point3D], distance_threshold: float = 0.05
    ) -> List[Point3D]:
        """Remove duplicate 3D points that are very close together"""
        if len(points) <= 1:
            return points

        # Simple greedy approach - can be improved with spatial data structures
        filtered_points = [points[0]]

        for point in points[1:]:
            is_duplicate = False
            for existing in filtered_points:
                distance = np.linalg.norm(point.position - existing.position)
                if distance < distance_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_points.append(point)

        print(f"Removed {len(points) - len(filtered_points)} duplicate points")
        return filtered_points
