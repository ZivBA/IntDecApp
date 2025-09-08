"""
Point-based view renderer for virtual navigation
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from .types import Point3D, CameraPose, RenderConfig


class VirtualCamera:
    """Virtual camera for rendering novel views"""

    def __init__(self, intrinsics: np.ndarray):
        self.intrinsics = intrinsics
        self.position = np.array([0.0, 0.0, 0.0])
        self.rotation = np.eye(3)

    def set_pose(
        self, position: np.ndarray, look_at: np.ndarray, up: np.ndarray = None
    ):
        """Set camera pose using position and look-at target"""
        self.position = position.copy()

        if up is None:
            up = np.array([0.0, 0.0, 1.0])  # Z-up coordinate system

        # Create rotation matrix from look-at
        forward = look_at - position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up_corrected = np.cross(right, forward)

        # Create rotation matrix (camera coordinate system)
        self.rotation = np.column_stack([right, up_corrected, -forward])

    def get_projection_matrix(self) -> np.ndarray:
        """Get 3x4 projection matrix"""
        # Create extrinsic matrix [R | t]
        t = -self.rotation.T @ self.position
        extrinsics = np.hstack([self.rotation.T, t.reshape(-1, 1)])

        # P = K * [R | t]
        return self.intrinsics @ extrinsics


class PointRenderer:
    """Render novel views from 3D point cloud using point splatting"""

    def __init__(self, config: RenderConfig):
        self.config = config
        self.point_size = 3  # Size of each point when rendered
        self.depth_tolerance = 0.1  # Tolerance for depth buffering

    def render(
        self,
        points_3d: List[Point3D],
        virtual_camera: VirtualCamera,
        depth_smoothing: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render a novel view from 3D points

        Returns:
            Tuple of (rendered_image, depth_map)
        """
        width, height = self.config.image_width, self.config.image_height

        # Initialize buffers
        image_buffer = np.full(
            (height, width, 3), self.config.background_color, dtype=np.uint8
        )
        depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
        weight_buffer = np.zeros((height, width), dtype=np.float32)

        # Get projection matrix
        P = virtual_camera.get_projection_matrix()

        # Project all points
        rendered_points = []
        for point in points_3d:
            projected = self._project_point(point, P)
            if projected is not None:
                rendered_points.append(projected)

        if not rendered_points:
            print("Warning: No points visible from this viewpoint")
            return image_buffer, depth_buffer

        # Sort by depth (far to near for proper alpha blending)
        rendered_points.sort(key=lambda x: x["depth"], reverse=True)

        # Render points with splatting
        for rp in rendered_points:
            self._splat_point(rp, image_buffer, depth_buffer, weight_buffer)

        # Post-processing
        if depth_smoothing:
            image_buffer = self._smooth_image(image_buffer, weight_buffer)

        # Convert depth buffer (remove infinite values)
        depth_map = np.where(depth_buffer == np.inf, 0, depth_buffer)

        return image_buffer, depth_map

    def _project_point(self, point: Point3D, P: np.ndarray) -> Optional[dict]:
        """Project 3D point to image coordinates"""
        # Convert to homogeneous coordinates
        point_4d = np.append(point.position, 1.0)

        # Project to image
        projected = P @ point_4d

        # Check if point is in front of camera
        if projected[2] <= 0:
            return None

        # Convert to pixel coordinates
        x = projected[0] / projected[2]
        y = projected[1] / projected[2]
        depth = projected[2]

        # Check if point is within image bounds
        if 0 <= x < self.config.image_width and 0 <= y < self.config.image_height:
            return {
                "x": int(x),
                "y": int(y),
                "depth": depth,
                "color": point.color,
                "confidence": point.confidence,
            }

        return None

    def _splat_point(
        self,
        rendered_point: dict,
        image_buffer: np.ndarray,
        depth_buffer: np.ndarray,
        weight_buffer: np.ndarray,
    ):
        """Splat a single point onto the image buffers"""
        x, y = rendered_point["x"], rendered_point["y"]
        depth = rendered_point["depth"]
        color = rendered_point["color"]
        confidence = rendered_point["confidence"]

        # Point splatting - render point as small circle
        for dy in range(-self.point_size, self.point_size + 1):
            for dx in range(-self.point_size, self.point_size + 1):
                px, py = x + dx, y + dy

                # Check bounds
                if (
                    0 <= px < self.config.image_width
                    and 0 <= py < self.config.image_height
                ):
                    # Gaussian-like falloff for smoother splatting
                    distance = np.sqrt(dx * dx + dy * dy)
                    if distance <= self.point_size:
                        weight = confidence * np.exp(
                            -(distance**2) / (2 * (self.point_size / 2) ** 2)
                        )

                        # Depth test with tolerance
                        if depth < depth_buffer[py, px] + self.depth_tolerance:
                            # Blend with existing color based on weight
                            if weight_buffer[py, px] > 0:
                                # Weighted average
                                total_weight = weight_buffer[py, px] + weight
                                image_buffer[py, px] = (
                                    (
                                        image_buffer[py, px] * weight_buffer[py, px]
                                        + color * weight
                                    )
                                    / total_weight
                                ).astype(np.uint8)
                                weight_buffer[py, px] = total_weight
                            else:
                                image_buffer[py, px] = color
                                weight_buffer[py, px] = weight

                            depth_buffer[py, px] = min(depth_buffer[py, px], depth)

    def _smooth_image(self, image: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Apply smoothing to reduce holes and artifacts"""
        # Identify holes (areas with very low weight)
        holes_mask = weights < 0.1

        if not np.any(holes_mask):
            return image

        # Inpaint small holes
        mask = (holes_mask * 255).astype(np.uint8)
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

        # Blend inpainted result with original
        result = image.copy()
        result[holes_mask] = inpainted[holes_mask]

        return result


class NavigationSystem:
    """System for managing virtual camera navigation in reconstructed space"""

    def __init__(self, points_3d: List[Point3D], original_poses: List[CameraPose]):
        self.points_3d = points_3d
        self.original_poses = original_poses

        # Compute scene bounds
        if points_3d:
            positions = np.array([p.position for p in points_3d])
            self.scene_center = np.mean(positions, axis=0)
            self.scene_bounds = {
                "min": np.min(positions, axis=0),
                "max": np.max(positions, axis=0),
            }
        else:
            self.scene_center = np.array([0, 0, 0])
            self.scene_bounds = {
                "min": np.array([-1, -1, -1]),
                "max": np.array([1, 1, 1]),
            }

    def get_walking_positions(
        self, n_positions: int = 8, height: float = 1.6
    ) -> List[np.ndarray]:
        """Generate positions for walking through the room"""
        positions = []

        # Create positions in a circle around room center
        room_size = np.linalg.norm(self.scene_bounds["max"] - self.scene_bounds["min"])
        radius = room_size * 0.3  # Stay closer to center

        for i in range(n_positions):
            angle = (i / n_positions) * 2 * np.pi
            x = self.scene_center[0] + radius * np.cos(angle)
            y = self.scene_center[1] + radius * np.sin(angle)
            z = self.scene_center[2] + height - 0.8  # Adjust for head height

            positions.append(np.array([x, y, z]))

        return positions

    def get_overview_positions(self) -> List[np.ndarray]:
        """Get positions for room overview (elevated views)"""
        room_height = self.scene_bounds["max"][2] - self.scene_bounds["min"][2]

        positions = []

        # Corner views
        corners = [
            [
                self.scene_bounds["min"][0],
                self.scene_bounds["min"][1],
                self.scene_center[2] + room_height,
            ],
            [
                self.scene_bounds["max"][0],
                self.scene_bounds["min"][1],
                self.scene_center[2] + room_height,
            ],
            [
                self.scene_bounds["max"][0],
                self.scene_bounds["max"][1],
                self.scene_center[2] + room_height,
            ],
            [
                self.scene_bounds["min"][0],
                self.scene_bounds["max"][1],
                self.scene_center[2] + room_height,
            ],
        ]

        for corner in corners:
            positions.append(np.array(corner))

        return positions

    def create_virtual_camera(self, image_shape: tuple) -> VirtualCamera:
        """Create virtual camera with appropriate intrinsics"""
        # Use intrinsics from original camera if available
        if self.original_poses and self.original_poses[0].intrinsics is not None:
            K = self.original_poses[0].intrinsics.copy()
        else:
            # Estimate intrinsics
            height, width = image_shape[:2]
            focal_length = width / (2 * np.tan(np.radians(35)))  # ~70 degree FOV
            K = np.array(
                [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]]
            )

        return VirtualCamera(K)
