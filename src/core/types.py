"""
Core data types for 3D reconstruction POC
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class CameraPose:
    """Camera pose in 3D space"""

    position: np.ndarray  # (3,) translation
    rotation: np.ndarray  # (3, 3) rotation matrix
    intrinsics: Optional[np.ndarray] = None  # (3, 3) camera matrix


@dataclass
class ImageFeatures:
    """Feature points extracted from an image"""

    keypoints: np.ndarray  # (N, 2) pixel coordinates
    descriptors: np.ndarray  # (N, descriptor_dim) feature descriptors
    image_idx: int


@dataclass
class FeatureMatch:
    """Match between features in two images"""

    image1_idx: int
    image2_idx: int
    matches: np.ndarray  # (M, 2) indices into keypoints arrays
    distances: np.ndarray  # (M,) match distances


@dataclass
class Point3D:
    """3D point with color"""

    position: np.ndarray  # (3,) xyz coordinates
    color: np.ndarray  # (3,) rgb values [0-255]
    confidence: float = 1.0


@dataclass
class ReconstructionResult:
    """Result of 3D reconstruction"""

    success: bool
    points_3d: List[Point3D]
    camera_poses: List[CameraPose]
    processing_time: float
    error_message: Optional[str] = None


@dataclass
class RenderConfig:
    """Configuration for view rendering"""

    image_width: int = 640
    image_height: int = 480
    camera_intrinsics: Optional[np.ndarray] = None
    background_color: Tuple[int, int, int] = (128, 128, 128)
