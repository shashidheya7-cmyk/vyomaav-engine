"""VYOMAAV Camera Geometry Engine - Intrinsics, Distortion, & Confidence Projection."""

import numpy as np
from PIL import Image, ExifTags
from typing import Tuple, Dict, Any, Optional


class PinholeCameraModel:
    """Manages Camera Intrinsics K, Extrinsics [R|t], and Confidence-Weighted Unprojection."""

    def __init__(
        self,
        K: Optional[np.ndarray] = None,
        distortion: Optional[np.ndarray] = None,
        fov_deg: float = 60.0
    ):
        self.K = K
        self.distortion = distortion if distortion is not None else np.zeros(5, dtype=np.float32)
        self.fov_deg = fov_deg

    def estimate_from_exif(self, image: Image.Image) -> Dict[str, Any]:
        """Extracts focal length and lens specs from image EXIF data if present."""
        w, h = image.size
        exif_data = image._getexif() if hasattr(image, '_getexif') and image._getexif() else {}
        
        focal_mm = None
        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                if tag_name == "FocalLength":
                    try:
                        focal_mm = float(value)
                    except (TypeError, ValueError):
                        pass

        # Estimate K matrix
        if focal_mm and "FocalLengthIn35mmFilm" in [ExifTags.TAGS.get(k) for k in exif_data.keys()]:
            # 35mm film sensor equivalent width is 36mm
            f_x = (focal_mm / 36.0) * max(w, h)
            f_y = f_x
        else:
            fov_rad = np.radians(self.fov_deg)
            f_x = w / (2.0 * np.tan(fov_rad / 2.0))
            f_y = f_x

        c_x, c_y = w / 2.0, h / 2.0
        self.K = np.array([
            [f_x, 0.0, c_x],
            [0.0, f_y, c_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        return {"K": self.K, "focal_px": f_x, "sensor_center": (c_x, c_y)}

    def unproject_depth_with_confidence(
        self,
        depth_map: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
        min_confidence: float = 0.35
    ) -> Dict[str, np.ndarray]:
        """Unprojects 2D Depth + Confidence Map into 3D Point Cloud with uncertainty weighting."""
        h, w = depth_map.shape
        if self.K is None:
            self.estimate_from_exif(Image.new("RGB", (w, h)))

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        c_x, c_y = self.K[0, 2], self.K[1, 2]
        f_x, f_y = self.K[0, 0], self.K[1, 1]

        Z = depth_map.flatten()
        X = (u.flatten() - c_x) * Z / f_x
        Y = (v.flatten() - c_y) * Z / f_y

        points_3d = np.stack([X, Y, Z], axis=-1)

        if confidence_map is not None:
            conf_flat = confidence_map.flatten()
            valid_mask = (conf_flat >= min_confidence) & (Z > 0)
            return {
                "points": points_3d[valid_mask],
                "confidence": conf_flat[valid_mask],
                "valid_mask": valid_mask
            }

        valid_mask = Z > 0
        return {
            "points": points_3d[valid_mask],
            "confidence": np.ones(np.sum(valid_mask), dtype=np.float32),
            "valid_mask": valid_mask
        }
