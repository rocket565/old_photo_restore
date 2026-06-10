from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class DamageMaskGenerator:
    """Detect likely scratches, cracks, stains, and missing photo regions."""

    def __init__(self, dilate_iterations: int = 2, min_component_area: int = 3):
        self.dilate_iterations = dilate_iterations
        self.min_component_area = min_component_area
        self.last_preprocessed: np.ndarray | None = None

    def build(self, image_bgr: np.ndarray, manual_mask_path: Path | None = None) -> np.ndarray:
        enhanced = self.preprocess(image_bgr)
        self.last_preprocessed = enhanced

        h, w = enhanced.shape
        threshold = self._bright_scratch_threshold(enhanced)
        _, mask = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY)

        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0, sigmaY=2.0)
        bright_residual = cv2.subtract(enhanced, blur)
        residual_threshold = max(16, int(np.percentile(bright_residual, 98.5)))
        residual_mask = (bright_residual >= residual_threshold).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, residual_mask)

        mask = self._filter_scratch_components(mask, image_area=h * w, enhanced=enhanced)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = self._filter_components(mask, image_area=h * w, max_area_ratio=0.018)

        dark_missing = cv2.inRange(enhanced, 0, 10)
        bright_missing = cv2.inRange(enhanced, 248, 255)
        border_damage = self._border_connected(dark_missing | bright_missing)
        border_damage = self._filter_components(
            border_damage, image_area=h * w, max_area_ratio=0.12
        )
        mask = cv2.bitwise_or(mask, border_damage)

        if manual_mask_path:
            manual = cv2.imread(str(manual_mask_path), cv2.IMREAD_GRAYSCALE)
            if manual is None:
                raise FileNotFoundError(f"Cannot read manual mask: {manual_mask_path}")
            manual = cv2.resize(manual, (w, h), interpolation=cv2.INTER_NEAREST)
            _, manual = cv2.threshold(manual, 1, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_or(mask, manual)

        return mask

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(denoised)

    def _bright_scratch_threshold(self, enhanced: np.ndarray) -> int:
        percentile_threshold = int(np.percentile(enhanced, 96.2))
        return int(np.clip(percentile_threshold, 205, 228))

    def _filter_components(
        self, mask: np.ndarray, image_area: int, max_area_ratio: float = 0.18
    ) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = np.zeros_like(mask)
        max_area = image_area * max_area_ratio
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if self.min_component_area <= area <= max_area:
                out[labels == label] = 255
        return out

    def _filter_scratch_components(
        self,
        mask: np.ndarray,
        image_area: int,
        enhanced: np.ndarray | None = None,
    ) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = np.zeros_like(mask)
        max_line_area = max(80, int(image_area * 0.006))
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_area:
                continue

            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            longest = max(width, height)
            shortest = max(1, min(width, height))
            aspect = longest / shortest
            fill_ratio = area / max(1, width * height)
            if enhanced is not None:
                contrast = self._component_local_contrast(labels == label, enhanced)
                min_contrast = 6 if longest >= 8 and aspect >= 2.0 else 8
                if contrast < min_contrast:
                    continue

            is_speck = area <= 56 and longest <= 18 and fill_ratio <= 0.90
            is_small_defect = area <= 180 and fill_ratio <= 0.68
            is_thin_line = (
                area <= max_line_area
                and longest >= 8
                and aspect >= 2.0
                and fill_ratio <= 0.58
            )
            if is_speck or is_small_defect or is_thin_line:
                out[labels == label] = 255
        return out

    def _component_local_contrast(self, component: np.ndarray, enhanced: np.ndarray) -> float:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
        ring = dilated & ~component
        if np.count_nonzero(ring) < 4:
            return 0.0
        comp_values = enhanced[component]
        ring_values = enhanced[ring]
        return float(np.median(comp_values) - np.median(ring_values))

    def _border_connected(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        flood = np.zeros((h + 2, w + 2), np.uint8)
        connected = mask.copy()
        for point in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            cv2.floodFill(connected, flood, point, 128)
        return np.where(connected == 128, 255, 0).astype(np.uint8)
