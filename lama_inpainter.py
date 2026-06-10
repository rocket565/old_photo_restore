from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import RestorationConfig


class LaMaInpaintingAdapter:
    """Adapter for IOPaint/LaMa JIT inpainting weights."""

    def __init__(self, config: RestorationConfig):
        self.config = config
        self.model_path = Path(config.lama_model_path)
        self._model = None
        self._device = None

    def is_available(self) -> bool:
        return self.model_path.exists() and self.model_path.stat().st_size > 10_000_000

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self.is_available():
            raise FileNotFoundError(f"LaMa model not found: {self.model_path}")
        if np.count_nonzero(mask) == 0:
            return image_bgr.copy()

        import torch

        if self._model is None:
            self._device = torch.device(self.config.resolved_device())
            self._model = torch.jit.load(str(self.model_path), map_location="cpu")
            self._model = self._model.to(self._device).eval()

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        rgb_pad = self._pad_to_modulo(rgb, 8, mode="symmetric")
        mask_pad = self._pad_to_modulo(mask, 8, mode="constant")

        image_tensor = torch.from_numpy(self._norm_image(rgb_pad)).unsqueeze(0).to(self._device)
        mask_tensor = torch.from_numpy(self._norm_image(mask_pad)).unsqueeze(0).to(self._device)
        mask_tensor = (mask_tensor > 0).float()

        with torch.inference_mode():
            result = self._model(image_tensor, mask_tensor)

        result = result[0].permute(1, 2, 0).detach().cpu().numpy()
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        result = result[:h, :w]
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    def _norm_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        image = np.transpose(image, (2, 0, 1))
        return image.astype(np.float32) / 255.0

    def _pad_to_modulo(self, image: np.ndarray, modulo: int, mode: str) -> np.ndarray:
        h, w = image.shape[:2]
        out_h = h if h % modulo == 0 else (h // modulo + 1) * modulo
        out_w = w if w % modulo == 0 else (w // modulo + 1) * modulo
        pad_h = out_h - h
        pad_w = out_w - w
        if image.ndim == 2:
            pad_width = ((0, pad_h), (0, pad_w))
        else:
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))
        if mode == "constant":
            return np.pad(image, pad_width, mode=mode, constant_values=0)
        return np.pad(image, pad_width, mode=mode)
