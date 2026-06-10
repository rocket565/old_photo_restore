from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from .config import RestorationConfig


class DDColorAdapter:
    """Thin adapter around the cloned DDColor repository."""

    def __init__(self, config: RestorationConfig):
        self.config = config
        self._colorizer = None

    def colorize(self, image_bgr: np.ndarray) -> np.ndarray:
        if self._colorizer is None:
            self._colorizer = self._build_colorizer()
        return self._colorizer.process(image_bgr)

    def _build_colorizer(self):
        repo = self.config.ddcolor_repo.resolve()
        if not repo.exists():
            raise FileNotFoundError(f"DDColor repo not found: {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        import torch
        from ddcolor import ColorizationPipeline, DDColor, build_ddcolor_model

        device = torch.device(self.config.resolved_device())
        local_model_path = self._local_model_path()
        if local_model_path:
            model = build_ddcolor_model(
                DDColor,
                model_path=str(local_model_path),
                input_size=self.config.ddcolor_input_size,
                model_size=self.config.ddcolor_model_size,
                device=device,
            )
        else:
            from huggingface_hub import PyTorchModelHubMixin

            class DDColorHF(DDColor, PyTorchModelHubMixin):
                def __init__(self, config=None, **kwargs):
                    if isinstance(config, dict):
                        kwargs = {**config, **kwargs}
                    super().__init__(**kwargs)

            model_name = self.config.ddcolor_model_name
            if not Path(model_name).is_dir() and "/" not in model_name:
                model_name = f"piddnad/{model_name}"
            try:
                model = DDColorHF.from_pretrained(model_name)
            except Exception as exc:
                raise RuntimeError(
                    "DDColor weights were not found locally and the server could "
                    "not download them from Hugging Face. Put the weight file at "
                    "`weights/ddcolor/pytorch_model.pt`, or set the DDColor local "
                    "weight path in the UI."
                ) from exc
            model = model.to(device)
            model.eval()

        return ColorizationPipeline(
            model, input_size=self.config.ddcolor_input_size, device=device
        )

    def _local_model_path(self) -> Path | None:
        candidates = []
        if self.config.ddcolor_model_path:
            candidates.append(Path(self.config.ddcolor_model_path))
        candidates.append(Path("weights/ddcolor/pytorch_model.pt"))
        candidates.append(Path("pretrain/ddcolor_modelscope.pth"))
        candidates.append(Path("DDColor/pretrain/ddcolor_modelscope.pth"))

        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None


def read_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def write_image_bgr(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise IOError(f"Failed to write image: {path}")
