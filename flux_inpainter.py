from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import RestorationConfig, RestorationPlan


class FluxInpaintingAdapter:
    """Lazy loader for FLUX-ControlNet-Inpainting."""

    def __init__(self, config: RestorationConfig):
        self.config = config
        self._pipe = None

    def inpaint(
        self,
        image_bgr: np.ndarray,
        mask_u8: np.ndarray,
        plan: RestorationPlan,
    ) -> np.ndarray:
        if np.count_nonzero(mask_u8) == 0:
            return image_bgr.copy()
        if self._pipe is None:
            self._pipe = self._build_pipe()

        import torch

        h, w = image_bgr.shape[:2]
        size = self._target_size(w, h)
        image = self._bgr_to_pil(image_bgr).resize(size, Image.LANCZOS)
        mask = Image.fromarray(mask_u8).convert("RGB").resize(size, Image.NEAREST)
        generator = torch.Generator(device=self.config.resolved_device()).manual_seed(
            self.config.seed
        )

        result = self._pipe(
            prompt=plan.inpaint_prompt,
            height=size[1],
            width=size[0],
            control_image=image,
            control_mask=mask,
            num_inference_steps=self.config.flux_steps,
            generator=generator,
            controlnet_conditioning_scale=self.config.flux_control_scale,
            guidance_scale=self.config.flux_guidance_scale,
            negative_prompt=plan.negative_prompt,
            true_guidance_scale=self.config.flux_true_guidance_scale,
        ).images[0]
        result = result.resize((w, h), Image.LANCZOS)
        result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        return self._blend_masked_result(image_bgr, result_bgr, mask_u8)

    def _build_pipe(self):
        repo = self.config.flux_repo.resolve()
        if not repo.exists():
            raise FileNotFoundError(f"FLUX-ControlNet-Inpainting repo not found: {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        self._check_transformers_torch_compat()

        import torch
        from controlnet_flux import FluxControlNetModel
        from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
        from transformer_flux import FluxTransformer2DModel

        device = self.config.resolved_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        base_model = self._resolve_model_ref(self.config.flux_base_model, "FLUX.1-dev base model")
        controlnet_model = self._resolve_model_ref(
            self.config.flux_controlnet_model, "FLUX ControlNet-Inpainting model"
        )
        base_kwargs = self._local_only_kwargs(base_model)
        controlnet_kwargs = self._local_only_kwargs(controlnet_model)
        controlnet = FluxControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=dtype, **controlnet_kwargs
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            base_model,
            subfolder="transformer",
            torch_dtype=dtype,
            **base_kwargs,
        )
        pipe = FluxControlNetInpaintingPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            transformer=transformer,
            torch_dtype=dtype,
            **base_kwargs,
        )
        if device == "cuda":
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to(device)
        return pipe

    def _target_size(self, width: int, height: int) -> tuple[int, int]:
        limit = int(self.config.working_size)
        scale = limit / max(width, height)
        if scale >= 1:
            out_w, out_h = width, height
        else:
            out_w, out_h = round(width * scale), round(height * scale)
        out_w = max(64, (out_w // 8) * 8)
        out_h = max(64, (out_h // 8) * 8)
        return out_w, out_h

    def _bgr_to_pil(self, image_bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    def _blend_masked_result(
        self,
        original_bgr: np.ndarray,
        generated_bgr: np.ndarray,
        mask_u8: np.ndarray,
    ) -> np.ndarray:
        _, hard_mask = cv2.threshold(mask_u8, 1, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(hard_mask) == 0:
            return original_bgr.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blend_mask = cv2.dilate(hard_mask, kernel, iterations=1)
        blend_mask = cv2.GaussianBlur(blend_mask, (0, 0), sigmaX=1.1, sigmaY=1.1)
        alpha = (blend_mask.astype(np.float32) / 255.0)[:, :, None]
        out = (
            original_bgr.astype(np.float32) * (1.0 - alpha)
            + generated_bgr.astype(np.float32) * alpha
        )
        return np.clip(out, 0, 255).astype(np.uint8)

    def _resolve_model_ref(self, value: str, label: str) -> str:
        path = Path(value)
        if path.exists():
            if label.startswith("FLUX.1-dev"):
                required = [
                    path / "model_index.json",
                    path / "transformer",
                    path / "vae",
                    path / "text_encoder",
                    path / "text_encoder_2",
                    path / "tokenizer",
                    path / "tokenizer_2",
                ]
                missing = [str(item) for item in required if not item.exists()]
                if missing:
                    raise FileNotFoundError(
                        "本地 FLUX.1-dev 基础模型不完整。"
                        "FLUX-ControlNet-Inpainting-Alpha 只是 ControlNet 权重，"
                        "仍然需要完整的 FLUX.1-dev 基础模型。缺少："
                        + ", ".join(missing)
                    )
            return str(path)

        if "/" in value:
            return value
        raise FileNotFoundError(f"{label} not found: {value}")

    def _local_only_kwargs(self, model_ref: str) -> dict[str, bool]:
        return {"local_files_only": True} if Path(model_ref).exists() else {}

    def _check_transformers_torch_compat(self) -> None:
        try:
            from importlib.metadata import version

            torch_version = version("torch")
            transformers_version = version("transformers")
        except Exception:
            return

        torch_parts = self._major_minor(torch_version)
        transformers_parts = self._major_minor(transformers_version)
        if torch_parts < (2, 4) and transformers_parts >= (4, 57):
            raise RuntimeError(
                "The installed transformers version requires PyTorch >= 2.4, "
                f"but torch is {torch_version}. Please run "
                "`pip install transformers==4.44.2`, or upgrade PyTorch."
            )

    def _major_minor(self, version_text: str) -> tuple[int, int]:
        parts = version_text.split("+", 1)[0].split(".")
        try:
            return int(parts[0]), int(parts[1])
        except Exception:
            return 0, 0
