from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RestorationPlan:
    """Structured plan produced by an LLM or by the local fallback planner."""

    scene: str = "old photograph"
    damage: list[str] = field(default_factory=list)
    color_style: str = "natural historical photo colors"
    inpaint_prompt: str = (
        "restore the damaged parts of an old photograph, preserve the original "
        "identity, composition, clothing, lighting, and film texture"
    )
    color_prompt: str = (
        "natural colorization for an old photo, realistic skin tones, muted "
        "film colors, preserve the original content"
    )
    negative_prompt: str = (
        "modern objects, changed face, different person, plastic skin, cartoon, "
        "over-saturated colors, distorted hands, extra fingers, text artifacts"
    )
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RestorationPlan":
        allowed = cls.__dataclass_fields__.keys()
        clean = {key: data[key] for key in allowed if key in data}
        for key in ("damage", "notes"):
            value = clean.get(key)
            if isinstance(value, str):
                clean[key] = [value]
            elif value is None:
                clean[key] = []
        return cls(**clean)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RestorationConfig:
    input_path: Path
    output_dir: Path
    manual_mask_path: Path | None = None
    manual_protect_mask_path: Path | None = None

    device: str = "auto"
    seed: int = 24
    working_size: int = 1024

    enable_flux: bool = True
    enable_colorization: bool = True
    enable_llm: bool = True
    enable_cv_repair: bool = True
    enable_llm_mask: bool = True
    enable_lama_repair: bool = True
    flux_requires_manual_mask: bool = True
    flux_allow_auto_mask_fallback: bool = False

    ddcolor_repo: Path = Path("DDColor")
    ddcolor_model_path: Path | None = Path("weights/ddcolor/pytorch_model.pt")
    ddcolor_model_name: str = "ddcolor_modelscope"
    ddcolor_input_size: int = 512
    ddcolor_model_size: str = "large"

    flux_repo: Path = Path("FLUX-Controlnet-Inpainting")
    flux_base_model: str = "weights/flux/FLUX.1-dev"
    flux_controlnet_model: str = (
        "weights/flux/FLUX.1-dev/FLUX.1-dev-Controlnet-Inpainting-Alpha"
    )
    flux_steps: int = 28
    flux_control_scale: float = 0.9
    flux_guidance_scale: float = 3.5
    flux_true_guidance_scale: float = 1.0

    lama_model_path: Path = Path("weights/lama/big-lama.pt")

    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout: int = 60

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
