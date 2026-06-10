from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import RestorationConfig, RestorationPlan


@dataclass
class ImageAnalysis:
    width: int
    height: int
    is_grayscale: bool
    mean_brightness: float
    mean_saturation: float
    damage_ratio: float

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "width": self.width,
            "height": self.height,
            "is_grayscale": self.is_grayscale,
            "mean_brightness": round(self.mean_brightness, 3),
            "mean_saturation": round(self.mean_saturation, 3),
            "damage_ratio": round(self.damage_ratio, 4),
        }


class RestorationPlanner:
    def create_plan(
        self,
        image_path: Path,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        config: RestorationConfig,
    ) -> tuple[RestorationPlan, str]:
        analysis = self._analyze(image_bgr, mask)
        prompt = self._build_llm_prompt(analysis)

        if config.enable_llm and config.llm_base_url and config.llm_model:
            try:
                return self._create_llm_plan(image_path, prompt, config), prompt
            except Exception as exc:
                fallback = self._heuristic_plan(analysis)
                fallback.notes.append(f"LLM planner failed, used heuristic plan: {exc}")
                return fallback, prompt

        plan = self._heuristic_plan(analysis)
        if config.enable_llm:
            plan.notes.append(
                "No LLM endpoint configured. Set --llm-base-url and --llm-model "
                "to enable OpenAI-compatible JSON planning."
            )
        return plan, prompt

    def create_safe_mask_regions(
        self,
        image_path: Path,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        config: RestorationConfig,
    ) -> tuple[list[dict], str]:
        analysis = self._analyze(image_bgr, mask)
        prompt = self._build_mask_prompt(analysis)
        if not (config.enable_llm and config.llm_base_url and config.llm_model):
            return [], prompt

        try:
            data = self._request_json(image_path, prompt, config)
        except Exception:
            return [], prompt

        regions = data.get("regions", data if isinstance(data, list) else [])
        if not isinstance(regions, list):
            return [], prompt
        return [region for region in regions if isinstance(region, dict)], prompt

    def review_result(
        self,
        original_path: Path,
        final_path: Path,
        config: RestorationConfig,
    ) -> tuple[dict, str]:
        prompt = self._build_review_prompt(config)
        if not (config.enable_llm and config.llm_base_url and config.llm_model):
            return {}, prompt

        try:
            return self._request_json([original_path, final_path], prompt, config), prompt
        except Exception as exc:
            return {
                "overall_pass": False,
                "issues": [
                    {
                        "type": "review_failed",
                        "severity": "low",
                        "description": f"LLM review failed: {exc}",
                    }
                ],
            }, prompt

    def _analyze(self, image_bgr: np.ndarray, mask: np.ndarray) -> ImageAnalysis:
        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        channel_diff = np.max(image_bgr, axis=2) - np.min(image_bgr, axis=2)
        is_grayscale = bool(np.percentile(channel_diff, 95) < 8)
        return ImageAnalysis(
            width=w,
            height=h,
            is_grayscale=is_grayscale,
            mean_brightness=float(np.mean(gray) / 255.0),
            mean_saturation=float(np.mean(saturation)),
            damage_ratio=float(np.count_nonzero(mask) / mask.size),
        )

    def _heuristic_plan(self, analysis: ImageAnalysis) -> RestorationPlan:
        damage = []
        if analysis.damage_ratio > 0.002:
            damage.append("scratches or local stains")
        if analysis.mean_brightness < 0.32:
            damage.append("underexposure")
        if analysis.mean_brightness > 0.72:
            damage.append("faded highlights")
        if analysis.is_grayscale or analysis.mean_saturation < 0.08:
            damage.append("missing or faded color")

        scene = "historical black and white photograph" if analysis.is_grayscale else "aged color photograph"
        return RestorationPlan(
            scene=scene,
            damage=damage,
            color_style=(
                "natural film colorization, restrained saturation, realistic skin "
                "tones, period-appropriate clothing and background colors"
            ),
            inpaint_prompt=(
                f"{scene}, repair scratches, cracks, stains, and missing emulsion; "
                "preserve identity, pose, clothing, background, lens softness, grain, "
                "eyes, eyelids, mouth, skin texture, and the original old photo "
                "character; fill only the masked physical photo damage"
            ),
            color_prompt=(
                "colorize the old photograph naturally, keep luminance structure, "
                "avoid changing faces or objects, use believable historical colors"
            ),
            negative_prompt=(
                "new objects, modern fashion, changed identity, altered face, "
                "over-sharpened skin, waxy skin, neon colors, anime, painting, "
                "blurred eyes, missing eyes, changed gaze, smoothed eyelids, white "
                "patches on skin, distorted hands, duplicated people, text artifacts"
            ),
        )

    def _build_llm_prompt(self, analysis: ImageAnalysis) -> str:
        return (
            "You are planning an old photo restoration workflow. Return one JSON "
            "object with keys: scene, damage, color_style, inpaint_prompt, "
            "color_prompt, negative_prompt, notes. The inpaint_prompt will be sent "
            "directly to FLUX inpainting, so write it as a conservative restoration "
            "instruction: repair only masked physical photo damage such as dust, "
            "scratches, cracks, stains, and missing emulsion. It must preserve the "
            "original identity, gaze, eyes, eyelids, mouth, hands, clothing, "
            "composition, lighting, film grain, and scan texture. Do not ask for "
            "beautification, modernization, sharpening, face reconstruction, or "
            "creative redrawing. The negative_prompt must explicitly reject changed "
            "faces, missing or blurred eyes, waxy/smoothed skin, new white patches, "
            "new highlights, modern objects, text artifacts, and over-saturated "
            "colors. Describe only restoration/colorization instructions.\n\n"
            f"Image analysis: {json.dumps(analysis.to_dict(), ensure_ascii=False)}"
        )

    def _build_mask_prompt(self, analysis: ImageAnalysis) -> str:
        return (
            "You are assisting mask creation for LaMa/OpenCV old photo restoration. "
            "Return strict JSON with one key, regions. regions must be a list of "
            "at most 18 objects. Each object must contain bbox [x1,y1,x2,y2] in "
            "original image pixels, reason, confidence from 0 to 1, and repair_mode "
            "chosen from fine_scratch, white_speck, stain, border_damage, or skip.\n\n"
            "White mask means repair; black means preserve. Be slightly assertive "
            "about obvious physical damage, but never about facial structure. Select only physical "
            "photo damage: bright scratches, white specks, dust, mold spots, stains, "
            "paper tears, missing emulsion, and damaged borders/background. Prefer "
            "narrow boxes around scratch bands or small boxes around specks. For "
            "large background areas, the box may be broader, but it must not include "
            "faces or important object contours unless the visible damage is isolated.\n\n"
            "Hard red lines: do NOT select eyes, eyelids, nose/mouth edges, face "
            "outline, hairline, eyeglass frames, buttons, frog closures, clothing "
            "seams/edges, hands/fingers, baby hat rib texture, or any main body "
            "contour. Face/skin/clothing may be included only when the box is tight "
            "around an isolated white speck or a very thin scratch; never select a "
            "whole face, whole hand, whole torso, or broad skin/clothing area. Pay "
            "special attention to small bright scratches or white specks on children/"
            "babies' faces and clothing, adult clothing, and plain background, but "
            "only return very tight boxes around the actual white damage. Include "
            "multiple small boxes when several isolated white scratches or specks "
            "are visible in the same person or garment. If a "
            "candidate overlaps an important contour, either shrink the bbox tightly "
            "to the damage or omit it. If uncertain about important structure, omit it; "
            "if uncertain only because the scratch is faint but isolated, include a tight box.\n\n"
            "The downstream system will intersect your bboxes with automatic bright "
            "scratch detection and a redline guard, so choose safe damaged regions, "
            "not artistic repair instructions. If there is no safe automatic region, "
            "return {\"regions\": []}.\n\n"
            f"Image analysis: {json.dumps(analysis.to_dict(), ensure_ascii=False)}"
        )

    def _build_review_prompt(self, config: RestorationConfig) -> str:
        runtime_context = {
            "colorization_enabled": bool(config.enable_colorization),
            "flux_enabled": bool(config.enable_flux),
            "flux_requires_manual_mask": bool(config.flux_requires_manual_mask),
            "llm_mask_enabled": bool(config.enable_llm_mask),
            "lama_repair_enabled": bool(config.enable_lama_repair),
            "opencv_fine_repair_enabled": bool(config.enable_cv_repair),
        }
        return (
            "You are a strict quality reviewer for old photo restoration. You will "
            "receive two images: first the original scan, then the restored final. "
            f"Runtime context: {json.dumps(runtime_context, ensure_ascii=False)}. "
            "If colorization_enabled is false, do not report colorization problems; "
            "review the result as a grayscale restoration. "
            "Return strict JSON with keys: overall_pass, score from 0 to 100, "
            "issues, recommended_action, color_notes. issues must be a list of "
            "objects with type, severity, region, description, and recommendation. "
            "Check for: changed faces or gaze, missing or blurred eyes, waxy skin, "
            "new white patches on faces, over-smoothed facial details, remaining "
            "obvious scratches, unnatural colorization, over-orange skin, modernized "
            "clothing/background, and any new artifacts. Be conservative: preserving "
            "identity and original photo texture matters more than removing every "
            "scratch. If the result is acceptable despite minor residual scratches, "
            "say so."
        )

    def _create_llm_plan(
        self, image_path: Path | list[Path], prompt: str, config: RestorationConfig
    ) -> RestorationPlan:
        url = config.llm_base_url.rstrip("/")
        data = self._request_json(image_path, prompt, config)
        return RestorationPlan.from_dict(data)

    def _request_json(
        self, image_path: Path | list[Path], prompt: str, config: RestorationConfig
    ) -> dict:
        url = config.llm_base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"

        image_paths = image_path if isinstance(image_path, list) else [image_path]
        user_content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{self._mime_type(path)};base64,"
                            f"{self._image_b64(path)}"
                        )
                    },
                }
            )

        payload = {
            "model": config.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You return strict JSON for old photo restoration.",
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        api_key = config.llm_api_key or os.environ.get("LLM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.llm_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        content = body["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return json.loads(content)

    def _image_b64(self, image_path: Path) -> str:
        return base64.b64encode(image_path.read_bytes()).decode("ascii")

    def _mime_type(self, image_path: Path) -> str:
        mime, _ = mimetypes.guess_type(str(image_path))
        return mime or "image/png"
