from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .colorizer import DDColorAdapter, read_image_bgr, write_image_bgr
from .config import RestorationConfig
from .flux_inpainter import FluxInpaintingAdapter
from .lama_inpainter import LaMaInpaintingAdapter
from .mask import DamageMaskGenerator
from .planner import RestorationPlanner


class OldPhotoRestorationPipeline:
    def __init__(self, config: RestorationConfig):
        self.config = config
        self.mask_generator = DamageMaskGenerator()
        self.planner = RestorationPlanner()
        self.flux = FluxInpaintingAdapter(config) if config.enable_flux else None
        self.lama = LaMaInpaintingAdapter(config) if config.enable_lama_repair else None
        self.colorizer = DDColorAdapter(config) if config.enable_colorization else None

    def run(self) -> Path:
        out_dir = self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        original = read_image_bgr(self.config.input_path)
        write_image_bgr(out_dir / "00_input.png", original)

        mask = self.mask_generator.build(original, self.config.manual_mask_path)
        manual_protect_mask = self._load_manual_protect_mask(mask.shape)
        if np.count_nonzero(manual_protect_mask) > 0:
            mask = self._apply_manual_protect_mask(mask, manual_protect_mask)
            write_image_bgr(
                out_dir / "01_manual_protect_mask.png",
                cv2.cvtColor(manual_protect_mask, cv2.COLOR_GRAY2BGR),
            )
        if self.mask_generator.last_preprocessed is not None:
            write_image_bgr(
                out_dir / "00_preprocessed.png",
                cv2.cvtColor(self.mask_generator.last_preprocessed, cv2.COLOR_GRAY2BGR),
            )
        mask_ratio = float(np.count_nonzero(mask) / mask.size)
        write_image_bgr(out_dir / "01_damage_mask.png", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

        plan, llm_prompt = self.planner.create_plan(
            self.config.input_path, original, mask, self.config
        )
        (out_dir / "llm_prompt.txt").write_text(llm_prompt, encoding="utf-8")
        (out_dir / "plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        llm_safe_mask = np.zeros_like(mask)
        llm_regions = []
        if self.config.enable_llm_mask:
            llm_regions, llm_mask_prompt = self.planner.create_safe_mask_regions(
                self.config.input_path, original, mask, self.config
            )
            llm_safe_mask = self._build_llm_safe_mask(original, mask, llm_regions)
            (out_dir / "llm_mask_prompt.txt").write_text(llm_mask_prompt, encoding="utf-8")
            (out_dir / "llm_safe_regions.json").write_text(
                json.dumps(llm_regions, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            write_image_bgr(
                out_dir / "01_llm_safe_mask.png",
                cv2.cvtColor(llm_safe_mask, cv2.COLOR_GRAY2BGR),
            )
            face_defect_mask = self._build_face_surface_defect_mask(original)
            if np.count_nonzero(face_defect_mask) > 0:
                llm_safe_mask = cv2.bitwise_or(llm_safe_mask, face_defect_mask)
                write_image_bgr(
                    out_dir / "01_face_defect_mask.png",
                    cv2.cvtColor(face_defect_mask, cv2.COLOR_GRAY2BGR),
                )
                write_image_bgr(
                    out_dir / "01_llm_safe_mask.png",
                    cv2.cvtColor(llm_safe_mask, cv2.COLOR_GRAY2BGR),
                )

        current = original
        repair_backend = "disabled"
        if self.config.enable_cv_repair and mask_ratio > 0:
            repair_mask = self._build_repair_mask(mask, llm_safe_mask)
            redline_guard = self._build_structure_redline_guard(original)
            write_image_bgr(
                out_dir / "01_redline_guard.png",
                cv2.cvtColor(redline_guard, cv2.COLOR_GRAY2BGR),
            )
            background_boost_mask = self._build_background_scratch_boost_mask(
                original, redline_guard
            )
            if np.count_nonzero(background_boost_mask) > 0:
                write_image_bgr(
                    out_dir / "01_background_scratch_boost_mask.png",
                    cv2.cvtColor(background_boost_mask, cv2.COLOR_GRAY2BGR),
                )
                repair_mask = cv2.bitwise_or(repair_mask, background_boost_mask)
            repair_mask = self._apply_redline_guard(original, repair_mask, redline_guard)
            repair_mask = self._apply_manual_protect_mask(repair_mask, manual_protect_mask)
            write_image_bgr(
                out_dir / "01_repair_mask.png",
                cv2.cvtColor(repair_mask, cv2.COLOR_GRAY2BGR),
            )
            current, repair_backend = self._repair_with_best_backend(current, repair_mask)
            residual_mask = self._build_residual_bright_damage_mask(original, current)
            residual_mask = self._apply_manual_protect_mask(residual_mask, manual_protect_mask)
            if np.count_nonzero(residual_mask) > 0:
                current, residual_backend = self._repair_residual_damage(current, residual_mask)
                repair_backend = f"{repair_backend}+residual-{residual_backend}"
                write_image_bgr(
                    out_dir / "02_residual_damage_mask.png",
                    cv2.cvtColor(residual_mask, cv2.COLOR_GRAY2BGR),
                )
            if repair_backend == "lama":
                write_image_bgr(out_dir / "02_lama_repaired.png", current)
            else:
                write_image_bgr(out_dir / "02_cv_repaired.png", current)

        flux_mask, flux_mask_source = self._select_flux_mask(
            mask, llm_safe_mask, manual_protect_mask
        )
        flux_mask_ratio = float(np.count_nonzero(flux_mask) / flux_mask.size)

        flux_status = "disabled"
        if self.flux is not None:
            if flux_mask_ratio <= 0:
                flux_status = "skipped: empty damage mask"
            elif self.config.flux_requires_manual_mask and not self.config.manual_mask_path:
                flux_status = "skipped: manual mask required"
            else:
                flux_status = f"run: {flux_mask_source}"

        use_flux = (
            self.flux is not None
            and flux_mask_ratio > 0
            and (not self.config.flux_requires_manual_mask or self.config.manual_mask_path)
        )
        if use_flux:
            before_flux = current
            current = self.flux.inpaint(current, flux_mask, plan)
            current = self._suppress_new_face_bright_artifacts(before_flux, current)
            write_image_bgr(out_dir / "03_flux_repaired.png", current)

        color_status = "disabled"
        if self.colorizer is not None:
            pre_color = current
            if self._is_already_color_photo(pre_color):
                current = self._enhance_existing_color_photo(pre_color)
                color_status = "enhanced: input already has color"
            else:
                current = self.colorizer.colorize(current)
                current = self._stabilize_colorization(pre_color, current)
                color_status = "run"
            write_image_bgr(out_dir / "04_ddcolor_colorized.png", current)

        final_path = out_dir / "final.png"
        write_image_bgr(final_path, current)
        quality_review = {}
        if self.config.enable_llm:
            quality_review, review_prompt = self.planner.review_result(
                self.config.input_path, final_path, self.config
            )
            (out_dir / "quality_review_prompt.txt").write_text(
                review_prompt, encoding="utf-8"
            )
            if quality_review:
                (out_dir / "quality_review.json").write_text(
                    json.dumps(quality_review, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        self._write_report(
            out_dir,
            final_path,
            plan,
            mask_ratio,
            flux_status,
            repair_backend,
            color_status,
            llm_regions,
            quality_review,
        )
        return final_path

    def _write_report(
        self,
        out_dir: Path,
        final_path: Path,
        plan,
        mask_ratio: float,
        flux_status: str,
        repair_backend: str,
        color_status: str,
        llm_regions: list[dict],
        quality_review: dict,
    ) -> None:
        lines = [
            "# Old Photo Restoration Report",
            "",
            f"- Input: `{self.config.input_path}`",
            f"- Final output: `{final_path}`",
            f"- Device: `{self.config.resolved_device()}`",
            f"- FLUX enabled: `{self.config.enable_flux}`",
            f"- FLUX status: `{flux_status}`",
            f"- FLUX auto mask fallback: `{self.config.flux_allow_auto_mask_fallback}`",
            f"- DDColor enabled: `{self.config.enable_colorization}`",
            f"- DDColor status: `{color_status}`",
            f"- LaMa repair enabled: `{self.config.enable_lama_repair}`",
            f"- LaMa model path: `{self.config.lama_model_path}`",
            f"- Repair backend: `{repair_backend}`",
            f"- OpenCV fine repair enabled: `{self.config.enable_cv_repair}`",
            f"- LLM mask enabled: `{self.config.enable_llm_mask}`",
            f"- FLUX requires manual mask: `{self.config.flux_requires_manual_mask}`",
            f"- Damage mask coverage: `{mask_ratio * 100:.4f}%`",
            "",
            "## Plan",
            "",
            f"- Scene: {plan.scene}",
            f"- Damage: {', '.join(plan.damage) if plan.damage else 'none detected'}",
            f"- Color style: {plan.color_style}",
            f"- Inpaint prompt: {plan.inpaint_prompt}",
            f"- Negative prompt: {plan.negative_prompt}",
        ]
        if self.config.enable_llm_mask:
            lines.extend(["", "## LLM Mask Regions", ""])
            if llm_regions:
                lines.extend(f"- `{region}`" for region in llm_regions)
            else:
                lines.append("- No safe automatic mask regions selected.")
        if plan.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {note}" for note in plan.notes)
        if quality_review:
            lines.extend(["", "## LLM Quality Review", ""])
            lines.append(f"- Overall pass: `{quality_review.get('overall_pass')}`")
            lines.append(f"- Score: `{quality_review.get('score', 'n/a')}`")
            action = quality_review.get("recommended_action")
            if action:
                lines.append(f"- Recommended action: {action}")
            color_notes = quality_review.get("color_notes")
            if color_notes:
                lines.append(f"- Color notes: {color_notes}")
            issues = quality_review.get("issues")
            if isinstance(issues, list) and issues:
                lines.extend(["", "### Issues", ""])
                for issue in issues:
                    lines.append(f"- `{issue}`")
        (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    def _cv_inpaint(self, image_bgr, mask, radius: int = 2):
        return cv2.inpaint(image_bgr, mask, radius, cv2.INPAINT_TELEA)

    def _repair_with_best_backend(self, image_bgr, mask):
        if np.count_nonzero(mask) == 0:
            return image_bgr.copy(), "empty"

        face_guard = self._build_face_guard(image_bgr)
        face_mask = self._build_face_micro_repair_mask(image_bgr, mask, face_guard)
        non_face_mask = mask.copy()
        if np.count_nonzero(face_guard) > 0:
            non_face_mask[face_guard > 0] = 0

        current = image_bgr.copy()
        backends = []
        large_non_face_mask = self._select_lama_region_mask(non_face_mask)
        fine_non_face_mask = non_face_mask.copy()
        if np.count_nonzero(large_non_face_mask) > 0:
            fine_non_face_mask[large_non_face_mask > 0] = 0

        if np.count_nonzero(large_non_face_mask) > 0 and self.lama is not None and self.lama.is_available():
            try:
                current = self.lama.inpaint(current, large_non_face_mask)
                backends.append("lama")
            except Exception:
                current = self._cv_inpaint(current, large_non_face_mask, radius=3)
                backends.append("opencv-r3-large")

        if np.count_nonzero(fine_non_face_mask) > 0:
            current = self._cv_inpaint(current, fine_non_face_mask, radius=2)
            backends.append("opencv-r2-fine")

        if np.count_nonzero(face_mask) > 0:
            current = self._cv_inpaint(current, face_mask, radius=1)
            backends.append("face-opencv-r1")

        current = self._suppress_new_face_bright_artifacts(image_bgr, current)
        return current, "+".join(backends) if backends else "empty"

    def _build_repair_mask(self, auto_mask: np.ndarray, llm_safe_mask: np.ndarray) -> np.ndarray:
        if np.count_nonzero(llm_safe_mask) == 0:
            return auto_mask
        return cv2.bitwise_or(auto_mask, llm_safe_mask)

    def _load_manual_protect_mask(self, shape: tuple[int, int]) -> np.ndarray:
        protect = np.zeros(shape, dtype=np.uint8)
        if not self.config.manual_protect_mask_path:
            return protect
        manual = cv2.imread(str(self.config.manual_protect_mask_path), cv2.IMREAD_GRAYSCALE)
        if manual is None:
            raise FileNotFoundError(
                f"Cannot read manual protect mask: {self.config.manual_protect_mask_path}"
            )
        h, w = shape
        manual = cv2.resize(manual, (w, h), interpolation=cv2.INTER_NEAREST)
        _, manual = cv2.threshold(manual, 1, 255, cv2.THRESH_BINARY)
        return manual

    def _apply_manual_protect_mask(
        self, mask: np.ndarray, protect_mask: np.ndarray
    ) -> np.ndarray:
        if np.count_nonzero(mask) == 0 or np.count_nonzero(protect_mask) == 0:
            return mask
        out = mask.copy()
        out[protect_mask > 0] = 0
        return out

    def _build_structure_redline_guard(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0, sigmaY=2.0)
        dark_residual = cv2.subtract(blur, gray)
        dark_edges = (dark_residual >= max(14, int(np.percentile(dark_residual, 97.4)))).astype(np.uint8) * 255

        canny = cv2.Canny(gray, 45, 130)
        structure = cv2.bitwise_and(canny, dark_edges)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        structure = cv2.dilate(structure, kernel, iterations=1)

        face_guard = self._build_face_guard(image_bgr)
        face_roi = self._build_face_surface_roi(image_bgr)
        guard = cv2.bitwise_or(structure, face_guard)
        guard = cv2.bitwise_or(guard, face_roi)
        return guard

    def _apply_redline_guard(
        self,
        image_bgr: np.ndarray,
        repair_mask: np.ndarray,
        redline_guard: np.ndarray,
    ) -> np.ndarray:
        if np.count_nonzero(repair_mask) == 0:
            return repair_mask

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0, sigmaY=2.0)
        bright_residual = cv2.subtract(gray, blur)
        face_guard = self._build_face_guard(image_bgr)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(repair_mask, 8)
        out = np.zeros_like(repair_mask)
        for label in range(1, num_labels):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            longest = max(width, height)
            shortest = max(1, min(width, height))
            fill_ratio = area / max(1, width * height)

            face_overlap = np.count_nonzero(component & (face_guard > 0)) / max(1, area)
            redline_overlap = np.count_nonzero(component & (redline_guard > 0)) / max(1, area)
            median_bright = float(np.median(bright_residual[component]))
            is_tiny_face_damage = (
                face_overlap > 0
                and (
                    (area <= 32 and longest <= 11)
                    or (
                        area <= 150
                        and shortest <= 5
                        and longest / shortest >= 3.4
                        and fill_ratio <= 0.45
                    )
                )
                and median_bright >= 10
            )
            is_confident_surface_damage = (
                median_bright >= 16
                and (area <= 120 or (longest / shortest >= 2.3 and fill_ratio <= 0.54))
            )

            if face_overlap > 0 and not is_tiny_face_damage:
                continue
            if redline_overlap >= 0.35 and not is_confident_surface_damage:
                continue
            out[component] = 255
        return out

    def _build_background_scratch_boost_mask(
        self, image_bgr: np.ndarray, redline_guard: np.ndarray
    ) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.8, sigmaY=2.8)
        bright_residual = cv2.subtract(enhanced, blur)

        threshold = max(12, int(np.percentile(bright_residual[redline_guard == 0], 98.0)))
        candidate = (bright_residual >= threshold).astype(np.uint8) * 255
        very_bright = cv2.inRange(enhanced, 212, 255)
        candidate = cv2.bitwise_or(candidate, very_bright)
        candidate[redline_guard > 0] = 0

        return self._filter_background_scratch_components(candidate, image_area=gray.size)

    def _filter_background_scratch_components(
        self, mask: np.ndarray, image_area: int
    ) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = np.zeros_like(mask)
        max_area = max(120, int(image_area * 0.005))
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 2 or area > max_area:
                continue
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            longest = max(width, height)
            shortest = max(1, min(width, height))
            fill_ratio = area / max(1, width * height)
            is_speck = area <= 68 and longest <= 18 and fill_ratio <= 0.90
            is_scratch = longest >= 7 and longest / shortest >= 1.8 and fill_ratio <= 0.68
            if is_speck or is_scratch:
                out[labels == label] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.dilate(out, kernel, iterations=1)

    def _select_lama_region_mask(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = np.zeros_like(mask)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            fill_ratio = area / max(1, width * height)
            shortest = max(1, min(width, height))
            is_block_damage = area >= 260 and fill_ratio >= 0.30 and shortest >= 8
            is_wide_missing_region = area >= 700 and shortest >= 14
            if is_block_damage or is_wide_missing_region:
                out[labels == label] = 255
        return out

    def _repair_residual_damage(self, image_bgr, mask):
        face_guard = self._build_face_guard(image_bgr)
        face_mask = self._build_face_micro_repair_mask(image_bgr, mask, face_guard)
        non_face_mask = mask.copy()
        if np.count_nonzero(face_guard) > 0:
            non_face_mask[face_guard > 0] = 0

        current = image_bgr.copy()
        backends = []
        if np.count_nonzero(non_face_mask) > 0:
            current = self._cv_inpaint(current, non_face_mask, radius=2)
            backends.append("opencv-r2")
        if np.count_nonzero(face_mask) > 0:
            current = self._cv_inpaint(current, face_mask, radius=1)
            backends.append("face-opencv-r1")
        return current, "+".join(backends) if backends else "empty"

    def _build_face_micro_repair_mask(
        self, image_bgr: np.ndarray, mask: np.ndarray, face_guard: np.ndarray | None = None
    ) -> np.ndarray:
        if face_guard is None:
            face_guard = self._build_face_guard(image_bgr)
        if np.count_nonzero(face_guard) == 0:
            return np.zeros_like(mask)
        face_candidate = cv2.bitwise_and(mask, face_guard)
        return self._filter_face_micro_defects(face_candidate)

    def _build_residual_bright_damage_mask(
        self, original_bgr: np.ndarray, repaired_bgr: np.ndarray
    ) -> np.ndarray:
        gray = cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.2, sigmaY=2.2)
        bright_residual = cv2.subtract(gray, blur)
        threshold = max(18, int(np.percentile(bright_residual, 99.0)))
        bright = (bright_residual >= threshold).astype(np.uint8) * 255

        edges = cv2.Canny(gray, 35, 105)
        line_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        line_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        lines = cv2.bitwise_or(
            cv2.morphologyEx(edges, cv2.MORPH_CLOSE, line_kernel_h),
            cv2.morphologyEx(edges, cv2.MORPH_CLOSE, line_kernel_v),
        )
        candidate = cv2.bitwise_or(bright, cv2.bitwise_and(bright, lines))

        face_guard = self._build_face_guard(original_bgr)
        non_face = candidate.copy()
        if np.count_nonzero(face_guard) > 0:
            non_face[face_guard > 0] = 0

        non_face = self._filter_residual_non_face_defects(non_face, image_area=gray.size)
        face_micro = self._build_face_micro_repair_mask(original_bgr, candidate, face_guard)
        out = cv2.bitwise_or(non_face, face_micro)
        if np.count_nonzero(face_guard) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            non_face_pixels = cv2.bitwise_and(out, cv2.bitwise_not(face_guard))
            non_face_pixels = cv2.dilate(non_face_pixels, kernel, iterations=1)
            out[face_guard == 0] = non_face_pixels[face_guard == 0]
        return out

    def _filter_residual_non_face_defects(
        self, mask: np.ndarray, image_area: int
    ) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = np.zeros_like(mask)
        max_area = max(60, int(image_area * 0.004))
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 2 or area > max_area:
                continue
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            longest = max(width, height)
            shortest = max(1, min(width, height))
            fill_ratio = area / max(1, width * height)
            is_speck = area <= 60 and longest <= 16
            is_scratch = longest >= 7 and longest / shortest >= 2.0 and fill_ratio <= 0.68
            if is_speck or is_scratch:
                out[labels == label] = 255
        return out

    def _select_flux_mask(
        self,
        auto_mask: np.ndarray,
        llm_safe_mask: np.ndarray,
        manual_protect_mask: np.ndarray,
    ):
        if self.config.manual_mask_path:
            manual_mask = self._load_binary_mask(self.config.manual_mask_path, auto_mask.shape)
            manual_mask = self._apply_manual_protect_mask(manual_mask, manual_protect_mask)
            return manual_mask, "manual mask only"
        if self.config.flux_allow_auto_mask_fallback:
            if np.count_nonzero(llm_safe_mask) > 0:
                return llm_safe_mask, "llm safe mask"
            return auto_mask, "auto mask fallback"
        return np.zeros_like(auto_mask), "no safe automatic mask"

    def _load_binary_mask(self, path: Path, shape: tuple[int, int]) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {path}")
        h, w = shape
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        return mask

    def _build_llm_safe_mask(
        self, image_bgr: np.ndarray, auto_mask: np.ndarray, regions: list[dict]
    ) -> np.ndarray:
        h, w = auto_mask.shape[:2]
        out = np.zeros_like(auto_mask)
        face_guard = self._build_face_guard(image_bgr)
        for region in regions[:18]:
            bbox = region.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                confidence = float(region.get("confidence", 0.0))
                x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            except (TypeError, ValueError):
                continue
            if confidence < 0.40:
                continue

            x1, x2 = sorted((max(0, x1), min(w, x2)))
            y1, y2 = sorted((max(0, y1), min(h, y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
            touches_border = x1 == 0 or y1 == 0 or x2 == w or y2 == h
            if touches_border:
                band = max(40, min(w, h) // 6)
                if x1 == 0:
                    x2 = min(x2, band)
                if x2 == w:
                    x1 = max(x1, w - band)
                if y1 == 0:
                    y2 = min(y2, band)
                if y2 == h:
                    y1 = max(y1, h - band)
                area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
            elif confidence < 0.52:
                continue

            if area_ratio < 0.00006 or area_ratio > 0.20:
                continue

            auto_crop = auto_mask[y1:y2, x1:x2]
            local_crop = self._local_llm_region_defects(image_bgr[y1:y2, x1:x2])
            crop = cv2.bitwise_or(auto_crop, local_crop)
            guard_crop = face_guard[y1:y2, x1:x2]
            if np.count_nonzero(guard_crop) > 0:
                face_allowed = self._filter_face_micro_defects(crop)
                crop[guard_crop > 0] = face_allowed[guard_crop > 0]
            if np.count_nonzero(crop) == 0:
                continue
            out[y1:y2, x1:x2] = cv2.bitwise_or(out[y1:y2, x1:x2], crop)

        if np.count_nonzero(out) == 0:
            return out
        face_pixels = cv2.bitwise_and(out, face_guard)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
        out = cv2.dilate(out, kernel, iterations=1)
        out[face_guard > 0] = face_pixels[face_guard > 0]
        return out

    def _local_llm_region_defects(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.size == 0:
            return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.6, sigmaY=1.6)
        bright_residual = cv2.subtract(gray, blur)
        bright_threshold = max(7, int(np.percentile(bright_residual, 96.0)))
        bright = (bright_residual >= bright_threshold).astype(np.uint8) * 255

        dark_residual = cv2.subtract(blur, gray)
        dark_threshold = max(10, int(np.percentile(dark_residual, 97.8)))
        dark = (dark_residual >= dark_threshold).astype(np.uint8) * 255

        edges = cv2.Canny(gray, 40, 110)
        thin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        line_candidates = cv2.bitwise_or(
            cv2.morphologyEx(edges, cv2.MORPH_CLOSE, thin_kernel),
            cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel),
        )
        high_contrast = cv2.bitwise_or(bright, dark)
        defect = cv2.bitwise_or(high_contrast, cv2.bitwise_and(line_candidates, high_contrast))
        return self._filter_local_defects(defect, image_area=gray.size)

    def _filter_local_defects(self, mask: np.ndarray, image_area: int) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = np.zeros_like(mask)
        max_area = max(24, int(image_area * 0.024))
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 2 or area > max_area:
                continue
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            longest = max(width, height)
            shortest = max(1, min(width, height))
            fill_ratio = area / max(1, width * height)
            if area <= 100 or (longest >= 7 and longest / shortest >= 1.8 and fill_ratio <= 0.82):
                out[labels == label] = 255
        return out

    def _filter_face_micro_defects(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = np.zeros_like(mask)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 2 or area > 120:
                continue
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            longest = max(width, height)
            shortest = max(1, min(width, height))
            fill_ratio = area / max(1, width * height)
            is_speck = area <= 32 and longest <= 11 and fill_ratio <= 0.86
            is_thin_scratch = (
                area <= 150
                and 8 <= longest <= 90
                and shortest <= 5
                and longest / shortest >= 3.2
                and fill_ratio <= 0.45
            )
            if is_speck or is_thin_scratch:
                out[labels == label] = 255
        return out

    def _filter_face_defects(self, mask: np.ndarray) -> np.ndarray:
        return self._filter_face_micro_defects(mask)

    def _build_face_guard(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        guard = np.zeros(gray.shape, dtype=np.uint8)
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return guard

        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            return guard

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(max(24, gray.shape[1] // 18), max(24, gray.shape[0] // 18)),
        )
        for x, y, fw, fh in faces:
            pad_x = int(fw * 0.18)
            pad_y = int(fh * 0.22)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(gray.shape[1], x + fw + pad_x)
            y2 = min(gray.shape[0], y + fh + pad_y)
            cv2.rectangle(guard, (x1, y1), (x2, y2), 255, thickness=-1)
        return guard

    def _build_face_surface_defect_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        face_roi = self._build_face_surface_roi(image_bgr)
        if np.count_nonzero(face_roi) == 0:
            return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0, sigmaY=2.0)
        bright_residual = cv2.subtract(gray, blur)
        bright_threshold = max(24, int(np.percentile(bright_residual[face_roi > 0], 98.8)))
        bright = (bright_residual >= bright_threshold).astype(np.uint8) * 255
        bright = cv2.bitwise_and(bright, face_roi)
        return self._filter_face_micro_defects(bright)

    def _build_face_surface_roi(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        roi = np.zeros(gray.shape, dtype=np.uint8)
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return roi

        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            return roi

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(max(24, gray.shape[1] // 18), max(24, gray.shape[0] // 18)),
        )
        for x, y, fw, fh in faces:
            center = (int(x + fw * 0.5), int(y + fh * 0.52))
            axes = (max(8, int(fw * 0.38)), max(8, int(fh * 0.42)))
            cv2.ellipse(roi, center, axes, 0, 0, 360, 255, thickness=-1)
        return roi

    def _suppress_new_face_bright_artifacts(
        self, before_bgr: np.ndarray, after_bgr: np.ndarray
    ) -> np.ndarray:
        face_guard = self._build_face_guard(before_bgr)
        if np.count_nonzero(face_guard) == 0:
            return after_bgr

        before_gray = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)
        new_bright = (
            (after_gray.astype(np.int16) - before_gray.astype(np.int16) > 26)
            & (after_gray > 178)
            & (face_guard > 0)
        ).astype(np.uint8) * 255

        artifact_mask = self._filter_new_face_artifact_components(new_bright)
        if np.count_nonzero(artifact_mask) == 0:
            return after_bgr

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        artifact_mask = cv2.dilate(artifact_mask, kernel, iterations=1)
        out = after_bgr.copy()
        alpha = (artifact_mask.astype(np.float32) / 255.0)[:, :, None] * 0.85
        out = (out.astype(np.float32) * (1.0 - alpha) + before_bgr.astype(np.float32) * alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _filter_new_face_artifact_components(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = np.zeros_like(mask)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 2 or area > 220:
                continue
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            fill_ratio = area / max(1, width * height)
            if fill_ratio <= 0.9:
                out[labels == label] = 255
        return out

    def _stabilize_colorization(
        self, base_bgr: np.ndarray, colorized_bgr: np.ndarray
    ) -> np.ndarray:
        base_lab = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2LAB)
        color_lab = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        color_lab[:, :, 0] = (
            base_lab[:, :, 0].astype(np.float32) * 0.72
            + color_lab[:, :, 0] * 0.28
        )
        color_lab[:, :, 1] = 128.0 + (color_lab[:, :, 1] - 128.0) * 0.82
        color_lab[:, :, 2] = 128.0 + (color_lab[:, :, 2] - 128.0) * 0.74

        muted = cv2.cvtColor(
            np.clip(color_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
        )
        hsv = cv2.cvtColor(muted, cv2.COLOR_BGR2HSV).astype(np.float32)
        skin_like = (
            (hsv[:, :, 0] >= 4)
            & (hsv[:, :, 0] <= 28)
            & (hsv[:, :, 1] > 58)
            & (hsv[:, :, 2] > 80)
        )
        face_guard = self._build_face_guard(base_bgr) > 0
        warm_skin = skin_like & face_guard
        hsv[:, :, 1][warm_skin] *= 0.90
        hsv[:, :, 2][warm_skin] *= 0.96
        stabilized = cv2.cvtColor(
            np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR
        )
        return cv2.addWeighted(stabilized, 0.93, base_bgr, 0.07, 0.0)

    def _is_already_color_photo(self, image_bgr: np.ndarray) -> bool:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)
        value = hsv[:, :, 2]
        valid = (value > 35) & (value < 245)
        if np.count_nonzero(valid) < max(64, int(image_bgr.shape[0] * image_bgr.shape[1] * 0.05)):
            return False

        valid_saturation = saturation[valid]
        mean_sat = float(np.mean(valid_saturation))
        p80_sat = float(np.percentile(valid_saturation, 80))
        p95_sat = float(np.percentile(valid_saturation, 95))
        return mean_sat >= 12.0 and p80_sat >= 20.0 and p95_sat >= 34.0

    def _enhance_existing_color_photo(self, image_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
        restored_l = cv2.addWeighted(clahe.apply(l_channel), 0.38, l_channel, 0.62, 0.0)

        lab_f = cv2.merge([restored_l, a_channel, b_channel]).astype(np.float32)
        lab_f[:, :, 1] = 128.0 + (lab_f[:, :, 1] - 128.0) * 1.28
        lab_f[:, :, 2] = 128.0 + (lab_f[:, :, 2] - 128.0) * 1.24
        enhanced = cv2.cvtColor(
            np.clip(lab_f, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
        )

        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.22
        hsv[:, :, 2] *= 1.025
        return cv2.cvtColor(
            np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR
        )
