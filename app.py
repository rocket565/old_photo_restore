from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def _sanitize_runtime_env() -> None:
    value = os.environ.get("OMP_NUM_THREADS")
    try:
        is_valid = value is None or int(value) > 0
    except ValueError:
        is_valid = False
    if not is_valid:
        os.environ["OMP_NUM_THREADS"] = "1"
    if Path("weights/flux/FLUX.1-dev/model_index.json").exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_sanitize_runtime_env()
_load_dotenv()


def build_config_from_args(args: argparse.Namespace):
    from old_photo_restore import RestorationConfig

    return RestorationConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        manual_mask_path=args.manual_mask,
        manual_protect_mask_path=args.manual_protect_mask,
        device=args.device,
        seed=args.seed,
        working_size=args.working_size,
        enable_flux=not args.no_flux,
        enable_colorization=not args.no_color,
        enable_llm=not args.no_llm,
        enable_llm_mask=not args.no_llm_mask,
        enable_lama_repair=not args.no_lama_repair,
        enable_cv_repair=not args.no_cv_repair,
        flux_requires_manual_mask=(not args.flux_auto_mask) or args.flux_manual_mask_only,
        flux_allow_auto_mask_fallback=args.flux_auto_mask,
        ddcolor_repo=args.ddcolor_repo,
        ddcolor_model_path=args.ddcolor_model_path,
        ddcolor_model_name=args.ddcolor_model_name,
        ddcolor_input_size=args.ddcolor_input_size,
        ddcolor_model_size=args.ddcolor_model_size,
        flux_repo=args.flux_repo,
        flux_base_model=args.flux_base_model,
        flux_controlnet_model=args.flux_controlnet_model,
        flux_steps=args.flux_steps,
        flux_control_scale=args.flux_control_scale,
        flux_guidance_scale=args.flux_guidance_scale,
        flux_true_guidance_scale=args.flux_true_guidance_scale,
        lama_model_path=args.lama_model_path,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_timeout=args.llm_timeout,
    )


def run_cli(args: argparse.Namespace) -> None:
    from old_photo_restore import OldPhotoRestorationPipeline

    config = build_config_from_args(args)
    final_path = OldPhotoRestorationPipeline(config).run()
    print(f"Restoration complete: {final_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-guided old photo restoration with FLUX inpainting and DDColor."
    )
    parser.add_argument("input", type=Path, nargs="?", help="Input old photo path")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("outputs/restore"))
    parser.add_argument("--ui", action="store_true", help="Launch the Gradio interface")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=6006)

    parser.add_argument("--manual-mask", type=Path, default=None)
    parser.add_argument(
        "--manual-protect-mask",
        type=Path,
        default=None,
        help="Manual protection mask. White pixels are force-preserved.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--working-size", type=int, default=1024)

    parser.add_argument("--no-flux", action="store_true", help="Skip FLUX inpainting")
    parser.add_argument("--no-color", action="store_true", help="Skip DDColor colorization")
    parser.add_argument("--no-llm", action="store_true", help="Use the local planner only")
    parser.add_argument("--no-llm-mask", action="store_true", help="Disable LLM safe automatic mask regions")
    parser.add_argument("--no-lama-repair", action="store_true", help="Disable LaMa repair and use OpenCV fallback")
    parser.add_argument("--no-cv-repair", action="store_true", help="Skip OpenCV fine scratch repair")
    parser.add_argument(
        "--flux-auto-mask",
        action="store_true",
        help="Allow FLUX to fall back to the full automatic damage mask.",
    )
    parser.add_argument(
        "--flux-manual-mask-only",
        action="store_true",
        help="Restrict FLUX to manually supplied masks only.",
    )

    parser.add_argument("--ddcolor-repo", type=Path, default=Path("DDColor"))
    parser.add_argument("--ddcolor-model-path", type=Path, default=None)
    parser.add_argument("--ddcolor-model-name", default="ddcolor_modelscope")
    parser.add_argument("--ddcolor-input-size", type=int, default=512)
    parser.add_argument("--ddcolor-model-size", default="large", choices=["tiny", "large"])

    parser.add_argument("--flux-repo", type=Path, default=Path("FLUX-Controlnet-Inpainting"))
    parser.add_argument("--flux-base-model", default="weights/flux/FLUX.1-dev")
    parser.add_argument(
        "--flux-controlnet-model",
        default="weights/flux/FLUX.1-dev/FLUX.1-dev-Controlnet-Inpainting-Alpha",
    )
    parser.add_argument("--flux-steps", type=int, default=28)
    parser.add_argument("--flux-control-scale", type=float, default=0.9)
    parser.add_argument("--flux-guidance-scale", type=float, default=3.5)
    parser.add_argument("--flux-true-guidance-scale", type=float, default=1.0)
    parser.add_argument("--lama-model-path", type=Path, default=Path("weights/lama/big-lama.pt"))

    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
    )
    parser.add_argument(
        "--llm-api-key",
        default=(
            os.environ.get("LLM_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("QWEN_API_KEY")
        ),
    )
    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "qwen-vl-plus"))
    parser.add_argument("--llm-timeout", type=int, default=60)
    return parser.parse_args()


def launch_ui(args: argparse.Namespace) -> None:
    import gradio as gr

    def restore_from_ui(
        image_path: str | None,
        manual_mask_path: str | None,
        brush_mask: Any,
        output_name: str,
        device: str,
        enable_flux: bool,
        enable_colorization: bool,
        enable_llm: bool,
        enable_llm_mask: bool,
        enable_lama_repair: bool,
        enable_cv_repair: bool,
        flux_requires_manual_mask: bool,
        flux_allow_auto_mask_fallback: bool,
        seed: int,
        working_size: int,
        ddcolor_model_path: str,
        ddcolor_model_name: str,
        ddcolor_input_size: int,
        ddcolor_model_size: str,
        flux_base_model: str,
        flux_controlnet_model: str,
        flux_steps: int,
        flux_control_scale: float,
        flux_guidance_scale: float,
        flux_true_guidance_scale: float,
        lama_model_path: str,
        llm_base_url: str,
        llm_model: str,
        llm_api_key: str,
        progress: gr.Progress = gr.Progress(track_tqdm=True),
    ) -> tuple[str | None, list[tuple[str, str]], dict[str, Any], str, list[str]]:
        from old_photo_restore import OldPhotoRestorationPipeline, RestorationConfig

        if not image_path:
            raise gr.Error("请先上传一张老照片。")

        progress(0.05, desc="准备修复任务")
        output_dir = _ui_output_dir(output_name)
        resolved_manual_mask, resolved_protect_mask = _resolve_ui_manual_masks(
            image_path, manual_mask_path, brush_mask, output_dir
        )
        config = RestorationConfig(
            input_path=Path(image_path),
            output_dir=output_dir,
            manual_mask_path=resolved_manual_mask,
            manual_protect_mask_path=resolved_protect_mask,
            device=device,
            seed=int(seed),
            working_size=int(working_size),
            enable_flux=bool(enable_flux),
            enable_colorization=bool(enable_colorization),
            enable_llm=bool(enable_llm),
            enable_llm_mask=bool(enable_llm_mask),
            enable_lama_repair=bool(enable_lama_repair),
            enable_cv_repair=bool(enable_cv_repair),
            flux_requires_manual_mask=bool(flux_requires_manual_mask),
            flux_allow_auto_mask_fallback=bool(flux_allow_auto_mask_fallback),
            ddcolor_repo=Path("DDColor"),
            ddcolor_model_path=Path(ddcolor_model_path) if ddcolor_model_path else None,
            ddcolor_model_name=ddcolor_model_name or "ddcolor_modelscope",
            ddcolor_input_size=int(ddcolor_input_size),
            ddcolor_model_size=ddcolor_model_size,
            flux_repo=Path("FLUX-Controlnet-Inpainting"),
            flux_base_model=flux_base_model or "weights/flux/FLUX.1-dev",
            flux_controlnet_model=(
                flux_controlnet_model
                or "weights/flux/FLUX.1-dev/FLUX.1-dev-Controlnet-Inpainting-Alpha"
            ),
            flux_steps=int(flux_steps),
            flux_control_scale=float(flux_control_scale),
            flux_guidance_scale=float(flux_guidance_scale),
            flux_true_guidance_scale=float(flux_true_guidance_scale),
            lama_model_path=Path(lama_model_path) if lama_model_path else Path("weights/lama/big-lama.pt"),
            llm_base_url=llm_base_url or None,
            llm_api_key=llm_api_key or None,
            llm_model=llm_model or None,
        )

        progress(0.15, desc="正在执行修复流水线")
        final_path = OldPhotoRestorationPipeline(config).run()
        progress(0.95, desc="正在整理输出结果")

        plan = _read_json(output_dir / "plan.json")
        report = _read_text(output_dir / "report.md")
        gallery = _gallery_items(output_dir)
        files = [str(path) for path in sorted(output_dir.glob("*")) if path.is_file()]
        return str(final_path), gallery, plan, report, files

    with gr.Blocks(title="老旧照片智能修复与上色") as demo:
        gr.Markdown("# 老旧照片智能修复与上色")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="老照片输入", type="filepath", height=340)
                manual_mask = gr.Image(label="手动修复掩膜", type="filepath", height=220)
                with gr.Accordion("掩膜操作说明", open=True):
                    gr.Markdown(
                        """
                        **画笔掩膜的作用**：只告诉 AI 哪里要修，不是手动画照片内容。

                        - 画白/涂抹区域：要修复的划痕、白斑、霉点、破损。
                        - 黑色或透明区域：保持原样，不让 LaMa/OpenCV 动。
                        - 画错的位置：用橡皮擦掉，或清空后重画。
                        - 推荐只比实际划痕宽 1-2 像素，越贴合越不容易糊。
                        - 脸上只涂孤立白点或极细白划痕，不要涂整片皮肤。
                        - 不要涂眼睛、嘴、鼻子边缘、脸型轮廓、眼镜框、纽扣、盘扣、衣服边线、手指轮廓、婴儿帽纹理。
                        - 不画掩膜时，会使用自动掩膜 + LLM 辅助掩膜；画了以后，会和自动掩膜合并。
                        """
                    )
                brush_mask = gr.ImageEditor(
                    label="画笔修复掩膜（白色=修复，黑色=保留）",
                    type="numpy",
                    height=260,
                    image_mode="RGBA",
                    brush=gr.Brush(
                        default_size=6,
                        colors=["#ffffff", "#000000"],
                        default_color="#ffffff",
                        color_mode="defaults",
                    ),
                    eraser=False,
                )
                output_name = gr.Textbox(label="输出名称", placeholder="example")
                run_button = gr.Button("开始修复", variant="primary")

            with gr.Column(scale=1):
                final_image = gr.Image(label="最终结果", type="filepath", height=430)
                output_files = gr.File(label="输出文件", file_count="multiple")

        with gr.Row():
            with gr.Column(scale=1):
                enable_cv_repair = gr.Checkbox(label="启用自动细划痕修复", value=True)
                enable_flux = gr.Checkbox(label="启用 FLUX 大面积补全", value=False)
                enable_colorization = gr.Checkbox(label="启用 DDColor 自然上色", value=True)
                enable_llm = gr.Checkbox(label="启用 LLM 修复规划", value=True)
                enable_llm_mask = gr.Checkbox(label="启用 LLM 安全掩膜", value=True)
                enable_lama_repair = gr.Checkbox(label="启用 LaMa 模型修复", value=True)
                flux_requires_manual_mask = gr.Checkbox(label="FLUX 仅用于手动掩膜", value=True)
                flux_allow_auto_mask_fallback = gr.Checkbox(label="FLUX 允许自动掩膜兜底", value=False)
            with gr.Column(scale=1):
                device = gr.Radio(["auto", "cuda", "cpu"], label="运行设备", value="auto")
                seed = gr.Number(label="随机种子", value=24, precision=0)
                working_size = gr.Slider(512, 1536, value=1024, step=64, label="工作分辨率")

        with gr.Accordion("DDColor 上色设置", open=False):
            ddcolor_model_path = gr.Textbox(
                label="本地权重路径",
                placeholder="weights/ddcolor/pytorch_model.pt",
            )
            ddcolor_model_name = gr.Textbox(label="模型名称", value="ddcolor_modelscope")
            with gr.Row():
                ddcolor_input_size = gr.Slider(256, 1024, value=512, step=64, label="输入尺寸")
                ddcolor_model_size = gr.Radio(["large", "tiny"], value="large", label="模型规模")

        with gr.Accordion("FLUX 局部修复设置", open=False):
            flux_base_model = gr.Textbox(
                label="基础模型",
                value="weights/flux/FLUX.1-dev",
            )
            flux_controlnet_model = gr.Textbox(
                label="ControlNet 模型",
                value="weights/flux/FLUX.1-dev/FLUX.1-dev-Controlnet-Inpainting-Alpha",
            )
            with gr.Row():
                flux_steps = gr.Slider(8, 50, value=28, step=1, label="推理步数")
                flux_control_scale = gr.Slider(0.0, 1.5, value=0.9, step=0.05, label="控制强度")
            with gr.Row():
                flux_guidance_scale = gr.Slider(1.0, 10.0, value=3.5, step=0.1, label="提示词引导")
                flux_true_guidance_scale = gr.Slider(
                    0.0, 8.0, value=1.0, step=0.1, label="真实引导强度"
                )

        with gr.Accordion("LaMa 模型修复设置", open=False):
            lama_model_path = gr.Textbox(
                label="LaMa 权重路径",
                value="weights/lama/big-lama.pt",
            )

        with gr.Accordion("LLM 修复规划设置", open=False):
            llm_base_url = gr.Textbox(label="接口地址", value=args.llm_base_url or "")
            llm_model = gr.Textbox(label="模型名称", value=args.llm_model or "")
            llm_api_key = gr.Textbox(label="API 密钥", type="password", value=args.llm_api_key or "")

        with gr.Tabs():
            with gr.Tab("中间结果"):
                gallery = gr.Gallery(label="修复过程", columns=4, height=360)
            with gr.Tab("修复计划"):
                plan_json = gr.JSON(label="结构化修复计划")
            with gr.Tab("修复报告"):
                report_md = gr.Markdown()

        inputs = [
            image,
            manual_mask,
            brush_mask,
            output_name,
            device,
            enable_flux,
            enable_colorization,
            enable_llm,
            enable_llm_mask,
            enable_lama_repair,
            enable_cv_repair,
            flux_requires_manual_mask,
            flux_allow_auto_mask_fallback,
            seed,
            working_size,
            ddcolor_model_path,
            ddcolor_model_name,
            ddcolor_input_size,
            ddcolor_model_size,
            flux_base_model,
            flux_controlnet_model,
            flux_steps,
            flux_control_scale,
            flux_guidance_scale,
            flux_true_guidance_scale,
            lama_model_path,
            llm_base_url,
            llm_model,
            llm_api_key,
        ]
        outputs = [final_image, gallery, plan_json, report_md, output_files]
        image.change(_image_editor_value_from_upload, inputs=image, outputs=brush_mask)
        run_button.click(restore_from_ui, inputs=inputs, outputs=outputs)

    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


def _ui_output_dir(output_name: str) -> Path:
    if output_name:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in output_name)
    else:
        safe = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "gradio" / safe


def _image_editor_value_from_upload(image_path: str | None) -> dict[str, Any] | None:
    if not image_path:
        return None
    return {
        "background": image_path,
        "layers": [],
        "composite": None,
    }


def _resolve_ui_manual_masks(
    image_path: str | None,
    manual_mask_path: str | None,
    brush_mask: Any,
    output_dir: Path,
) -> tuple[Path | None, Path | None]:
    masks = []
    if manual_mask_path:
        masks.append(Path(manual_mask_path))

    brush_repair, brush_protect = _extract_brush_masks(brush_mask)
    protect_path = None
    if (brush_repair is not None or brush_protect is not None) and image_path:
        import cv2
        import numpy as np

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None:
            h, w = image.shape[:2]
            output_dir.mkdir(parents=True, exist_ok=True)
            if brush_protect is not None:
                brush_protect = cv2.resize(brush_protect, (w, h), interpolation=cv2.INTER_NEAREST)
                _, brush_protect = cv2.threshold(brush_protect, 1, 255, cv2.THRESH_BINARY)
                if np.count_nonzero(brush_protect) > 0:
                    protect_path = output_dir / "ui_brush_protect_mask.png"
                    cv2.imwrite(str(protect_path), brush_protect)
            if brush_repair is not None:
                brush_repair = cv2.resize(brush_repair, (w, h), interpolation=cv2.INTER_NEAREST)
                _, brush_repair = cv2.threshold(brush_repair, 1, 255, cv2.THRESH_BINARY)
                if brush_protect is not None:
                    brush_repair[brush_protect > 0] = 0
                if np.count_nonzero(brush_repair) > 0:
                    brush_path = output_dir / "ui_brush_repair_mask.png"
                    cv2.imwrite(str(brush_path), brush_repair)
                    masks.append(brush_path)

    if not masks:
        return None, protect_path
    if len(masks) == 1:
        return masks[0], protect_path

    import cv2
    import numpy as np

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path else None
    if image is None:
        return masks[-1], protect_path
    h, w = image.shape[:2]
    merged = np.zeros((h, w), dtype=np.uint8)
    for path in masks:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        merged = cv2.bitwise_or(merged, mask)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "ui_manual_mask_merged.png"
    cv2.imwrite(str(merged_path), merged)
    return merged_path, protect_path


def _extract_brush_masks(value: Any):
    if value is None:
        return None, None

    import cv2
    import numpy as np

    candidates = []
    composite_fallback = False
    if isinstance(value, dict):
        layers = value.get("layers") or []
        if isinstance(layers, list):
            candidates.extend(layer for layer in layers if layer is not None)
        if not candidates and value.get("composite") is not None:
            candidates.append(value["composite"])
            composite_fallback = True
    else:
        candidates.append(value)

    repair = None
    protect = None
    for candidate in candidates:
        add_mask, protect_mask = _brush_layer_to_masks(
            candidate,
            allow_opaque_protect=not composite_fallback,
        )
        if add_mask is not None:
            repair = add_mask if repair is None else cv2.bitwise_or(repair, add_mask)
        if protect_mask is not None:
            protect = protect_mask if protect is None else cv2.bitwise_or(protect, protect_mask)

    if repair is not None and protect is not None:
        repair[protect > 0] = 0
    if repair is not None and np.count_nonzero(repair) == 0:
        repair = None
    if protect is not None and np.count_nonzero(protect) == 0:
        protect = None
    return repair, protect


def _brush_layer_to_masks(candidate: Any, allow_opaque_protect: bool = True):
    import cv2
    import numpy as np

    arr = np.asarray(candidate)
    if arr.size == 0:
        return None, None

    if arr.ndim == 3 and arr.shape[2] >= 4:
        rgba = arr.astype(np.uint8)
        active = rgba[:, :, 3] > 8
        gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
        repair = (active & (gray >= 128)).astype(np.uint8) * 255
        protect = (active & (gray < 128)).astype(np.uint8) * 255
    elif arr.ndim == 3:
        gray = cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        repair = (gray >= 220).astype(np.uint8) * 255
        if allow_opaque_protect:
            protect = (gray <= 35).astype(np.uint8) * 255
        else:
            protect = np.zeros_like(repair)
    elif arr.ndim == 2:
        gray = arr.astype(np.uint8)
        repair = (gray >= 220).astype(np.uint8) * 255
        protect = np.zeros_like(repair)
    else:
        return None, None

    repair = repair if np.count_nonzero(repair) > 0 else None
    protect = protect if np.count_nonzero(protect) > 0 else None
    return repair, protect


def _gallery_items(output_dir: Path) -> list[tuple[str, str]]:
    names = [
        ("00_input.png", "原图"),
        ("00_preprocessed.png", "预处理图"),
        ("01_damage_mask.png", "损伤掩膜"),
        ("01_llm_safe_mask.png", "LLM 安全掩膜"),
        ("01_face_defect_mask.png", "面部小瑕疵掩膜"),
        ("01_redline_guard.png", "结构红线保护"),
        ("01_background_scratch_boost_mask.png", "背景划痕增强掩膜"),
        ("01_repair_mask.png", "最终修复掩膜"),
        ("02_lama_repaired.png", "LaMa 模型修复"),
        ("02_cv_repaired.png", "自动细划痕修复"),
        ("02_residual_damage_mask.png", "残留瑕疵掩膜"),
        ("03_flux_repaired.png", "FLUX 大面积补全"),
        ("04_ddcolor_colorized.png", "DDColor 上色结果"),
        ("final.png", "最终结果"),
    ]
    return [
        (str(output_dir / filename), caption)
        for filename, caption in names
        if (output_dir / filename).exists()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.ui or args.input is None:
        launch_ui(args)
    else:
        run_cli(args)


if __name__ == "__main__":
    main()
