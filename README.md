# LLM-Guided Old Photo Restoration

This project combines:

- **DDColor** for whole-image natural colorization.
- **FLUX.1-dev ControlNet Inpainting** for scratch, crack, stain, and missing-region restoration.
- **An LLM/VLM planning layer** for structured restoration prompts and conservative quality guidance.

The intended workflow is:

```text
old photo
  -> damage mask detection
  -> LLM restoration plan
  -> FLUX local inpainting
  -> DDColor full-image colorization
  -> final image + intermediate artifacts + report
```

## Setup

The workspace already contains:

- `FLUX-Controlnet-Inpainting/`
- `DDColor/`

Install Python dependencies in your environment:

```bash
pip install -r requirements.txt
```

FLUX requires a large GPU. The Alimama README recommends around 1024px inputs and high VRAM. DDColor can run more easily, but still benefits from CUDA.

## Run

Full pipeline:

```bash
python app.py
```

CLI mode:

```bash
python app.py path/to/old_photo.jpg -o outputs/example
```

Colorization only:

```bash
python app.py path/to/old_photo.jpg -o outputs/color_only --no-flux
```

FLUX repair only:

```bash
python app.py path/to/old_photo.jpg -o outputs/repair_only --no-color
```

Use a manual damage mask:

```bash
python app.py path/to/old_photo.jpg --manual-mask path/to/mask.png -o outputs/manual
```

Use an OpenAI-compatible LLM endpoint:

```bash
set LLM_BASE_URL=http://localhost:1234/v1
set LLM_MODEL=your-vision-model
python app.py path/to/old_photo.jpg -o outputs/llm
```

Use local FLUX weights:

```bash
python app.py path/to/old_photo.jpg -o outputs/local_flux \
  --flux-base-model weights/flux/FLUX.1-dev \
  --flux-controlnet-model weights/flux/FLUX.1-dev-Controlnet-Inpainting-Alpha
```

`FLUX.1-dev-Controlnet-Inpainting-Alpha` is the ControlNet/Inpainting
adapter. It still requires a complete `FLUX.1-dev` base model directory
containing files such as `model_index.json`, `transformer/`, `vae/`, and the
text encoder/tokenizer folders.

The pipeline writes:

- `00_input.png`
- `01_damage_mask.png`
- `llm_prompt.txt`
- `plan.json`
- `02_flux_repaired.png` if FLUX is enabled
- `03_ddcolor_colorized.png` if DDColor is enabled
- `final.png`
- `report.md`

## Notes

FLUX.1-dev related weights are commonly licensed for non-commercial or research use. Check the model licenses before using the pipeline commercially.
