# Model Paths

Place model files under the project root:

```text
weights/
  ddcolor/
    pytorch_model.pt

  flux/
    FLUX.1-dev/
      model_index.json
      scheduler/
      tokenizer/
      tokenizer_2/
      text_encoder/
      text_encoder_2/
      transformer/
      vae/

      FLUX.1-dev-Controlnet-Inpainting-Alpha/
        config.json
        diffusion_pytorch_model.safetensors
```

Run a preflight check:

```bash
/root/miniconda3/bin/python check_models.py
```

Start the Gradio app:

```bash
./start_app.sh
```

The app defaults to local paths:

```text
DDColor: weights/ddcolor/pytorch_model.pt
FLUX base: weights/flux/FLUX.1-dev
FLUX ControlNet: weights/flux/FLUX.1-dev/FLUX.1-dev-Controlnet-Inpainting-Alpha
```
