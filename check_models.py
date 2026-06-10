from __future__ import annotations

from pathlib import Path


def check_file(path: Path) -> bool:
    ok = path.exists() and path.is_file()
    print(f"[{'OK' if ok else 'MISS'}] {path}")
    return ok


def check_dir(path: Path) -> bool:
    ok = path.exists() and path.is_dir()
    print(f"[{'OK' if ok else 'MISS'}] {path}/")
    return ok


def main() -> None:
    ok = True

    print("DDColor")
    ok &= check_file(Path("weights/ddcolor/pytorch_model.pt"))

    print("\nFLUX.1-dev base model")
    base = Path("weights/flux/FLUX.1-dev")
    ok &= check_file(base / "model_index.json")
    for name in ["scheduler", "tokenizer", "tokenizer_2"]:
        ok &= check_dir(base / name)
    ok &= check_file(base / "scheduler" / "scheduler_config.json")
    ok &= check_file(base / "tokenizer" / "tokenizer_config.json")
    ok &= check_file(base / "tokenizer_2" / "tokenizer_config.json")

    for name in ["text_encoder", "text_encoder_2", "transformer", "vae"]:
        ok &= check_dir(base / name)
        ok &= check_file(base / name / "config.json")
        if not any((base / name).glob("*.safetensors")):
            print(f"[MISS] {base / name}/*.safetensors")
            ok = False

    print("\nFLUX ControlNet Inpainting Alpha")
    control = base / "FLUX.1-dev-Controlnet-Inpainting-Alpha"
    ok &= check_file(control / "config.json")
    ok &= check_file(control / "diffusion_pytorch_model.safetensors")

    if ok:
        print("\nAll required model paths are present.")
    else:
        raise SystemExit("\nSome model paths are missing. Please place weights before enabling that module.")


if __name__ == "__main__":
    main()
