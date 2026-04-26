#!/usr/bin/env python3
"""
CatVTON-based batch pre-renderer for YYNOT try-on app.
Drop-in replacement for prerender.py using CatVTON instead of FASHN v1.5.

CatVTON achieves SOTA on VITON-HD (FID 5.43 vs FASHN's higher score) and uses
<8GB VRAM vs ~20GB for IDM-VTON. Better boundary/neckline accuracy via superior
garment structure capture.

─── SERVER SETUP (run once) ────────────────────────────────────────────────────

  cd ~/yynot-tryon

  # Clone CatVTON alongside FASHN
  git clone https://github.com/Zheng-Chong/CatVTON catvton
  cd catvton

  # Install dependencies (use existing venv or create new one)
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install diffusers==0.27.2 transformers accelerate omegaconf
  pip install opencv-python-headless Pillow einops

  # Weights are pulled from HuggingFace automatically on first run:
  #   zhengchong/CatVTON        (attention checkpoint, ~2GB)
  #   booksforcharlie/stable-diffusion-inpainting  (base, ~4GB)
  #   DensePose + SCHP weights are fetched from the CatVTON HF repo

  cd ~/yynot-tryon

─── USAGE ──────────────────────────────────────────────────────────────────────

  # Full batch re-render (replaces all FASHN renders)
  python prerender_catvton.py --force

  # Specific pairs only
  python prerender_catvton.py --models 01350_00 --garments shirt1 --force

  # Quality vs speed
  python prerender_catvton.py --steps 50   # balanced (default)
  python prerender_catvton.py --steps 100  # best quality, ~2× slower

────────────────────────────────────────────────────────────────────────────────
"""

import sys, os, json, argparse
from pathlib import Path
from PIL import Image


def get_cloth_type(filename: str) -> str:
    """Map garment filename → CatVTON cloth_type: upper / lower / overall."""
    name = filename.lower()
    if any(k in name for k in ["pant", "jean", "trouser", "short", "skirt", "bottom"]):
        return "lower"
    if any(k in name for k in ["dress", "jumpsuit", "overall", "romper", "suit"]):
        return "overall"
    return "upper"


def slug(path) -> str:
    return Path(path).stem


def load_catalog() -> dict:
    p = Path("catalog.json")
    if p.exists():
        return json.loads(p.read_text())
    return {"brand": "Prototype II", "models": [], "garments": []}


def save_catalog(cat: dict):
    Path("catalog.json").write_text(json.dumps(cat, indent=2))


def main():
    parser = argparse.ArgumentParser(description="CatVTON batch renderer")
    parser.add_argument("--force",    action="store_true", help="Re-render existing outputs")
    parser.add_argument("--models",   nargs="*",           help="Limit to specific model slugs")
    parser.add_argument("--garments", nargs="*",           help="Limit to specific garment slugs")
    parser.add_argument("--steps",    type=int,   default=50,  help="Inference steps (default 50)")
    parser.add_argument("--guidance", type=float, default=2.5, help="Guidance scale (default 2.5)")
    parser.add_argument("--seed",     type=int,   default=42,  help="Random seed")
    parser.add_argument("--catvton-dir", type=str, default="catvton",
                        help="Path to cloned CatVTON repo (default: ./catvton)")
    args = parser.parse_args()

    catvton_dir = Path(args.catvton_dir).resolve()
    if not catvton_dir.exists():
        print(f"ERROR: CatVTON directory not found at {catvton_dir}")
        print("Run: git clone https://github.com/Zheng-Chong/CatVTON catvton")
        sys.exit(1)

    sys.path.insert(0, str(catvton_dir))

    model_dir   = Path("assets/models")
    garment_dir = Path("assets/garments")
    render_dir  = Path("assets/renders")
    render_dir.mkdir(exist_ok=True)

    model_files   = sorted(model_dir.glob("*.[jJwWpP][pPeEnN][gGbBgG]*"))
    garment_files = sorted(garment_dir.glob("*.[jJwWpP][pPeEnN][gGbBgG]*"))

    if not model_files:
        print("ERROR: No model images found in assets/models/"); sys.exit(1)
    if not garment_files:
        print("ERROR: No garment images found in assets/garments/"); sys.exit(1)

    if args.models:
        model_files = [f for f in model_files if slug(f) in args.models]
    if args.garments:
        garment_files = [f for f in garment_files if slug(f) in args.garments]

    pairs = [(m, g) for m in model_files for g in garment_files]
    total = len(pairs)
    print(f"\n📦 {len(model_files)} models × {len(garment_files)} garments = {total} renders\n")

    # Lazy-load (expensive — do once)
    pipeline     = None
    automasker   = None
    mask_proc    = None
    resize_fn    = None

    cat = load_catalog()
    cat["models"] = sorted(set(cat.get("models", []) + [slug(f) for f in model_files]))
    existing_ids = {g["id"] for g in cat.get("garments", [])}
    for gf in garment_files:
        gid = slug(gf)
        if gid not in existing_ids:
            cat.setdefault("garments", []).append({
                "id": gid,
                "name": gid.replace("-", " ").replace("_", " ").title(),
                "category": "tops",
                "thumbnail": str(gf),
                "renders": {}
            })
    save_catalog(cat)

    success, skipped, failed = 0, 0, 0

    for i, (model_path, garment_path) in enumerate(pairs, 1):
        m_slug = slug(model_path)
        g_slug = slug(garment_path)
        out_path = render_dir / f"{m_slug}__{g_slug}.jpg"
        label = f"({i}/{total}) {m_slug} × {g_slug}"

        if out_path.exists() and not args.force:
            print(f"  ⏭  skip  {label}")
            skipped += 1
            for g in cat["garments"]:
                if g["id"] == g_slug:
                    g.setdefault("renders", {})[m_slug] = str(out_path)
            continue

        print(f"  🎨 render {label} ...", end=" ", flush=True)

        try:
            if pipeline is None:
                print("\n  ⏳ Loading CatVTON pipeline (first time)...")
                import torch
                from diffusers.image_processor import VaeImageProcessor

                # CatVTON imports from the cloned repo
                from model.cloth_masker import AutoMasker
                from model.pipeline import CatVTONPipeline
                from utils import resize_and_padding

                resize_fn = resize_and_padding
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

                pipeline = CatVTONPipeline(
                    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
                    attn_ckpt="zhengchong/CatVTON",
                    attn_ckpt_version="mix",
                    weight_dtype=dtype,
                    use_tf32=True,
                    device="cuda",
                )
                mask_proc = VaeImageProcessor(
                    vae_scale_factor=8,
                    do_normalize=False,
                    do_binarize=True,
                    do_convert_grayscale=True,
                )
                automasker = AutoMasker(
                    densepose_ckpt=str(catvton_dir / "DensePose"),
                    schp_ckpt=str(catvton_dir / "SCHP"),
                    device="cuda",
                )
                print("  ✅ Pipeline loaded\n")

            import torch

            person  = Image.open(model_path).convert("RGB")
            garment = Image.open(garment_path).convert("RGB")
            cloth_type = get_cloth_type(g_slug)

            # CatVTON target resolution
            W, H = 768, 1024
            person_r  = resize_fn(person,  (W, H))
            garment_r = resize_fn(garment, (W, H))

            # Auto-generate clothing region mask from person image
            mask_result = automasker(person_r, cloth_type)
            mask = mask_proc.preprocess(mask_result["mask"], height=H, width=W)[0]

            generator = torch.Generator(device="cuda").manual_seed(args.seed)

            output = pipeline(
                image=person_r,
                condition_image=garment_r,
                mask=mask,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
            )

            # pipeline returns list of PIL images
            result_img = output[0] if isinstance(output, list) else output.images[0]
            result_img.save(str(out_path), quality=92)
            print(f"✅ saved → {out_path}")

            for g in cat["garments"]:
                if g["id"] == g_slug:
                    g.setdefault("renders", {})[m_slug] = str(out_path)

            save_catalog(cat)
            success += 1

        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    save_catalog(cat)
    print(f"\n{'─'*50}")
    print(f"  ✅ rendered : {success}")
    print(f"  ⏭  skipped  : {skipped}")
    print(f"  ❌ failed   : {failed}")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()
