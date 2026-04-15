#!/Users/Henry/Desktop/clothing/yynot-tryon/fashn-vton-1.5/.venv/bin/python3
"""
Batch pre-renderer for YYNOT try-on app.
Runs FASHN VTON v1.5 inference for every model × garment combination.
Skips pairs that already have a render unless --force is passed.
"""

import sys, os, json, argparse
from pathlib import Path
from PIL import Image

def get_category(filename: str) -> str:
    name = filename.lower()
    if any(k in name for k in ["pant","jean","trouser","short","skirt","bottom"]):
        return "bottoms"
    if any(k in name for k in ["dress","jumpsuit","overall","romper","suit"]):
        return "one-pieces"
    return "tops"  # default

def slug(path: str) -> str:
    return Path(path).stem

def load_catalog() -> dict:
    p = Path("catalog.json")
    if p.exists():
        return json.loads(p.read_text())
    return {"brand": "YYNOT?", "models": [], "garments": []}

def save_catalog(cat: dict):
    Path("catalog.json").write_text(json.dumps(cat, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-render existing outputs")
    parser.add_argument("--models", nargs="*", help="Limit to specific model slugs")
    parser.add_argument("--garments", nargs="*", help="Limit to specific garment slugs")
    parser.add_argument("--steps", type=int, default=10, help="Diffusion steps (default 10; use 30 for best quality)")
    parser.add_argument("--weights", type=str, default="fashn-vton-1.5/weights", help="Path to FASHN weights directory")
    args = parser.parse_args()

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

    # Filter if --models / --garments passed
    if args.models:
        model_files = [f for f in model_files if slug(f) in args.models]
    if args.garments:
        garment_files = [f for f in garment_files if slug(f) in args.garments]

    pairs = [(m, g) for m in model_files for g in garment_files]
    total = len(pairs)
    print(f"\n📦 {len(model_files)} models × {len(garment_files)} garments = {total} renders\n")

    # Lazy-load pipeline (expensive — do it once)
    pipeline = None
    success, skipped, failed = 0, 0, 0

    # Build catalog structure
    cat = load_catalog()
    model_slugs   = [slug(f) for f in model_files]
    garment_slugs = [slug(f) for f in garment_files]
    cat["models"] = sorted(set(cat.get("models", []) + model_slugs))

    # Ensure garment entries exist
    existing_ids = {g["id"] for g in cat.get("garments", [])}
    for gf in garment_files:
        gid = slug(gf)
        if gid not in existing_ids:
            cat.setdefault("garments", []).append({
                "id": gid,
                "name": gid.replace("-", " ").replace("_", " ").title(),
                "category": get_category(gid),
                "thumbnail": str(gf),
                "renders": {}
            })
    save_catalog(cat)

    for i, (model_path, garment_path) in enumerate(pairs, 1):
        m_slug = slug(model_path)
        g_slug = slug(garment_path)
        out_path = render_dir / f"{m_slug}__{g_slug}.jpg"
        label = f"({i}/{total}) {m_slug} × {g_slug}"

        if out_path.exists() and not args.force:
            print(f"  ⏭  skip  {label}")
            skipped += 1
            # Still update catalog
            for g in cat["garments"]:
                if g["id"] == g_slug:
                    g.setdefault("renders", {})[m_slug] = str(out_path)
            continue

        print(f"  🎨 render {label} ...", end=" ", flush=True)

        try:
            if pipeline is None:
                print("\n  ⏳ Loading FASHN pipeline (first time)...")
                from fashn_vton import TryOnPipeline
                pipeline = TryOnPipeline(weights_dir=args.weights)
                print("  ✅ Pipeline loaded\n")

            person  = Image.open(model_path).convert("RGB")
            garment = Image.open(garment_path).convert("RGB")
            category = get_category(g_slug)

            result = pipeline(
                person_image=person,
                garment_image=garment,
                category=category,
                num_timesteps=args.steps,
            )
            result.images[0].save(str(out_path), quality=92)
            print(f"✅ saved → {out_path}")

            # Update catalog
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
