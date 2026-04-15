import json, os
from pathlib import Path

IMG_EXTS = {'.jpg','.jpeg','.png','.webp'}

def slug(p): return Path(p).stem

def category(name):
    n = name.lower()
    if any(k in n for k in ["pant","jean","trouser","short","skirt","bottom"]): return "bottoms"
    if any(k in n for k in ["dress","jump","overall","romper"]): return "one-pieces"
    return "tops"

models   = sorted([f for f in os.listdir("assets/models")   if Path(f).suffix.lower() in IMG_EXTS])
garments = sorted([f for f in os.listdir("assets/garments") if Path(f).suffix.lower() in IMG_EXTS])
renders  = os.listdir("assets/renders") if os.path.exists("assets/renders") else []

render_map = {}
for r in renders:
    stem = Path(r).stem
    if "__" in stem:
        m, g = stem.split("__", 1)
        render_map.setdefault(g, {})[m] = f"assets/renders/{r}"

model_slugs = [slug(m) for m in models]

garment_list = []
for g in garments:
    gid = slug(g)
    garment_list.append({
        "id": gid,
        "name": gid.replace("-"," ").replace("_"," ").title(),
        "category": category(gid),
        "thumbnail": f"assets/garments/{g}",
        "renders": render_map.get(gid, {})
    })

cat = {
    "brand": "YYNOT?",
    "models": model_slugs,
    "garments": garment_list
}

Path("catalog.json").write_text(json.dumps(cat, indent=2))
print(f"✅ catalog.json written")
print(f"   {len(model_slugs)} models: {model_slugs}")
print(f"   {len(garment_list)} garments")
print(f"   {sum(len(v) for v in render_map.values())} renders indexed")
