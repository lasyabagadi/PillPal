import argparse
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random
import shutil

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED

def collect_images(root: Path):
    classes = []
    items = []
    for cls_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        classes.append(cls_dir.name)
        for img in cls_dir.rglob("*"):
            if img.is_file() and is_image(img):
                items.append((img, cls_dir.name))
    if not items:
        raise SystemExit(f"No images found under {root}. Ensure raw_dir has class subfolders.")
    return classes, items


def save_resized(src: Path, dst: Path, size: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im = im.resize((size, size), Image.BICUBIC)
        im.save(dst, quality=95)


def main():
    ap = argparse.ArgumentParser(description="Resize + split dataset into train/val (ImageFolder)")
    ap.add_argument("--raw_dir", type=str, required=True, help="Path to raw data with class subfolders")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dir; creates train/ and val/")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy_non_images", action="store_true", help="Copy non-image files if present (off by default)")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if out_dir.exists():
        print(f"[i] Removing existing output dir: {out_dir}")
        shutil.rmtree(out_dir)

    classes, items = collect_images(raw_dir)
    print(f"[i] Found {len(items)} images across {len(classes)} classes.")

    random.seed(args.seed)
    random.shuffle(items)

    n_val = max(1, int(len(items) * args.val_split))
    val_set = set(items[:n_val])

    for img_path, cls in tqdm(items, desc="Processing"):
        split = "val" if (img_path, cls) in val_set else "train"
        relname = img_path.stem + img_path.suffix.lower()
        dst = out_dir / split / cls / relname
        try:
            save_resized(img_path, dst, args.img_size)
        except Exception as e:
            print(f"[w] Skipping {img_path}: {e}")

    # Write class index mapping
    idx_path = out_dir / "classes.txt"
    with open(idx_path, "w") as f:
        for i, cls in enumerate(sorted(set(c for _, c in items))):
            f.write(f"{i}\t{cls}\n")
    print(f"[✓] Wrote {idx_path}")
