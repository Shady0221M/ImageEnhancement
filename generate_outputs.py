"""Run every technique x every parameter combo on every test image.
Writes outputs/images/... and outputs/manifest.csv.

Usage (repo root, venv active):  python generate_outputs.py
"""
import itertools
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils.io_utils import load_image_gray
from utils.metrics import all_metrics
from pipeline.registry import TECHNIQUES

IN_ROOT = Path("test_images/prepared")
OUT_ROOT = Path("outputs")
VARIATIONS = ["blurred_light_texture", "blurred_strong_edges",
              "uneven_illumination", "noisy", "control_sharp"]
REFERENCE = "control_sharp"
MAX_SIDE = 512          # downscale larger images (keeps files small and runs fast)
JPEG_QUALITY = 92


def load(path):
    img = load_image_gray(path)
    h, w = img.shape
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0, 1).astype(np.float32)


def save(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    cv2.imwrite(str(path), u8, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def combos(grid):
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        yield dict(zip(keys, vals))


def param_id(combo):
    return "__".join(f"{k}-{v}" for k, v in combo.items())


def main():
    rows = []
    sources = sorted(p for p in IN_ROOT.iterdir() if p.is_dir())
    for src in sources:
        ref = load(src / f"{REFERENCE}.png")
        for var in VARIATIONS:
            path = src / f"{var}.png"
            if not path.exists():
                print("[skip] missing", path)
                continue
            img = load(path)
            if img.shape != ref.shape:
                ref_r = cv2.resize(ref, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                ref_r = ref
            save(OUT_ROOT / "images" / src.name / var / "input.jpg", img)
            base = all_metrics(img, ref_r)
            print(f"{src.name}/{var}")

            for tname, cfg in TECHNIQUES.items():
                for combo in combos(cfg["grid"]):
                    params = {**cfg["fixed"], **combo}
                    try:
                        t0 = time.perf_counter()
                        out, _ = cfg["fn"](img, params)
                        ms = (time.perf_counter() - t0) * 1000
                    except Exception as e:
                        print(f"  [fail] {tname} {combo}: {e}")
                        continue
                    assert out.shape == img.shape, (tname, out.shape)
                    out = np.clip(out, 0, 1).astype(np.float32)

                    rel = Path("images") / src.name / var / tname / f"{param_id(combo)}.jpg"
                    save(OUT_ROOT / rel, out)

                    m = all_metrics(out, ref_r)
                    row = {"source": src.name, "variation": var, "technique": tname,
                           "file": rel.as_posix(), "time_ms": ms}
                    row.update({f"p_{k}": v for k, v in combo.items()})
                    row.update(m)
                    row.update({f"in_{k}": v for k, v in base.items()})
                    row["d_ssim"] = m["ssim"] - base["ssim"]
                    row["d_psnr"] = m["psnr"] - base["psnr"]
                    row["sharp_gain"] = m["sharpness"] / (base["sharpness"] + 1e-12)
                    rows.append(row)

    df = pd.DataFrame(rows)
    OUT_ROOT.mkdir(exist_ok=True)
    df.to_csv(OUT_ROOT / "manifest.csv", index=False)
    print(f"\nDone: {len(df)} outputs. Manifest: {OUT_ROOT / 'manifest.csv'}")


if __name__ == "__main__":
    main()