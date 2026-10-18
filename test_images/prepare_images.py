
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import zipfile

# ========== CONFIGURATION ==========

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "original"
OUTPUT_DIR = BASE_DIR / "prepared"
ZIP_PATH = BASE_DIR / "image_variations.zip"

MAX_SIZE = 768
LIGHT_BLUR_RADIUS = 1.2
STRONG_BLUR_RADIUS = 4.0
NOISE_STD = 0.06

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

# ========== CHECK INPUT ==========

if not INPUT_DIR.is_dir():
    raise SystemExit(
        f"Input folder not found: {INPUT_DIR}"
    )

files = sorted(
    p for p in INPUT_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in EXTENSIONS
)

if not files:
    raise SystemExit(
        f"No images found in {INPUT_DIR}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== PREPARE EACH IMAGE ==========

for index, path in enumerate(files):
    print(f"\nProcessing {path.name}...")

    # Load image and correct orientation if metadata exists.
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    # Resize longest side to at most 768 px.
    image.thumbnail(
        (MAX_SIZE, MAX_SIZE),
        Image.Resampling.LANCZOS
    )

    # Separate folder for each source photograph.
    folder = OUTPUT_DIR / path.stem
    folder.mkdir(parents=True, exist_ok=True)

    # 1. Clean reference / ground truth.
    image.save(folder / "control_sharp.png")

    # Use float values in [0, 1] for degradation.
    img = np.asarray(image, dtype=np.float32) / 255.0

    # 2. Slight blur: Laplacian sharpening experiments.
    image.filter(
        ImageFilter.GaussianBlur(radius=LIGHT_BLUR_RADIUS)
    ).save(folder / "blurred_light_texture.png")

    # 3. Strong blur: frequency-domain unsharp masking.
    image.filter(
        ImageFilter.GaussianBlur(radius=STRONG_BLUR_RADIUS)
    ).save(folder / "blurred_strong_edges.png")

    # 4. Simulated uneven illumination.
    # Smooth horizontal gradient from darker to brighter.
    h, w = img.shape[:2]

    gradient = np.linspace(
        0.55, 1.20, w, dtype=np.float32
    )
    illumination = np.broadcast_to(
        gradient[None, :, None], (h, w, 1)
    )

    uneven = np.clip(img * illumination, 0.0, 1.0)

    Image.fromarray(
        np.round(uneven * 255).astype(np.uint8)
    ).save(folder / "uneven_illumination.png")

    # 5. Gaussian noise.
    # Each image gets a reproducible, distinct noise pattern.
    rng = np.random.default_rng(42 + index)

    noise = rng.normal(
        0.0, NOISE_STD, size=img.shape
    ).astype(np.float32)

    noisy = np.clip(img + noise, 0.0, 1.0)

    Image.fromarray(
        np.round(noisy * 255).astype(np.uint8)
    ).save(folder / "noisy.png")

    print(f"  Size: {image.width} x {image.height}")
    print("  Created 5 PNG images.")

# ========== CREATE ZIP ARCHIVE ==========

# Rebuild the archive with the latest prepared images.
with zipfile.ZipFile(
    ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED
) as archive:

    for output_file in sorted(OUTPUT_DIR.rglob("*.png")):
        archive.write(
            output_file,
            arcname=output_file.relative_to(BASE_DIR)
        )

total_images = sum(
    1 for p in OUTPUT_DIR.rglob("*.png")
    if p.is_file()
)

print("\n========== COMPLETE ==========")
print(f"Original photos: {len(files)}")
print(f"Generated PNGs: {total_images}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"ZIP archive: {ZIP_PATH}")
print("Original TIFF files were not modified.")
