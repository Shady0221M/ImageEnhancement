
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import zipfile

INPUT_DIR = Path("original")
OUTPUT_DIR = Path("variations")

OUTPUT_DIR.mkdir(exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

images = [
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in extensions
]

if not images:
    raise SystemExit("No images found inside the originals folder.")

for path in images:
    # Convert to RGB for consistent processing
    original = Image.open(path).convert("RGB")
    img = np.asarray(original).astype(np.float32) / 255.0

    folder = OUTPUT_DIR / path.stem
    folder.mkdir(exist_ok=True)

    # 1. Preserve original sharp image
    original.save(folder / "01_original.png")

    # 2. Slight blur: useful for Laplacian sharpening
    slightly_blurred = original.filter(
        ImageFilter.GaussianBlur(radius=1.2)
    )
    slightly_blurred.save(folder / "02_slight_blur.png")

    # 3. Strong blur: useful for unsharp masking
    strongly_blurred = original.filter(
        ImageFilter.GaussianBlur(radius=4.0)
    )
    strongly_blurred.save(folder / "03_strong_blur.png")

    # 4. Uneven illumination: gradual dark-to-bright gradient
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w]

    # Smooth illumination gradient across the image
    illumination = 0.55 + 0.65 * (x / max(w - 1, 1))
    illumination = illumination.astype(np.float32)

    uneven = np.clip(img * illumination[..., None], 0, 1)
    uneven_img = Image.fromarray(
        (uneven * 255).astype(np.uint8)
    )
    uneven_img.save(folder / "04_uneven_illumination.png")

    # 5. Add Gaussian noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.06, img.shape).astype(np.float32)

    noisy = np.clip(img + noise, 0, 1)
    noisy_img = Image.fromarray(
        (noisy * 255).astype(np.uint8)
    )
    noisy_img.save(folder / "05_gaussian_noise.png")

    print(f"Created variations for: {path.name}")

# Package all variations into one ZIP file
zip_path = Path("image_variations.zip")

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for file in OUTPUT_DIR.rglob("*.png"):
        z.write(file, file.relative_to(OUTPUT_DIR.parent))

print(f"\nDone! Images saved in: {OUTPUT_DIR.resolve()}")
print(f"ZIP archive created: {zip_path.resolve()}")
