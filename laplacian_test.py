"""
Laplacian experiment runner — corrected version.

Run from project root:
    python laplacian_test.py

Expected layout:
test_images/prepared/<source-folder>/
    control_sharp.png
    blurred_light_texture.png
    blurred_strong_edges.png
    noisy.png
    uneven_illumination.png

Runs:
Spatial: padding × kernel × strength.
Frequency: grayscale/RGB × padding × kernel × strength.
Outputs: intermediate-step plots, final images, and Excel metrics.
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from modules.laplacian import sharpen_spatial, sharpen_frequency

INPUT_ROOT = Path("test_images/prepared")
OUTPUT_ROOT = Path("results/laplacian_clean")
VARIATIONS = [
    "blurred_light_texture",
    "blurred_strong_edges",
    "noisy",
    "uneven_illumination",
]
PADDINGS = ["constant", "edge", "wrap"]
KERNEL_TYPES = ["4-neighbour", "8-neighbour"]
STRENGTHS = [0.25, 0.5, 0.75]


def read_rgb01(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def calculate_metrics(reference, output):
    reference = np.clip(reference, 0, 1)
    output = np.clip(output, 0, 1)
    mse = float(np.mean((reference - output) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
    if reference.ndim == 3:
        similarity = float(ssim(reference, output, data_range=1.0, channel_axis=2))
    else:
        similarity = float(ssim(reference, output, data_range=1.0))
    return mse, psnr, similarity


def save_intermediates(title, items, path):
    cols = 3
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, array) in zip(axes, items):
        array = np.asarray(array)
        if array.ndim == 3 and array.shape[2] == 3:
            ax.imshow(np.clip(array, 0, 1))
        else:
            cmap = "magma" if "spectrum" in name.lower() or "response" in name.lower() else "gray"
            ax.imshow(array, cmap=cmap)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_output(output, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(output, 0, 1) * 255).astype(np.uint8)
    if arr.ndim == 2:
        cv_image = arr
    else:
        cv_image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), cv_image):
        raise IOError(f"Failed to save output: {path}")


def run_spatial(gray, reference_gray, folder, variation, results):
    for padding in ("constant", "edge", "wrap"):
        for kernel in ("4-neighbour", "8-neighbour"):
            for strength in STRENGTHS:
                output, stage = sharpen_spatial(
                    gray, strength=strength, kernel_type=kernel, padding=padding
                )
                mse, psnr, similarity = calculate_metrics(reference_gray, output)
                results.append({
                    "domain": "spatial", "source_folder": folder,
                    "variation": variation, "color_mode": "grayscale",
                    "padding": padding, "kernel": kernel, "strength": strength,
                    "MSE": mse, "PSNR_dB": psnr, "SSIM": similarity,
                })
                base = OUTPUT_ROOT / "spatial" / folder / variation
                filename = f"{padding}_{kernel.replace('-', '')}_k{strength}.png"
                save_output(output, base / "outputs" / filename)
                if padding == "edge" and kernel == "4-neighbour" and strength == 0.5:
                    save_intermediates(
                        f"Spatial | {folder} | {variation} | edge | 4-neighbour | k=0.5",
                        [
                            ("Input", stage["input"]),
                            ("Laplacian kernel", stage["kernel"]),
                            ("Laplacian response", stage["laplacian_display"]),
                            ("Response (signed)", stage["laplacian_response"]),
                            ("Output before clipping", stage["sharpened_before_clip"]),
                            ("Final output", output),
                            ("Clean reference", reference_gray),
                        ],
                        base / "intermediate_steps.png",
                    )


def run_frequency(rgb, reference_rgb, folder, variation, results):
    for color_mode in ("grayscale", "rgb"):
        image = to_gray(rgb) if color_mode == "grayscale" else rgb
        reference = to_gray(reference_rgb) if color_mode == "grayscale" else reference_rgb
        for padding in ("constant", "edge", "wrap"):
            for kernel in ("4-neighbour", "8-neighbour"):
                for strength in STRENGTHS:
                    output, info = sharpen_frequency(
                        image, strength=strength, kernel_type=kernel,
                        padding=padding, color_mode=color_mode
                    )
                    mse, psnr, similarity = calculate_metrics(reference, output)
                    results.append({
                        "domain": "frequency", "source_folder": folder,
                        "variation": variation, "color_mode": color_mode,
                        "padding": padding, "kernel": kernel, "strength": strength,
                        "MSE": mse, "PSNR_dB": psnr, "SSIM": similarity,
                    })
                    base = OUTPUT_ROOT / "frequency" / folder / variation / color_mode
                    filename = f"{padding}_{kernel.replace('-', '')}_k{strength}.png"
                    save_output(output, base / "outputs" / filename)

                    if padding == "edge" and kernel == "4-neighbour" and strength == 0.5:
                        if color_mode == "grayscale":
                            stage = info["stages"]
                            items = [
                                ("Input", image),
                                ("Input FFT magnitude", stage["input_spectrum"]),
                                ("Laplacian frequency response", stage["laplacian_frequency_response"]),
                                ("Sharpening transfer function", stage["sharpening_transfer_function"]),
                                ("Filtered FFT magnitude", stage["filtered_spectrum"]),
                                ("Inverse FFT before clipping", stage["inverse_fft_before_clip"]),
                                ("Final output", output),
                                ("Clean reference", reference),
                            ]
                        else:
                            stage = info["channels"][0]  # red-channel intermediate stages
                            items = [
                                ("Input RGB", image),
                                ("Red channel input spectrum", stage["input_spectrum"]),
                                ("Laplacian frequency response", stage["laplacian_frequency_response"]),
                                ("Sharpening transfer function", stage["sharpening_transfer_function"]),
                                ("Red channel filtered spectrum", stage["filtered_spectrum"]),
                                ("Red channel inverse FFT", stage["inverse_fft_before_clip"]),
                                ("Final RGB output", output),
                                ("Clean RGB reference", reference),
                            ]
                        save_intermediates(
                            f"Frequency | {folder} | {variation} | {color_mode} | edge | 4-neighbour | k=0.5",
                            items, base / "intermediate_steps.png"
                        )


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = []
    folders = sorted(p for p in INPUT_ROOT.glob("*") if p.is_dir())
    if not folders:
        raise FileNotFoundError(
            f"No prepared image folders found under {INPUT_ROOT.resolve()}"
        )

    for folder in folders:
        reference_path = folder / "control_sharp.png"
        if not reference_path.exists():
            print(f"[SKIP] Missing reference: {reference_path}")
            continue
        reference_rgb = read_rgb01(reference_path)

        for variation in VARIATIONS:
            image_path = folder / f"{variation}.png"
            if not image_path.exists():
                print(f"[SKIP] Missing variation: {image_path}")
                continue

            print(f"Processing {folder.name}/{variation}")
            rgb = read_rgb01(image_path)
            if rgb.shape[:2] != reference_rgb.shape[:2]:
                reference = cv2.resize(
                    reference_rgb, (rgb.shape[1], rgb.shape[0]),
                    interpolation=cv2.INTER_AREA
                )
            else:
                reference = reference_rgb

            run_spatial(to_gray(rgb), to_gray(reference), folder.name, variation, all_results)
            run_frequency(rgb, reference, folder.name, variation, all_results)

    if not all_results:
        raise RuntimeError("No experiments ran. Verify prepared image names and folders.")

    results_df = pd.DataFrame(all_results)
    excel_path = OUTPUT_ROOT / "laplacian_experiment_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="All Results")
        summary = (
            results_df.groupby(
                ["domain", "variation", "color_mode", "padding", "kernel", "strength"],
                dropna=False
            )
            .agg(
                mean_MSE=("MSE", "mean"),
                mean_PSNR_dB=("PSNR_dB", "mean"),
                mean_SSIM=("SSIM", "mean"),
                image_count=("MSE", "count"),
            )
            .reset_index()
        )
        summary.to_excel(writer, index=False, sheet_name="Summary")

    print("\nAll experiments completed.")
    print(f"Metrics workbook: {excel_path}")
    print(f"Images and intermediate plots: {OUTPUT_ROOT}")
    print("Metrics compare each result against control_sharp.png; inspect images too.")


if __name__ == "__main__":
    main()
