import cv2
import numpy as np
from skimage.metrics import structural_similarity


def psnr(ref, img):
    mse = float(np.mean((ref.astype(np.float64) - img.astype(np.float64)) ** 2))
    return 100.0 if mse < 1e-12 else float(10 * np.log10(1.0 / mse))


def ssim(ref, img):
    return float(structural_similarity(ref, img, data_range=1.0))


def sharpness(img):
    """Variance of Laplacian (higher = sharper, but noise also raises it)."""
    return float(cv2.Laplacian(img.astype(np.float32), cv2.CV_32F).var())


def contrast(img):
    """Global contrast: standard deviation of intensity."""
    return float(img.std())


def illumination_cv(img):
    """Unevenness of lighting: coefficient of variation of a heavily blurred image.
    Lower = more uniform illumination."""
    sigma = 0.08 * min(img.shape)
    low = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    return float(low.std() / (low.mean() + 1e-6))


def noise_sigma(img):
    """Immerkaer's fast noise-level estimate."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = img.shape
    conv = cv2.filter2D(img.astype(np.float32), -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) / (6.0 * (w - 2) * (h - 2)) * np.abs(conv).sum())


def all_metrics(img, ref):
    """img: image to score; ref: clean reference (control_sharp)."""
    return {
        "ssim": ssim(ref, img),
        "psnr": psnr(ref, img),
        "sharpness": sharpness(img),
        "contrast": contrast(img),
        "illum_cv": illumination_cv(img),
        "noise": noise_sigma(img),
    }