import cv2
import numpy as np


def load_image_gray(path):
    """Load an image file as a 2D grayscale float32 array in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img.astype(np.float32) / 255.0


def to_gray_float(img):
    """Convert any array (uint8 or float, gray or BGR) to 2D float32 in [0, 1]."""
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def clip01(img):
    """Clip an array to the valid display range [0, 1]."""
    return np.clip(img, 0.0, 1.0)


def normalize01(img):
    """Stretch an array to [0, 1] (useful for viewing intermediates)."""
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def save_image(path, img):
    """Save a [0, 1] float image as an 8-bit PNG/JPG."""
    cv2.imwrite(str(path), (clip01(img) * 255).round().astype(np.uint8))