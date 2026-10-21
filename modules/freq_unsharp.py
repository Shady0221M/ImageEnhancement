import numpy as np
from utils.io_utils import clip01, normalize01

DEFAULTS = {
    "filter_type": "gaussian",   # "ideal" | "gaussian" | "butterworth"
    "cutoff": 30.0,              # D0, in pixels of the padded spectrum
    "strength": 1.0,             # k: 1 = unsharp masking, >1 = high-boost
    "order": 2,                  # Butterworth order n
}


def pad_image(img):
    """Zero-pad to 2M x 2N (image in the top-left) to avoid wrap-around."""
    M, N = img.shape
    padded = np.zeros((2 * M, 2 * N), dtype=np.float64)
    padded[:M, :N] = img
    return padded


def distance_grid(P, Q):
    """D(u,v): distance from the centre of the shifted spectrum."""
    u = np.arange(P) - P // 2
    v = np.arange(Q) - Q // 2
    V, U = np.meshgrid(v, u)
    return np.sqrt(U ** 2 + V ** 2)


def lowpass_filter(shape, filter_type, d0, order=2):
    """H_LP(u,v) for Ideal, Gaussian or Butterworth."""
    if d0 <= 0:
        raise ValueError("cutoff must be > 0")
    D = distance_grid(*shape)
    ft = filter_type.lower()
    if ft == "ideal":
        return (D <= d0).astype(np.float64)
    if ft == "gaussian":
        return np.exp(-(D ** 2) / (2 * d0 ** 2))
    if ft == "butterworth":
        return 1.0 / (1.0 + (D / d0) ** (2 * order))
    raise ValueError(f"Unknown filter_type: {filter_type}")


def highboost_filter(h_lp, k):
    """H_hb = 1 + k * (1 - H_LP). k=1 is unsharp masking, k>1 is high-boost."""
    return 1.0 + k * (1.0 - h_lp)


def fft_roundtrip_error(img):
    """Max abs error of forward-then-inverse FFT (should be ~1e-12)."""
    img = img.astype(np.float64)
    F = np.fft.fftshift(np.fft.fft2(img))
    back = np.fft.ifft2(np.fft.ifftshift(F)).real
    return float(np.max(np.abs(back - img)))


def apply_freq_unsharp(image, params=None):
    """
    Frequency-domain unsharp masking / high-boost filtering.

    image : 2D float array in [0, 1]
    params: dict with filter_type, cutoff, strength, order (all optional)
    returns (output, intermediates); output has the same shape, float32, [0, 1]
    """
    p = {**DEFAULTS, **(params or {})}
    M, N = image.shape

    padded = pad_image(image.astype(np.float64))
    F = np.fft.fftshift(np.fft.fft2(padded))

    h_lp = lowpass_filter(padded.shape, p["filter_type"], p["cutoff"], p["order"])
    H = highboost_filter(h_lp, p["strength"])

    G = F * H
    g = np.fft.ifft2(np.fft.ifftshift(G)).real
    out = clip01(g[:M, :N]).astype(np.float32)

    intermediates = {
        "spectrum": normalize01(np.log1p(np.abs(F))),
        "lowpass": h_lp,
        "filter": H,
        "filtered_spectrum": normalize01(np.log1p(np.abs(G))),
    }
    return out, intermediates