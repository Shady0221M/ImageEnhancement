"""
Laplacian sharpening module — corrected version.

Supports:
- Spatial domain: constant, edge, wrap padding.
- Frequency domain: grayscale or RGB, constant/edge/wrap padding.
- 4-neighbour and 8-neighbour kernels.
- Returns intermediate arrays for visualization.

Images passed to sharpening functions must be float arrays normalized to [0, 1].
"""
import numpy as np
import cv2

KERNELS = {
    "4-neighbour": np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=np.float32),
    "8-neighbour": np.array([[1, 1, 1],
                             [1, -8, 1],
                             [1, 1, 1]], dtype=np.float32),
}
PADDING_MODES = {"constant", "edge", "wrap"}
COLOR_MODES = {"grayscale", "rgb"}


def _validate_gray(image, strength, kernel_type, padding):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image.")
    if image.size == 0 or not np.isfinite(image).all():
        raise ValueError("Image is empty or contains NaN/inf.")
    if image.min() < -1e-6 or image.max() > 1.000001:
        raise ValueError("Image values must be normalized to [0, 1].")
    if kernel_type not in KERNELS:
        raise ValueError(f"kernel_type must be one of {list(KERNELS)}")
    if padding not in PADDING_MODES:
        raise ValueError(f"padding must be one of {sorted(PADDING_MODES)}")
    if strength < 0:
        raise ValueError("strength must be non-negative.")
    return image


def convolve2d(image, kernel, padding="edge"):
    """Apply 3x3 correlation with the requested padding."""
    image = np.asarray(image, dtype=np.float32)
    if padding == "constant":
        padded = np.pad(image, 1, mode="constant", constant_values=0)
    elif padding in {"edge", "wrap"}:
        padded = np.pad(image, 1, mode=padding)
    else:
        raise ValueError(f"Unsupported padding: {padding}")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return np.einsum("ijkl,kl->ij", windows, kernel).astype(np.float32)


def _display_signed(array):
    """Map signed response to [0,1] for display only."""
    array = np.asarray(array, dtype=np.float32)
    maximum = float(np.max(np.abs(array)))
    if maximum < 1e-12:
        return np.full_like(array, 0.5)
    return 0.5 + 0.5 * array / maximum


def sharpen_spatial(image, strength=0.5, kernel_type="4-neighbour",
                    padding="edge"):
    """Spatial sharpening: output = image - strength * Laplacian response."""
    image = _validate_gray(image, strength, kernel_type, padding)
    kernel = KERNELS[kernel_type]
    response = convolve2d(image, kernel, padding)
    raw = image - strength * response
    output = np.clip(raw, 0.0, 1.0)
    return output, {
        "input": image,
        "kernel": kernel.copy(),
        "laplacian_response": response,
        "laplacian_display": _display_signed(response),
        "sharpened_before_clip": raw,
        "sharpened": output,
    }


def _pad_frequency(image, padding, pad_widths):
    """Pad an image before FFT filtering. Argument order is intentional."""
    if padding == "constant":
        return np.pad(image, pad_widths, mode="constant", constant_values=0)
    if padding in {"edge", "wrap"}:
        return np.pad(image, pad_widths, mode=padding)
    raise ValueError(f"Unsupported padding: {padding}")


def _discrete_laplacian_transfer(shape, kernel):
    """DFT of a discrete Laplacian kernel with its centre shifted to the origin."""
    kh, kw = kernel.shape
    spatial_kernel = np.zeros(shape, dtype=np.float32)
    spatial_kernel[:kh, :kw] = kernel
    spatial_kernel = np.roll(spatial_kernel, -(kh // 2), axis=0)
    spatial_kernel = np.roll(spatial_kernel, -(kw // 2), axis=1)
    return np.fft.fft2(spatial_kernel)


def _filter_frequency_channel(channel, strength, kernel_type, padding, pad=16):
    """Filter one normalized 2D channel and return the output plus intermediate stages."""
    channel = np.asarray(channel, dtype=np.float32)
    if channel.ndim != 2:
        raise ValueError("Frequency channel must be 2D.")
    if channel.min() < -1e-6 or channel.max() > 1.000001:
        raise ValueError("Channel values must be normalized to [0, 1].")

    pad_widths = ((pad, pad), (pad, pad))
    padded = _pad_frequency(channel, padding, pad_widths)
    F = np.fft.fft2(padded)

    # Exact frequency response of the chosen discrete kernel.
    L_complex = _discrete_laplacian_transfer(padded.shape, KERNELS[kernel_type])
    L = np.real(L_complex)
    H = 1.0 - strength * L
    G = F * H

    reconstructed = np.fft.ifft2(G).real
    cropped = reconstructed[pad:-pad, pad:-pad] if pad else reconstructed
    output = np.clip(cropped, 0.0, 1.0)

    stages = {
        "padded_input": padded,
        "input_spectrum": np.log1p(np.abs(np.fft.fftshift(F))),
        "laplacian_frequency_response": np.fft.fftshift(L),
        "sharpening_transfer_function": np.fft.fftshift(H),
        "filtered_spectrum": np.log1p(np.abs(np.fft.fftshift(G))),
        "inverse_fft_before_clip": cropped,
        "sharpened": output,
    }
    return output, stages


def sharpen_frequency(image, strength=0.5, kernel_type="4-neighbour",
                      padding="edge", color_mode="grayscale"):
    """
    Frequency-domain Laplacian sharpening.

    Input:
      grayscale: [H,W] or RGB [H,W,3] if color_mode='grayscale'
      RGB mode: RGB [H,W,3], each channel filtered independently

    Returns (output, intermediate_dict).
    """
    if kernel_type not in KERNELS:
        raise ValueError(f"kernel_type must be one of {list(KERNELS)}")
    if padding not in PADDING_MODES:
        raise ValueError(f"padding must be one of {sorted(PADDING_MODES)}")
    if color_mode not in COLOR_MODES:
        raise ValueError(f"color_mode must be one of {sorted(COLOR_MODES)}")
    if strength < 0:
        raise ValueError("strength must be non-negative.")

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        if color_mode == "rgb":
            raise ValueError("RGB mode requires an RGB image shaped [H,W,3].")
        gray = arr
        _validate_gray(gray, strength, kernel_type, padding)
        output, stages = _filter_frequency_channel(
            gray, strength, kernel_type, padding
        )
        return output, {"color_mode": "grayscale", "stages": stages}

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Input must be grayscale [H,W] or RGB [H,W,3].")
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("Image is empty or contains NaN/inf.")
    if arr.min() < -1e-6 or arr.max() > 1.000001:
        raise ValueError("RGB values must be normalized to [0, 1].")

    if color_mode == "grayscale":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        output, stages = _filter_frequency_channel(
            gray, strength, kernel_type, padding
        )
        return output, {"color_mode": "grayscale", "stages": stages}

    outputs, channel_stages = [], []
    for channel_index in range(3):
        result, stages = _filter_frequency_channel(
            arr[:, :, channel_index], strength, kernel_type, padding
        )
        outputs.append(result)
        channel_stages.append(stages)
    output = np.stack(outputs, axis=2)
    return output, {
        "color_mode": "rgb",
        "channels": channel_stages,
        "sharpened": output,
    }
