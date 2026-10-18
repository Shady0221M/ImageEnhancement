"""Homomorphic illumination correction and contrast enhancement.

Run from the project directory::
    python -m modules.homomorphic input.jpg -o output.png
"""

import argparse
from pathlib import Path

import numpy as np


def apply_homomorphic(image, params=None):
    """Return (enhanced, intermediates) for a grayscale image in [0, 1].

    Parameters (dictionary): gamma_l=0.5 attenuates illumination;
    gamma_h=1.5 boosts detail; cutoff=30.0 is the Gaussian radius in
    frequency bins; c=1.0 controls the transition; padding=32 reflects
    pixels at the edges to reduce FFT boundary artifacts.

    This enhances existing detail, including in blurred images; it is not
    a blur inversion/deconvolution algorithm. All intermediate arrays use
    the padded shape, while the final output has the original shape.
    """
    defaults = dict(gamma_l=0.5, gamma_h=1.5, cutoff=30.0, c=1.0, padding=32)
    params = {} if params is None else dict(params)
    unknown = params.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    defaults.update(params)
    gamma_l, gamma_h, cutoff, c = (
        float(defaults[key]) for key in ('gamma_l', 'gamma_h', 'cutoff', 'c')
    )
    if not np.isfinite([gamma_l, gamma_h, cutoff, c]).all():
        raise ValueError('Filter parameters must be finite.')
    if not (0 < gamma_l <= 1 <= gamma_h and cutoff > 0 and c > 0):
        raise ValueError('Require 0 < gamma_l <= 1 <= gamma_h, cutoff > 0, c > 0.')
    padding = defaults['padding']
    if isinstance(padding, bool) or not isinstance(padding, (int, np.integer)) or padding < 0:
        raise ValueError('padding must be a nonnegative integer.')
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or image.size == 0:
        raise ValueError('Expected a nonempty 2D grayscale image.')
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('Image intensities must be finite and in [0, 1].')

    # 1. f = illumination * reflectance (a model, not a unique decomposition).
    padded = np.pad(image, padding, mode='reflect') if padding else image.copy()
    # 2. A small offset makes the natural logarithm defined for black pixels.
    epsilon = 1e-6
    log_image = np.log(padded + epsilon)
    # 3. Center the DFT so the illumination frequencies lie near the origin.
    spectrum = np.fft.fftshift(np.fft.fft2(log_image))
    rows, cols = padded.shape
    u = np.arange(rows, dtype=float) - rows // 2
    v = np.arange(cols, dtype=float) - cols // 2
    distance = np.hypot(u[:, None], v[None, :])
    # 4. Modified Gaussian high-pass filter: H(0)=gamma_l, H(infinity)=gamma_h.
    with np.errstate(over='ignore'):
        transfer = gamma_l + (gamma_h - gamma_l) * (
            -np.expm1(-c * (distance / cutoff) ** 2)
        )
    filtered_spectrum = spectrum * transfer
    # 5. Undo the centering before taking the real inverse DFT.
    filtered_log = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)).real
    # 6. Reverse the log offset. Stabilize extreme gains before exponentiation.
    restored = np.exp(np.clip(filtered_log, -745, 700)) - epsilon
    cropped = restored[padding:padding + image.shape[0], padding:padding + image.shape[1]]
    # Stretch only if needed to fit the shared [0, 1] output contract.
    lo, hi = cropped.min(), cropped.max()
    if hi > lo and (lo < 0 or hi > 1):
        enhanced = (cropped - lo) / (hi - lo)
    else:
        enhanced = np.clip(cropped, 0, 1)
    return enhanced.astype(np.float32), {
        'log_image': log_image,
        'spectrum': spectrum,
        'filter': transfer,
        'filtered_spectrum': filtered_spectrum,
        'filtered_log': filtered_log,
        'restored_image': restored,
    }


def main():
    """Read an image, apply the filter, and save a grayscale result."""
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Input image (color is converted to grayscale)')
    parser.add_argument('-o', '--output', type=Path, default=Path('homomorphic_output.png'))
    parser.add_argument('--gamma-l', type=float, default=0.5)
    parser.add_argument('--gamma-h', type=float, default=1.5)
    parser.add_argument('--cutoff', type=float, default=30.0)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--padding', type=int, default=32)
    parser.add_argument('--intermediates', type=Path, help='Optional .npz file for all stages')
    args = parser.parse_args()
    try:
        image = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Cannot read image: {args.input}')
        result, stages = apply_homomorphic(image.astype(np.float64) / 255, {
            key: getattr(args, key) for key in ('gamma_l', 'gamma_h', 'cutoff', 'c', 'padding')
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), np.rint(result * 255).astype(np.uint8)):
            raise ValueError(f'Cannot save image: {args.output}')
        if args.intermediates:
            args.intermediates.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.intermediates, **stages)
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f'Error: {error}\n')
    print(f'Saved enhanced image to {args.output}')


if __name__ == '__main__':
    main()
