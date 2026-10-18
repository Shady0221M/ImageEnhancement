Python 3.13.16 required - [Download](https://www.python.org/downloads/release/python-31316/)

**Open terminal:**

```bash
git clone https://github.com/Shady0221M/ImageEnhancement.git
cd ImageEnhancement
code .
```

**Inside VS Code:**

```bash
git pull
py -3.13 --version
py -3.13 -m venv .venv
```

Activate the virtual environment using whichever command works:

```powershell
.\.venv\Scripts\Activate.ps1
```

```bat
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

# ImageEnhancement

Image Processing Collaborative Project- IP- CSPE74(090,119,125)

**Input format.** Grayscale image, float values 0 to 1, as a 2D array.

**Parameters.** Passed as a single dictionary. Each member lists their parameter names and default values now (for example, strength, cutoff, filter type for Member 2).

**Output format.** Processed image (same shape, float, 0 to 1) plus a dictionary of intermediate outputs. Agree on the dictionary keys each module will provide.

**Function names.** One main function per module file, named consistently (e.g. `apply_<technique>`), with the same argument order.

**Color handling.** Decide now. Recommendation: grayscale only for the core project, luminance-only as a bonus if time permits.

**Padding and edge handling.** Who pads, and where? Recommendation: each module handles its own padding internally and returns the original size.

**Shared helpers and owners:**

- Member 1 owns image loading, grayscale conversion, normalization (`io_utils.py`)
- Member 2 owns spectrum display scaling and plotting (`display.py`)
- Member 3 owns metrics (`metrics.py`)

**Who edits app.py.** Recommendation: Member 1 only, until the integration week.
## Homomorphic filtering

Run from the `ImageEnhancement` directory after installing dependencies:

```bash
python -m modules.homomorphic test_images/blurred.jpg -o results/enhanced.png
```

Replace the input path with your image. Color images are converted to grayscale.
You can also run `python modules/homomorphic.py input.jpg -o output.png` directly.

The module follows the illumination–reflectance model `f = i * r`, takes
`ln(f + 1e-6)`, computes the centered FFT, applies a modified Gaussian
high-pass filter, computes the inverse FFT, and exponentiates the result.
The filter is `H = gamma_l + (gamma_h - gamma_l) * (1 - exp(-c * D² / cutoff²))`.
Reflection padding reduces edge artifacts; output is cropped to the input size.
The result is stretched if necessary to fit [0, 1].

```python
from modules.homomorphic import apply_homomorphic
from utils.io_utils import load_image_gray, save_image

image = load_image_gray("test_images/blurred.jpg")
result, stages = apply_homomorphic(image, {
    "gamma_l": 0.5, "gamma_h": 1.5, "cutoff": 30.0, "c": 1.0, "padding": 32,
})
save_image("enhanced.png", result)
```

`gamma_l` controls low-frequency illumination attenuation (0 < value <= 1),
`gamma_h` controls detail amplification (>= 1), `cutoff` is a positive radius
in frequency bins of the padded image, and `c` is a positive transition factor.
Try a smaller `gamma_h` if noise is amplified. The module enhances contrast
and existing edges in blurred images; it does not undo motion or defocus blur.

Use `--intermediates results/stages.npz` to save `log_image`, `spectrum`,
`filter`, `filtered_spectrum`, `filtered_log`, and `restored_image`.
These arrays include padding; the returned image does not.
Use `--help` to see all command-line options.
