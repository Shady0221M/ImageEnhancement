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