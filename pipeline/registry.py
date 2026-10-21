from modules.laplacian import apply_laplacian
from modules.freq_unsharp import apply_freq_unsharp
from modules.homomorphic import apply_homomorphic

# "grid"  = parameters the user can vary on the website
# "fixed" = parameters held constant for every run
TECHNIQUES = {
    "laplacian": {
        "fn": apply_laplacian,
        "fixed": {"padding": "edge"},
        "grid": {
            "kernel_type": ["4-neighbour", "8-neighbour"],
            "strength": [0.25, 0.5, 0.75, 1.0],
        },
    },
    "freq_unsharp": {
        "fn": apply_freq_unsharp,
        "fixed": {"order": 2},
        "grid": {
            "filter_type": ["ideal", "butterworth", "gaussian"],
            "cutoff": [10, 30, 60, 120],
            "strength": [0.5, 1.0, 2.0],
        },
    },
    "homomorphic": {
        "fn": apply_homomorphic,
        "fixed": {"c": 1.0, "padding": 32},
        "grid": {
            "gamma_l": [0.3, 0.5, 0.8],
            "gamma_h": [1.5, 2.0, 3.0],
            "cutoff": [10, 30, 60],
        },
    },
}