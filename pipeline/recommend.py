import pandas as pd

LABELS = {
    "laplacian": "Laplacian",
    "freq_unsharp": "Frequency unsharp masking",
    "homomorphic": "Homomorphic filtering",
}
METRIC_COLS = ["d_ssim", "d_psnr", "ssim", "psnr", "sharpness",
               "sharp_gain", "contrast", "illum_cv", "noise", "time_ms"]


def param_cols(df, technique):
    g = df[df.technique == technique]
    return [c for c in df.columns if c.startswith("p_") and g[c].notna().any()]


def ranking(df, variation, technique):
    """Parameter sets for one image condition and filter, best first (mean over sources)."""
    g = df[(df.variation == variation) & (df.technique == technique)]
    pc = param_cols(df, technique)
    r = g.groupby(pc)[METRIC_COLS].mean().reset_index()
    return r.sort_values("d_ssim", ascending=False).reset_index(drop=True)


def params_str(row, pc):
    return ", ".join(f"{c[2:]}={row[c]}" for c in pc)


def best_overall(df, variation):
    """Best parameter set of each filter for one image condition."""
    out = []
    for t in LABELS:
        if t not in set(df.technique):
            continue
        r = ranking(df, variation, t).iloc[0]
        out.append({"Filter": LABELS[t], "Best parameters": params_str(r, param_cols(df, t)),
                    "ΔSSIM": r.d_ssim, "SSIM": r.ssim, "ΔPSNR (dB)": r.d_psnr,
                    "Sharpness gain ×": r.sharp_gain, "Time (ms)": r.time_ms})
    return pd.DataFrame(out).sort_values("ΔSSIM", ascending=False).reset_index(drop=True)


def suitability_matrix(df):
    """Rows = image condition, columns = filter, value = best achievable ΔSSIM."""
    rows = {}
    for v in sorted(df.variation.unique()):
        rows[v] = {LABELS[t]: ranking(df, v, t).iloc[0].d_ssim
                   for t in LABELS if t in set(df.technique)}
    return pd.DataFrame(rows).T