from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.recommend import (LABELS, ranking, best_overall,
                                param_cols, suitability_matrix)

OUT = Path("outputs")
st.set_page_config(page_title="Image Enhancement Studio", layout="wide")


@st.cache_data
def load():
    return pd.read_csv(OUT / "manifest.csv")


if not (OUT / "manifest.csv").exists():
    st.error("Run `python generate_outputs.py` first.")
    st.stop()
df = load()

st.title("Image Enhancement Studio")
st.caption("Pre-computed results: choose an image, a filter and its parameters.")

# ---------------- sidebar ----------------
with st.sidebar:
    source = st.selectbox("Source photo", sorted(df.source.unique()))
    variation = st.selectbox("Image condition", sorted(df.variation.unique()))
    technique = st.selectbox("Filter", list(LABELS), format_func=LABELS.get)

    g = df[(df.source == source) & (df.variation == variation) & (df.technique == technique)]
    pcols = param_cols(df, technique)
    best = ranking(df, variation, technique).iloc[0]

    st.markdown("---")
    use_best = st.checkbox("Use recommended parameters", value=False)
    chosen = {}
    for c in pcols:
        name = c[2:]
        opts = sorted(g[c].unique().tolist())
        if use_best:
            chosen[c] = best[c]
            st.write(f"**{name}** = {best[c]}")
        elif pd.api.types.is_numeric_dtype(g[c]):
            chosen[c] = st.select_slider(name, options=opts)
        else:
            chosen[c] = st.radio(name, opts)


def pick(frame, sel):
    for c, v in sel.items():
        frame = frame[frame[c] == v]
    return frame.iloc[0]


r = pick(g, chosen)
input_path = OUT / "images" / source / variation / "input.jpg"

tab1, tab2, tab3, tab4 = st.tabs(["Result", "Parameter sweep", "Which filter suits this image?", "About the metrics"])

# ---------------- tab 1: result ----------------
with tab1:
    c1, c2 = st.columns(2)
    c1.image(str(input_path), caption=f"Input ({variation})", use_container_width=True)
    c2.image(str(OUT / r["file"]), caption=LABELS[technique], use_container_width=True)

    m = st.columns(4)
    m[0].metric("SSIM vs clean original", f"{r.ssim:.3f}", f"{r.ssim - r.in_ssim:+.3f}")
    m[1].metric("PSNR (dB)", f"{r.psnr:.2f}", f"{r.psnr - r.in_psnr:+.2f}")
    m[2].metric("Sharpness (Var. of Laplacian)", f"{r.sharpness:.4f}", f"{r.sharpness - r.in_sharpness:+.4f}")
    m[3].metric("Contrast (std)", f"{r.contrast:.3f}", f"{r.contrast - r.in_contrast:+.3f}")
    n = st.columns(3)
    n[0].metric("Illumination unevenness", f"{r.illum_cv:.3f}", f"{r.illum_cv - r.in_illum_cv:+.3f}", delta_color="inverse")
    n[1].metric("Estimated noise", f"{r.noise:.4f}", f"{r.noise - r.in_noise:+.4f}", delta_color="inverse")
    n[2].metric("Processing time (ms)", f"{r.time_ms:.1f}")

# ---------------- tab 2: sweep ----------------
with tab2:
    st.write("Vary one parameter; the others stay at the sidebar values.")
    vary = st.selectbox("Parameter to vary", [c[2:] for c in pcols])
    vc = "p_" + vary
    sub = g
    for c, v in chosen.items():
        if c != vc:
            sub = sub[sub[c] == v]
    sub = sub.sort_values(vc)
    cols = st.columns(min(len(sub), 4))
    for i, (_, row) in enumerate(sub.iterrows()):
        with cols[i % len(cols)]:
            st.image(str(OUT / row["file"]), use_container_width=True,
                     caption=f"{vary}={row[vc]} | SSIM {row.ssim:.3f}")
    chart = st.line_chart if pd.api.types.is_numeric_dtype(sub[vc]) else st.bar_chart
    a, b = st.columns(2)
    a.caption("SSIM vs clean original")
    with a:
        chart(sub.set_index(vc)[["ssim"]])
    b.caption("Sharpness (Variance of Laplacian)")
    with b:
        chart(sub.set_index(vc)[["sharpness"]])

# ---------------- tab 3: recommendations ----------------
with tab3:
    st.subheader(f"Best setting of each filter for '{variation}'")
    st.caption("Averaged over all source photos. Ranked by ΔSSIM (improvement toward the clean original).")
    st.dataframe(best_overall(df, variation), use_container_width=True)

    st.subheader(f"Top 5 parameter sets for {LABELS[technique]}")
    top = ranking(df, variation, technique).head(5)
    top = top.rename(columns={c: c[2:] for c in pcols})
    st.dataframe(top[[c[2:] for c in pcols] + ["d_ssim", "ssim", "d_psnr", "sharp_gain"]],
                 use_container_width=True)

    st.subheader("Suitability matrix (best ΔSSIM)")
    st.caption("Green = the filter helps this kind of image. Red = it makes it worse.")
    st.dataframe(suitability_matrix(df).style.background_gradient(cmap="RdYlGn", axis=None).format("{:+.3f}"),
                 use_container_width=True)

# ---------------- tab 4: metrics ----------------
with tab4:
    st.markdown("""
- **SSIM / PSNR** compare the result to the clean `control_sharp` original. Higher is closer to the original.
- **ΔSSIM** is the improvement over the unprocessed input. It is the score used to pick the "best" parameters.
- **Sharpness** is the variance of the Laplacian. Noise also raises it, so read it together with the noise estimate.
- **Contrast** is the standard deviation of intensities.
- **Illumination unevenness** is the coefficient of variation of a heavily blurred copy. Lower means more even lighting.
- **Estimated noise** is Immerkaer's noise-level estimate. Lower is better.
""")