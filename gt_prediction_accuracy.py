# Validation Pipeline Step 4:
# Fuzzy Set-Based Accuracy Assessment for Tree Species Fractions Classification

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

#--------Setup--------|
input       = ""      # input geopackage with ground truth and prediction values for tree species 0-100% from step 4
gt_prefix   = "gt_"
pred_prefix = "pred_"

'''
________________________________________________________________________________
Reading input-dataset and build dataframe for ground truth and prediction per Polygon

R      : (N, K)  Ground-Truth-Memberships
S      : (N, K)  Prediction-Memberships
W      : (N)    area-weight per polygon
'''

def load_data(path: str, gt_prefix: str, pred_prefix: str):

    gdf = gpd.read_file(path)

    gt_species   = {c[len(gt_prefix):]   for c in gdf.columns if c.startswith(gt_prefix)}
    pred_species = {c[len(pred_prefix):] for c in gdf.columns if c.startswith(pred_prefix)}
    species = sorted(gt_species & pred_species)

    R = gdf[[f"{gt_prefix}{a}"   for a in species]].to_numpy(dtype=float) # GT-Matrix
    S = gdf[[f"{pred_prefix}{a}" for a in species]].to_numpy(dtype=float) # Pred-Matrix

    R = np.nan_to_num(R)
    S = np.nan_to_num(S)

    # Normalise to 1
    if R.max() > 1.5: R /= 100.0
    if S.max() > 1.5: S /= 100.0

    for arr in (R, S):
        row_sums = arr.sum(axis=1, keepdims=True)
        arr[:] = np.where(row_sums > 0, arr / row_sums, 0.0)

    # area-weight for the weighted average between polygons
    areas = gdf.geometry.area.values.astype(float)
    W = areas / areas.sum()
    return R, S, W, species

'''
________________________________________________________________________________
Compute Accuracy metrics across all polygons with area-weights

    diag_k   = Σ_n  W[n] · min(S[n,k], R[n,k])    - diagonal values for min(pred | gt) per tree species
    gt_sum_k = Σ_n  W[n] · R[n,k]                 - GT sum per species
    pd_sum_k = Σ_n  W[n] · S[n,k]                 - pred sum per species

    PA_k = diag_k / gt_sum_k    - Producer's Accuracy
    UA_k = diag_k / pd_sum_k    - User's Accuracy
    OA   = Σ_k diag_k           - Overall

'''
def compute_accuracy(R: np.ndarray, S: np.ndarray,
                     W: np.ndarray, species: list) -> tuple[pd.DataFrame, float]:
    W_col = W.reshape(-1, 1)   # (N,1) for broadcasting with (N,K)

    diag   = (W_col * np.minimum(S, R)).sum(axis=0)   # (K,)
    gt_sum = (W_col * R).sum(axis=0)                  # (K,)
    pd_sum = (W_col * S).sum(axis=0)                  # (K,)

    PA = np.where(gt_sum > 0, diag / gt_sum, np.nan)
    UA = np.where(pd_sum > 0, diag / pd_sum, np.nan)
    OA = diag.sum()

    df = pd.DataFrame({"Species": species, "PA": PA, "UA": UA})
    return df, OA

#-----Plot tree species accuracies-------|

# color scheme
BG     = "#0f1117"
PANEL  = "#1c1f2e"
TEXT   = "#e0e0ee"
GREY   = "#7a7f99"
BLUE   = "#4a8fd4"
ORANGE = "#f0883e"
GOLD   = "#FFD700"

def _apply_style(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3e")
    ax.tick_params(colors=GREY, labelsize=9)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)
    ax.title.set_color(TEXT)
    ax.yaxis.grid(True, color="#2a2d3e", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)

def plot_accuracy(df: pd.DataFrame, oa: float) -> None:

    K = len(df)
    x = np.arange(K)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, K * 0.9), 6), facecolor=BG)
    _apply_style(ax)

    # barplot
    ax.bar(x - bar_width / 2, df.PA * 100, bar_width,
           color=BLUE,   alpha=0.85, label="Producer Accuracy (PA)")
    ax.bar(x + bar_width / 2, df.UA * 100, bar_width,
           color=ORANGE, alpha=0.85, label="User Accuracy (UA)")

    # reference-line
    ax.axhline(oa * 100, color=GOLD, lw=1.5, ls="--",
               label=f"Overall Accuracy = {oa * 100:.1f} %")

    # numbers on bars
    for i, row in df.iterrows():
        if not np.isnan(row.PA):
            ax.text(x[i] - bar_width / 2, row.PA * 100 + 1.5,
                    f"{row.PA * 100:.0f}", ha="center", fontsize=7.5, color=BLUE)
        if not np.isnan(row.UA):
            ax.text(x[i] + bar_width / 2, row.UA * 100 + 1.5,
                    f"{row.UA * 100:.0f}", ha="center", fontsize=7.5, color=ORANGE)

    ax.set_xticks(x)
    ax.set_xticklabels(df.Species, rotation=45, ha="right")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%, area-weighted)", color=GREY)
    ax.set_title(
        "classification quality for tree species (area-weighted)\n"
        "Producer Accuracy (PA) · User Accuracy (UA)",
        fontsize=11, color=TEXT)
    ax.legend(fontsize=8.5, facecolor=PANEL, edgecolor="#2a2d3e",
              labelcolor=TEXT, loc="upper right")

    fig.tight_layout()
    plt.show()


#-------main-------|
def main():
    R, S, W, species = load_data(input, gt_prefix, pred_prefix)
    df, oa         = compute_accuracy(R, S, W, species)
    plot_accuracy(df, oa)


if __name__ == "__main__":
    main()