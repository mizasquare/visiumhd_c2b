import os
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad

# -----------------------------
# Config
# -----------------------------
H5AD_IN     = "cellbin_out_SH/roi_LHb/cellbins_clustered.h5ad"  # clustered file from previous step
TIFF_PATH   = "28-D1.tif"
OUT_DIR     = "cluster_exports_SH_LHb"
os.makedirs(OUT_DIR, exist_ok=True)
# H5AD_IN     = "cellbin_out_SH/cellbins_clustered.h5ad"  # clustered file from previous step
# TIFF_PATH   = "28-D1.tif"
# OUT_DIR     = "cluster_exports_SH"
# os.makedirs(OUT_DIR, exist_ok=True)

# Overlay look
POINT_SIZE_BG  = 0.12   # other clusters (gray)
POINT_SIZE_FG  = 0.8    # current cluster
ALPHA_BG       = 0.35
ALPHA_FG       = 0.9

# ROI thumbnails
MAKE_ROI       = True
ROI_PAD        = 200     # pixels margin around cluster’s bbox
ROI_MAX_SIZE   = 3000    # cap longer side to avoid huge PNGs

# Markers
N_TOP_MARKERS_TABLE = 100    # export this many per cluster to CSV
N_TOP_PLOT          = 10     # plot this many per cluster
DE_METHOD           = "wilcoxon"  # 'wilcoxon' is a solid default
MIN_COUNTS_FILTER   = None    # e.g. 50; set None to skip

# -----------------------------
# Load data
# -----------------------------
adata: ad.AnnData = sc.read_h5ad(H5AD_IN)
if "leiden" not in adata.obs.columns:
    raise RuntimeError("No 'leiden' in adata.obs. Run clustering first.")

# (optional) filter very low-count cells before DE
if MIN_COUNTS_FILTER:
    if "n_counts" not in adata.obs:
        adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    sc.pp.filter_cells(adata, min_counts=MIN_COUNTS_FILTER)

# Ensure centroids present
for col in ("centroid_x", "centroid_y"):
    if col not in adata.obs.columns:
        raise RuntimeError(f"Missing {col} in adata.obs")

# Read histology
img = tiff.imread(TIFF_PATH)
H, W = img.shape[:2]

# Colors per cluster
clusters = sorted(adata.obs["leiden"].astype(str).unique(), key=lambda s: int(s) if s.isdigit() else s)
cmap = plt.get_cmap("tab20")
cluster_color = {cl: cmap(i % 20) for i, cl in enumerate(clusters)}

# Build a DataFrame for plotting
df = adata.obs[["leiden", "centroid_x", "centroid_y"]].copy()
df["leiden"] = df["leiden"].astype(str)
# in-bounds
df = df[(df["centroid_x"]>=0)&(df["centroid_x"]<W)&(df["centroid_y"]>=0)&(df["centroid_y"]<H)]

# -----------------------------
# 1) Per-cluster overlays
# -----------------------------
print("Rendering per-cluster overlays…")
for cl in clusters:
    out_png = os.path.join(OUT_DIR, f"overlay_cluster_{cl}.png")

    # split fg/bg
    fg = df[df["leiden"] == cl]
    bg = df[df["leiden"] != cl]

    plt.figure(figsize=(10, 10), dpi=220)
    plt.imshow(img)
    # background cells (light gray)
    plt.scatter(bg["centroid_x"], bg["centroid_y"], s=POINT_SIZE_BG,
                c="lightgray", alpha=ALPHA_BG, rasterized=True)
    # current cluster
    plt.scatter(fg["centroid_x"], fg["centroid_y"], s=POINT_SIZE_FG,
                c=[cluster_color[cl]], alpha=ALPHA_FG, rasterized=True)
    plt.title(f"Cluster {cl} (n={len(fg)})", fontsize=10)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Optional ROI crop around this cluster
    if MAKE_ROI and len(fg) > 0:
        x0, x1 = int(fg["centroid_x"].min()) - ROI_PAD, int(fg["centroid_x"].max()) + ROI_PAD
        y0, y1 = int(fg["centroid_y"].min()) - ROI_PAD, int(fg["centroid_y"].max()) + ROI_PAD
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        crop = img[y0:y1, x0:x1]

        # rescale if too big
        ch, cw = crop.shape[:2]
        scale = 1.0
        if max(ch, cw) > ROI_MAX_SIZE:
            scale = ROI_MAX_SIZE / float(max(ch, cw))
            import skimage.transform as skt
            crop = skt.rescale(crop, scale, channel_axis=2, preserve_range=True, anti_aliasing=True).astype(img.dtype)

        # shift coords if scaled
        fx = (fg["centroid_x"] - x0) * scale
        fy = (fg["centroid_y"] - y0) * scale
        bx = (bg["centroid_x"] - x0) * scale
        by = (bg["centroid_y"] - y0) * scale

        out_roi = os.path.join(OUT_DIR, f"overlay_cluster_{cl}_roi.png")
        plt.figure(figsize=(8, 8), dpi=240)
        plt.imshow(crop)
        plt.scatter(bx, by, s=POINT_SIZE_BG/scale, c="lightgray", alpha=ALPHA_BG, rasterized=True)
        plt.scatter(fx, fy, s=POINT_SIZE_FG/scale, c=[cluster_color[cl]], alpha=ALPHA_FG, rasterized=True)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(out_roi, bbox_inches="tight", pad_inches=0)
        plt.close()

# -----------------------------
# 2) Marker discovery per cluster
# -----------------------------
print("Computing markers…")
# Work on a copy to avoid altering the on-disk file
adata_mark = adata.copy()

# Normalize/log if not already (idempotent if you did before)
if "log1p" not in adata_mark.uns_keys():
    sc.pp.normalize_total(adata_mark, target_sum=1e4)
    sc.pp.log1p(adata_mark)

sc.tl.rank_genes_groups(
    adata_mark, groupby="leiden",
    method=DE_METHOD, use_raw=False,
    n_genes=adata_mark.n_vars
)

# Export full tables (top N for convenience)
def rank_to_df(adata, n_top=N_TOP_MARKERS_TABLE):
    """Collect rank_genes_groups to tidy DataFrame per cluster."""
    rg = adata.uns["rank_genes_groups"]
    groups = rg["names"].dtype.names
    rows = []
    for g in groups:
        names = rg["names"][g][:n_top]
        pvals = rg["pvals_adj"][g][:n_top] if "pvals_adj" in rg else rg["pvals"][g][:n_top]
        scores = rg["scores"][g][:n_top]
        lfc = rg["logfoldchanges"][g][:n_top] if "logfoldchanges" in rg else [np.nan]*len(names)
        for i, gene in enumerate(names):
            rows.append({"cluster": g, "gene": gene, "score": float(scores[i]),
                         "logFC": float(lfc[i]) if lfc[i] is not None else np.nan,
                         "p_adj": float(pvals[i])})
    return pd.DataFrame(rows)

markers_df = rank_to_df(adata_mark, n_top=N_TOP_MARKERS_TABLE)
markers_csv = os.path.join(OUT_DIR, "markers_topN.csv")
markers_df.to_csv(markers_csv, index=False)
print(f"Saved markers table → {markers_csv}")

# Quick plots: top-N per cluster (dotplot & heatmap)
sc.settings.figdir = OUT_DIR  # where Scanpy saves figures
try:
    sc.pl.rank_genes_groups(adata_mark, n_genes=N_TOP_PLOT, sharey=False, save="_ranked_markers.png")
except Exception:
    pass
try:
    sc.pl.rank_genes_groups_dotplot(adata_mark, n_genes=N_TOP_PLOT, standard_scale="var", save="_dotplot.png")
except Exception:
    pass
try:
    sc.pl.rank_genes_groups_heatmap(adata_mark, n_genes=N_TOP_PLOT, swap_axes=True, figsize=(6,8), save="_heatmap.png")
except Exception:
    pass

print("Done.")

# -----------------------------
# Heterogeneity heatmap (tile-wise Shannon entropy)
# -----------------------------
from scipy.ndimage import gaussian_filter

# Params
BIN_PX        = 64       # tile size in pixels (e.g., 32, 64, 96)
MIN_POINTS    = 30       # skip tiles with very few cells
SMOOTH_SIGMA  = 1.0      # Gaussian blur (in tile units). 0 = no smoothing
CMAP_HEAT     = "magma"
OUT_HET       = os.path.join(OUT_DIR, "heterogeneity_entropy.png")
OUT_HET_OVLY  = os.path.join(OUT_DIR, "overlay_heterogeneity.png")
ROI           = None     # e.g., (y0, y1, x0, x1) or leave as None

print("Building heterogeneity (entropy) map…")

# Prepare cluster index mapping (robust to non-digit labels)
cluster_list = clusters[:]  # already sorted above
cl2idx = {c: i for i, c in enumerate(cluster_list)}
ncl = len(cluster_list)

# Bin centroids to tiles
nb_y = int(np.ceil(H / BIN_PX))
nb_x = int(np.ceil(W / BIN_PX))
tile_counts = np.zeros((nb_y, nb_x, ncl), dtype=np.int32)

iy = np.clip((df["centroid_y"].values // BIN_PX).astype(int), 0, nb_y - 1)
ix = np.clip((df["centroid_x"].values // BIN_PX).astype(int), 0, nb_x - 1)
ic = np.array([cl2idx[c] for c in df["leiden"].values], dtype=int)  # <-- FIXED

# Accumulate counts per tile × cluster
np.add.at(tile_counts, (iy, ix, ic), 1)

# Compute Shannon entropy per tile; normalize to [0,1]
tot = tile_counts.sum(axis=2)
with np.errstate(divide="ignore", invalid="ignore"):
    p = tile_counts / np.where(tot[..., None] > 0, tot[..., None], 1)
    ent = -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=2)  # H in bits

# Normalize by max possible entropy for the observed #clusters in each tile
k_obs = (tile_counts > 0).sum(axis=2)
max_ent = np.where(k_obs > 0, np.log2(k_obs), 1.0)
het = np.where(tot >= MIN_POINTS, ent / max_ent, np.nan)  # [0,1] with NaNs for sparse tiles

# Optional smoothing in tile space (NaN-aware)
if SMOOTH_SIGMA and SMOOTH_SIGMA > 0:
    m = ~np.isnan(het)
    het_filled = np.where(m, het, 0.0)
    m_s = gaussian_filter(m.astype(float), SMOOTH_SIGMA, mode="nearest")
    h_s = gaussian_filter(het_filled, SMOOTH_SIGMA, mode="nearest")
    het = np.where(m_s > 1e-6, h_s / np.maximum(m_s, 1e-6), np.nan)

# Upsample to pixel grid
tile_img = np.kron(het, np.ones((BIN_PX, BIN_PX), dtype=float))
tile_img = tile_img[:H, :W]

# Save entropy heatmap alone
plt.figure(figsize=(10, 10), dpi=200)
plt.imshow(tile_img, cmap=CMAP_HEAT, vmin=0, vmax=1)
plt.axis("off")
cbar = plt.colorbar(fraction=0.03, pad=0.01)
cbar.set_label("Local cluster heterogeneity (normalized entropy)")
plt.tight_layout(pad=0)
plt.savefig(OUT_HET, bbox_inches="tight", pad_inches=0)
plt.close()
print(f"Saved heterogeneity heatmap → {OUT_HET}")

# Save overlay on histology
plt.figure(figsize=(10, 10), dpi=200)
plt.imshow(img)
plt.imshow(tile_img, cmap=CMAP_HEAT, vmin=0, vmax=1, alpha=0.55)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(OUT_HET_OVLY, bbox_inches="tight", pad_inches=0)
plt.close()
print(f"Saved heterogeneity overlay → {OUT_HET_OVLY}")

# Optional: ROI version
if ROI is not None:
    OUT_HET_ROI = os.path.join(OUT_DIR, "heterogeneity_entropy_roi.png")
    OUT_HET_OVLY_ROI = os.path.join(OUT_DIR, "overlay_heterogeneity_roi.png")
    y0, y1, x0, x1 = ROI

    plt.figure(figsize=(8, 8), dpi=250)
    plt.imshow(tile_img[y0:y1, x0:x1], cmap=CMAP_HEAT, vmin=0, vmax=1)
    plt.axis("off")
    cbar = plt.colorbar(fraction=0.03, pad=0.01)
    cbar.set_label("Local heterogeneity (normalized)")
    plt.tight_layout(pad=0)
    plt.savefig(OUT_HET_ROI, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved heterogeneity ROI → {OUT_HET_ROI}")

    plt.figure(figsize=(8, 8), dpi=250)
    plt.imshow(img[y0:y1, x0:x1])
    plt.imshow(tile_img[y0:y1, x0:x1], cmap=CMAP_HEAT, vmin=0, vmax=1, alpha=0.6)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(OUT_HET_OVLY_ROI, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved overlay heterogeneity ROI → {OUT_HET_OVLY_ROI}")
