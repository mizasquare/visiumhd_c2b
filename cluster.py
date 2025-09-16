#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering + quick overlays.

Runner (pipeline_runner.py) patches these constants at runtime:
  - H5AD_IN, H5AD_OUT, TIFF_PATH, OUTDIR
You can also run standalone by editing them below.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

import scanpy as sc

# -----------------------------
# Config (will be patched by runner)
# -----------------------------
H5AD_IN   = "cellbin_out_SH/roi_LHb/cellbins_LHb.h5ad"
H5AD_OUT  = "cellbin_out_SH/roi_LHb/cellbins_LHb_clustered.h5ad"
TIFF_PATH = "28-D1.tif"
OUTDIR    = "cluster_outputs_placeholder"   # <- pipeline_runner replaces this

# Algorithm knobs (runner may patch these too)
RESOLUTION  = 1.0
N_PCS       = 50
MIN_COUNTS  = 50
N_NEIGHBORS     = 20
NEIGHBOR_METRIC = "euclidean"   # e.g., "euclidean", "cosine"
N_HVG           = 3000
UMAP_MIN_DIST   = 0.5
UMAP_SPREAD     = 1.0
LEIDEN_N_ITER   = 2
RANDOM_SEED     = 0

# Derived outputs
OUT_PNG      = os.path.join(OUTDIR, "overlay_leiden.png")
OUT_PNG_ROI  = os.path.join(OUTDIR, "overlay_leiden_roi.png")

# -----------------------------
# Helpers
# -----------------------------
def _ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _color_map_for_clusters(cat_series: pd.Series):
    """Return a reproducible color per cluster label."""
    cats = pd.Categorical(cat_series).categories.tolist()
    # Use matplotlib tab20 cycling deterministically
    import matplotlib as mpl
    base_cmap = mpl.cm.get_cmap("tab20", max(20, len(cats)))
    color_map = {c: base_cmap(i % base_cmap.N) for i, c in enumerate(cats)}
    return color_map

def _auto_roi_bounds(x, y, pad_frac=0.03):
    """Tight bounds around data with small padding."""
    x = np.asarray(x); y = np.asarray(y)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    dx, dy = xmax - xmin, ymax - ymin
    if dx == 0: dx = 1.0
    if dy == 0: dy = 1.0
    return (
        int(max(0, np.floor(xmin - pad_frac * dx))),
        int(np.ceil(xmax + pad_frac * dx)),
        int(max(0, np.floor(ymin - pad_frac * dy))),
        int(np.ceil(ymax + pad_frac * dy)),
    )

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    sc.set_figure_params(figsize=(4, 4), dpi=120)

    # ---------------- Load AnnData ----------------
    print("Loading AnnData…")
    adata = sc.read_h5ad(H5AD_IN)

    # Optional light filter on counts (keeps behavior with your logs)
    if MIN_COUNTS and "total_counts" in adata.obs.columns:
        before = adata.n_obs
        sc.pp.filter_cells(adata, min_counts=MIN_COUNTS)
        print(f"Filtered cells by min_counts={MIN_COUNTS}: {before} -> {adata.n_obs}")
    else:
        print(f"Skipping min_counts filter (MIN_COUNTS={MIN_COUNTS}, total_counts not present)")

    # ---------------- Scanpy pipeline ----------------
    print("Running Scanpy pipeline…")
    # If data are already counts (integers), harmony with neighbors; otherwise proceed (Scanpy warns on float)
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)

    # HVGs
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3")
    adata = adata[:, adata.var["highly_variable"]].copy()

    # Scale + PCA
    sc.pp.scale(adata, zero_center=True, max_value=10)
    sc.tl.pca(adata, n_comps=N_PCS, random_state=RANDOM_SEED)

    # Neighbors
    use_pcs = min(N_PCS, adata.obsm["X_pca"].shape[1])
    sc.pp.neighbors(
        adata,
        n_pcs=use_pcs,
        n_neighbors=N_NEIGHBORS,
        metric=NEIGHBOR_METRIC,
        method="umap"
    )

    # UMAP
    sc.tl.umap(adata, min_dist=UMAP_MIN_DIST, spread=UMAP_SPREAD, random_state=RANDOM_SEED)

    # Leiden
    sc.tl.leiden(
        adata,
        resolution=RESOLUTION,
        key_added="leiden",
        n_iterations=LEIDEN_N_ITER
    )

    # Count table (nice to have in logs)
    cl_counts = adata.obs["leiden"].value_counts().sort_index()
    print("Clusters: leiden")
    print(cl_counts)

    # Save clustered data
    outdir_parent = Path(OUTDIR)
    outdir_parent.mkdir(parents=True, exist_ok=True)
    adata.write(H5AD_OUT)
    print(f"Saved clustered AnnData → {H5AD_OUT}")

    # ---------------- Overlay (full frame) ----------------
    print("Loading histology TIFF (for overlay)…")
    img = tiff.imread(TIFF_PATH)
    H, W = img.shape[:2]
    print(f"TIFF shape: H={H}, W={W}")

    # Ensure we have centroids
    for col in ("centroid_x", "centroid_y"):
        if col not in adata.obs.columns:
            raise RuntimeError(f"Missing {col} in adata.obs. Upstream aggregator must export centroids.")

    df = adata.obs[["centroid_x", "centroid_y", "leiden"]].copy()
    df["leiden"] = df["leiden"].astype(str)
    color_map = _color_map_for_clusters(df["leiden"])

    plt.figure(figsize=(H/4000 + 4, W/4000 + 4))  # small proportional tweak
    plt.imshow(img)
    plt.scatter(df["centroid_x"].values, df["centroid_y"].values,
                s=0.6, c=[color_map[c] for c in df["leiden"].values],
                alpha=0.9, rasterized=True)
    plt.axis("off")

    # Build legend once (compact)
    import matplotlib.patches as mpatches
    legend_elems = [mpatches.Patch(color=color_map[c], label=str(c)) for c in sorted(color_map.keys(), key=lambda x: int(x) if x.isdigit() else x)]
    plt.legend(handles=legend_elems, title="Leiden", loc="lower right", frameon=True, fontsize=6)

    plt.tight_layout(pad=0)
    _ensure_parent(OUT_PNG)
    plt.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved overlay → {OUT_PNG}")

    # ---------------- ROI-cropped overlay (auto by centroids) ----------------
    x0, x1, y0, y1 = _auto_roi_bounds(df["centroid_x"], df["centroid_y"], pad_frac=0.03)
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)

    plt.figure(figsize=((y1-y0)/2000 + 3, (x1-x0)/2000 + 3))
    plt.imshow(img[y0:y1, x0:x1])
    plt.scatter(df["centroid_x"].values - x0, df["centroid_y"].values - y0,
                s=0.8, c=[color_map[c] for c in df["leiden"].values],
                alpha=0.9, rasterized=True)
    plt.axis("off")
    plt.tight_layout(pad=0)
    _ensure_parent(OUT_PNG_ROI)
    plt.savefig(OUT_PNG_ROI, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved ROI overlay → {OUT_PNG_ROI}")
