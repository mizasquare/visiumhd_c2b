#!/usr/bin/env python3
"""
Build cell-by-gene matrices from polygonal nuclei masks (GeoJSON) and Visium counts.

Modes
-----
1) Full image (default): save directly under --outdir-base
2) ROI mode (one or more ImageJ .roi files): polygons AND spots are restricted to
   the union of ROIs; results saved under --outdir-base/roi_<tag>/

Examples
--------
# ROI mode
python build_cellxgene_from_polygons.py \
  --tiff 28-A1.tif \
  --geojson stardist_output_GH_LHb/nuclei_masks_dilated_v.geojson \
  --tenx-h5 filtered_feature_bc_matrix_GH.h5 \
  --positions tissue_positions_GH.parquet \
  --outdir-base cellbin_out_GH_LHb \
  --roi 28-A1.tif.LHbl.roi 28-A1.tif.LHbr.roi

# Full image
python build_cellxgene_from_polygons.py \
  --tiff 28-A1.tif \
  --geojson stardist_output_GH_LHb/nuclei_masks_dilated_v.geojson \
  --tenx-h5 filtered_feature_bc_matrix_GH.h5 \
  --positions tissue_positions_GH.parquet \
  --outdir-base cellbin_out_GH_LHb
"""

import os, re, json, gzip, math, struct, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tifffile as tiff

from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree

# Optional AnnData
try:
    import anndata as ad
except Exception:
    ad = None

# ---------------------------
# Defaults (only used if flags omitted)
# ---------------------------
TIFF_PATH   = "28-1.tif"
GEOJSON_IN  = "stardist_output_SH/nuclei_masks_dilated_v.geojson"
TENX_H5     = "filtered_feature_bc_matrix_SH.h5"
TENX_DIR    = None
POS_PARQUET = "tissue_positions_SH.parquet"
OUTDIR_BASE = "cellbin_out_SH"

ASSIGN_BATCH = 200_000
AGGR_BATCH   = 10_000
USE_COVERS   = True  # Point-in-polygon predicate: covers (robust) vs contains

# ---------------------------
# Utilities
# ---------------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def derive_roi_tag(roi_paths):
    """
    Try to derive a compact tag (e.g., 'LHb') from filenames:
      <img>.LHbl.roi + <img>.LHbr.roi -> 'LHb'
    Fallback: common prefix or first stem.
    """
    stems = []
    for p in roi_paths or []:
        b = os.path.basename(p)
        m = re.search(r'\.tif\.([^.]+)\.roi$', b, flags=re.IGNORECASE)
        stems.append(m.group(1) if m else Path(b).stem)
    if not stems:
        return "roi"
    if len(stems) == 1:
        return stems[0]
    prefix = os.path.commonprefix(stems)
    prefix = re.sub(r'[_\W]+$', '', prefix)
    return prefix if len(prefix) >= 3 else min(stems, key=len)

def read_polygons_from_geojson(geojson_path: str):
    """Return list[Polygon] from a GeoJSON FeatureCollection."""
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    polys = []
    for feat in gj["features"]:
        geom = shape(feat["geometry"])
        if isinstance(geom, Polygon):
            polys.append(geom)
        elif isinstance(geom, MultiPolygon):
            # Merge into a single polygon (exterior union)
            merged = unary_union(list(geom.geoms))
            # merged may be MultiPolygon again; keep pieces
            if isinstance(merged, Polygon):
                polys.append(merged)
            elif isinstance(merged, MultiPolygon):
                polys.extend(list(merged.geoms))
        else:
            # attempt a robust buffer(0) to polygonize if possible
            p = geom.buffer(0)
            if isinstance(p, Polygon):
                polys.append(p)
    return polys

def read_tiff_shape_hw(tiff_path: str):
    """Return (H, W) without loading entire image to memory."""
    try:
        with tiff.TiffFile(tiff_path) as tf:
            page = tf.pages[0]
            h, w = page.shape[:2]
            return int(h), int(w)
    except Exception:
        img = tiff.imread(tiff_path)
        return int(img.shape[0]), int(img.shape[1])

def _coerce_positions(tp: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame with columns: barcode, x, y  (pixel coords)
    Accepts common Visium field names and renames.
    """
    df = tp.copy()
    # Normalize barcode
    if "barcode" not in df.columns:
        for cand in ["barcode_id", "Barcode", "barcodes", "spot_id", "BC"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "barcode"})
                break
    if "barcode" not in df.columns:
        raise ValueError("Positions parquet must include a 'barcode' column")
    # Normalize x/y
    if not ({"x", "y"} <= set(df.columns)):
        # Visium commonly stores row/col in fullres pixels
        row_col_candidates = (
            ("pxl_col_in_fullres", "pxl_row_in_fullres"),  # x, y
            ("pxl_col", "pxl_row"),
            ("col", "row"),
        )
        for cx, cy in row_col_candidates:
            if cx in df.columns and cy in df.columns:
                df = df.rename(columns={cx: "x", cy: "y"})
                break
    if not ({"x", "y"} <= set(df.columns)):
        raise ValueError("Positions parquet must have x/y pixel coordinates or pxl_col_in_fullres/pxl_row_in_fullres")
    # Keep only necessary cols
    return df[["barcode", "x", "y"]].copy()

def read_tenx_mtx(mtx_dir: str):
    from scipy.io import mmread
    mtx = Path(mtx_dir) / "matrix.mtx.gz"
    feats = Path(mtx_dir) / "features.tsv.gz"
    bcs   = Path(mtx_dir) / "barcodes.tsv.gz"
    X = mmread(gzip.open(mtx, "rb")).tocsr().transpose()  # spots x genes
    var = pd.read_csv(gzip.open(feats, "rt"), sep="\t", header=None)
    if var.shape[1] >= 2:
        var.columns = ["gene_id", "gene_name"] + [f"v{i}" for i in range(2, var.shape[1])]
    else:
        var.columns = ["gene_id"]
    obs = pd.read_csv(gzip.open(bcs, "rt"), sep="\t", header=None)
    obs.columns = ["barcode"]
    return X, obs, var

def read_tenx_h5(h5_path: str):
    import h5py
    with h5py.File(h5_path, "r") as f:
        M = f["matrix"]
        data   = M["data"][()]
        indices= M["indices"][()]
        indptr = M["indptr"][()]
        shape_ = M["shape"][()]
        X = sp.csc_matrix((data, indices, indptr), shape=shape_).transpose().tocsr()  # spots x genes
        var_names = [x.decode("utf-8") for x in M["features"]["name"][()]]
        gene_ids  = [x.decode("utf-8") for x in M["features"]["id"][()]]
        barcodes  = [x.decode("utf-8") for x in M["barcodes"][()]]
        var = pd.DataFrame({"gene_id": gene_ids, "gene_name": var_names})
        obs = pd.DataFrame({"barcode": barcodes})
    return X, obs, var

# ---------------------------
# ImageJ .roi (binary) → Polygon
# ---------------------------
COORDINATES     = 64
VERSION_OFFSET  = 4
TYPE_OFFSET     = 6
TOP_OFFSET      = 8
LEFT_OFFSET     = 10
N_COORDS_OFFSET = 16
OPTIONS_OFFSET  = 50
SUB_PIXEL_RESOLUTION = 128
TYPE_POLYGON=0; TYPE_FREEHAND=7; TYPE_TRACED=8

def _beshort(b, off):  import struct as _s; return _s.unpack_from(">h", b, off)[0]
def _beushort(b, off): import struct as _s; return _s.unpack_from(">H", b, off)[0]
def _beint(b, off):    import struct as _s; return _s.unpack_from(">i", b, off)[0]

def imagej_roi_to_polygon(path: str) -> Polygon:
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"Iout":
        raise ValueError("Not an ImageJ ROI")
    version = _beshort(data, VERSION_OFFSET)
    rtype   = data[TYPE_OFFSET]
    left    = _beshort(data, LEFT_OFFSET)
    top     = _beshort(data, TOP_OFFSET)
    n       = _beushort(data, N_COORDS_OFFSET) or _beint(data, 18)
    if n <= 0: raise ValueError("ROI has no vertices")
    if rtype not in (TYPE_POLYGON, TYPE_FREEHAND, TYPE_TRACED):
        raise ValueError(f"ROI type {rtype} is not an area polygon")
    base_xs = COORDINATES; base_ys = base_xs + 2*n
    xs_rel = np.frombuffer(data, dtype=">i2", count=n, offset=base_xs).astype(np.float64)
    ys_rel = np.frombuffer(data, dtype=">i2", count=n, offset=base_ys).astype(np.float64)
    options = _beshort(data, OPTIONS_OFFSET)
    if (options & SUB_PIXEL_RESOLUTION) and version >= 222:
        base_xf = COORDINATES + 4*n; base_yf = base_xf + 4*n
        xs = np.frombuffer(data, dtype=">f4", count=n, offset=base_xf).astype(np.float64)
        ys = np.frombuffer(data, dtype=">f4", count=n, offset=base_yf).astype(np.float64)
    else:
        xs = left + xs_rel; ys = top + ys_rel
    coords = np.column_stack([xs, ys])
    if not np.array_equal(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly

# ---------------------------
# Core
# ---------------------------
def run_aggregation(
    tiff_path: str,
    geojson_in: str,
    pos_parquet: str,
    outdir_base: str,
    tenx_h5: str | None = None,
    tenx_dir: str | None = None,
    roi_paths: list[str] | None = None,
    assign_batch: int = ASSIGN_BATCH,
    aggr_batch: int = AGGR_BATCH,
    use_covers: bool = USE_COVERS,
):
    roi_paths = list(roi_paths or [])
    # Decide output folder
    if roi_paths:
        tag = derive_roi_tag(roi_paths) or "roi"
        outdir = os.path.join(outdir_base, f"roi_{tag}")
    else:
        outdir = outdir_base
    _ensure_dir(outdir)

    print("• Reading histology size…")
    H, W = read_tiff_shape_hw(tiff_path)
    print(f"  histology H={H}, W={W}")

    print("• Reading nuclei polygons (GeoJSON)…")
    polygons = read_polygons_from_geojson(geojson_in)
    print(f"  polygons: {len(polygons):,}")

    print("• Reading Visium positions…")
    tp_raw = pd.read_parquet(pos_parquet)
    tp = _coerce_positions(tp_raw)
    print(f"  positions: {len(tp):,}")

    # Load counts
    print("• Reading Visium counts…")
    if tenx_h5 and Path(tenx_h5).exists():
        X_spots_genes, obs, var = read_tenx_h5(tenx_h5)
    elif tenx_dir:
        X_spots_genes, obs, var = read_tenx_mtx(tenx_dir)
    else:
        raise RuntimeError("Provide either --tenx-h5 or --tenx-dir")
    print(f"  matrix spots×genes = {X_spots_genes.shape[0]} × {X_spots_genes.shape[1]}")

    # Align matrix rows to positions (tp) order
    idx_map = pd.Series(np.arange(len(obs)), index=obs["barcode"])
    tp = tp.loc[tp["barcode"].isin(idx_map.index)].reset_index(drop=True)
    row_idx = idx_map.loc[tp["barcode"]].to_numpy()
    X_spots_genes = X_spots_genes[row_idx, :]
    print(f"  after align: {X_spots_genes.shape}")

    # ROI restriction (if any): filter BOTH polygons and spots
    if roi_paths:
        print("• Loading ROI(s) and restricting to their union…")
        for p in roi_paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"ROI file not found: {p}")
        roi_union = unary_union([imagej_roi_to_polygon(p) for p in roi_paths])
        roi_prep  = prep(roi_union)

        # Keep polygons whose representative point is inside ROI
        polys_in = []
        for poly in polygons:
            rp = poly.representative_point()
            if roi_prep.contains(rp):
                polys_in.append(poly)
        print(f"  polygons in ROI: {len(polys_in):,} / {len(polygons):,}")
        polygons = polys_in

        # Keep spots inside ROI
        mask_spots = tp.apply(lambda r: roi_prep.contains(Point(float(r["x"]), float(r["y"]))), axis=1)
        tp = tp.loc[mask_spots].reset_index(drop=True)
        X_spots_genes = X_spots_genes[mask_spots.values, :]
        print(f"  spots in ROI: {len(tp):,}")

    if len(polygons) == 0:
        raise RuntimeError("No polygons to aggregate (after ROI filtering).")

    # Assign spots → cells
    print("• Building spatial index (STRtree) and assigning spots to polygons…")
    tree = STRtree(polygons)
    geom_list = list(tree.geometries)
    geom2idx = {id(g): i for i, g in enumerate(geom_list)}
    # before assignment loop, right after building tree/geom_list/geom2idx:
    if use_covers and not hasattr(geom_list[0], "covers"):
        # emulate covers: point may lie in interior or boundary
        def _pred(poly, pt):
            return poly.contains(pt) or poly.touches(pt)
    else:
        def _pred(poly, pt):
            return poly.covers(pt) if use_covers else poly.contains(pt)

    spot2cell = np.full(len(tp), -1, dtype=np.int32)
    n_batches = math.ceil(len(tp) / assign_batch)

    for b in range(n_batches):
        s = b * assign_batch
        e = min((b + 1) * assign_batch, len(tp))
        sub = tp.iloc[s:e]
        pts = [Point(float(x), float(y)) for x, y in zip(sub["x"].to_numpy(), sub["y"].to_numpy())]

        if hasattr(tree, "query_bulk"):
            # Shapely 2.x fast path → (2, K) array of [point_idx, geom_idx]
            ij = tree.query_bulk(pts)
            # iterate by column to avoid fancy broadcasting surprises
            for k in range(ij.shape[1]):
                pi = int(ij[0, k])
                gi = int(ij[1, k])
                pt = pts[pi]
                poly = geom_list[gi]
                if _pred(poly, pt):
                    spot2cell[s + pi] = geom2idx[id(poly)]
        else:
            # Shapely 1.8 (or 2.x without query_bulk) → query(pt) may return indices OR geometry objects
            for j, pt in enumerate(pts):
                cands = tree.query(pt)
                # Normalize candidates to a list of polygon objects
                polys = []
                if cands is None:
                    polys = []
                elif hasattr(cands, "__len__") and len(cands) == 0:
                    polys = []
                else:
                    # numpy array of ints (indices) OR list of ints
                    if isinstance(cands, np.ndarray) and np.issubdtype(cands.dtype, np.integer):
                        polys = [geom_list[int(i)] for i in cands.tolist()]
                    elif isinstance(cands, (list, tuple)) and all(isinstance(i, (int, np.integer)) for i in cands):
                        polys = [geom_list[int(i)] for i in cands]
                    else:
                        # assume actual geometry objects
                        polys = list(cands)

                for poly in polys:
                    if _pred(poly, pt):
                        spot2cell[s + j] = geom2idx[id(poly)]
                        break

        if (b + 1) % 5 == 0 or b == n_batches - 1:
            assigned_so_far = int((spot2cell[:e] >= 0).sum())
            print(f"  batch {b + 1}/{n_batches} — assigned so far: {assigned_so_far:,}")

    assigned_mask = (spot2cell >= 0)
    assigned_total = int(assigned_mask.sum())
    print(f"  assigned {assigned_total:,} / {len(tp):,} spots ({assigned_total/len(tp)*100:.1f}%)")

    # Assignment yield diagnostics
    print("• Computing assignment yield diagnostics…")
    row_sums = np.asarray(X_spots_genes.sum(axis=1)).ravel()
    counts_total   = float(row_sums.sum())
    counts_assigned   = float(row_sums[assigned_mask].sum())
    counts_unassigned = counts_total - counts_assigned
    assigned_frac   = (counts_assigned / counts_total) if counts_total > 0 else np.nan
    _ensure_dir(outdir)
    with open(os.path.join(outdir, "assignment_yield.json"), "w") as f:
        json.dump({
            "n_spots_total": int(len(tp)),
            "n_spots_assigned": int(assigned_mask.sum()),
            "spots_assigned_frac": float(assigned_mask.mean()) if len(tp) else float("nan"),
            "counts_total": counts_total,
            "counts_assigned": counts_assigned,
            "counts_unassigned": counts_unassigned,
            "counts_assigned_frac": float(assigned_frac) if not np.isnan(assigned_frac) else None,
        }, f, indent=2)

    per_spot = tp[["barcode", "x", "y"]].copy()
    per_spot["spot_counts"] = row_sums
    per_spot["assigned"] = assigned_mask.astype(np.int8)
    per_spot.to_csv(os.path.join(outdir, "per_spot_counts_assigned.csv"), index=False)

    # Gene-wise assigned fractions (simple sanity report)
    assigned_rows   = np.where(assigned_mask)[0]
    unassigned_rows = np.where(~assigned_mask)[0]
    if len(assigned_rows):
        assigned_gene_counts   = np.asarray(X_spots_genes[assigned_rows, :].sum(axis=0)).ravel()
    else:
        assigned_gene_counts   = np.zeros(X_spots_genes.shape[1], dtype=np.float64)
    if len(unassigned_rows):
        unassigned_gene_counts = np.asarray(X_spots_genes[unassigned_rows, :].sum(axis=0)).ravel()
    else:
        unassigned_gene_counts = np.zeros(X_spots_genes.shape[1], dtype=np.float64)
    gene_total = assigned_gene_counts + unassigned_gene_counts
    gene_df = pd.DataFrame({
        "gene_id": var["gene_id"].values,
        "gene_name": (var["gene_name"].values if "gene_name" in var.columns else var["gene_id"].values),
        "assigned_counts": assigned_gene_counts,
        "unassigned_counts": unassigned_gene_counts,
        "total_counts": gene_total,
    })
    gene_df["assigned_frac"] = np.divide(
        gene_df["assigned_counts"], gene_df["total_counts"],
        out=np.zeros_like(gene_total, dtype=float), where=(gene_total > 0)
    )
    gene_df["unassigned_frac"] = 1.0 - gene_df["assigned_frac"]
    gene_df.to_csv(os.path.join(outdir, "gene_assigned_fractions.csv"), index=False)
    (gene_df.loc[gene_df["total_counts"] >= 100]
        .sort_values("unassigned_frac", ascending=False)
        .head(30)[["gene_name","total_counts","unassigned_frac"]]
        .to_csv(os.path.join(outdir, "gene_unassigned_top30.csv"), index=False))
    print("  wrote assignment_yield.json, per_spot_counts_assigned.csv, gene_assigned_fractions.csv, gene_unassigned_top30.csv")

    # Aggregate: cells × genes
    print("• Aggregating counts per cell…")
    n_cells = len(polygons)
    n_genes = X_spots_genes.shape[1]
    cell_X = sp.csr_matrix((n_cells, n_genes), dtype=X_spots_genes.dtype)

    valid_rows = np.where(spot2cell >= 0)[0]
    target_rows = spot2cell[valid_rows]

    for start in range(0, len(valid_rows), aggr_batch):
        rows = valid_rows[start:start + aggr_batch]
        sub = X_spots_genes[rows]
        data, indices, indptr = sub.data, sub.indices, sub.indptr
        repeats = np.diff(indptr)
        tr = np.repeat(target_rows[start:start + aggr_batch], repeats)
        coo = sp.coo_matrix((data, (tr, indices)), shape=(n_cells, n_genes))
        cell_X += coo.tocsr()

    print(f"  built cell matrix: {cell_X.shape}, nnz={cell_X.nnz:,}")

    # Per-cell metadata
    centroids = [p.centroid.coords[0] for p in polygons]
    areas     = [p.area for p in polygons]
    n_spots_per_cell = np.bincount(spot2cell[spot2cell>=0], minlength=n_cells)

    cells_df = pd.DataFrame({
        "cell_id"    : np.arange(n_cells, dtype=int),
        "centroid_x" : [c[0] for c in centroids],
        "centroid_y" : [c[1] for c in centroids],
        "area_px2"   : areas,
        "n_spots"    : n_spots_per_cell
    })

    # Save sidecars
    print("• Saving matrix and sidecars…")
    sp.save_npz(os.path.join(outdir, "cell_X.npz"), cell_X)
    cells_df.to_csv(os.path.join(outdir, "cells_meta.csv"), index=False)
    per_spot.assign(cell_id=spot2cell)[["barcode","x","y","cell_id"]].to_csv(
        os.path.join(outdir, "spot_to_cell.csv"), index=False)
    var.to_csv(os.path.join(outdir, "genes.tsv"), sep="\t", index=False)

    # Save AnnData (if available)
    if ad is not None:
        adata = ad.AnnData(cell_X)
        # var
        adata.var["gene_id"] = var["gene_id"].astype(str).values
        if "gene_name" in var.columns:
            adata.var["gene_symbol"] = var["gene_name"].astype(str).values
            adata.var_names = pd.Index(adata.var["gene_symbol"])
        else:
            adata.var_names = pd.Index(adata.var["gene_id"])
        adata.var.index.name = None
        adata.var_names_make_unique()
        # obs
        adata.obs = cells_df.set_index(pd.Index([f"cell_{i}" for i in cells_df["cell_id"]]))
        adata.obs["cell_id"] = cells_df["cell_id"].values
        adata.obs["centroid_x"] = cells_df["centroid_x"].values
        adata.obs["centroid_y"] = cells_df["centroid_y"].values
        # uns
        adata.uns["spot_to_cell_path"] = os.path.abspath(os.path.join(outdir, "spot_to_cell.csv"))
        adata.uns["histology_shape_hw"] = [int(H), int(W)]
        adata.uns["source_geojson"]     = os.path.abspath(geojson_in)
        if roi_paths:
            adata.uns["roi_paths"] = [os.path.abspath(p) for p in roi_paths]
        out_h5ad = os.path.join(outdir, "cellbins.h5ad")
        adata.write(out_h5ad)
        print(f"  wrote AnnData: {out_h5ad}")
    else:
        print("  anndata not installed; skipped .h5ad")

    print("Done.")

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate Visium counts into nuclei polygons; supports ROI-restricted mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--tiff',        dest='tiff_path',   default=TIFF_PATH)
    p.add_argument('--geojson',     dest='geojson_in',  default=GEOJSON_IN)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--tenx-h5',     dest='tenx_h5',     default=TENX_H5)
    g.add_argument('--tenx-dir',    dest='tenx_dir',    default=TENX_DIR)
    p.add_argument('--positions',   dest='pos_parquet', default=POS_PARQUET)
    p.add_argument('--outdir-base', dest='outdir_base', default=OUTDIR_BASE)
    p.add_argument('--roi',         dest='roi_paths', nargs='*', default=None,
                   help='One or more ImageJ .roi files (pixel coords). If provided, ROI mode is enabled.')
    p.add_argument('--assign-batch', type=int, default=ASSIGN_BATCH)
    p.add_argument('--aggr-batch',   type=int, default=AGGR_BATCH)
    p.add_argument('--use-covers',   action='store_true', default=USE_COVERS)
    p.add_argument('--no-use-covers', action='store_false', dest='use_covers')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_aggregation(
        tiff_path=args.tiff_path,
        geojson_in=args.geojson_in,
        pos_parquet=args.pos_parquet,
        outdir_base=args.outdir_base,
        tenx_h5=args.tenx_h5,
        tenx_dir=args.tenx_dir,
        roi_paths=args.roi_paths,
        assign_batch=args.assign_batch,
        aggr_batch=args.aggr_batch,
        use_covers=args.use_covers,
    )
