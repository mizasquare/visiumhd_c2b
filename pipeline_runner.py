#!/usr/bin/env python3
"""
End-to-end runner for your spatial pipeline with QC stage:
    1) stardister -> nuclei polygons (GeoJSON) + preview
    2) voronoi_dilation -> dilated polygons (GeoJSON) + overlay
    3) aggregation -> cells×genes (h5ad) [either full image or ROI-limited]
    3.5) QC (robust Scanpy-style) -> <input>_qcfiltered.h5ad
    4) clustering (Scanpy) on QC-filtered h5ad
    5) cluster overlays, markers, heterogeneity maps
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ------------------------------
# Utilities
# ------------------------------

def run(cmd, cwd=None):
    print("\n$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd)
    ret = proc.wait()
    if ret != 0:
        raise SystemExit(f"Command failed with exit code {ret}: {' '.join(str(c) for c in cmd)}")

def as_py_string(p):
    return repr(str(p))

def patch_constants(src_path: Path, dst_path: Path, repl: dict):
    """Make a patched copy of a Python file, replacing module-level constants.
    repl: { 'NAME': 'value_literal', ... } (values inserted literally; quote strings yourself)
    """
    import re as _re
    text = src_path.read_text(encoding="utf-8")

    def sub_one(txt, name, lit):
        pattern = rf"^(\s*){_re.escape(name)}\s*=.*$"
        replacement = rf"\1{name} = {lit}"
        new, n = _re.subn(pattern, replacement, txt, flags=_re.MULTILINE)
        if n == 0:
            header_pat = _re.compile(r"(# +[-]+\n# +CONFIG.*?\n# +[-]+)", _re.DOTALL)
            m = header_pat.search(txt)
            ins = f"\n{name} = {lit}\n"
            if m:
                idx = m.end()
                new = txt[:idx] + ins + txt[idx:]
            else:
                new = ins + txt
        return new

    for k, v in repl.items():
        text = sub_one(text, k, v)

    dst_path.write_text(text, encoding="utf-8")

def prefer_roi_h5ad(base_dir: Path) -> Path:
    """Search recursively for *.h5ad under base_dir; prefer ROI export if present."""
    cands = sorted(base_dir.rglob("*.h5ad"))
    if not cands:
        return None
    roi_pref = [
        p for p in cands
        if ("roi" in p.stem.lower()) or ("roi" in p.parent.name.lower())
    ]
    return roi_pref[0] if roi_pref else cands[0]


# ------------------------------
# Orchestration
# ------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--tiff', required=True, help='Full-res histology TIFF')
    g_counts = ap.add_mutually_exclusive_group(required=True)
    g_counts.add_argument('--tenx-h5', help='10x HDF5 path (filtered_feature_bc_matrix_*.h5)')
    g_counts.add_argument('--tenx-dir', help='10x MTX dir containing matrix.mtx.gz / features.tsv.gz / barcodes.tsv.gz')
    ap.add_argument('--positions', required=True, help='tissue_positions.parquet file')
    ap.add_argument('--roi', nargs='*', default=None, help='ImageJ .roi paths for ROI-limited aggregation (0+ list). If omitted, full-image mode is used.')
    ap.add_argument('--outbase', required=True, help='Short tag for outputs (e.g., SH, GH, SH_LHb)')

    # ---- StarDist knobs (forwarded to stardister.py) ----
    ap.add_argument('--sd-prob', type=float, default=None, help='StarDist probability threshold (forwarded to stardister.py --prob)')
    ap.add_argument('--sd-nms', type=float, default=None, help='StarDist NMS IoU threshold (forwarded to stardister.py --nms)')

    # ---- Voronoi dilation knobs (forwarded to voronoi_dilation.py) ----
    ap.add_argument('--voro-dilation-um', type=float, default=None,
                    help='Voronoi-capped dilation radius in micrometers (forwards to --dilation)')
    ap.add_argument('--voro-px-um', type=float, default=None,
                    help='Pixel size in micrometers (forwards to --px). If omitted, voronoi script default applies.')

    # Optional clustering overrides
    ap.add_argument('--res', type=float, default=None, help='Leiden resolution (cluster.py)')
    ap.add_argument('--n-pcs', type=int, default=None, help='Number of PCs (cluster.py)')
    ap.add_argument('--min-counts', type=int, default=None, help='Min counts per cell filter (cluster.py)')
    ap.add_argument('--cl-n-neighbors', type=int, default=None, help='Neighbors: number of neighbors (cluster.py)')
    ap.add_argument('--cl-metric', default=None, help='Neighbors metric (e.g. euclidean, cosine)')
    ap.add_argument('--cl-hvg', type=int, default=None, help='Highly variable genes: n_top_genes (cluster.py)')
    ap.add_argument('--cl-umap-min-dist', type=float, default=None, help='UMAP min_dist (cluster.py)')
    ap.add_argument('--cl-umap-spread', type=float, default=None, help='UMAP spread (cluster.py)')
    ap.add_argument('--cl-leiden-iters', type=int, default=None, help='Leiden n_iterations (cluster.py)')
    ap.add_argument('--cl-seed', type=int, default=None, help='Random seed for reproducibility (cluster.py)')
    # ---------------------- NEW: QC options ----------------------
    ap.add_argument('--qc-species', choices=['mouse','human'], default='mouse', help='Species for mito/ribo prefixes (QC stage)')
    ap.add_argument('--qc-min-genes', type=int, default=200, help='Absolute floor for detected genes (pre-MAD)')
    ap.add_argument('--qc-min-umis',  type=int, default=500, help='Absolute floor for UMIs (pre-MAD)')
    ap.add_argument('--qc-max-mito',  type=float, default=30.0, help='Absolute cap for mito%% (0 to disable; MAD-only)')
    ap.add_argument('--qc-mad-mult',  type=float, default=3.0, help='MAD multiplier (robust lower nUMI/nGene; upper mito%%)')
    ap.add_argument('--qc-max-top5-frac', type=float, default=None, help='Optional cap on top-5 gene dominance per cell (e.g., 0.6)')
    ap.add_argument('--qc-run-scrublet', action='store_true', help='Run Scrublet if available')
    ap.add_argument('--qc-outdir-name', default='qc_out', help='Name of QC subfolder next to aggregated h5ad')
    # -------------------------------------------------------------

    args = ap.parse_args()

    root = Path.cwd()
    work = root / '.run_work'
    work.mkdir(exist_ok=True)

    # Resolve inputs
    tiff_path = Path(args.tiff).resolve()
    pos_path  = Path(args.positions).resolve()
    tenx_h5   = Path(args.tenx_h5).resolve() if args.tenx_h5 else None
    tenx_dir  = Path(args.tenx_dir).resolve() if args.tenx_dir else None
    roi_list  = [Path(p).resolve() for p in (args.roi or [])]

    out_stardist_dir = root / f'stardist_output_{args.outbase}'
    out_stardist_dir.mkdir(exist_ok=True)

    out_voro_dir = root / f'voronoi_output_{args.outbase}'
    out_voro_dir.mkdir(exist_ok=True)

    out_cellbin  = root / f'cellbin_out_{args.outbase}'
    out_cellbin.mkdir(parents=True, exist_ok=True)

    # Stage 0: discover scripts
    s_stardister = root / 'stardister.py'
    s_voronoi    = root / 'voronoi_dilation.py'
    s_cell_agr   = root / 'cell_agr.py'                 # legacy full-image aggregator (constant-patched)
    s_build_cellxgene_from_polygons = root / 'build_cellxgene_from_polygons.py'           # CLI ROI/full-image aggregator (preferred)  [:contentReference[oaicite:3]{index=3}]
    s_build_roi  = root / 'build_cellxgene_from_polygons.py'  # alternate ROI aggregator if present
    s_qc         = root / 'scRNA_qc_pipeline.py'        # QC stage script                         [:contentReference[oaicite:4]{index=4}]
    s_cluster    = root / 'cluster.py'                  # your clustering script (unchanged)      [:contentReference[oaicite:5]{index=5}]
    s_overlay    = root / 'cluster_overlay.py'          # overlays + markers                      [:contentReference[oaicite:6]{index=6}]

    for p in [s_stardister, s_voronoi, s_cell_agr, s_build_cellxgene_from_polygons, s_cluster, s_overlay, s_qc]:
        if not p.exists():
            raise SystemExit(f"Missing required script: {p}")

    # ---------------------------------
    # 1) Stardist segmentation -> polygons
    # ---------------------------------
    cmd_stardist = [
        sys.executable,
        str(s_stardister),
        "-i", str(tiff_path),
        "-o", str(out_stardist_dir),
        "--preview_max_side", "8000",
    ]
    if args.sd_prob is not None:
        cmd_stardist += ["--prob", str(args.sd_prob)]
    if args.sd_nms is not None:
        cmd_stardist += ["--nms", str(args.sd_nms)]
    run(cmd_stardist)


    geojson_native = out_stardist_dir / 'nuclei_masks_native.geojson'
    if not geojson_native.exists():
        cands = list(out_stardist_dir.rglob('*.geojson'))
        if not cands:
            raise SystemExit("Stardist stage completed but no GeoJSON was found in the output directory.")
        geojson_native = cands[0]
        print(f"[warn] Using discovered GeoJSON: {geojson_native}")

    # ---------------------------------
    # 2) Voronoi-capped dilation
    # ---------------------------------
    geojson_dilated = out_voro_dir / "nuclei_masks_dilated_v.geojson"

    cmd_voro = [
        sys.executable, str(s_voronoi),
        "--in", str(geojson_native),
        "--out", str(geojson_dilated),
        "--tiff", str(tiff_path),
        "--overlay", str(out_voro_dir / "overlay_before_after.png"),
    ]
    if args.voro_dilation_um is not None:
        cmd_voro += ["--dilation", str(args.voro_dilation_um)]
    if args.voro_px_um is not None:
        cmd_voro += ["--px", str(args.voro_px_um)]
    run(cmd_voro)

    # ---------------------------------
    # 3) Aggregation (cells × genes)
    # ---------------------------------
    agr_work = work / 'cell_agr'
    agr_work.mkdir(parents=True, exist_ok=True)

    h5ad_aggregated = None

    if roi_list:
        # Prefer CLI-based ROI aggregator if available (build_cellxgene_... or cell_agr_roi.py)
        agg_script = s_build_roi if s_build_roi.exists() else s_build_cellxgene_from_polygons
        print(f"[info] Aggregator used (CLI): {agg_script.name}")
        cmd = [
            sys.executable, str(agg_script),
            "--tiff", str(tiff_path),
            "--geojson", str(geojson_dilated),
            ("--tenx-h5" if tenx_h5 else "--tenx-dir"), str(tenx_h5 if tenx_h5 else tenx_dir),
            "--positions", str(pos_path),
            "--outdir-base", str(out_cellbin),
        ]
        if roi_list:
            cmd += ["--roi", *[str(p) for p in roi_list]]
        run(cmd)

        # Prefer ROI h5ad under out_cellbin
        h5ad_aggregated = prefer_roi_h5ad(out_cellbin)
        if h5ad_aggregated is None:
            raise SystemExit('Aggregation (ROI) produced no .h5ad file.')
    else:
        # Full-image mode with legacy aggregator (constant-patched)
        agr_src = s_cell_agr
        agr_dst = agr_work / 'cell_agr_run.py'
        out_cellbin.mkdir(parents=True, exist_ok=True)
        repl = {
            'TIFF_PATH'   : as_py_string(tiff_path),
            'GEOJSON_IN'  : as_py_string(geojson_dilated),
            'TENX_H5'     : as_py_string(tenx_h5) if tenx_h5 else 'None',
            'TENX_DIR'    : as_py_string(tenx_dir) if tenx_dir else 'None',
            'POS_PARQUET' : as_py_string(pos_path),
            'OUTDIR'      : as_py_string(out_cellbin),
        }
        patch_constants(agr_src, agr_dst, repl)
        run([sys.executable, str(agr_dst)])

        h5ad_aggregated = prefer_roi_h5ad(out_cellbin)
        if h5ad_aggregated is None:
            raise SystemExit('Aggregation produced no .h5ad file.')

    print(f"Aggregation h5ad → {h5ad_aggregated}")

    # ---------------------------------
    # 3.5) QC (robust Scanpy-style)  <-- NEW
    # ---------------------------------
    qc_outdir = h5ad_aggregated.parent / args.qc_outdir_name
    qc_outdir.mkdir(exist_ok=True)

    qc_cmd = [
        sys.executable, str(s_qc),
        "--in", str(h5ad_aggregated),
        "--species", args.qc_species,
        "--outdir", str(qc_outdir),
        "--mad-mult", str(args.qc_mad_mult),
        "--min-genes", str(args.qc_min_genes),
        "--min-umis", str(args.qc_min_umis),
        "--max-mito", str(args.qc_max_mito),
    ]
    if args.qc_max_top5_frac is not None:
        qc_cmd += ["--max-top5-frac", str(args.qc_max_top5_frac)]
    if args.qc_run_scrublet:
        qc_cmd += ["--run-scrublet"]

    run(qc_cmd)

    # Resolve the QC-filtered h5ad (<base>_qcfiltered.h5ad)
    qc_filtered = None
    for cand in qc_outdir.glob("*_qcfiltered.h5ad"):
        qc_filtered = cand
        break
    if qc_filtered is None:
        # best effort fallback: if only one h5ad is present use it
        h5ads = list(qc_outdir.glob("*.h5ad"))
        if len(h5ads) == 1:
            qc_filtered = h5ads[0]
    if qc_filtered is None:
        raise SystemExit("QC stage ran but no *_qcfiltered.h5ad was found.")

    print(f"QC-filtered h5ad → {qc_filtered}")

    # ---------------------------------
    # 4) Clustering (on QC-filtered h5ad)
    # ---------------------------------
    cl_work = work / 'cluster'
    cl_work.mkdir(parents=True, exist_ok=True)

    cl_src = s_cluster
    cl_dst = cl_work / 'cluster_run.py'

    # Put clustered .h5ad next to the QC-filtered input
    cluster_outdir = qc_filtered.parent
    os.makedirs(cluster_outdir, exist_ok=True)
    h5ad_out = qc_filtered.with_name(qc_filtered.stem.replace("_qcfiltered", "") + '_clustered.h5ad')

    repl = {
        'H5AD_IN'  : as_py_string(qc_filtered),
        'H5AD_OUT' : as_py_string(h5ad_out),
        'TIFF_PATH': as_py_string(tiff_path),
        'OUTDIR'   : as_py_string(cluster_outdir),
    }
    if args.res is not None:
        repl['RESOLUTION'] = str(args.res)
    if args.n_pcs is not None:
        repl['N_PCS'] = str(args.n_pcs)
    if args.min_counts is not None:
        repl['MIN_COUNTS'] = str(args.min_counts)
        if args.res is not None:
            repl['RESOLUTION'] = str(args.res)
        if args.n_pcs is not None:
            repl['N_PCS'] = str(args.n_pcs)
        if args.min_counts is not None:
            repl['MIN_COUNTS'] = str(args.min_counts)
        if args.cl_n_neighbors is not None:
            repl['N_NEIGHBORS'] = str(args.cl_n_neighbors)
        if args.cl_metric is not None:
           repl['NEIGHBOR_METRIC'] = as_py_string(args.cl_metric)
        if args.cl_hvg is not None:
            repl['N_HVG'] = str(args.cl_hvg)
        if args.cl_umap_min_dist is not None:
            repl['UMAP_MIN_DIST'] = str(args.cl_umap_min_dist)
        if args.cl_umap_spread is not None:
            repl['UMAP_SPREAD'] = str(args.cl_umap_spread)
        if args.cl_leiden_iters is not None:
            repl['LEIDEN_N_ITER'] = str(args.cl_leiden_iters)
        if args.cl_seed is not None:
            repl['RANDOM_SEED'] = str(args.cl_seed)

    patch_constants(cl_src, cl_dst, repl)
    os.makedirs(cluster_outdir, exist_ok=True)
    run([sys.executable, str(cl_dst)])

    # ---------------------------------
    # 5) Cluster overlays + markers + heterogeneity
    # ---------------------------------
    ov_work = work / 'cluster_overlay'
    ov_work.mkdir(parents=True, exist_ok=True)

    ov_src = s_overlay
    ov_dst = ov_work / 'cluster_overlay_run.py'

    out_dir = h5ad_out.parent / 'cluster_exports'
    out_dir.mkdir(exist_ok=True)

    repl = {
        'H5AD_IN'  : as_py_string(h5ad_out),
        'TIFF_PATH': as_py_string(tiff_path),
        'OUT_DIR'  : as_py_string(out_dir),
    }
    patch_constants(ov_src, ov_dst, repl)
    run([sys.executable, str(ov_dst)])

    print("\nAll stages completed successfully. Outputs:")
    print(f"  Stardist:        {out_stardist_dir}")
    print(f"  Dilated:         {geojson_dilated}")
    print(f"  Aggregation:     {out_cellbin}")
    print(f"  QC:              {qc_outdir}")
    print(f"  Clustered h5ad:  {h5ad_out}")
    print(f"  Overlays/markers:{out_dir}")


if __name__ == '__main__':
    main()


#example paramet:
#--tiff 28-D1.tif --tenx-h5 filtered_feature_bc_matrix_SH.h5 --positions tissue_positions_SH.parquet --roi 28-D1.tif.LHbl.roi 28-D1.tif.LHbr.roi --outbase SH_LHb --qc-species mouse --qc-min-genes 200 --qc-min-umis 500 --qc-max-mito 15  --qc-mad-mult 3 --qc-run-scrublet
#--tiff 28-A1.tif --tenx-h5 filtered_feature_bc_matrix_GH.h5 --positions tissue_positions_GH.parquet --roi 28-A1.tif.LHbl.roi 28-A1.tif.LHbr.roi --outbase GH_LHb --qc-species mouse --qc-min-genes 200 --qc-min-umis 500 --qc-max-mito 15  --qc-mad-mult 3 --qc-run-scrublet