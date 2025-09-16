#!/usr/bin/env python3
import argparse, json, os, sys, subprocess
from pathlib import Path

# ------------------------------
# Small utilities
# ------------------------------
def echo(msg): print(msg, flush=True)

def run(cmd, dry=False, cwd=None):
    echo("\n$ " + " ".join(str(c) for c in cmd))
    if dry: return 0
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd)
    rc = p.wait()
    if rc != 0:
        raise SystemExit(f"[error] command failed ({rc})")
    return rc

def fail(msg): raise SystemExit("[fatal] " + msg)

def exists(p): return Path(p).exists()

def find_first(base: Path, patterns):
    for pat in patterns:
        hits = sorted(base.rglob(pat))
        if hits: return hits[0]
    return None

# ------------------------------
# Lightweight integrity checks
# ------------------------------
def check_tiff(tiff_path: Path):
    if not tiff_path.exists(): fail(f"TIFF not found: {tiff_path}")
    if tiff_path.stat().st_size < 1024: fail(f"TIFF too small: {tiff_path}")

def check_parquet(parquet_path: Path):
    if not parquet_path.exists(): fail(f"Positions parquet not found: {parquet_path}")
    if parquet_path.stat().st_size < 128: fail(f"Parquet suspiciously small: {parquet_path}")

def check_10x(tenx_h5: Path|None, tenx_dir: Path|None):
    if tenx_h5:
        if not tenx_h5.exists(): fail(f"10x H5 not found: {tenx_h5}")
        if tenx_h5.stat().st_size < 1024: fail(f"10x H5 too small: {tenx_h5}")
        return
    if tenx_dir:
        mtx = tenx_dir / "matrix.mtx.gz"
        feats = tenx_dir / "features.tsv.gz"
        bar = tenx_dir / "barcodes.tsv.gz"
        for p in [mtx, feats, bar]:
            if not p.exists(): fail(f"10x MTX file missing: {p}")
            if p.stat().st_size < 64: fail(f"10x MTX file too small: {p}")
        return
    fail("Provide either --tenx-h5 or --tenx-dir")

def check_geojson_polygons(geojson_path: Path, want_dilated_hint=False):
    if not geojson_path.exists(): fail(f"GeoJSON not found: {geojson_path}")
    if geojson_path.stat().st_size < 128: fail(f"GeoJSON too small: {geojson_path}")
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        feats = data.get("features", [])
        if not feats: fail(f"No features in GeoJSON: {geojson_path}")
        g = feats[0].get("geometry", {})
        if g.get("type") not in ("Polygon", "MultiPolygon"):
            echo(f"[warn] First feature geometry type {g.get('type')} (expected Polygon/MultiPolygon).")
        if want_dilated_hint and "dilated" not in geojson_path.name.lower():
            echo(f"[hint] This GeoJSON name lacks 'dilated' — ensure you passed the Voronoi output, not the native polygons.")
    except Exception as e:
        fail(f"GeoJSON parse failed: {geojson_path} ({e})")

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Lean runner for pipeline (supports dry-run & reuse of StarDist/Voronoi outputs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument('--tiff', required=True, help='Full-res histology TIFF')
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--tenx-h5', help='10x HDF5 path')
    grp.add_argument('--tenx-dir', help='10x MTX directory (matrix.mtx.gz/features.tsv.gz/barcodes.tsv.gz)')
    ap.add_argument('--positions', required=True, help='tissue_positions.parquet')
    ap.add_argument('--roi', nargs='*', default=None, help='0+ ImageJ .roi files for ROI aggregation')
    ap.add_argument('--outbase', required=True, help='Output tag (e.g., SH, GH, SH_LHb)')

    # Reuse / skip heavy stages
    ap.add_argument('--reuse-stardist-dir', help='Use an existing StarDist output dir; skip StarDist')
    ap.add_argument('--stardist-geojson', help='Use this native polygons GeoJSON directly; skip StarDist')
    ap.add_argument('--reuse-voronoi-geojson', help='Use this dilated polygons GeoJSON; skip Voronoi')
    ap.add_argument('--force', action='store_true', help='Recompute even if outputs exist')

    # Dry-run
    ap.add_argument('--dry-run', action='store_true', help='Do not run scripts — only validate and print commands')

    # Paths to stage scripts (keep defaults consistent with your repo)
    ap.add_argument('--stardister', default='stardister.py')
    ap.add_argument('--voronoi', default='voronoi_dilation.py')
    ap.add_argument('--cell-agr-roi', default='build_cellxgene_from_polygons.py')
    ap.add_argument('--cell-agr-full', default='cell_agr.py')
    ap.add_argument('--qc', default='scRNA_qc_pipeline.py')
    ap.add_argument('--cluster', default='cluster.py')
    ap.add_argument('--overlay', default='cluster_overlay.py')

    # QC opts (passed through)
    ap.add_argument('--qc-species', choices=['mouse','human'], default='mouse')
    ap.add_argument('--qc-min-genes', type=int, default=200)
    ap.add_argument('--qc-min-umis',  type=int, default=500)
    ap.add_argument('--qc-max-mito',  type=float, default=20)
    ap.add_argument('--qc-mad-mult',  type=float, default=3.0)
    ap.add_argument('--qc-max-top5-frac', type=float, default=None)
    ap.add_argument('--qc-run-scrublet', action='store_true')
    ap.add_argument('--qc-outdir-name', default='qc_out',
                   help='Name of QC subfolder next to aggregated h5ad')

    # Clustering opts (passed through)
    ap.add_argument('--res', type=float, default=None)
    ap.add_argument('--n-pcs', type=int, default=None)
    ap.add_argument('--min-counts', type=int, default=None)

    args = ap.parse_args()

    # Resolve paths
    tiff_path = Path(args.tiff).resolve()
    pos_path  = Path(args.positions).resolve()
    tenx_h5   = Path(args.tenx_h5).resolve() if args.tenx_h5 else None
    tenx_dir  = Path(args.tenx_dir).resolve() if args.tenx_dir else None
    roi_list  = [Path(p).resolve() for p in (args.roi or [])]

    # Output dirs
    root = Path.cwd()
    out_stardist_dir = root / f"stardist_output_{args.outbase}"
    out_voro_dir     = root / f"voronoi_output_{args.outbase}"
    out_cellbin_dir  = root / f"cellbin_out_{args.outbase}"
    out_stardist_dir.mkdir(exist_ok=True)
    out_voro_dir.mkdir(exist_ok=True)
    out_cellbin_dir.mkdir(exist_ok=True)

    # Script paths
    s_stardister = Path(args.stardister).resolve()
    s_voronoi    = Path(args.voronoi).resolve()
    s_cell_roi   = Path(args.cell_agr_roi).resolve()
    s_cell_full  = Path(args.cell_agr_full).resolve()
    s_qc         = Path(args.qc).resolve()
    s_cluster    = Path(args.cluster).resolve()
    s_overlay    = Path(args.overlay).resolve()

    # ---------------- Integrity checks (fast) ----------------
    check_tiff(tiff_path)
    check_parquet(pos_path)
    check_10x(tenx_h5, tenx_dir)

    # ---------------- Decide StarDist input ----------------
    geojson_native = None
    if args.stardist_geojson:
        geojson_native = Path(args.stardist_geojson).resolve()
        check_geojson_polygons(geojson_native, want_dilated_hint=False)
        echo(f"[info] Using provided StarDist native GeoJSON: {geojson_native}")
    elif args.reuse_stardist_dir:
        reuse_dir = Path(args.reuse_stardist_dir).resolve()
        if not reuse_dir.exists(): fail(f"--reuse-stardist-dir not found: {reuse_dir}")
        cand = find_first(reuse_dir, ["nuclei_masks_native.geojson", "*native*.geojson", "*.geojson"])
        if not cand: fail(f"No GeoJSON found under {reuse_dir}")
        geojson_native = cand
        check_geojson_polygons(geojson_native, want_dilated_hint=False)
        echo(f"[info] Reusing StarDist native GeoJSON: {geojson_native}")
    else:
        default_native = out_stardist_dir / "nuclei_masks_native.geojson"
        if default_native.exists() and not args.force:
            geojson_native = default_native
            check_geojson_polygons(geojson_native, want_dilated_hint=False)
            echo(f"[info] Found existing StarDist output: {geojson_native}")
        else:
            echo("[plan] Run StarDist segmentation to produce native polygons.")
            run([sys.executable, str(s_stardister), "-i", str(tiff_path), "-o", str(out_stardist_dir), "--preview_max_side", "8000"], dry=args.dry_run)
            geojson_native = default_native

    # ---------------- Decide Voronoi output ----------------
    geojson_dil = None
    if args.reuse_voronoi_geojson:
        geojson_dil = Path(args.reuse_voronoi_geojson).resolve()
        check_geojson_polygons(geojson_dil, want_dilated_hint=True)
        echo(f"[info] Using provided Voronoi dilated GeoJSON: {geojson_dil}")
    else:
        default_dil = out_voro_dir / "nuclei_masks_dilated_v.geojson"
        if default_dil.exists() and not args.force:
            geojson_dil = default_dil
            check_geojson_polygons(geojson_dil, want_dilated_hint=True)
            echo(f"[info] Found existing dilated GeoJSON: {geojson_dil}")
        else:
            echo("[plan] Run Voronoi-capped dilation.")
            run([
                sys.executable, str(s_voronoi),
                "--in", str(geojson_native),
                "--out", str(default_dil),
                "--tiff", str(tiff_path),
                "--overlay", str(out_voro_dir / "overlay_before_after.png"),
            ], dry=args.dry_run)
            geojson_dil = default_dil

    if not args.dry_run:
        check_geojson_polygons(geojson_dil, want_dilated_hint=True)

    # ---------------- Aggregation ----------------
    echo("[plan] Aggregate counts to polygons (ROI or full).")
    if roi_list:
        cmd = [
            sys.executable, str(s_cell_roi),
            "--tiff", str(tiff_path),
            "--geojson", str(geojson_dil),
            ("--tenx-h5" if tenx_h5 else "--tenx-dir"), str(tenx_h5 if tenx_h5 else tenx_dir),
            "--positions", str(pos_path),
            "--outdir-base", str(out_cellbin_dir),
            "--roi", *[str(p) for p in roi_list]
        ]
        run(cmd, dry=args.dry_run)
    else:
        cmd = [
            sys.executable, str(s_cell_roi),
            "--tiff", str(tiff_path),
            "--geojson", str(geojson_dil),
            ("--tenx-h5" if tenx_h5 else "--tenx-dir"), str(tenx_h5 if tenx_h5 else tenx_dir),
            "--positions", str(pos_path),
            "--outdir-base", str(out_cellbin_dir)
        ]
        run(cmd, dry=args.dry_run)

    # discover produced h5ad
    h5ad_in = find_first(out_cellbin_dir, ["*roi*/*.h5ad", "*/*.h5ad", "*.h5ad"])
    if not h5ad_in:
        msg = f"No .h5ad found under {out_cellbin_dir}. (In dry-run, this is expected; in real run, it indicates an aggregation error.)"
        if args.dry_run: echo("[note] " + msg)
        else: fail(msg)
    else:
        echo(f"[info] Aggregated h5ad: {h5ad_in}")

    # ---------------- QC ----------------
    qc_out = out_cellbin_dir / args.qc_outdir_name
    qc_out.mkdir(exist_ok=True)
    echo("[plan] QC stage (MAD thresholds + optional Scrublet).")
    qc_cmd = [
        sys.executable, str(s_qc),
        "--in", str(h5ad_in),
        "--species", args.qc_species,
        "--outdir", str(qc_out),
        "--mad-mult", str(args.qc_mad_mult),
        "--min-genes", str(args.qc_min_genes),
        "--min-umis", str(args.qc_min_umis),
        "--max-mito", str(args.qc_max_mito),
    ]
    if args.qc_max_top5_frac is not None:
        qc_cmd += ["--max-top5-frac", str(args.qc_max_top5_frac)]
    if args.qc_run_scrublet:
        qc_cmd += ["--run-scrublet"]
    run(qc_cmd, dry=args.dry_run)

    qc_filtered = find_first(qc_out, ["*_qcfiltered.h5ad", "*.h5ad"])
    if not qc_filtered:
        msg = f"No QC-filtered .h5ad found in {qc_out} (expected *_qcfiltered.h5ad)."
        if args.dry_run: echo("[note] " + msg)
        else: fail(msg)
    else:
        echo(f"[info] QC-filtered h5ad: {qc_filtered}")

    # ---------------- Clustering ----------------
    echo("[plan] Clustering (Scanpy).")
    cluster_outdir = qc_out
    h5ad_out = qc_filtered.with_name(qc_filtered.stem.replace("_qcfiltered", "") + "_clustered.h5ad")
    os.environ["H5AD_IN"]  = str(qc_filtered)
    os.environ["H5AD_OUT"] = str(h5ad_out)
    os.environ["OUTDIR"]   = str(cluster_outdir)
    os.environ["TIFF_PATH"]= str(tiff_path)
    cl_cmd = [sys.executable, str(s_cluster)]
    if args.res is not None:      cl_cmd += ["--res", str(args.res)]
    if args.n_pcs is not None:    cl_cmd += ["--n-pcs", str(args.n_pcs)]
    if args.min_counts is not None: cl_cmd += ["--min-counts", str(args.min_counts)]
    run(cl_cmd, dry=args.dry_run)

    # ---------------- Overlays / markers ----------------
    echo("[plan] Cluster overlays + markers.")
    out_dir = cluster_outdir / "cluster_exports"
    out_dir.mkdir(exist_ok=True)
    os.environ["H5AD_IN"]  = str(h5ad_out)
    os.environ["OUT_DIR"]  = str(out_dir)
    os.environ["TIFF_PATH"]= str(tiff_path)
    run([sys.executable, str(s_overlay)], dry=args.dry_run)

    echo("\n[done] Lean plan complete." + (" (dry-run; nothing executed)" if args.dry_run else ""))

if __name__ == "__main__":
    main()
