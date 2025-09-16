#!/usr/bin/env python3

"""
scRNA-seq QC Pipeline (Scanpy-style)
------------------------------------
Computes per-cell QC metrics (UMIs, gene diversity, mito/ribo %, top-gene dominance),
plots distributions, applies robust MAD-based filters, and (optionally) runs Scrublet
doublet detection. Saves a filtered h5ad, QC CSV, plots, and a JSON report.

Usage examples:
  python scRNA_qc_pipeline.py --in data.h5ad --species mouse --outdir qc_out
  python scRNA_qc_pipeline.py --in ./filtered_feature_bc_matrix --tenx --species human --outdir qc_out --run-scrublet

Key options:
  --species {mouse,human}           Sets default gene prefixes (mito/ribo).
  --mito-prefix MT- / mt-           Override mito gene prefix (case-insensitive match).
  --ribo-prefixes Rps Rpl           Override ribosomal subunit prefixes (space-separated).
  --mad-mult 3.0                    MAD multiplier for robust thresholds.
  --min-genes 200                   Absolute floor for detected genes (before MAD filter; set 0 to disable).
  --min-umis  500                   Absolute floor for UMIs (before MAD filter; set 0 to disable).
  --max-mito  30                    Absolute cap (%) for mito fraction (after MAD filter; set 0 to use MAD-only).
  --run-scrublet                    Try Scrublet for doublet detection (if installed).
  --expected-doublet-rate 0.06      Scrublet expected doublet rate.

Outputs in --outdir:
  qc_metrics.csv
  qc_plots.pdf
  qc_report.json
  <input_basename>_qcfiltered.h5ad
"""
import argparse, json, os, sys, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORTANT: Use matplotlib only, no seaborn.
import scanpy as sc
import anndata as ad

def robust_median_mad(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826  # normalized MAD
    if mad == 0 or np.isnan(mad):
        # Fallback: use IQR/1.349 ~ std-like
        q75, q25 = np.nanpercentile(x, 75), np.nanpercentile(x, 25)
        mad = (q75 - q25) / 1.349 if (q75 > q25) else 1.0
        if mad == 0 or np.isnan(mad):
            mad = 1.0
    return med, mad

def detect_prefixes(species, mito_prefix, ribo_prefixes):
    if mito_prefix is None:
        mito_prefix = "mt-" if species == "mouse" else "MT-"
    if ribo_prefixes is None:
        if species == "mouse":
            ribo_prefixes = ["Rps", "Rpl"]
        else:
            ribo_prefixes = ["RPS", "RPL"]
    return mito_prefix, ribo_prefixes

def flag_genes(var_names, prefix):
    # case-insensitive startswith
    prefix_low = prefix.lower()
    return np.array([str(g).lower().startswith(prefix_low) for g in var_names], dtype=bool)

def flag_genes_multi(var_names, prefixes):
    prefixes_low = [p.lower() for p in prefixes]
    mask = np.zeros(len(var_names), dtype=bool)
    vlow = [str(g).lower() for g in var_names]
    for p in prefixes_low:
        mask |= np.fromiter((name.startswith(p) for name in vlow), dtype=bool, count=len(vlow))
    return mask

def compute_topk_fraction_per_cell(X, k=5):
    """Return fraction of counts explained by top-k genes per cell."""
    # Works for dense or csr/csc
    if hasattr(X, "tocsr"):
        X = X.tocsr()
        fracs = np.zeros(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            row = X.getrow(i).toarray().ravel()
            s = row.sum()
            if s <= 0:
                fracs[i] = np.nan
            else:
                fracs[i] = np.sort(row)[-k:].sum() / s
        return fracs
    else:
        X = np.asarray(X)
        s = X.sum(axis=1)
        topk = np.sort(X, axis=1)[:, -k:].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(s > 0, topk / s, np.nan)

def add_qc_metrics(adata, mito_prefix, ribo_prefixes):
    var_names = np.array(adata.var_names)
    mito_mask = flag_genes(var_names, mito_prefix)
    ribo_mask = flag_genes_multi(var_names, ribo_prefixes)

    # Ensure columns exist for Scanpy versions that expect qc_vars as column names
    adata.var["mt"] = mito_mask
    adata.var["ribo"] = ribo_mask

    # Standard Scanpy QC metrics
    # Pass column names to be robust across Scanpy versions
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo"],
        percent_top=None,
        log1p=False,
        inplace=True
    )

    # Top-5 gene dominance metric
    adata.obs["top5_frac"] = compute_topk_fraction_per_cell(adata.X, k=5)

    # Rename for clarity (Scanpy creates these columns if present)
    rename_map = {}
    if "total_counts" in adata.obs.columns:
        rename_map["total_counts"] = "n_umi"
    if "n_genes_by_counts" in adata.obs.columns:
        rename_map["n_genes_by_counts"] = "n_genes"
    if "pct_counts_mt" in adata.obs.columns:
        rename_map["pct_counts_mt"] = "pct_mito"
    if "pct_counts_ribo" in adata.obs.columns:
        rename_map["pct_counts_ribo"] = "pct_ribo"
    if rename_map:
        adata.obs.rename(columns=rename_map, inplace=True)

    # If pct_* columns didn't appear (e.g., no mt/ribo genes), create them as 0
    for col in ["n_umi", "n_genes", "pct_mito", "pct_ribo"]:
        if col not in adata.obs.columns:
            if col in ["pct_mito", "pct_ribo"]:
                adata.obs[col] = 0.0
            else:
                # Fallback to basic metrics
                if col == "n_umi":
                    sums = np.array(adata.X.sum(axis=1)).ravel() if hasattr(adata.X, "sum") else adata.X.sum(1)
                    adata.obs[col] = sums
                elif col == "n_genes":
                    if hasattr(adata.X, "tocsr"):
                        adata.obs[col] = adata.X.getnnz(axis=1)
                    else:
                        adata.obs[col] = (np.asarray(adata.X) > 0).sum(axis=1)

    return mito_mask, ribo_mask

def make_plots(adata, out_pdf):
    # Prepare figure with multiple panels (but single figure)
    pdf = out_pdf
    with plt.rc_context():
        plt.figure(figsize=(12, 10), dpi=150)

        # 2x2 grid + 1 wide
        # Panel 1: Violin-like via boxplot (UMIs)
        plt.subplot(2, 3, 1)
        plt.boxplot([adata.obs["n_umi"].values], showfliers=True)
        plt.title("Total UMIs per cell")
        plt.ylabel("UMIs")
        plt.xticks([1], ["cells"])

        # Panel 2: Genes detected
        plt.subplot(2, 3, 2)
        plt.boxplot([adata.obs["n_genes"].values], showfliers=True)
        plt.title("Genes detected per cell")
        plt.ylabel("n_genes")
        plt.xticks([1], ["cells"])

        # Panel 3: pct mito
        plt.subplot(2, 3, 3)
        plt.boxplot([adata.obs["pct_mito"].values], showfliers=True)
        plt.title("Mitochondrial %")
        plt.ylabel("% of UMIs")
        plt.xticks([1], ["cells"])

        # Panel 4: UMI vs Genes scatter
        plt.subplot(2, 3, 4)
        plt.scatter(adata.obs["n_umi"], adata.obs["n_genes"], s=4, alpha=0.35)
        plt.xlabel("UMIs")
        plt.ylabel("Genes")
        plt.title("UMIs vs Genes")

        # Panel 5: Mito % vs UMIs
        plt.subplot(2, 3, 5)
        plt.scatter(adata.obs["n_umi"], adata.obs["pct_mito"], s=4, alpha=0.35)
        plt.xlabel("UMIs")
        plt.ylabel("Mito %")
        plt.title("Mito % vs UMIs")

        # Panel 6: Top5 dominance
        plt.subplot(2, 3, 6)
        plt.boxplot([adata.obs["top5_frac"].values], showfliers=True)
        plt.title("Top-5 genes fraction")
        plt.ylabel("Fraction")

        plt.tight_layout()
        plt.savefig(pdf, bbox_inches="tight")
        plt.close()

def apply_filters(adata, args):
    obs = adata.obs

    # Absolute floors (pre-filter)
    keep = np.ones(len(obs), dtype=bool)
    if args.min_genes > 0:
        keep &= obs["n_genes"].values >= args.min_genes
    if args.min_umis > 0:
        keep &= obs["n_umi"].values >= args.min_umis

    # Robust filters with MAD
    # Lower bounds for n_umi and n_genes
    for col in ["n_umi", "n_genes"]:
        med, mad = robust_median_mad(obs[col].values)
        lower = med - args.mad_mult * mad
        keep &= obs[col].values >= lower

    # Upper bound for pct_mito
    med_mito, mad_mito = robust_median_mad(obs["pct_mito"].values)
    upper_mito = med_mito + args.mad_mult * mad_mito
    # Apply absolute cap if provided and smaller
    if args.max_mito > 0:
        upper_mito = min(upper_mito, args.max_mito)

    keep &= obs["pct_mito"].values <= upper_mito

    # Optionally cap extreme top-gene dominance
    if args.max_top5_frac is not None and args.max_top5_frac > 0:
        keep &= obs["top5_frac"].values <= args.max_top5_frac

    adata.obs["qc_keep"] = keep
    return {
        "lower_n_umi": float(med - args.mad_mult * mad),
        "lower_n_genes": float(robust_median_mad(obs["n_genes"].values)[0] - args.mad_mult * robust_median_mad(obs["n_genes"].values)[1]),
        "upper_pct_mito": float(upper_mito),
    }

def maybe_run_scrublet(adata, args):
    if not args.run_scrublet:
        adata.obs["doublet_score"] = np.nan
        adata.obs["doublet_call"] = False
        return {"scrublet_ran": False}

    try:
        import scrublet as scr
    except Exception as e:
        sys.stderr.write(f"[warn] Scrublet not available: {e}\n")
        adata.obs["doublet_score"] = np.nan
        adata.obs["doublet_call"] = False
        return {"scrublet_ran": False, "error": str(e)}

    # Scrublet expects counts matrix cells x genes (dense or sparse)
    counts_matrix = adata.X
    if hasattr(counts_matrix, "tocsr"):
        counts_matrix = counts_matrix.tocsr()

    scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=args.expected_doublet_rate)
    doublet_scores, predicted_doublets = scrub.scrub_doublets()
    adata.obs["doublet_score"] = doublet_scores
    adata.obs["doublet_call"] = predicted_doublets.astype(bool)

    return {
        "scrublet_ran": True,
        "expected_doublet_rate": args.expected_doublet_rate,
        "threshold": float(getattr(scrub, "threshold_", np.nan))
    }

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--in", dest="input_path", required=True, help="Input .h5ad or 10x mtx dir (use --tenx)")
    p.add_argument("--tenx", action="store_true", help="Interpret input as a 10x directory (matrix.mtx, features.tsv, barcodes.tsv)")
    p.add_argument("--species", choices=["mouse", "human"], default="mouse", help="Species for default gene prefixes")
    p.add_argument("--mito-prefix", default=None, help="Override mito gene prefix (e.g., MT- or mt-)")
    p.add_argument("--ribo-prefixes", nargs="+", default=None, help="Override ribosomal gene prefixes (e.g., RPS RPL)")
    p.add_argument("--outdir", default="qc_out", help="Output directory")
    p.add_argument("--mad-mult", type=float, default=3.0, help="MAD multiplier for robust thresholds")
    p.add_argument("--min-genes", type=int, default=200, help="Absolute minimum genes per cell (0 to disable)")
    p.add_argument("--min-umis", type=int, default=500, help="Absolute minimum UMIs per cell (0 to disable)")
    p.add_argument("--max-mito", type=float, default=30.0, help="Absolute cap on mito %% (0 to disable, use MAD only)")
    p.add_argument("--max-top5-frac", type=float, default=None, help="Optional cap on top-5 gene dominance (e.g., 0.6)")
    p.add_argument("--run-scrublet", action="store_true", help="Run Scrublet doublet detection if installed")
    p.add_argument("--expected-doublet-rate", type=float, default=0.06, help="Expected doublet rate for Scrublet")
    args = p.parse_args()

    in_path = Path(args.input_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.tenx:
        adata = sc.read_10x_mtx(in_path, var_names="gene_symbols", cache=False)
        base = in_path.name
    else:
        if in_path.suffix.lower() == ".h5ad":
            adata = sc.read_h5ad(in_path)
            base = in_path.stem
        else:
            # try generic readers
            adata = sc.read(in_path)
            base = in_path.stem

    # Ensure var_names are unique
    adata.var_names_make_unique()

    # Decide gene category prefixes
    mito_prefix, ribo_prefixes = detect_prefixes(args.species, args.mito_prefix, args.ribo_prefixes)

    # Compute QC metrics
    mito_mask, ribo_mask = add_qc_metrics(adata, mito_prefix, ribo_prefixes)

    # Run Scrublet (optional)
    scrub_info = maybe_run_scrublet(adata, args)

    # Plots
    plots_pdf = outdir / "qc_plots.pdf"
    make_plots(adata, str(plots_pdf))

    # Apply filters
    thresh = apply_filters(adata, args)

    # Export QC table
    qc_cols = ["n_umi", "n_genes", "pct_mito", "pct_ribo", "top5_frac", "doublet_score", "doublet_call", "qc_keep"]
    qc_df = adata.obs[qc_cols].copy()
    qc_csv = outdir / "qc_metrics.csv"
    qc_df.to_csv(qc_csv)

    # Save filtered object
    keep_mask = adata.obs["qc_keep"].values & (~adata.obs.get("doublet_call", pd.Series(False, index=adata.obs.index)).values)
    adata_filt = adata[keep_mask].copy()
    out_h5ad = outdir / f"{base}_qcfiltered.h5ad"
    adata_filt.write_h5ad(out_h5ad)

    # Report
    report = {
        "input": str(in_path),
        "n_cells_raw": int(adata.n_obs),
        "n_cells_kept": int(adata_filt.n_obs),
        "species": args.species,
        "mito_prefix": mito_prefix,
        "ribo_prefixes": ribo_prefixes,
        "thresholds": thresh,
        "scrublet": scrub_info,
        "filters": {
            "min_genes": args.min_genes,
            "min_umis": args.min_umis,
            "max_mito": args.max_mito,
            "mad_mult": args.mad_mult,
            "max_top5_frac": args.max_top5_frac
        },
        "outputs": {
            "qc_csv": str(qc_csv),
            "qc_plots_pdf": str(plots_pdf),
            "filtered_h5ad": str(out_h5ad)
        }
    }
    with open(outdir / "qc_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
