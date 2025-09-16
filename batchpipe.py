#!/usr/bin/env python3
import os, sys, subprocess, itertools, shlex
from pathlib import Path
from datetime import datetime

# --------- USER-EDITABLE GLOBALS ---------
# Path to the pipeline runner
PIPELINE = Path(__file__).parent / "pipeline_runner.py"

# Pixel size (um/px) for your histology TIFFs.
# Used to convert the requested Voronoi growth in *pixels* to micrometers.
PX_UM = 1.0

# Three conditions for StarDist thresholds
PROB_VALS = [0.35, 0.50]      # detection; higher = stricter
NMS_VALS  = [0.55, 0.65]      # merge overlap; lower = more aggressive merging

# Four growth radii in *pixels*
VORO_GROWTH_PX = [1, 3, 5]

# Two clustering configs: "current" (yours) and "coarser" (fewer clusters)
CLUSTERING = {
    "current": {
        "res": 1.2,
        "n_pcs": 60,
        "min_counts": 50,
        "n_neighbors": 12,
        "metric": "cosine",
        "hvg": 15000,
        "umap_min_dist": 0.2,
        "umap_spread": 1.0,
    },
    "coarser": {
        "res": 0.6,
        "n_pcs": 40,
        "min_counts": 50,
        "n_neighbors": 20,
        "metric": "cosine",
        "hvg": 8000,
        "umap_min_dist": 0.25,
        "umap_spread": 1.0,
    },
}

# Datasets (GH and SH) with their specific inputs
DATASETS = [
    {
        "tag": "GH_LHb",
        "tiff": "28-A1.tif",
        "tenx_h5": "filtered_feature_bc_matrix_GH.h5",
        "positions": "tissue_positions_GH.parquet",
        "roi": ["28-A1.tif.LHbl.roi", "28-A1.tif.LHbr.roi"],
        "qc": {"species": "mouse", "min_genes": 200, "min_umis": 200, "max_mito": 15, "mad_mult": 3, "run_scrublet": True},
    },
    {
        "tag": "SH_LHb",
        "tiff": "28-D1.tif",
        "tenx_h5": "filtered_feature_bc_matrix_SH.h5",
        "positions": "tissue_positions_SH.parquet",
        "roi": ["28-D1.tif.LHbl.roi", "28-D1.tif.LHbr.roi"],
        "qc": {"species": "mouse", "min_genes": 200, "min_umis": 200, "max_mito": 15, "mad_mult": 3, "run_scrublet": True},
    },
]

# Where to write logs and a manifest of runs
RUN_ROOT = Path("batch_runs")
# -----------------------------------------

def ensure_pipeline():
    global PIPELINE
    if not PIPELINE.exists():
        # Fall back to local working dir if launched elsewhere
        alt = Path("pipeline_runner.py")
        if alt.exists():
            PIPELINE = alt.resolve()
    if not PIPELINE.exists():
        raise SystemExit(f"Could not locate pipeline_runner.py at {PIPELINE}")

def build_base_args(ds, cluster_cfg, outbase):
    args = [
        sys.executable, str(PIPELINE),
        "--tiff", ds["tiff"],
        "--tenx-h5", ds["tenx_h5"],
        "--positions", ds["positions"],
        "--roi", *ds["roi"],
        "--outbase", outbase,                 # <-- unique per run
        "--qc-species", ds["qc"]["species"],
        "--qc-min-genes", str(ds["qc"]["min_genes"]),
        "--qc-min-umis",  str(ds["qc"]["min_umis"]),
        "--qc-max-mito",  str(ds["qc"]["max_mito"]),
        "--qc-mad-mult",  str(ds["qc"]["mad_mult"]),
    ]
    if ds["qc"]["run_scrublet"]:
        args.append("--qc-run-scrublet")

    # clustering (parametrized)
    args += [
        "--res", str(cluster_cfg["res"]),
        "--n-pcs", str(cluster_cfg["n_pcs"]),
        "--min-counts", str(cluster_cfg["min_counts"]),
        "--cl-n-neighbors", str(cluster_cfg["n_neighbors"]),
        "--cl-metric", cluster_cfg["metric"],
        "--cl-hvg", str(cluster_cfg["hvg"]),
        "--cl-umap-min-dist", str(cluster_cfg["umap_min_dist"]),
        "--cl-umap-spread", str(cluster_cfg["umap_spread"]),
    ]
    return args

def tag_from_params(prob, nms, growth_px, cl_key):
    return f"sdP{prob:.2f}_sdN{nms:.2f}_vpx{growth_px}_cl{cl_key}"

def main():
    ensure_pipeline()
    RUN_ROOT.mkdir(exist_ok=True)
    manifest_lines = []
    total = 0

    # Cartesian product over all requested conditions
    for ds, prob, nms, growth_px, cl_key in itertools.product(
        DATASETS, PROB_VALS, NMS_VALS, VORO_GROWTH_PX, CLUSTERING.keys()
    ):
        cluster_cfg = CLUSTERING[cl_key]
        run_tag = tag_from_params(prob, nms, growth_px, cl_key)

        # Each run gets its own work dir & log file
        run_dir = RUN_ROOT / f"{ds['tag']}__{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Make outbase unique so outputs do NOT collide
        outbase = f"{ds['tag']}__{run_tag}"

        # Build command
        cmd = build_base_args(ds, cluster_cfg, outbase)

        # StarDist knobs (forwarded to stardister.py)
        cmd += ["--sd-prob", str(prob), "--sd-nms", str(nms)]

        # Voronoi growth in micrometers (convert from pixels)
        dilation_um = float(growth_px) * float(PX_UM)
        cmd += ["--voro-dilation-um", str(dilation_um), "--voro-px-um", str(PX_UM)]

        # Logging & manifest
        log_path = run_dir / "run.log"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmd_str = " ".join(shlex.quote(x) for x in cmd)
        manifest_lines.append("\t".join([ts, ds["tag"], run_tag, cmd_str]))

        print(f"\n[RUN] {ds['tag']} | {run_tag}")
        print("CMD:", cmd_str)
        with open(log_path, "w") as lf:
            lf.write("CMD: " + cmd_str + "\n\n")
            lf.flush()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
            ret = proc.wait()
            lf.write(f"\n[exit_code] {ret}\n")
        if ret != 0:
            print(f"[WARN] Run failed (exit={ret}): {run_tag}")
        total += 1

    # Save manifest
    manifest = RUN_ROOT / "manifest.tsv"
    with open(manifest, "w") as f:
        f.write("timestamp\tdataset\trun_tag\tcommand\n")
        f.write("\n".join(manifest_lines) + "\n")

    print(f"\nAll scheduled runs finished. Total attempted: {total}")
    print(f"Manifest saved to: {manifest}")

if __name__ == "__main__":
    main()
