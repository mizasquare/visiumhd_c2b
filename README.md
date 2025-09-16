# visiumhd_c2b

Comprehensive pipeline for Visium HD / Visium CytAssist spatial transcriptomics.  The
repository contains command line tools that segment nuclei with StarDist, expand
polygons with Voronoi-capped dilation, aggregate 10x Genomics counts into
cell-by-gene matrices, run Scanpy-based QC, cluster cells, and generate spatial
overlays plus marker tables.  The `pipeline_runner.py` script wires the stages
into a reproducible end-to-end workflow, while individual modules remain usable
as stand-alone utilities.

## Key capabilities

- Patch-wise StarDist inference with adaptive tiling and deduplication
  (`stardister.py`).
- Voronoi-based polygon growth with optional before/after overlays
  (`voronoi_dilation.py`).
- Flexible cell-by-gene aggregation that supports full-slide and ROI-restricted
  exports (`build_cellxgene_from_polygons.py`).
- Robust QC following Scanpy conventions, including MAD-based thresholds and
  optional Scrublet doublet detection (`scRNA_qc_pipeline.py`).
- Leiden clustering, UMAP embeddings, whole-slide overlays, ROI thumbnails, and
  per-cluster marker discovery (`cluster.py`, `cluster_overlay.py`).
- Batch experimentation helper to sweep parameter grids across multiple samples
  (`batchpipe.py`).

## Repository layout

| Script | Purpose |
| --- | --- |
| `pipeline_runner.py` | Full pipeline with CLI knobs for each stage and QC parameters. |
| `pipeline_runner_lean.py` | Lean runner that can reuse existing StarDist/Voronoi outputs and perform integrity checks or dry runs. |
| `stardister.py` | Runs StarDist 2D on large histology TIFFs with adaptive tiling, tissue gating, and preview generation. |
| `voronoi_dilation.py` | Converts nuclei polygons into cell-like territories by dilating with Voronoi limits and can render overlays. |
| `build_cellxgene_from_polygons.py` | Aggregates 10x counts into per-cell AnnData/CellxGene outputs (full-frame or ROI-limited). |
| `scRNA_qc_pipeline.py` | Adds QC metrics, plots distributions, and filters AnnData objects. |
| `cluster.py` | Scanpy workflow for clustering + spatial overlays (invoked by the runner). |
| `cluster_overlay.py` | Post-hoc exports: per-cluster overlays, ROI crops, marker tables, marker plots. |
| `batchpipe.py` | Example batch runner that sweeps StarDist, Voronoi, and clustering parameters across datasets. |

## Installation

1. Create a fresh Python environment (Python ≥3.9 is recommended).
2. Install the required libraries.  GPU-enabled StarDist/TF is recommended but
   not mandatory.

```bash
pip install -U pip wheel setuptools
pip install numpy pandas scipy tifffile scikit-image shapely matplotlib \
    csbdeep stardist anndata scanpy tqdm scrublet
```

> **Note:** Some packages (e.g., `stardist`, `csbdeep`, `scanpy`) may need system
> dependencies such as a working compiler, libhdf5, and OpenMP.  On Linux, the
> [conda-forge](https://conda-forge.org/) channel is a convenient way to satisfy
> these requirements.

## Input requirements

The runners expect the canonical Visium HD / CytAssist inputs:

- High-resolution histology TIFF image (`--tiff`).
- 10x filtered counts, either as the `filtered_feature_bc_matrix*.h5` file
  (`--tenx-h5`) or the MTX directory (`--tenx-dir`).
- `tissue_positions.parquet` exported from Space Ranger or Xenium Explorer
  (`--positions`).
- Optional ImageJ `.roi` files for ROI-restricted analyses (`--roi`).

Place the data next to the scripts or provide absolute paths.  Outputs are
written next to the repository by default:

```
./stardist_output_<TAG>/
./voronoi_output_<TAG>/
./cellbin_out_<TAG>/
./cluster_outputs_<TAG>/
```

## Quickstart: run the full pipeline

```bash
python pipeline_runner.py \
  --tiff 28-A1.tif \
  --tenx-h5 filtered_feature_bc_matrix_GH.h5 \
  --positions tissue_positions_GH.parquet \
  --roi 28-A1.tif.LHbl.roi 28-A1.tif.LHbr.roi \
  --outbase GH_LHb \
  --qc-species mouse --qc-run-scrublet \
  --sd-prob 0.40 --sd-nms 0.60 \
  --voro-dilation-um 4.0 --voro-px-um 0.5 \
  --res 1.2 --n-pcs 60 --cl-n-neighbors 15 --cl-metric cosine
```

This command performs StarDist segmentation, Voronoi dilation, ROI-restricted
aggregation, QC filtering, clustering, overlays, and marker discovery.  Adjust
thresholds and clustering parameters as needed.  Intermediate files are cached
under stage-specific directories so that you can rerun later stages without
recomputing StarDist.

For iterative debugging or reuse of precomputed steps, switch to
`pipeline_runner_lean.py`:

```bash
python pipeline_runner_lean.py \
  --tiff 28-A1.tif \
  --tenx-h5 filtered_feature_bc_matrix_GH.h5 \
  --positions tissue_positions_GH.parquet \
  --reuse-stardist-dir stardist_output_GH_LHb \
  --reuse-voronoi-geojson voronoi_output_GH_LHb/nuclei_masks_dilated_v.geojson \
  --outbase GH_LHb --qc-species mouse --qc-run-scrublet --dry-run
```

The lean runner validates inputs, prints the exact commands that would be
executed, and can skip StarDist or Voronoi if you already have their outputs.
Use `--dry-run` to confirm paths before launching long jobs.

## Running individual stages

Each stage can also be executed manually:

1. **StarDist segmentation**
   ```bash
   python stardister.py -i 28-A1.tif -o stardist_output_GH_LHb \
     --prob 0.40 --nms 0.60 --preview
   ```
   Generates polygon GeoJSON (`nuclei_masks_native.geojson`) and preview PNGs.

2. **Voronoi-capped dilation**
   ```bash
   python voronoi_dilation.py \
     --tiff 28-A1.tif \
     --in stardist_output_GH_LHb/nuclei_masks_native.geojson \
     --out voronoi_output_GH_LHb/nuclei_masks_dilated_v.geojson \
     --dilation 4.0 --px 0.5 --overlay voronoi_output_GH_LHb/overlay.png
   ```

3. **Cell-by-gene aggregation**
   ```bash
   python build_cellxgene_from_polygons.py \
     --tiff 28-A1.tif \
     --geojson voronoi_output_GH_LHb/nuclei_masks_dilated_v.geojson \
     --tenx-h5 filtered_feature_bc_matrix_GH.h5 \
     --positions tissue_positions_GH.parquet \
     --outdir-base cellbin_out_GH_LHb \
     --roi 28-A1.tif.LHbl.roi 28-A1.tif.LHbr.roi
   ```
   Produces AnnData (`.h5ad`), CSV summaries, and coordinate tables; centroid
   columns are later consumed by the clustering/overlay scripts.

4. **QC + filtering**
   ```bash
   python scRNA_qc_pipeline.py \
     --in cellbin_out_GH_LHb/roi_LHb/cellbins.h5ad \
     --species mouse --mad-mult 3 --max-mito 20 \
     --outdir cellbin_out_GH_LHb/roi_LHb/qc_out --run-scrublet
   ```
   Writes `<input>_qcfiltered.h5ad`, a JSON report, QC metrics CSV, and plots.

5. **Clustering and overlays**
   ```bash
   python cluster.py
   python cluster_overlay.py
   ```
   Edit the module-level constants or let `pipeline_runner.py` patch them on the
   fly.  Outputs include clustered AnnData, UMAP, whole-slide overlays, ROI
   thumbnails, and ranked marker tables/plots.

## Batch experiments

`batchpipe.py` demonstrates how to sweep StarDist thresholds, Voronoi growth
radii, and clustering hyperparameters across multiple datasets.  Adjust the
`DATASETS`, `PROB_VALS`, `NMS_VALS`, `VORO_GROWTH_PX`, and `CLUSTERING`
dictionaries, then run:

```bash
python batchpipe.py
```

Each combination gets its own log under `batch_runs/<sample>__<param_tag>/`.

## Tips

- If you frequently reuse StarDist results, archive the entire
  `stardist_output_<TAG>` folder; it includes metadata and previews that help
  diagnose issues.
- The QC stage honors the `--qc-max-top5-frac` guardrail to flag highly dominant
  genes per cell—enable it when datasets contain ribosomal contamination.
- For large TIFFs, ensure enough RAM/disk scratch space; StarDist tiles can be
  run on GPU by configuring TensorFlow appropriately before launching the
  scripts.

