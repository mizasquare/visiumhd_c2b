# stardister_patchwise_directpoly_dedup.py
import os, json, time, random, argparse, gc
import numpy as np
import tifffile as tiff
import skimage.io as skio

from skimage import img_as_float32, img_as_float
from skimage.util import img_as_ubyte
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.measure import regionprops, block_reduce
from skimage.segmentation import find_boundaries
from skimage.morphology import (
    remove_small_objects, binary_opening, binary_closing, remove_small_holes, disk
)
from skimage.filters import threshold_otsu
from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regionprops

from shapely.geometry import Polygon, mapping, shape
from shapely.strtree import STRtree

from csbdeep.utils import normalize
from stardist.models import StarDist2D

from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------
# Default Config (override via main() args if needed)
# ---------------------------
SCALE               = 2.0       # upscale for inference
PATCH               = 2048
CELL_DIAM_SCALED    = 6.0
OVERLAP_NATIVE      = None      # if None, computed from CELL_DIAM_SCALED/SCALE
NTILES              = (8, 8, 1)
PROB                = 0.35      # consider 0.40–0.50 if you still see speckles
NMS                 = 0.60
MIN_AREA_SCALED     = 15        # min area (px) at SCALED (2x) space, before downsampling

# Dedup
IOU_THRESHOLD       = 0.50

# Tissue-gating knobs
TISSUE_MASK_MAX_SIDE = 4096     # build mask at ~4k long side
TISSUE_MIN_FRAC      = 0.02     # skip tiles with <2% tissue (final safety gate)

# >>> NEW: adaptive tiler knobs
TISSUE_TARGET_FRAC   = 0.60     # accept tile only if ≥60% tissue
MIN_PATCH            = 512      # do not go below this (native px) when subdividing
EDGE_LOW_FRAC        = 0.05     # fast-drop tiles with <5% tissue

# Preview / output
KEEP_UPSCALED_CANVAS = False    # set True to also keep 2× raster
PREVIEW_MAX_SIDE     = 8000     # downscale long side for preview
PREVIEW_DPI          = 300
MONTAGE_N_MAX        = 12       # sample cell crops


# ===========================
# Helpers
# ===========================
def build_tissue_mask(rgb, max_side=TISSUE_MASK_MAX_SIDE):
    H, W = rgb.shape[:2]
    ds = int(np.ceil(max(H, W) / max_side))
    ds = max(ds, 1)
    if ds > 1:
        rgb_lr = rescale(rgb, 1.0 / ds, channel_axis=2, anti_aliasing=True, preserve_range=True)
    else:
        rgb_lr = rgb

    gray = rgb_lr.mean(axis=2)  # simple brightness
    try:
        thr = threshold_otsu(gray)
    except Exception:
        thr = 0.95

    mask = gray < min(thr, 0.95)  # below bright glass background
    mask = binary_opening(mask, disk(2))
    mask = binary_closing(mask, disk(3))
    mask = remove_small_holes(mask, area_threshold=32)
    mask = remove_small_objects(mask, min_size=64)
    return mask.astype(bool), ds


def _ceil_div(a, b):
    return int(np.ceil(a / float(b)))


def _clip_box(y0, y1, x0, x1, H, W):
    return max(0, y0), min(H, y1), max(0, x0), min(W, x1)


def generate_adaptive_tiles(
    tissue_mask_lr: np.ndarray,
    DS: int,
    H: int, W: int,
    max_patch_native: int,
    min_patch_native: int = MIN_PATCH,
    target_frac: float = TISSUE_TARGET_FRAC,
    edge_low_frac: float = EDGE_LOW_FRAC,
    overlap_native: int = 0,
):
    """
    Build native-space tiles (y0,y1,x0,x1) that:
      - are between min_patch_native and max_patch_native in side length
      - contain >= target_frac tissue
      - discard near-empty tiles
      - subdivide ambiguous edge tiles (quadtree-ish)

    Works in LR mask space for speed; maps back to native at the end.
    """
    Hlr, Wlr = tissue_mask_lr.shape
    max_lr = _ceil_div(max_patch_native, DS)
    min_lr = max(1, _ceil_div(min_patch_native, DS))

    lab = sk_label(tissue_mask_lr, connectivity=1)
    comps = sk_regionprops(lab)
    tiles_native = []

    def process_box(y0_lr, y1_lr, x0_lr, x1_lr):
        h_lr = y1_lr - y0_lr
        w_lr = x1_lr - x0_lr
        if h_lr <= 0 or w_lr <= 0:
            return

        submask = tissue_mask_lr[y0_lr:y1_lr, x0_lr:x1_lr]
        area = float(h_lr * w_lr)
        if area == 0:
            return
        frac = float(submask.sum()) / area

        # near-empty → drop
        if frac <= edge_low_frac:
            return

        # too big → split
        if h_lr > max_lr or w_lr > max_lr:
            ym = y0_lr + h_lr // 2
            xm = x0_lr + w_lr // 2
            process_box(y0_lr, ym, x0_lr, xm)
            process_box(y0_lr, ym, xm, x1_lr)
            process_box(ym, y1_lr, x0_lr, xm)
            process_box(ym, y1_lr, xm, x1_lr)
            return

        # within size band → accept if dense enough
        if h_lr >= min_lr and w_lr >= min_lr and frac >= target_frac:
            y0 = y0_lr * DS
            y1 = min(y1_lr * DS, H)
            x0 = x0_lr * DS
            x1 = min(x1_lr * DS, W)

            # symmetric overlap padding
            pad = max(0, overlap_native // 2)
            y0, y1, x0, x1 = _clip_box(y0 - pad, y1 + pad, x0 - pad, x1 + pad, H, W)

            # enforce min size after padding
            if (y1 - y0) < min_patch_native:
                need = min_patch_native - (y1 - y0)
                grow_up = need // 2
                grow_dn = need - grow_up
                y0 = max(0, y0 - grow_up)
                y1 = min(H, y1 + grow_dn)
            if (x1 - x0) < min_patch_native:
                need = min_patch_native - (x1 - x0)
                grow_lt = need // 2
                grow_rt = need - grow_lt
                x0 = max(0, x0 - grow_lt)
                x1 = min(W, x1 + grow_rt)

            tiles_native.append((y0, y1, x0, x1))
            return

        # ambiguous edge region → split if possible
        if h_lr > min_lr or w_lr > min_lr:
            ym = y0_lr + h_lr // 2
            xm = x0_lr + w_lr // 2
            process_box(y0_lr, ym, x0_lr, xm)
            process_box(y0_lr, ym, xm, x1_lr)
            process_box(ym, y1_lr, x0_lr, xm)
            process_box(ym, y1_lr, xm, x1_lr)
        else:
            # too small to split further and not dense → reject
            return

    # seed on each connected tissue component’s bbox (with a small ring)
    for c in comps:
        y0_lr, x0_lr, y1_lr, x1_lr = c.bbox
        ring = max(1, min_lr // 2)
        y0_lr = max(0, y0_lr - ring); x0_lr = max(0, x0_lr - ring)
        y1_lr = min(Hlr, y1_lr + ring); x1_lr = min(Wlr, x1_lr + ring)
        process_box(y0_lr, y1_lr, x0_lr, x1_lr)

    tiles_native = sorted(list(set(tiles_native)))  # unique + deterministic order
    return tiles_native


def run_patchwise_inference(
    img_rgb, model, outdir,
    scale=SCALE, patch=PATCH, overlap_native=None,
    ntiles=NTILES, prob=PROB, nms=NMS,
    min_area_scaled=MIN_AREA_SCALED,
    keep_upscaled_canvas=KEEP_UPSCALED_CANVAS,
    tissue_min_frac=TISSUE_MIN_FRAC,
):
    H, W = img_rgb.shape[:2]

    # Compute overlap if not explicitly given (in native px)
    if overlap_native is None:
        overlap_native = int(max(8, round((2.5 * CELL_DIAM_SCALED) / scale)))
    print(f"Using PATCH={patch}, OVERLAP(native)={overlap_native}, OVERLAP(scaled)={overlap_native*scale}")

    # Whole-slide tissue mask (low-res)
    tissue_mask_lr, DS = build_tissue_mask(img_rgb)
    print(f"Tissue mask built at DS={DS}, tissue fraction={tissue_mask_lr.mean():.3f}")

    labels_full_native = np.zeros((H, W), dtype=np.uint32)
    next_id_native = 1

    if keep_upscaled_canvas:
        labels_full_up = np.zeros((int(H * scale), int(W * scale)), dtype=np.uint32)
        next_id_up = 1
    else:
        labels_full_up = None
        next_id_up = None

    features = []
    t0_all = time.time()

    # --------------------------
    # Adaptive tiles (native px)
    # --------------------------
    tiles = generate_adaptive_tiles(
        tissue_mask_lr, DS, H, W,
        max_patch_native=patch,
        min_patch_native=MIN_PATCH,
        target_frac=TISSUE_TARGET_FRAC,
        edge_low_frac=EDGE_LOW_FRAC,
        overlap_native=overlap_native,
    )
    if not tiles:
        print("Adaptive tiler produced 0 tiles; falling back to coarse grid.")
        tiles = []
        step = max(64, patch - overlap_native)
        for y0 in range(0, H, step):
            y1 = min(y0 + patch, H)
            for x0 in range(0, W, step):
                x1 = min(x0 + patch, W)
                # gate by tissue presence in LR mask
                ys0_lr, ys1_lr = y0 // DS, int(np.ceil(y1 / DS))
                xs0_lr, xs1_lr = x0 // DS, int(np.ceil(x1 / DS))
                patch_mask_lr = tissue_mask_lr[ys0_lr:ys1_lr, xs0_lr:xs1_lr]
                tissue_frac = float(patch_mask_lr.mean()) if patch_mask_lr.size else 0.0
                if tissue_frac >= TISSUE_TARGET_FRAC and (y1 - y0) >= MIN_PATCH and (x1 - x0) >= MIN_PATCH:
                    tiles.append((y0, y1, x0, x1))

    print(f"Adaptive tiles selected: {len(tiles)}")

    for (y0, y1, x0, x1) in tiles:
        patch_rgb = img_rgb[y0:y1, x0:x1]

        # final safety gate using local LR mask slice
        ys0_lr, ys1_lr = y0 // DS, int(np.ceil(y1 / DS))
        xs0_lr, xs1_lr = x0 // DS, int(np.ceil(x1 / DS))
        patch_mask_lr = tissue_mask_lr[ys0_lr:ys1_lr, xs0_lr:xs1_lr]
        tissue_frac = float(patch_mask_lr.mean()) if patch_mask_lr.size else 0.0
        if tissue_frac < tissue_min_frac or patch_rgb.std() < 1e-3:
            continue

        # normalize & upscale
        patch_norm = normalize(patch_rgb, 1, 99.8, axis=(0, 1))
        patch_up = rescale(
            patch_norm, scale, channel_axis=2,
            anti_aliasing=True, preserve_range=True
        ).astype(np.float32)

        # predict (on 2×)
        labels_up, details = model.predict_instances(
            patch_up, axes='YXC', n_tiles=ntiles,
            prob_thresh=prob, nms_thresh=nms,
            show_tile_progress=False
        )

        # filter small in scaled space
        labels_up = remove_small_objects(labels_up, min_size=min_area_scaled)

        # (A) paste to 2× canvas if requested
        if keep_upscaled_canvas and labels_up.max() > 0:
            labels_up_off = labels_up.copy()
            labels_up_off[labels_up_off > 0] += next_id_up
            next_id_up = int(labels_up_off.max()) + 1
            ys0_up, xs0_up = int(y0 * scale), int(x0 * scale)
            ys1_up, xs1_up = ys0_up + labels_up_off.shape[0], xs0_up + labels_up_off.shape[1]
            labels_full_up[ys0_up:ys1_up, xs0_up:xs1_up] = np.maximum(
                labels_full_up[ys0_up:ys1_up, xs0_up:xs1_up], labels_up_off
            )

        # (B) downsample to 1× and paste (mask off-tissue)
        if labels_up.max() > 0:
            labels_native_patch = rescale(
                labels_up.astype(float), 1.0 / scale,
                order=0, anti_aliasing=False, preserve_range=True
            ).astype(np.uint32)

            mask_native_patch = resize(
                patch_mask_lr.astype(float),
                (y1 - y0, x1 - x0),
                order=0, preserve_range=True, anti_aliasing=False
            ).astype(bool)
            labels_native_patch[~mask_native_patch] = 0

            if labels_native_patch.max() > 0:
                labels_native_patch[labels_native_patch > 0] += next_id_native
                next_id_native = int(labels_native_patch.max()) + 1

                labels_full_native[y0:y1, x0:x1] = np.maximum(
                    labels_full_native[y0:y1, x0:x1], labels_native_patch
                )

        # Collect polygons (convert to 1× coords) + centroid tissue gate
        if isinstance(details, dict) and ("coord" in details):
            for poly_pts in details["coord"]:
                poly_pts = np.asarray(poly_pts)
                if poly_pts.ndim == 2 and poly_pts.shape[0] == 2 and poly_pts.shape[1] != 2:
                    poly_pts = poly_pts.T
                # Y(row), X(col) → 1× and shift to native coords
                poly_pts[:, 0] = poly_pts[:, 0] / scale + y0
                poly_pts[:, 1] = poly_pts[:, 1] / scale + x0
                poly = Polygon([(float(x), float(y)) for y, x in poly_pts])  # (x, y)

                if not (poly.is_valid and poly.area >= min_area_scaled / (scale * scale)):
                    continue

                cy, cx = poly.centroid.y, poly.centroid.x  # (row, col)
                cy_lr, cx_lr = int(cy // DS), int(cx // DS)
                if (0 <= cy_lr < tissue_mask_lr.shape[0] and 0 <= cx_lr < tissue_mask_lr.shape[1]):
                    if not tissue_mask_lr[cy_lr, cx_lr]:
                        continue
                else:
                    continue

                features.append({
                    "type": "Feature",
                    "properties": {"cell_id": int(len(features) + 1)},
                    "geometry": mapping(poly)
                })

    elapsed_all = time.time() - t0_all
    print(f"Patchwise inference done in {elapsed_all/60:.1f} min")
    print(f"Total nuclei (native labels) = {next_id_native - 1}, polygons collected = {len(features)}")

    return labels_full_native, features, labels_full_up


def deduplicate_polygons_by_label(features, labels_native):
    """
    Fast O(N) dedup:
      - For each polygon, sample its centroid in the native label map.
      - Keep the first polygon seen for that label id; drop the rest.
      - If centroid hits background (0), we fall back to hashing rounded
        centroid coords to avoid duplicates from patch overlaps.

    This relies on labels_full_native already resolving overlaps.
    """
    if not features:
        print("No polygons to deduplicate.")
        return []

    H, W = labels_native.shape[:2]
    kept = []
    seen_labels = set()
    seen_bg_hash = set()

    for f in features:
        poly = shape(f["geometry"])
        cy, cx = poly.centroid.y, poly.centroid.x  # (row, col)
        iy, ix = int(round(cy)), int(round(cx))

        # out-of-bounds centroids -> treat as background
        if not (0 <= iy < H and 0 <= ix < W):
            lab = 0
        else:
            lab = int(labels_native[iy, ix])

        if lab > 0:
            if lab in seen_labels:
                continue
            seen_labels.add(lab)
            kept.append(f)
        else:
            # background centroid (rare if masking was correct):
            # dedup by 1px-rounded centroid hash
            h = (iy, ix)
            if h in seen_bg_hash:
                continue
            seen_bg_hash.add(h)
            kept.append(f)

    # Renumber cell_id
    for new_id, f in enumerate(kept, start=1):
        f["properties"]["cell_id"] = new_id

    removed = len(features) - len(kept)
    print(f"Centroid-label dedup removed {removed}; kept {len(kept)} polygons")
    return kept


def save_label_tiff(labels_native, outdir, filename="nuclei_labels_native.tiff"):
    os.makedirs(outdir, exist_ok=True)
    tiff_path = os.path.join(outdir, filename)
    try:
        skio.imsave(tiff_path, labels_native.astype(np.uint32), check_contrast=False)
    except Exception:
        tiff.imwrite(tiff_path, labels_native.astype(np.uint32), bigtiff=True)
    print(f"Saved label map: {tiff_path}")
    return tiff_path


def save_geojson(features, outdir, filename="nuclei_masks_native.geojson"):
    os.makedirs(outdir, exist_ok=True)
    geojson_path = os.path.join(outdir, filename)
    with open(geojson_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"Saved polygons: {geojson_path}")
    return geojson_path


def _compute_downscale_factor(h, w, max_side):
    if max(h, w) <= max_side:
        return 1
    # integer factor so block_reduce works for labels
    return max(2, int(np.ceil(max(h, w) / max_side)))


def save_overlay_preview(
    rgb, labels, out_png,
    max_side=8000,
    draw_centroids=False
):
    """
    Memory-lean overlay:
      - Integer downscale factor via block ops (labels) and local mean (RGB)
      - 1px boundary paint in uint8
    """
    H, W = rgb.shape[:2]
    f = _compute_downscale_factor(H, W, max_side)

    # Downscale RGB with local mean (channel-wise), then to uint8
    if f > 1:
        if rgb.ndim == 2:
            rgb_ds = downscale_local_mean(rgb, (f, f)).astype(np.float32)
            rgb_ds = np.repeat(rgb_ds[..., None], 3, axis=2)
        else:
            chans = []
            for c in range(rgb.shape[2]):
                chans.append(downscale_local_mean(rgb[..., c], (f, f)).astype(np.float32))
            rgb_ds = np.stack(chans, axis=2)
    else:
        rgb_ds = rgb.astype(np.float32, copy=False)
        if rgb_ds.ndim == 2:
            rgb_ds = np.repeat(rgb_ds[..., None], 3, axis=2)

    rgb_ds_u8 = img_as_ubyte(np.clip(rgb_ds, 0, 1)) if rgb_ds.dtype != np.uint8 else rgb_ds.copy()

    # Downscale labels with max pooling (keeps categorical ids)
    if f > 1:
        pad_y = (-H) % f
        pad_x = (-W) % f
        if pad_y or pad_x:
            labels_pad = np.pad(labels, ((0, pad_y), (0, pad_x)), mode='constant')
        else:
            labels_pad = labels
        labels_ds = block_reduce(labels_pad, block_size=(f, f), func=np.max)
        labels_ds = labels_ds[:rgb_ds_u8.shape[0], :rgb_ds_u8.shape[1]]
    else:
        labels_ds = labels

    # Boundaries
    bnd = find_boundaries(labels_ds, mode='inner')
    rgb_ds_u8[bnd, 0] = 0
    rgb_ds_u8[bnd, 1] = 255
    rgb_ds_u8[bnd, 2] = 0

    if draw_centroids:
        for rp in regionprops(labels_ds):
            cy, cx = rp.centroid
            y = int(round(cy)); x = int(round(cx))
            if 1 <= y < rgb_ds_u8.shape[0]-1 and 1 <= x < rgb_ds_u8.shape[1]-1:
                rgb_ds_u8[y, x-1:x+2] = (255, 255, 0)
                rgb_ds_u8[y-1:y+2, x] = (255, 255, 0)

    skio.imsave(out_png, rgb_ds_u8, check_contrast=False)
    print(f"Saved overlay preview: {out_png}")
    return out_png


def save_montage(img_rgb, labels_native, outdir, n_max=MONTAGE_N_MAX, tile=256, pad=8):
    """
    Lightweight montage using PIL. Extracts n_max random cells,
    crops with a small context, center-crops/pads to tile size,
    and composes to a grid without creating a huge figure.
    """
    max_label = int(labels_native.max()) if labels_native.size else 0
    if max_label <= 0:
        print("Montage skipped: no labels found.")
        return None

    sample_ids = random.sample(range(1, max_label + 1), min(n_max, max_label))
    ncols = min(4, max(1, int(np.ceil(np.sqrt(len(sample_ids))))))
    nrows = int(np.ceil(len(sample_ids) / ncols))
    canvas_w = ncols * (tile + pad) + pad
    canvas_h = nrows * (tile + pad) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(0, 0, 0))

    for idx, sid in enumerate(sample_ids):
        mask = (labels_native == sid)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        y0, y1 = max(0, ys.min()-20), min(labels_native.shape[0], ys.max()+20)
        x0, x1 = max(0, xs.min()-20), min(labels_native.shape[1], xs.max()+20)
        crop_rgb = img_rgb[y0:y1, x0:x1]

        if crop_rgb.dtype != np.uint8:
            crop_rgb_u8 = img_as_ubyte(np.clip(crop_rgb, 0, 1))
        else:
            crop_rgb_u8 = crop_rgb

        h, w = crop_rgb_u8.shape[:2]
        side = max(h, w)
        pad_y = (side - h) // 2
        pad_x = (side - w) // 2
        crop_sq = np.pad(
            crop_rgb_u8,
            ((pad_y, side - h - pad_y), (pad_x, side - w - pad_x), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        im = Image.fromarray(crop_sq, mode="RGB").resize((tile, tile), Image.BILINEAR)

        r = idx // ncols
        c = idx % ncols
        ox = pad + c * (tile + pad)
        oy = pad + r * (tile + pad)
        canvas.paste(im, (ox, oy))

    montage_path = os.path.join(outdir, "montage_sample.png")
    canvas.save(montage_path, format="PNG", optimize=True)
    print(f"Saved montage: {montage_path}")
    return montage_path


# ===========================
# Main orchestration
# ===========================
def main(
    input: str,
    outputdir: str,
    scale: float = SCALE,
    patch: int = PATCH,
    overlap_native: int | None = OVERLAP_NATIVE,
    ntiles=NTILES,
    prob: float = PROB,
    nms: float = NMS,
    min_area_scaled: int = MIN_AREA_SCALED,
    iou_threshold: float = IOU_THRESHOLD,
    keep_upscaled_canvas: bool = KEEP_UPSCALED_CANVAS,
    preview_max_side: int = PREVIEW_MAX_SIDE,
    preview_dpi: int = PREVIEW_DPI,
):
    os.makedirs(outputdir, exist_ok=True)

    # Load image
    img_rgb = tiff.imread(input, maxworkers=1)
    img_rgb = img_as_float32(img_rgb)
    print(f"Loaded image {input} with shape {img_rgb.shape}")

    # Model (load once)
    model = StarDist2D.from_pretrained('2D_versatile_he')

    # Patchwise inference (adaptive tiles)
    labels_full_native, features, labels_full_up = run_patchwise_inference(
        img_rgb, model, outputdir,
        scale=scale, patch=patch, overlap_native=overlap_native,
        ntiles=ntiles, prob=prob, nms=nms,
        min_area_scaled=min_area_scaled,
        keep_upscaled_canvas=keep_upscaled_canvas,
    )

    # Dedup polygons (fast, raster-guided)
    dedup_features = deduplicate_polygons_by_label(features, labels_full_native)

    # Save outputs
    tiff_path = save_label_tiff(labels_full_native, outputdir)
    geojson_path = save_geojson(dedup_features, outputdir)

    # Overlays / montage
    overlay_path = os.path.join(outputdir, "overlay_preview.png")
    save_overlay_preview(
        img_rgb, labels_full_native, overlay_path,
        max_side=preview_max_side, draw_centroids=True
    )

    save_montage(img_rgb, labels_full_native, outputdir)

    # Optionally save 2× canvas
    if keep_upscaled_canvas and labels_full_up is not None:
        up_tiff = os.path.join(outputdir, "nuclei_labels_upscaled_2x.tiff")
        try:
            skio.imsave(up_tiff, labels_full_up.astype(np.uint32), check_contrast=False)
        except Exception:
            tiff.imwrite(up_tiff, labels_full_up.astype(np.uint32), bigtiff=True)
        print(f"Saved upscaled label map: {up_tiff}")
    else:
        up_tiff = None

    return {
        "labels_tiff": tiff_path,
        "polygons_geojson": geojson_path,
        "overlay_png": overlay_path,
        "upscaled_tiff": up_tiff,
    }


# ===========================
# CLI
# ===========================
if __name__ == "__main__":
    # Uncomment if you want CLI usage
    parser = argparse.ArgumentParser(description="StarDist adaptive patchwise inference with thin-line overlay & raster-guided dedup.")
    parser.add_argument("-i", "--input", required=True, help="Input histology image (e.g., .tif)")
    parser.add_argument("-o", "--outputdir", required=True, help="Output directory")
    parser.add_argument("--scale", type=float, default=SCALE)
    parser.add_argument("--patch", type=int, default=PATCH)
    parser.add_argument("--overlap_native", type=int, default=-1, help="Overlap in native px; -1 = auto")
    parser.add_argument("--prob", type=float, default=PROB)
    parser.add_argument("--nms", type=float, default=NMS)
    parser.add_argument("--min_area_scaled", type=int, default=MIN_AREA_SCALED)
    parser.add_argument("--keep_upscaled_canvas", action="store_true", default=KEEP_UPSCALED_CANVAS)
    parser.add_argument("--preview_max_side", type=int, default=PREVIEW_MAX_SIDE)
    parser.add_argument("--preview_dpi", type=int, default=PREVIEW_DPI)
    args = parser.parse_args()
    overlap_val = None if args.overlap_native == -1 else args.overlap_native
    main(
        input=args.input,
        outputdir=args.outputdir,
        scale=args.scale,
        patch=args.patch,
        overlap_native=overlap_val,
        prob=args.prob,
        nms=args.nms,
        min_area_scaled=args.min_area_scaled,
        keep_upscaled_canvas=args.keep_upscaled_canvas,
        preview_max_side=args.preview_max_side,
        preview_dpi=args.preview_dpi,
    )
    # Example direct call:
    # main(input="28-D1.tif", outputdir="stardist_output_SH")
