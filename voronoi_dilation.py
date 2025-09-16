# voronoi_capped_dilation.py
import sys, json, os
from pathlib import Path

import numpy as np
from shapely.geometry import shape, mapping, MultiPoint, box, Polygon, MultiPolygon, Point
from shapely.ops import voronoi_diagram
from shapely.strtree import STRtree

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x  # fallback

import tifffile as tiff
import skimage.io as skio
from skimage.draw import polygon as sk_polygon
from skimage.measure import regionprops, block_reduce
from skimage.segmentation import find_boundaries
from skimage.transform import downscale_local_mean
from skimage.util import img_as_ubyte

# --- BEGIN CLI COMPAT WRAPPER ---
def _coerce_cli(argv):
    """
    Supports both:
      Positional:
        script.py <TIFF> <IN_GEOJSON> <OUT_GEOJSON> [dilation_um] [px_um] [overlay_png]
      Flags:
        script.py --tiff TIF --in IN_GEOJSON --out OUT_GEOJSON [--dilation 2.0] [--px 0.5] [--overlay OUT_PNG]
    Returns: [TIFF, IN_GEOJSON, OUT_GEOJSON, dilation, px_um, overlay_png or ""]
    """
    if any(a.startswith("--") for a in argv[1:]):
        import argparse
        ap = argparse.ArgumentParser(add_help=False)
        ap.add_argument("--tiff", required=True)
        ap.add_argument("--in", dest="in_geo", required=True)
        ap.add_argument("--out", dest="out_geo", required=True)
        ap.add_argument("--dilation", type=float, default=2.0)
        ap.add_argument("--px", type=float, default=1.0)
        ap.add_argument("--overlay", default="")
        ns, _ = ap.parse_known_args(argv[1:])
        return [ns.tiff, ns.in_geo, ns.out_geo, str(ns.dilation), str(ns.px), ns.overlay]
    # legacy positional mode — pad to len 6
    av = argv[1:]
    while len(av) < 6:
        av.append("")
    return av[:6]

_TI, _IN, _OUT, _DIL, _PX, _OV = _coerce_cli(sys.argv)
# --- END CLI COMPAT WRAPPER ---


# ---------------------------
# Overlay helpers
# ---------------------------
def _compute_downscale_factor_fallback(H, W, max_side):
    if max_side is None or max_side <= 0:
        return 1
    m = max(H, W)
    if m <= max_side:
        return 1
    # integer factor only (block ops)
    f = int(np.ceil(m / max_side))
    return max(1, f)

def load_rgb_from_path(path):
    """
    Load an RGB or grayscale TIFF/PNG/JPEG.
    Returns float32 image in [0,1] with shape (H,W,3).
    """
    if path.lower().endswith((".tif", ".tiff")):
        arr = tiff.imread(path)
    else:
        arr = skio.imread(path)

    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]  # drop alpha if present

    arr = arr.astype(np.float32, copy=False)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr

def rasterize_geojson_to_labels(geojson_path, H, W, start_label=1):
    """
    Returns a 2D int32 label image of shape (H, W).
    Each polygon feature gets a unique integer ID (0=background).
    Assumes GeoJSON coordinates are in *pixel units* that match (H, W).
    """
    with open(geojson_path, "r") as f:
        gj = json.load(f)

    labels = np.zeros((H, W), dtype=np.int32)
    lid = int(start_label)

    for feat in gj.get("features", []):
        geom = shape(feat["geometry"])
        # normalize to iterable of polygons
        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:
            continue

        for poly in polys:
            # exterior ring
            ex = np.asarray(poly.exterior.coords, dtype=np.float64)
            rr, cc = sk_polygon(
                np.clip(ex[:, 1], 0, H-1),
                np.clip(ex[:, 0], 0, W-1),
                shape=(H, W)
            )
            labels[rr, cc] = lid
            # holes -> reset to 0
            for ring in poly.interiors:
                hi = np.asarray(ring.coords, dtype=np.float64)
                rrh, cch = sk_polygon(
                    np.clip(hi[:, 1], 0, H-1),
                    np.clip(hi[:, 0], 0, W-1),
                    shape=(H, W)
                )
                labels[rrh, cch] = 0
            lid += 1

    return labels

def save_before_after_overlay(
    rgb_or_path, labels_before, labels_after, out_png,
    max_side=8000, draw_centroids=False, draw_growth_fill=True, growth_alpha=0.25
):
    if not out_png:
        return None  # explicitly skip overlay if empty path

    # accept path or array for RGB
    if isinstance(rgb_or_path, str):
        rgb = load_rgb_from_path(rgb_or_path)
    else:
        rgb = rgb_or_path

    if not (isinstance(labels_before, np.ndarray) and labels_before.ndim == 2):
        raise ValueError("labels_before must be a 2-D numpy array (H, W).")
    if not (isinstance(labels_after, np.ndarray) and labels_after.ndim == 2):
        raise ValueError("labels_after must be a 2-D numpy array (H, W).")
    if labels_before.shape != labels_after.shape:
        raise ValueError(f"labels_before shape {labels_before.shape} != labels_after shape {labels_after.shape}")

    H, W = rgb.shape[:2]
    if labels_before.shape != (H, W):
        raise ValueError(f"Label map shape {labels_before.shape} must match RGB shape {(H, W)}.")

    f = _compute_downscale_factor_fallback(H, W, max_side)

    # Downscale RGB (channel-wise local mean)
    if f > 1:
        if rgb.ndim == 2:
            rgb_ds = downscale_local_mean(rgb, (f, f)).astype(np.float32)
            rgb_ds = np.repeat(rgb_ds[..., None], 3, axis=2)
        else:
            chans = [downscale_local_mean(rgb[..., c], (f, f)).astype(np.float32) for c in range(rgb.shape[2])]
            rgb_ds = np.stack(chans, axis=2)
    else:
        rgb_ds = rgb.astype(np.float32, copy=False)
        if rgb_ds.ndim == 2:
            rgb_ds = np.repeat(rgb_ds[..., None], 3, axis=2)
    rgb_ds_u8 = img_as_ubyte(np.clip(rgb_ds, 0, 1)) if rgb_ds.dtype != np.uint8 else rgb_ds.copy()

    # Downscale labels (max-pooling preserves IDs)
    def _downscale_labels(lbl):
        if lbl.ndim != 2: raise ValueError("Label map must be 2-D.")
        if f <= 1: return lbl
        pad_y = (-H) % f
        pad_x = (-W) % f
        if pad_y or pad_x:
            lbl = np.pad(lbl, ((0, pad_y), (0, pad_x)), mode='constant')
        ds = block_reduce(lbl, block_size=(f, f), func=np.max)
        return ds[:rgb_ds_u8.shape[0], :rgb_ds_u8.shape[1]]

    lb = _downscale_labels(labels_before)
    la = _downscale_labels(labels_after)

    # Optional growth fills (semi-transparent)
    if draw_growth_fill:
        bb = (lb > 0)
        ba = (la > 0)
        grown  = np.logical_and(ba, ~bb)  # after minus before
        shrunk = np.logical_and(bb, ~ba)  # before minus after

        def blend(mask, color):
            if not np.any(mask): return
            for ch, val in enumerate(color):
                base = rgb_ds_u8[..., ch].astype(np.float32)
                base[mask] = (1.0 - growth_alpha) * base[mask] + growth_alpha * float(val)
                rgb_ds_u8[..., ch] = base.astype(np.uint8)

        blend(grown,  (0, 255, 255))   # cyan
        blend(shrunk, (255, 0, 255))   # magenta

    # Boundaries
    b_before = find_boundaries(lb, mode='inner')
    b_after  = find_boundaries(la, mode='inner')
    both = np.logical_and(b_before, b_after)
    only_before = np.logical_and(b_before, ~b_after)
    only_after  = np.logical_and(b_after,  ~b_before)

    # Paint: overlap=green, before-only=magenta, after-only=cyan
    rgb_ds_u8[only_before, 0] = 255; rgb_ds_u8[only_before, 1] = 0;   rgb_ds_u8[only_before, 2] = 255
    rgb_ds_u8[only_after,  0] = 0;   rgb_ds_u8[only_after,  1] = 255; rgb_ds_u8[only_after,  2] = 255
    rgb_ds_u8[both,         0] = 0;   rgb_ds_u8[both,         1] = 255; rgb_ds_u8[both,         2] = 0

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    skio.imsave(out_png, rgb_ds_u8, check_contrast=False)
    print(f"Saved before-after overlay: {out_png}")
    return out_png


# ---------------------------
# Voronoi-capped dilation
# ---------------------------
CLEAN_EPS = 0.0  # set >0 if you want post-ops cleanup via buffer(0)

def load_polygons(geojson_path):
    with open(geojson_path, "r") as f:
        gj = json.load(f)
    feats = gj["features"]
    geoms = [shape(feat["geometry"]) for feat in feats]
    return gj, feats, geoms

def compute_envelope(geoms, pad_px=50):
    minx = min(g.bounds[0] for g in geoms)
    miny = min(g.bounds[1] for g in geoms)
    maxx = max(g.bounds[2] for g in geoms)
    maxy = max(g.bounds[3] for g in geoms)
    return box(minx - pad_px, miny - pad_px, maxx + pad_px, maxy + pad_px)

def build_voronoi_cells(centroids, envelope):
    """
    Return a list of Voronoi cells (polygons), one per centroid.
    Works across Shapely versions where STRtree.query may return indices or geometries.
    """
    # ensure sequence of shapely Points
    pts = [p if isinstance(p, Point) else Point(p.x, p.y) for p in centroids]
    mp = MultiPoint(pts)
    v = voronoi_diagram(mp, envelope=envelope, tolerance=0.0, edges=False)
    cells = list(v.geoms)

    tree = STRtree(cells)

    def _as_geom(item):
        # Shapely may return numpy.int64 indices in some versions
        if isinstance(item, (int, np.integer)):
            return cells[int(item)]
        return item  # already a geometry

    centroid_to_cell = []
    for pt in pts:
        # Try modern API with predicate, fall back otherwise
        try:
            cands = tree.query(pt, predicate="intersects")
        except TypeError:
            cands = tree.query(pt)

        chosen = None
        for c in cands:
            cell = _as_geom(c)
            if cell.covers(pt):
                chosen = cell
                break

        if chosen is None:
            cand_geoms = [ _as_geom(c) for c in (cands if len(cands) > 0 else range(len(cells))) ]
            chosen = min(cand_geoms, key=lambda g: g.distance(pt))

        centroid_to_cell.append(chosen)

    return centroid_to_cell

def safe_buffer(poly, r_px):
    """Buffer + optional cleanup. r_px can be 0."""
    if r_px == 0:
        return poly
    g = poly.buffer(r_px)   # round joins are fine for soma-like growth
    if CLEAN_EPS != 0.0:
        g = g.buffer(0)
    return g

def dilate_voronoi_capped(
    geojson_in,
    geojson_out,
    dilation_um,
    pixel_size_um,
    border_pad_px=50,
):
    gj, feats, geoms = load_polygons(geojson_in)
    n = len(geoms)
    if n == 0:
        raise RuntimeError("No polygons found in the input GeoJSON.")

    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be > 0.")
    grow_px = float(dilation_um) / float(pixel_size_um)

    centroids = [g.centroid for g in geoms]
    envelope = compute_envelope(geoms, pad_px=border_pad_px)
    cells = build_voronoi_cells(centroids, envelope)

    out_geoms = []
    clipped_count = 0
    area_gain = []
    for g, cell in tqdm(zip(geoms, cells), total=n, desc="Dilating (Voronoi-capped)"):
        grown = safe_buffer(g, grow_px)
        clipped = grown.intersection(cell)
        if CLEAN_EPS != 0.0:
            clipped = clipped.buffer(0)

        out = clipped if not clipped.is_empty else g
        if out.area + 1e-9 < grown.area:
            clipped_count += 1

        area_gain.append(max(out.area - g.area, 0.0))
        out_geoms.append(out)

    for feat, newg in zip(feats, out_geoms):
        feat["geometry"] = mapping(newg)

    Path(geojson_out).parent.mkdir(parents=True, exist_ok=True)
    with open(geojson_out, "w") as f:
        json.dump(gj, f)

    gain_arr = np.array(area_gain, dtype=float)
    print("---- Voronoi-capped dilation summary ----")
    print(f"Polygons                 : {n:,}")
    print(f"Dilation radius (µm)     : {dilation_um:.3f}")
    print(f"Radius (px)              : {grow_px:.3f}")
    print(f"Clipped by Voronoi (cnt) : {clipped_count:,} ({clipped_count/n*100:.1f}%)")
    print(f"Area gain per polygon    : mean={gain_arr.mean():.2f} px², median={np.median(gain_arr):.2f} px²")
    print(f"Output written to        : {geojson_out}")
    return geojson_out


# ---------------------------
# CLI / Main
# ---------------------------
def main(
    rgb_path,
    geojson_in,
    geojson_out,
    overlay_png="",
    dilation_um=2.0,
    pixel_size_um=1.0,
    border_pad_px=50,
    max_side=8000,
    draw_centroids=False,
    draw_growth_fill=True,
    growth_alpha=0.25,
):
    # 1) Perform dilation first (produces geojson_out)
    out_gj = dilate_voronoi_capped(
        geojson_in=geojson_in,
        geojson_out=geojson_out,
        dilation_um=dilation_um,
        pixel_size_um=pixel_size_um,
        border_pad_px=border_pad_px,
    )

    # 2–3) Overlay is optional — only if a non-empty path is provided
    if overlay_png:
        rgb = load_rgb_from_path(rgb_path)
        H, W = rgb.shape[:2]
        labels_before = rasterize_geojson_to_labels(geojson_in, H, W)
        labels_after  = rasterize_geojson_to_labels(out_gj,     H, W)

        save_before_after_overlay(
            rgb_or_path=rgb,
            labels_before=labels_before,
            labels_after=labels_after,
            out_png=overlay_png,
            max_side=max_side,
            draw_centroids=draw_centroids,
            draw_growth_fill=draw_growth_fill,
            growth_alpha=growth_alpha,
        )
    return 0


if __name__ == "__main__":
    # Ensure output dirs exist for both geojson and (optional) overlay
    if _OUT:
        os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    if _OV:
        os.makedirs(os.path.dirname(_OV), exist_ok=True)

    dilation_um   = float(_DIL) if _DIL else 2.0
    pixel_size_um = float(_PX)  if _PX  else 1.0
    overlay_png   = _OV or ""

    sys.exit(main(
        rgb_path=_TI,
        geojson_in=_IN,
        geojson_out=_OUT,
        overlay_png=overlay_png,
        dilation_um=dilation_um,
        pixel_size_um=pixel_size_um,
        border_pad_px=50,
        max_side=8000,
        draw_centroids=False,
        draw_growth_fill=True,
        growth_alpha=0.25,
    ))
