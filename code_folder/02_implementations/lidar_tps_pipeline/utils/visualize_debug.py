#!/usr/bin/env python3
"""
visualize_debug.py -- Extended debug visualization for lidar_tps_pipeline.

Reads the C++ --debug output and produces additional composite images:
  08_canvas_rotation.jpg       -- rotation-only warp stitched to canvas
  09_canvas_tps.jpg            -- TPS-warped canvas (seam path overlay)
  10_rotation_vs_tps_*.jpg     -- per-camera side-by-side rotation vs TPS
  11_disp_quiver_*.jpg         -- TPS displacement field as quiver arrows
  12_overlap_zoom_lc.jpg       -- zoomed FL<->FC overlap region comparison
  12_overlap_zoom_cr.jpg       -- zoomed FC<->FR overlap region comparison
  13_pipeline_strip.jpg        -- full pipeline strip: source -> rotation -> TPS -> final

Usage:
  python3 utils/visualize_debug.py [debug_dir]

  debug_dir defaults to:
    output/lidar_ring_stitch/debug/frame_0000
"""

import sys
import os
import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -- config --------------------------------------------------------------------
JPEG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 92]

# -- helpers -------------------------------------------------------------------

def load(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img

def save(path, img):
    cv2.imwrite(path, img, JPEG_QUALITY)
    print(f"  saved {os.path.basename(path)}")

def canvas_stitch(fl, fc, fr):
    """
    Stitch 3 canvas-sized images with feather blending in overlap zones.
    In the overlap between two cameras, alpha-blend based on each camera's
    coverage fraction so seams are smooth rather than hard-cut.
    """
    H, W = fl.shape[:2]

    fl_f = fl.astype(np.float32)
    fc_f = fc.astype(np.float32)
    fr_f = fr.astype(np.float32)

    # Valid masks per camera (any channel > 0)
    v_fl = (fl.max(axis=2) > 0).astype(np.float32)
    v_fc = (fc.max(axis=2) > 0).astype(np.float32)
    v_fr = (fr.max(axis=2) > 0).astype(np.float32)

    # Smooth the masks to get soft feather weights in overlap zones
    k = max(3, W // 150) | 1   # kernel size ~ 0.7% of canvas width, must be odd
    v_fl_s = cv2.GaussianBlur(v_fl, (k, k), 0)
    v_fc_s = cv2.GaussianBlur(v_fc, (k, k), 0)
    v_fr_s = cv2.GaussianBlur(v_fr, (k, k), 0)

    # Weighted sum -- cameras contribute proportionally to their blurred coverage
    denom = v_fl_s + v_fc_s + v_fr_s
    denom = np.where(denom < 1e-6, 1.0, denom)

    out_f = (fl_f * v_fl_s[:, :, None] +
             fc_f * v_fc_s[:, :, None] +
             fr_f * v_fr_s[:, :, None]) / denom[:, :, None]

    return np.clip(out_f, 0, 255).astype(np.uint8)

def side_by_side(*imgs, gap=8, label_h=30, labels=None):
    """Horizontally concatenate images with a thin black gap and optional labels."""
    H = max(im.shape[0] for im in imgs) + label_h
    W_total = sum(im.shape[1] for im in imgs) + gap * (len(imgs) - 1)
    out = np.zeros((H, W_total, 3), dtype=np.uint8)
    x = 0
    for k, im in enumerate(imgs):
        h, w = im.shape[:2]
        out[label_h:label_h + h, x:x + w] = im
        if labels and k < len(labels):
            cv2.putText(out, labels[k], (x + 6, label_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
        x += w + gap
    return out

def zoom_region(img, x0, x1, scale=2):
    """Crop [x0:x1] columns from full canvas and upscale."""
    crop = img[:, x0:x1]
    h, w = crop.shape[:2]
    return cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)

def quiver_from_dispfields(disp_x_path, disp_y_path, warped_path, title, step=32):
    """
    Overlay TPS displacement arrows onto warped canvas image.
    Auto-crops to the camera's actual coverage bounding box (removes black margins).
    Returns BGR image.
    """
    dx_img = cv2.imread(disp_x_path)
    dy_img = cv2.imread(disp_y_path)
    bg     = cv2.imread(warped_path)
    if dx_img is None or dy_img is None or bg is None:
        return None

    H, W = bg.shape[:2]

    # Crop to non-black bounding box so the plot fills the figure
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    cols_any = np.where(bg_gray.max(axis=0) > 8)[0]
    rows_any = np.where(bg_gray.max(axis=1) > 8)[0]
    if len(cols_any) == 0 or len(rows_any) == 0:
        return None
    x0, x1 = int(cols_any[0]), int(cols_any[-1]) + 1
    y0, y1 = int(rows_any[0]), int(rows_any[-1]) + 1

    bg_crop    = bg[y0:y1, x0:x1]
    Hc, Wc     = bg_crop.shape[:2]

    # The disp images are JET-mapped uint8 with midpoint=128 -> 0, so invert
    dx_float = (dx_img[:, :, 0].astype(np.float32) - 128.0) / 127.0
    dy_float = (dy_img[:, :, 0].astype(np.float32) - 128.0) / 127.0

    # Upsample to canvas size if needed (disp was computed at half-res)
    if dx_float.shape != (H, W):
        dx_float = cv2.resize(dx_float, (W, H), interpolation=cv2.INTER_LINEAR)
        dy_float = cv2.resize(dy_float, (W, H), interpolation=cv2.INTER_LINEAR)

    # Crop disp fields to same bounding box
    dx_crop = dx_float[y0:y1, x0:x1]
    dy_crop = dy_float[y0:y1, x0:x1]

    # Build quiver grid over cropped region
    ys = np.arange(step // 2, Hc, step)
    xs = np.arange(step // 2, Wc, step)
    XX, YY = np.meshgrid(xs, ys)
    U = dx_crop[YY, XX]
    V = dy_crop[YY, XX]

    # Mask: only draw arrows where background has actual content
    bg_gray_crop = bg_gray[y0:y1, x0:x1]
    valid_mask = bg_gray_crop[YY, XX] > 8

    U_m = np.where(valid_mask, U, np.nan)
    V_m = np.where(valid_mask, V, np.nan)

    # Magnitude for colour
    mag = np.sqrt(U**2 + V**2)
    mag_masked = np.where(valid_mask, mag, np.nan)
    mag_max = max(np.nanmax(mag_masked) if np.any(valid_mask) else 1e-3, 1e-3)

    fig, ax = plt.subplots(figsize=(Wc / 120, Hc / 120), dpi=120)
    ax.imshow(cv2.cvtColor(bg_crop, cv2.COLOR_BGR2RGB))
    q = ax.quiver(XX, YY, U_m, V_m, mag_masked,
                  cmap='plasma', clim=(0, mag_max),
                  angles='xy', scale_units='xy', scale=0.05,
                  width=0.002, headwidth=4, headlength=5, alpha=0.85)
    plt.colorbar(q, ax=ax, fraction=0.02, pad=0.01,
                 label='TPS correction (px, normalised)')
    ax.set_title(title, fontsize=11, pad=5)
    ax.axis('off')
    plt.tight_layout(pad=0.3)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


# -- main ----------------------------------------------------------------------

def main(debug_dir):
    print(f"Visualizing: {debug_dir}")

    def p(name): return os.path.join(debug_dir, name)
    def exists(name): return os.path.isfile(p(name))

    cams = ["FL", "FC", "FR"]

    # -- 08: rotation-only canvas ----------------------------------------------
    if all(exists(f"02b_rotation_{c}.jpg") for c in cams):
        rot_imgs = [load(p(f"02b_rotation_{c}.jpg")) for c in cams]
        canvas_rot = canvas_stitch(*rot_imgs)
        save(p("08_canvas_rotation.jpg"), canvas_rot)

    # -- 09: TPS canvas with seam overlay (alias to 05_seam_paths) -------------
    if exists("05_seam_paths.jpg"):
        canvas_tps = load(p("05_seam_paths.jpg"))
        save(p("09_canvas_tps.jpg"), canvas_tps)

    # -- 10: per-camera rotation vs TPS side-by-side ---------------------------
    for c in cams:
        rot_f  = p(f"02b_rotation_{c}.jpg")
        tps_f  = p(f"02_warped_{c}.jpg")
        ctrl_f = p(f"03_ctrl_pts_{c}.jpg")
        if not (exists(f"02b_rotation_{c}.jpg") and exists(f"02_warped_{c}.jpg")):
            continue
        rot_img  = load(rot_f)
        tps_img  = load(tps_f)
        ctrl_img = load(ctrl_f) if exists(f"03_ctrl_pts_{c}.jpg") else tps_img
        strip = side_by_side(rot_img, tps_img, ctrl_img,
                             labels=["Rotation only", "TPS warped", "Ctrl pts"])
        save(p(f"10_rotation_vs_tps_{c}.jpg"), strip)

    # -- 11: displacement quiver overlays --------------------------------------
    for c in cams:
        dx_f = p(f"06_disp_x_{c}.jpg")
        dy_f = p(f"06_disp_y_{c}.jpg")
        w_f  = p(f"02_warped_{c}.jpg")
        if not (exists(f"06_disp_x_{c}.jpg") and exists(f"06_disp_y_{c}.jpg")
                and exists(f"02_warped_{c}.jpg")):
            continue
        quiver_img = quiver_from_dispfields(dx_f, dy_f, w_f,
                                            f"TPS displacement -- {c}")
        if quiver_img is not None:
            save(p(f"11_disp_quiver_{c}.jpg"), quiver_img)

    # -- 12a: TPS overlap strip -- before seam cut ------------------------------
    # Left panel:  camera A cropped to its valid right-edge coverage
    # Middle panel: camera B cropped to its valid left-edge coverage
    # Right panel: 50% blend cropped to the true overlap zone (both valid)
    for tag, cam_a, cam_b in [("lc", "FL", "FC"), ("cr", "FC", "FR")]:
        if not (exists(f"02_warped_{cam_a}.jpg") and exists(f"02_warped_{cam_b}.jpg")):
            continue
        wa = load(p(f"02_warped_{cam_a}.jpg"))
        wb = load(p(f"02_warped_{cam_b}.jpg"))
        H, W = wa.shape[:2]

        col_valid_a = wa.max(axis=(0, 2)) > 8   # shape (W,)
        col_valid_b = wb.max(axis=(0, 2)) > 8

        # True overlap: both valid
        both = col_valid_a & col_valid_b
        overlap_cols = np.where(both)[0]
        if len(overlap_cols) < 10:
            continue
        xo0, xo1 = int(overlap_cols[0]), int(overlap_cols[-1]) + 1

        # Camera A: show its valid region around the right edge (entering overlap)
        cols_a = np.where(col_valid_a)[0]
        xa0 = max(int(cols_a[0]), xo0 - (xo1 - xo0))   # ~one overlap-width to the left
        xa1 = xo1
        xa0 = max(0, xa0)

        # Camera B: show its valid region around the left edge (entering overlap)
        cols_b = np.where(col_valid_b)[0]
        xb0 = xo0
        xb1 = min(int(cols_b[-1]) + 1, xo1 + (xo1 - xo0))
        xb1 = min(W, xb1)

        strip_a     = wa[:, xa0:xa1]
        strip_b     = wb[:, xb0:xb1]
        strip_blend = cv2.addWeighted(wa[:, xo0:xo1], 0.5,
                                      wb[:, xo0:xo1], 0.5, 0)

        # Resize strips to same height (already same H) -- just concatenate
        # Make all strips the same width for cleaner layout by padding to max width
        max_w = max(strip_a.shape[1], strip_b.shape[1], strip_blend.shape[1])
        def pad_w(img, target_w):
            h, w = img.shape[:2]
            if w >= target_w: return img
            pad = np.zeros((h, target_w - w, 3), dtype=np.uint8)
            return np.hstack([img, pad])

        result = side_by_side(pad_w(strip_a, max_w),
                              pad_w(strip_b, max_w),
                              pad_w(strip_blend, max_w),
                              labels=[f"{cam_a} (TPS, right edge ->overlap)",
                                      f"{cam_b} (TPS, <-overlap left edge)",
                                      "50% blend -- true overlap zone"])
        save(p(f"12a_tps_overlap_{tag}.jpg"), result)

    # -- 12: overlap zone zoom comparisons -------------------------------------
    # Need the final canvas to know seam positions
    seam_f = p("05_seam_paths.jpg")
    if exists("05_seam_paths.jpg"):
        canvas = load(seam_f)
        H, W = canvas.shape[:2]

        # Find rough seam columns by locating red (FL<->FC) and green (FC<->FR) pixels
        red_mask   = (canvas[:, :, 2] > 200) & (canvas[:, :, 1] < 50) & (canvas[:, :, 0] < 50)
        green_mask = (canvas[:, :, 1] > 200) & (canvas[:, :, 2] < 50) & (canvas[:, :, 0] < 50)

        red_cols   = np.where(red_mask.any(axis=0))[0]
        green_cols = np.where(green_mask.any(axis=0))[0]

        seam_lc = int(np.median(red_cols))   if len(red_cols)   > 10 else W // 3
        seam_cr = int(np.median(green_cols)) if len(green_cols) > 10 else 2 * W // 3

        zoom_w = min(600, W // 6)

        # FL<->FC overlap zoom: rotation vs TPS vs final
        if all(exists(f"02b_rotation_{c}.jpg") and exists(f"02_warped_{c}.jpg")
               for c in ["FL", "FC"]):
            rot_fl  = load(p("02b_rotation_FL.jpg"))
            rot_fc  = load(p("02b_rotation_FC.jpg"))
            tps_fl  = load(p("02_warped_FL.jpg"))
            tps_fc  = load(p("02_warped_FC.jpg"))

            x0 = max(0, seam_lc - zoom_w)
            x1 = min(W, seam_lc + zoom_w)

            rot_canvas_lc = canvas_stitch(rot_fl, rot_fc, np.zeros_like(rot_fl))
            rot_zoom = zoom_region(rot_canvas_lc, x0, x1)
            tps_canvas_lc = canvas_stitch(tps_fl, tps_fc, np.zeros_like(tps_fl))
            tps_zoom  = zoom_region(tps_canvas_lc, x0, x1)
            final_zoom = zoom_region(canvas, x0, x1)

            strip = side_by_side(rot_zoom, tps_zoom, final_zoom,
                                 labels=["Rotation only", "TPS warped", "Final + seam"])
            save(p("12_overlap_zoom_lc.jpg"), strip)

        # FC<->FR overlap zoom
        if all(exists(f"02b_rotation_{c}.jpg") and exists(f"02_warped_{c}.jpg")
               for c in ["FC", "FR"]):
            rot_fc  = load(p("02b_rotation_FC.jpg"))
            rot_fr  = load(p("02b_rotation_FR.jpg"))
            tps_fc  = load(p("02_warped_FC.jpg"))
            tps_fr  = load(p("02_warped_FR.jpg"))

            x0 = max(0, seam_cr - zoom_w)
            x1 = min(W, seam_cr + zoom_w)

            rot_canvas_cr = canvas_stitch(np.zeros_like(rot_fc), rot_fc, rot_fr)
            rot_zoom  = zoom_region(rot_canvas_cr, x0, x1)
            tps_canvas_cr = canvas_stitch(np.zeros_like(tps_fc), tps_fc, tps_fr)
            tps_zoom  = zoom_region(tps_canvas_cr, x0, x1)
            final_zoom = zoom_region(canvas, x0, x1)

            strip = side_by_side(rot_zoom, tps_zoom, final_zoom,
                                 labels=["Rotation only", "TPS warped", "Final + seam"])
            save(p("12_overlap_zoom_cr.jpg"), strip)

    # -- 13: full pipeline strip -----------------------------------------------
    # Row 1: source FL, FC, FR
    # Row 2: rotation-only canvas
    # Row 3: TPS canvas (seam paths)
    src_strip_parts = []
    for c in cams:
        if exists(f"01_source_{c}.jpg"):
            src_strip_parts.append(load(p(f"01_source_{c}.jpg")))

    rows = []
    if src_strip_parts:
        # Resize sources to same height
        tgt_h = src_strip_parts[0].shape[0]
        resized = []
        for im in src_strip_parts:
            scale = tgt_h / im.shape[0]
            resized.append(cv2.resize(im, (int(im.shape[1]*scale), tgt_h)))
        row_src = side_by_side(*resized, labels=["FL source", "FC source", "FR source"])
        rows.append(row_src)

    if exists("08_canvas_rotation.jpg"):
        rows.append(load(p("08_canvas_rotation.jpg")))
    if exists("05_seam_paths.jpg"):
        rows.append(load(p("05_seam_paths.jpg")))

    if rows:
        # Resize all to same width
        tgt_w = max(r.shape[1] for r in rows)
        scaled = []
        labels_strip = ["Source images (FL / FC / FR)",
                        "Rotation-only canvas",
                        "TPS-warped canvas + seam paths"]
        for k, r in enumerate(rows):
            scale = tgt_w / r.shape[1]
            sr = cv2.resize(r, (tgt_w, int(r.shape[0] * scale)))
            # Add label bar
            bar = np.zeros((30, tgt_w, 3), dtype=np.uint8)
            if k < len(labels_strip):
                cv2.putText(bar, labels_strip[k], (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 1)
            scaled.append(np.vstack([bar, sr]))

        strip_full = np.vstack(scaled)
        save(p("13_pipeline_strip.jpg"), strip_full)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_dir = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        debug_dir = os.path.join(script_dir, "..", "output",
                                 "lidar_ring_stitch", "debug", "frame_0000")
    main(os.path.realpath(debug_dir))
