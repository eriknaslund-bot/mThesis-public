"""Full-canvas pairwise stitch: reference (left) + comp blend + right non-overlap.

Canvas layout (UDIS++ paper style):
  [left non-overlap] [comp blend in overlap zone] [right non-overlap]

Cylindrical mode (default, matches training):
  Both images are projected to cylinder; overlap crops fed to the model;
  canvas assembled from cylindrical images -- no homography needed.

Planar mode (--no-cylindrical):
  Right image warped by rotation homography into left camera frame.

Usage:
    python vis_stitch_full.py --pair 1 --frame 0 \\
        --weights weights/udis_pp_lidar_full_comp/best.pth --lidar
"""

import argparse, json, os, sys
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(__file__))
from compute_overlaps import load_calib, compute_overlap
from dataset import ADJACENT_PAIRS, _compute_orb_matches
from cylindrical import build_cylindrical_remap, compute_cylindrical_overlap

NET_H, NET_W = 512, 704  # overridden by weights config.json or --img_h/--img_w


def bgr_to_tensor(bgr, h, w):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return TF.to_tensor(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))

def bgr_to_tensor_full(bgr):
    """Convert full-resolution BGR ndarray to float tensor without resize."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)

def tensor_to_bgr(t):
    arr = (t.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def imagenet_norm(t):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1).to(t.device)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1).to(t.device)
    return (t - mean) / std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames',  default='/home/Erik/mThesis/argo2_data/training/subset_14logs_frames.json')
    parser.add_argument('--calib',   default='/home/Erik/mThesis/argo2_data/training/calibration.json')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--pair',    type=int, default=1,
                        help='Pair index into ADJACENT_PAIRS (0-4)')
    parser.add_argument('--frame',   type=int, default=0)
    parser.add_argument('--out',     default='output/stitch_full')
    parser.add_argument('--device',  default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--img_h',   type=int, default=None)
    parser.add_argument('--img_w',   type=int, default=None)
    parser.add_argument('--lidar',   action='store_true', default=True,
                        help='Use UDISppLidar (default True)')
    parser.add_argument('--no-lidar', dest='lidar', action='store_false')
    parser.add_argument('--cylindrical', action='store_true', default=True,
                        help='Cylindrical projection mode (default True, matches training)')
    parser.add_argument('--no-cylindrical', dest='cylindrical', action='store_false')
    parser.add_argument('--no_calib_prewarp', action='store_true', default=False,
                        help='Skip calibration H pre-warp on right crop (match training with --no_calib_prewarp)')
    parser.add_argument('--warp-only', action='store_true', default=False,
                        help='Feather-blend left+warped_right instead of using composition output')
    parser.add_argument('--adapt', action='store_true', default=False,
                        help='Run iterative warp adaption (Sec. 3.3) before inference')
    parser.add_argument('--adapt-iters', type=int, default=50,
                        help='Max iterations for warp adaption (default: 50)')
    parser.add_argument('--adapt-lr', type=float, default=1e-4,
                        help='Learning rate for warp adaption (default: 1e-4)')
    parser.add_argument('--orb', action='store_true', default=False,
                        help='Augment TPS control points with ORB keypoint correspondences at inference')
    parser.add_argument('--grid_h', type=int, default=None)
    parser.add_argument('--grid_w', type=int, default=None)
    parser.add_argument('--orb_max_ctrl_pts', type=int, default=0,
                        help='Cap on ORB control points added to TPS (0 = no cap). '
                             'Should match the value used during training.')
    args = parser.parse_args()

    # Resolve NET_H/NET_W from config.json
    global NET_H, NET_W
    cfg_path = os.path.join(os.path.dirname(args.weights), 'config.json')
    _cfg_args = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            _cfg_args = json.load(f).get('args', {})
        NET_H = _cfg_args.get('img_h', NET_H)
        NET_W = _cfg_args.get('img_w', NET_W)
    if args.img_h: NET_H = args.img_h
    if args.img_w: NET_W = args.img_w
    grid_h = args.grid_h or _cfg_args.get('grid_h', 13)
    grid_w = args.grid_w or _cfg_args.get('grid_w', 13)

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    name_l, name_r = ADJACENT_PAIRS[args.pair]
    print(f'Pair {args.pair}: {name_l} -> {name_r}  |  NET {NET_H}x{NET_W}')

    with open(args.frames) as f:
        frame = json.load(f)[args.frame]

    cams  = load_calib(args.calib)
    cam_l = cams[name_l]
    cam_r = cams[name_r]

    img_l_bgr = cv2.imread(frame[name_l])
    img_r_bgr = cv2.imread(frame[name_r])

    # -- Load model --------------------------------------------------------
    if args.lidar:
        from udis_pp_lidar import UDISppLidar
        model = UDISppLidar(img_h=NET_H, img_w=NET_W,
                            grid_h=grid_h, grid_w=grid_w).to(device)
    else:
        from udis_pp import UDISpp
        _warp_compose = _cfg_args.get('warp_compose', False)
        model = UDISpp(img_h=NET_H, img_w=NET_W,
                       warp_compose=_warp_compose,
                       orb_max_ctrl_pts=args.orb_max_ctrl_pts,
                       grid_h=grid_h, grid_w=grid_w).to(device)

    sd = torch.load(args.weights, map_location=device)
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f'Loaded: {args.weights}')

    if args.cylindrical:
        # -- Cylindrical mode (full-image, matches training) ---------------
        f_cyl = float(cam_l['fx'])
        map_x_l, map_y_l = build_cylindrical_remap(cam_l, f_cyl)
        map_x_r, map_y_r = build_cylindrical_remap(cam_r, f_cyl)

        cyl_l = cv2.remap(img_l_bgr, map_x_l, map_y_l,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cyl_r = cv2.remap(img_r_bgr, map_x_r, map_y_r,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        l_crop, r_crop = compute_cylindrical_overlap(cam_l, cam_r, f_cyl)
        if l_crop is None:
            print('ERROR: no cylindrical overlap found'); return
        lu0, lu1, lv0, lv1 = l_crop
        ru0, ru1, rv0, rv1 = r_crop

        # Pre-align cyl_r to cyl_l's canvas (matches dataset.py cylindrical_full branch)
        canvas_offset = int(round((lu0 - ru0 + lu1 - ru1) / 2))
        H_cyl, W_cyl  = cyl_l.shape[:2]
        aligned_r     = np.zeros_like(cyl_l)
        if canvas_offset >= 0:
            src_end = min(W_cyl - canvas_offset, cyl_r.shape[1])
            if src_end > 0:
                aligned_r[:, canvas_offset:canvas_offset + src_end] = cyl_r[:, :src_end]
        else:
            r_start = -canvas_offset
            dst_end = min(W_cyl, cyl_r.shape[1] - r_start)
            if dst_end > 0:
                aligned_r[:, :dst_end] = cyl_r[:, r_start:r_start + dst_end]

        # Feed full images to model -- same as training (full_image=True, fixed_H=I)
        t_l = bgr_to_tensor(cyl_l,    NET_H, NET_W).unsqueeze(0).to(device)
        t_r = bgr_to_tensor(aligned_r, NET_H, NET_W).unsqueeze(0).to(device)
        ln  = imagenet_norm(t_l.squeeze(0)).unsqueeze(0).to(device)
        rn  = imagenet_norm(t_r.squeeze(0)).unsqueeze(0).to(device)

        I3 = torch.eye(3, device=device).unsqueeze(0)
        if args.adapt:
            model.train()
            model._detach_backbone = True
            out, snaps = model.adapt(
                ln, rn, t_l, t_r,
                max_iter=args.adapt_iters,
                tau=1e-4,
                lr=args.adapt_lr,
                save_iters=[0, args.adapt_iters // 2, args.adapt_iters],
                fixed_H=I3,
                full_image=True,
            )
            model.eval()
            model._detach_backbone = False
            for it, snap in sorted(snaps.items()):
                wr = snap['img_wr_local']
                v  = snap['valid_tps']
                err = ((t_l.cpu() - wr.cpu()).abs() * v.cpu()).sum() / v.cpu().sum().clamp(min=1)
                print(f'  adapt iter {it:3d}  L1={err.item():.5f}')
        else:
            with torch.no_grad():
                out = model(ln, rn, t_l, t_r, fixed_H=I3, full_image=True)

        # stitched covers the full cyl_l canvas (NET resolution -> resize to original)
        stitched_net  = tensor_to_bgr(out['stitched'].squeeze(0))
        stitched_full = cv2.resize(stitched_net, (W_cyl, H_cyl))

        # -- Apply TPS warp to cyl_r via cv2.remap ------------------------
        # tps_grid (B, NET_H, NET_W, 2): normalised source coords in aligned_r space.
        # aligned_r pixel x = (norm+1)/2*(W_cyl-1).
        # cyl_r pixel x = aligned_r pixel x - canvas_offset.
        tps_grid_np   = out['tps_grid'][0].detach().cpu().numpy()  # (NET_H, NET_W, 2)
        tps_grid_full = cv2.resize(tps_grid_np, (W_cyl, H_cyl))   # (H_cyl, W_cyl, 2)

        map_x = (tps_grid_full[:, :, 0] + 1.0) / 2.0 * (W_cyl - 1) - canvas_offset
        map_y = (tps_grid_full[:, :, 1] + 1.0) / 2.0 * (H_cyl - 1)
        warped_cyl_r = cv2.remap(cyl_r,
                                 map_x.astype(np.float32),
                                 map_y.astype(np.float32),
                                 cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

        # Canvas assembly:
        #  [0, lu0)      -- left non-overlap: raw cyl_l
        #  [lu0, lu1)    -- overlap zone: model composition blend
        #  [lu1, lu1+..) -- right non-overlap: raw cyl_r (TPS ~0 far from seam)
        # Note: warped_cyl_r is W_cyl-wide (cyl_l-space remap); right non-overlap
        # extends beyond it, so use raw cyl_r[:, ru1:] directly.
        r_right  = cyl_r[:, ru1:]
        canvas_w = lu1 + r_right.shape[1]
        canvas     = np.zeros((H_cyl, canvas_w, 3), dtype=np.uint8)
        canvas_ref = np.zeros((H_cyl, canvas_w, 3), dtype=np.uint8)

        canvas[:, :lu0]    = cyl_l[:, :lu0]                    # non-overlap: raw left
        canvas[:, lu0:lu1] = stitched_full[:, lu0:lu1]         # overlap: model blend
        canvas[:, lu1:lu1 + r_right.shape[1]] = r_right
        canvas_ref[:, :lu1] = cyl_l[:, :lu1]
        canvas_ref[:, lu1:lu1 + r_right.shape[1]] = r_right

        # Reference: feather blend at overlap
        crop_l    = cyl_l[lv0:lv1, lu0:lu1]
        crop_r_rs = cv2.resize(cyl_r[rv0:rv1, ru0:ru1], (lu1 - lu0, lv1 - lv0))
        feather   = np.linspace(1, 0, lu1 - lu0, dtype=np.float32)[None, :, None]
        ref_blend = (feather * crop_l.astype(np.float32) +
                     (1 - feather) * crop_r_rs.astype(np.float32)
                     ).clip(0, 255).astype(np.uint8)
        canvas_ref[lv0:lv0 + ref_blend.shape[0], lu0:lu0 + ref_blend.shape[1]] = ref_blend

    else:
        # -- Planar calibrated mode (--no-cylindrical) ----------------------------
        # Matches CalibratedAV2PairDataset with use_overlap_crop=True, apply_calib_H=True:
        # 1. Crop overlap strips from full planar images.
        # 2. Warp right crop into left frame via rotation homography H_crop.
        # 3. Feed to model with fixed_H=None (GlobalWarpNet sees identity input).
        # 4. Canvas: left_full | model_blend_in_overlap | right_nonov_warped_to_left_frame
        from compute_overlaps import compute_overlap as _compute_overlap
        l_crop_b, r_crop_b = _compute_overlap(cam_l, cam_r)
        if l_crop_b is None:
            print('ERROR: no planar overlap found'); return
        lu0, lu1, lv0, lv1 = l_crop_b
        ru0, ru1, rv0, rv1 = r_crop_b
        cw = lu1 - lu0
        ch = lv1 - lv0

        # Rotation homography: right_crop pixels -> left_crop pixels
        R_rel = cam_l['R'].T @ cam_r['R']
        Kl = np.array([[cam_l['fx'], 0, cam_l['cx']],
                       [0, cam_l['fy'], cam_l['cy']], [0,0,1]], dtype=np.float64)
        Kr = np.array([[cam_r['fx'], 0, cam_r['cx']],
                       [0, cam_r['fy'], cam_r['cy']], [0,0,1]], dtype=np.float64)
        H_full = Kl @ R_rel @ np.linalg.inv(Kr)
        T_r     = np.array([[1,0,ru0],[0,1,rv0],[0,0,1]], dtype=np.float64)
        T_l_inv = np.array([[1,0,-lu0],[0,1,-lv0],[0,0,1]], dtype=np.float64)
        H_crop  = T_l_inv @ H_full @ T_r

        crop_l = img_l_bgr[lv0:lv1, lu0:lu1]
        crop_r = img_r_bgr[rv0:rv1, ru0:ru1]
        if not args.no_calib_prewarp:
            crop_r = cv2.warpPerspective(
                crop_r, H_crop, (cw, ch),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Optional ORB keypoint matching (run on BGR crops before resize)
        orb_t = None
        if args.orb and not args.lidar:
            crop_l_rgb = cv2.cvtColor(crop_l, cv2.COLOR_BGR2RGB)
            crop_r_rgb = cv2.cvtColor(crop_r, cv2.COLOR_BGR2RGB)
            orb_t = _compute_orb_matches(
                crop_l_rgb, crop_r_rgb, NET_H, NET_W
            ).unsqueeze(0).to(device)  # (1, N, 4)
            n_real = int(((orb_t[0, :, :2] - orb_t[0, :, 2:]).norm(dim=-1) > 1.0).sum().item())
            print(f'  ORB: {n_real} real matches')

        t_l = bgr_to_tensor(crop_l,   NET_H, NET_W).unsqueeze(0).to(device)
        t_r = bgr_to_tensor(crop_r,   NET_H, NET_W).unsqueeze(0).to(device)
        ln  = imagenet_norm(t_l.squeeze(0)).unsqueeze(0).to(device)
        rn  = imagenet_norm(t_r.squeeze(0)).unsqueeze(0).to(device)

        if args.adapt:
            model.train()
            model._detach_backbone = True
            out, snaps = model.adapt(
                ln, rn, t_l, t_r,
                max_iter=args.adapt_iters,
                tau=1e-4,
                lr=args.adapt_lr,
                save_iters=[0, args.adapt_iters // 2, args.adapt_iters],
                fixed_H=None,
                full_image=False,
            )
            model.eval()
            model._detach_backbone = False
            # Print per-snapshot crop alignment error for diagnostics
            for it, snap in sorted(snaps.items()):
                wr = snap['img_wr_local']
                v  = snap['valid_tps']
                err = ((t_l.cpu() - wr.cpu()).abs() * v.cpu()).sum() / v.cpu().sum().clamp(min=1)
                print(f'  adapt iter {it:3d}  L1={err.item():.5f}')
        else:
            with torch.no_grad():
                out = model(ln, rn, t_l, t_r, fixed_H=None, orb_matches=orb_t)

        H_mat = out.get('H_mat')
        if H_mat is not None:
            delta = H_mat[0].detach().cpu().numpy() - np.eye(3)
            print(f'  H deviation from I: max={np.abs(delta).max():.4f}  mean={np.abs(delta).mean():.4f}')

        if args.warp_only:
            # Feather-blend crop_l and img_wr_local to show alignment after warp only
            warp_net = tensor_to_bgr(out['img_wr_local'].squeeze(0))
            warp_r   = cv2.resize(warp_net, (cw, ch)).astype(np.float32)
            left_f   = cv2.resize(crop_l,   (cw, ch)).astype(np.float32)
            alpha    = np.linspace(1, 0, cw, dtype=np.float32)[None, :, None]
            stitched_crop = (alpha * left_f + (1 - alpha) * warp_r).clip(0, 255).astype(np.uint8)
        else:
            stitched_net  = tensor_to_bgr(out['stitched'].squeeze(0))
            stitched_crop = cv2.resize(stitched_net, (cw, ch))

        # Full canvas: extend to lu1 + right-non-overlap width; warp full right image
        # into this wider canvas via H_full so the non-overlap continues the same
        # perspective as the overlap blend (avoids the coordinate-frame jump).
        H_l, W_l = img_l_bgr.shape[:2]
        W_r = img_r_bgr.shape[1]
        canvas_w = lu1 + (W_r - ru1)
        img_r_wide = cv2.warpPerspective(
            img_r_bgr, H_full, (canvas_w, H_l),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        canvas     = np.zeros((H_l, canvas_w, 3), dtype=np.uint8)
        canvas_ref = np.zeros((H_l, canvas_w, 3), dtype=np.uint8)
        canvas[:, :lu1]          = img_l_bgr[:, :lu1]   # raw left (non-overlap + overlap bg)
        canvas[:, lu1:]          = img_r_wide[:, lu1:]   # H_full-warped right non-overlap
        canvas[lv0:lv1, lu0:lu1] = stitched_crop         # model blend replaces overlap zone
        canvas_ref[:, :lu1]      = img_l_bgr[:, :lu1]
        canvas_ref[:, lu1:]      = img_r_wide[:, lu1:]

        # Also save the model output crop directly (UDIS2-style, no canvas assembly)
        crop_r_w = crop_r

    # -- Save outputs ------------------------------------------------------
    mode = 'cyl' if args.cylindrical else 'planar'
    tag  = f'p{args.pair}_{name_l[:8]}_{name_r[:8]}_{mode}'

    cv2.imwrite(os.path.join(args.out, f'{tag}_stitch.jpg'),   canvas)
    cv2.imwrite(os.path.join(args.out, f'{tag}_ref.jpg'),      canvas_ref)
    if not args.cylindrical:
        cv2.imwrite(os.path.join(args.out, f'{tag}_crop.jpg'), stitched_crop)

    # Debug panel: left | right (aligned) | stitched | mask_r
    mask_r_net = out['mask_r'].squeeze().detach().cpu().numpy()  # NET_H x NET_W
    def pnl(img): return cv2.resize(img, (NET_W, NET_H))
    mask_disp = cv2.cvtColor(
        (cv2.resize(mask_r_net, (NET_W, NET_H)) * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR)
    if args.cylindrical:
        panel_l = pnl(cyl_l)
        panel_r = pnl(aligned_r)
        panel_s = pnl(stitched_full)
        panel_cols = [panel_l, panel_r, panel_s, mask_disp]
        labels     = ['left', 'right_aligned', 'stitched', 'mask_R']
    else:
        panel_l = pnl(crop_l)
        panel_s = pnl(stitched_crop)
        if args.lidar:
            panel_r = pnl(crop_r_w)
            panel_cols = [panel_l, panel_r, panel_s, mask_disp]
            labels     = ['left', 'right_aligned', 'stitched', 'mask_R']
        else:
            # 5-column: separate GlobalH warp from TPS warp
            panel_r_g = pnl(tensor_to_bgr(out['img_wr_global'].squeeze(0)))
            panel_r_l = pnl(tensor_to_bgr(out['img_wr_local'].squeeze(0)))
            panel_cols = [panel_l, panel_r_g, panel_r_l, panel_s, mask_disp]
            labels     = ['left', 'right_GlobalH', 'right_TPS', 'stitched', 'mask_R']
    panel = np.hstack(panel_cols)
    for i, lbl in enumerate(labels):
        cv2.putText(panel, lbl, (i*NET_W+4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)
        cv2.putText(panel, lbl, (i*NET_W+4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.imwrite(os.path.join(args.out, f'{tag}_panel.jpg'), panel)

    print(f'Saved to {args.out}/')
    print(f'  {tag}_stitch.jpg  ({canvas.shape[1]}x{canvas.shape[0]})')
    print(f'  {tag}_ref.jpg')
    print(f'  {tag}_panel.jpg')


if __name__ == '__main__':
    main()
