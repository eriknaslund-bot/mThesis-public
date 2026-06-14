"""Compute exact overlap pixel ranges between adjacent AV2 cameras.

Uses the full calibration (intrinsics + extrinsics) to project each camera's
boundary rays into its neighbour's image plane, giving the exact pixel strip
[u_start, u_end] in each camera that overlaps with the adjacent one.

For distant objects (>> baseline ~0.4 m) the baseline shift is negligible
compared to the focal length, so we treat cameras as co-located (pure rotation).
The baseline is corrected by an additional horizontal shift derived from the
known inter-camera translation projected onto the image plane.

Returns a dict:
    { pair_name: { 'left': (u0,u1,v0,v1), 'right': (u0,u1,v0,v1) } }
where (u0,u1,v0,v1) are pixel crop coords in the respective camera image.
"""

import json
import numpy as np


# ring_front_center excluded: different image resolution from all other ring cameras.
CAMERA_ORDER = [
    'ring_rear_left',
    'ring_side_left',
    'ring_front_left',
    'ring_front_right',
    'ring_side_right',
    'ring_rear_right',
]


def quat_to_mat(qw, qx, qy, qz):
    """Unit quaternion -> 3x3 rotation matrix (camera->ego)."""
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if n < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])


def load_calib(calib_json: str) -> dict:
    with open(calib_json) as f:
        raw = json.load(f)
    cams = {}
    for name, v in raw.items():
        if 'ring_' not in name:
            continue
        R = quat_to_mat(v['qw'], v['qx'], v['qy'], v['qz'])
        cams[name] = {
            'R':  R,                      # camera->ego rotation
            't':  np.array([v['tx_m'], v['ty_m'], v['tz_m']]),  # camera position in ego
            'fx': v['fx'], 'fy': v['fy'],
            'cx': v['cx'], 'cy': v['cy'],
            'W':  v['width'], 'H': v['height'],
        }
    return cams


def pixel_to_ray_ego(u, v, cam):
    """Pixel (u,v) -> unit ray direction in ego frame."""
    p_cam = np.array([(u - cam['cx']) / cam['fx'],
                      (v - cam['cy']) / cam['fy'],
                      1.0])
    p_cam /= np.linalg.norm(p_cam)
    return cam['R'] @ p_cam        # ego-frame direction


def ray_ego_to_pixel(ray_ego, cam):
    """Ego-frame direction -> (u, v) pixel in cam. Returns None if behind cam."""
    # Project into camera frame: p_cam = R^T @ ray_ego
    p_cam = cam['R'].T @ ray_ego
    if p_cam[2] <= 0:
        return None
    u = cam['fx'] * p_cam[0] / p_cam[2] + cam['cx']
    v = cam['fy'] * p_cam[1] / p_cam[2] + cam['cy']
    return u, v


def compute_overlap(cam_l, cam_r, n_samples: int = 200):
    """Find overlap pixel strips between adjacent cameras L and R.

    Strategy:
      1. Sample a dense grid of boundary + interior rays from cam_l.
      2. Project each into cam_r.
      3. Keep rays that land inside cam_r -- those are the overlap rays.
      4. Record their u range in both images.
      5. Repeat from cam_r -> cam_l.

    Returns:
        (l_u0, l_u1, l_v0, l_v1) -- crop in cam_l image
        (r_u0, r_u1, r_v0, r_v1) -- crop in cam_r image
        or None if no overlap found.
    """
    Wl, Hl = cam_l['W'], cam_l['H']
    Wr, Hr = cam_r['W'], cam_r['H']

    # Sample a dense grid of (u,v) across cam_l
    us_l = np.linspace(0, Wl-1, n_samples)
    vs_l = np.linspace(0, Hl-1, n_samples)
    ug, vg = np.meshgrid(us_l, vs_l)
    ug, vg = ug.ravel(), vg.ravel()

    l_us_valid, l_vs_valid = [], []
    r_us_valid, r_vs_valid = [], []

    for u_l, v_l in zip(ug, vg):
        ray = pixel_to_ray_ego(u_l, v_l, cam_l)
        proj = ray_ego_to_pixel(ray, cam_r)
        if proj is None:
            continue
        u_r, v_r = proj
        if 0 <= u_r < Wr and 0 <= v_r < Hr:
            l_us_valid.append(u_l)
            l_vs_valid.append(v_l)
            r_us_valid.append(u_r)
            r_vs_valid.append(v_r)

    if len(l_us_valid) < 10:
        return None, None

    l_u0 = int(np.floor(min(l_us_valid)))
    l_u1 = int(np.ceil(max(l_us_valid)))
    l_v0 = int(np.floor(min(l_vs_valid)))
    l_v1 = int(np.ceil(max(l_vs_valid)))

    r_u0 = int(np.floor(min(r_us_valid)))
    r_u1 = int(np.ceil(max(r_us_valid)))
    r_v0 = int(np.floor(min(r_vs_valid)))
    r_v1 = int(np.ceil(max(r_vs_valid)))

    # Clamp to image bounds (exclusive-end indices for Python slicing)
    l_crop = (max(0, l_u0), min(Wl, l_u1), max(0, l_v0), min(Hl, l_v1))
    r_crop = (max(0, r_u0), min(Wr, r_u1), max(0, r_v0), min(Hr, r_v1))
    return l_crop, r_crop


def compute_all_overlaps(calib_json: str) -> dict:
    cams = load_calib(calib_json)
    results = {}

    for i in range(len(CAMERA_ORDER)):
        name_l = CAMERA_ORDER[i]
        name_r = CAMERA_ORDER[(i + 1) % len(CAMERA_ORDER)]
        cam_l  = cams[name_l]
        cam_r  = cams[name_r]

        l_crop, r_crop = compute_overlap(cam_l, cam_r)
        pair_key = f'{name_l[:12]}|{name_r[:12]}'

        if l_crop is None:
            print(f'  {pair_key}: NO OVERLAP FOUND')
            continue

        lu0, lu1, lv0, lv1 = l_crop
        ru0, ru1, rv0, rv1 = r_crop

        l_w = lu1 - lu0
        r_w = ru1 - ru0
        pct_l = 100 * l_w / cam_l['W']
        pct_r = 100 * r_w / cam_r['W']

        print(f'{pair_key}')
        print(f'  {name_l[:20]:20s}  u:[{lu0:4d},{lu1:4d}] v:[{lv0:4d},{lv1:4d}]  '
              f'({l_w}px wide, {pct_l:.0f}% of W={cam_l["W"]})')
        print(f'  {name_r[:20]:20s}  u:[{ru0:4d},{ru1:4d}] v:[{rv0:4d},{rv1:4d}]  '
              f'({r_w}px wide, {pct_r:.0f}% of W={cam_r["W"]})')

        results[i] = {
            'left_cam':  name_l,
            'right_cam': name_r,
            'left_crop': l_crop,   # (u0,u1,v0,v1)
            'right_crop': r_crop,
        }

    return results


if __name__ == '__main__':
    import sys
    calib = sys.argv[1] if len(sys.argv) > 1 \
        else '/home/Erik/mThesis/argo2_data/extracted/calibration.json'
    print(f'Using calibration: {calib}\n')
    compute_all_overlaps(calib)
