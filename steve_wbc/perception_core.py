#!/usr/bin/env python3
"""
perception_core.py
==================
Stateless perception helpers for the MMO-700 camera pipeline.

  render_rgbd(renderer, model, data, cam_name) → (rgb H×W×3, depth H×W)
  detect_red_object(rgb, cfg)                  → (found, cx, cy, mask)
  pixel_to_bearing(cx, cy, fovy, img_w, img_h) → (h_angle, v_angle)  [rad]
  unproject_pixel_to_world(model, data, ...)   → (pt_world, raw_depth)

All simulation stepping is the caller's responsibility.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import mujoco

IMG_H = 480
IMG_W = 640


@dataclass
class PerceptionConfig:
    """Tunable parameters for the active-perception pipeline."""

    # Red-object detector
    # Stage 1 — absolute floor: R channel must exceed this
    red_r_min:     int   = 120    # lower than before to catch shadowed faces

    # Stage 2 — ratio: R/(G+1) and R/(B+1) must both exceed this
    # rgba="0.85 0.15 0.10" → R/G ≈ 5.7, R/B ≈ 8.5 in the rendered image.
    # Setting 2.5 is tight enough to reject orange/pink/specular, loose enough
    # for shadowed or slightly de-saturated surfaces.
    red_ratio_min: float = 2.5

    min_pixels:    int   = 30     # minimum blob size to count as a detection

    # Wrist scan sweep (velocity-based shoulder_pan)
    scan_start:   float = -0.8    # shoulder_pan lower bound (rad)
    scan_end:     float =  0.8    # shoulder_pan upper bound (rad)
    scan_vel:     float =  0.35   # sweep speed (rad/s)
    render_every: int   = 20      # render every N sim steps during scan

    # Visual servo
    servo_gain_pan:     float = 0.006   # rad/s per horizontal pixel error
    servo_gain_lift:    float = 0.004   # rad/s per vertical pixel error
    servo_pix_thr:      float = 18.0    # convergence threshold (px)
    servo_render_every: int   = 6       # render every N steps during servo

    # Localisation
    box_half_height: float = 0.02   # top-surface → centre correction (m)

    # Camera name for wrist perception
    cam_name: str = "wrist_cam"


# ── Low-level primitives ───────────────────────────────────────────────────────

def render_rgbd(
    renderer: mujoco.Renderer,
    model:    mujoco.MjModel,
    data:     mujoco.MjData,
    cam_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rgb H×W×3 uint8, depth H×W float32) for *cam_name*."""
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    depth = renderer.render().copy()
    return rgb, depth


def render_rgb_only(
    renderer: mujoco.Renderer,
    model:    mujoco.MjModel,
    data:     mujoco.MjData,
    cam_name: str,
) -> np.ndarray:
    """Return rgb H×W×3 uint8 for *cam_name* (no depth pass)."""
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    return renderer.render().copy()


def detect_red_object(
    rgb: np.ndarray,
    cfg: PerceptionConfig = PerceptionConfig(),
) -> Tuple[bool, float, float, np.ndarray]:
    """
    Detect red pixels robustly using absolute-floor + ratio tests.

    The box material is rgba="0.85 0.15 0.10", so rendered R ≈ 200-230,
    G ≈ 25-50, B ≈ 15-40 under typical MuJoCo lighting.

    Two-stage mask:
      Stage 1 (absolute)  — R channel is meaningfully bright
      Stage 2 (ratios)    — R dominates both G and B by a factor > red_ratio_min
                            This rejects:
                              • white specular highlights  (all channels high)
                              • dark shadows              (all channels low)
                              • orange / yellow           (G too high vs R)
                              • pink / magenta            (B too high vs R)

    Returns (found, cx_pixel, cy_pixel, mask).
    """
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    # Stage 1: R must be above the absolute floor
    bright_red = r > cfg.red_r_min

    # Stage 2: ratio checks — R/(G+1) and R/(B+1) must both exceed red_ratio_min
    dominant_over_g = (r / (g + 1.0)) > cfg.red_ratio_min
    dominant_over_b = (r / (b + 1.0)) > cfg.red_ratio_min

    mask = bright_red & dominant_over_g & dominant_over_b

    ys, xs = np.where(mask)
    if len(ys) < cfg.min_pixels:
        return False, 0.0, 0.0, mask
    return True, float(np.median(xs)), float(np.median(ys)), mask


def pixel_to_bearing(
    cx: float, cy: float,
    fovy: float,
    img_w: int = IMG_W,
    img_h: int = IMG_H,
) -> Tuple[float, float]:
    """
    Convert a pixel centroid to horizontal and vertical bearing angles (rad).

    Positive h_angle → object is to the right of centre.
    Positive v_angle → object is below centre (lower in image).

    Parameters
    ----------
    cx, cy : pixel column / row of detected centroid
    fovy   : camera vertical field-of-view (degrees)

    Returns
    -------
    h_angle, v_angle : float (radians)
    """
    f = 0.5 * img_h / np.tan(np.deg2rad(fovy) / 2.0)
    h_angle = np.arctan2(cx - img_w / 2.0, f)
    v_angle = np.arctan2(cy - img_h / 2.0, f)
    return float(h_angle), float(v_angle)


def unproject_pixel_to_world(
    model:     mujoco.MjModel,
    data:      mujoco.MjData,
    cam_name:  str,
    px:        float,
    py:        float,
    depth_img: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Un-project pixel (px col, py row) with depth into world 3-D.

    Returns (pt_world ndarray(3) or None, raw_depth float).
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fovy   = model.cam_fovy[cam_id]
    f      = 0.5 * IMG_H / np.tan(np.deg2rad(fovy) / 2.0)
    cx_img = IMG_W / 2.0
    cy_img = IMG_H / 2.0

    d = float(depth_img[int(round(py)), int(round(px))])
    if d <= 1e-6:
        return None, d

    x_cam = (px - cx_img) * d / f
    y_cam = (cy_img - py) * d / f   # flip row → +Y up
    z_cam = -d                       # camera looks along −Z

    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    pt_world = cam_pos + cam_mat @ np.array([x_cam, y_cam, z_cam])
    return pt_world, d
