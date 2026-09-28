#!/usr/bin/env python3
"""
vgn_mujoco_bridge.py
====================
Bridge between the MuJoCo MMO-700 simulation (Python 3.13) and VGN
(Python 3.10 in vgn_env conda, torch + open3d).

The Problem
-----------
vgn_env packages (torch, open3d) are compiled for Python 3.10. The MuJoCo
environment runs Python 3.13. Mixing them in the same interpreter causes ABI errors.

The Solution
------------
VGN inference runs in a subprocess using vgn_env's Python 3.10 interpreter
via the companion script `vgn_worker.py`. Depth images + TSDF params are
written to a temporary .npz file; results come back as a JSON line on stdout.

Public API
----------
VGNBridge(model_path, tsdf_size, tsdf_resolution)
    .init_tsdf(box_approx_pos)
    .collect_view(model, data, renderer, cam_name, box_approx_pos) -> int
    .run_inference() -> bool
    .best_grasp_world() -> (pos_world, rot_mat_3x3, width_m, score) | None

Helpers
-------
read_lidar(model, data, sensor_name) -> float   # metres (+inf = no obstacle)
scan_arm_configs(n_views) -> list[float]         # shoulder_pan angles for scan
VGN_OK : bool                                    # True if vgn_env Python found
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import mujoco
import numpy as np

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_WORKER_PATH = _HERE / "vgn_worker.py"


def find_vgn_dir() -> Path:
    """VGN checkout (with src/ and data/models/vgn_conv.pth). Override with $VGN_DIR."""
    candidates = [Path(os.environ["VGN_DIR"])] if os.environ.get("VGN_DIR") else []
    candidates += [
        _HERE.parent / "vgn",                                   # porta_ros2_ws/scripts
        _HERE.parent.parent / "porta_ros2_ws" / "vgn",          # steve_ros2_ws/scripts
        *[p / "porta_ros2_ws" / "vgn" for p in _HERE.parents],  # steve_wbc package in the source tree
        Path.home() / "Master_Thesis" / "porta_ros2_ws" / "vgn",
        Path.home() / "porta_ros2_ws" / "vgn",
    ]
    for c in candidates:
        if (c / "src" / "vgn").is_dir():
            return c
    return candidates[0] if candidates else Path.home() / "porta_ros2_ws" / "vgn"


_VGN_DIR = find_vgn_dir()
_VGN_SRC = _VGN_DIR / "src"


def _resolve_vgn_python() -> Path:
    """Dynamically locate the vgn_env Python binary across macOS and Linux."""
    # vgn_env first: the current interpreter usually lacks torch/open3d
    search_candidates = [
        *([Path(os.environ["VGN_PYTHON"])] if os.environ.get("VGN_PYTHON") else []),
        Path("/Users/shrikar/anaconda3/envs/vgn_env/bin/python3"),
        Path("/Users/shrikar/miniconda3/envs/vgn_env/bin/python3"),
        Path.home() / "anaconda3" / "envs" / "vgn_env" / "bin" / "python3",
        Path.home() / "miniconda3" / "envs" / "vgn_env" / "bin" / "python3",
        Path.home() / "anaconda3" / "envs" / "vgn_env" / "bin" / "python",
        Path.home() / "miniconda3" / "envs" / "vgn_env" / "bin" / "python",
        Path("/home/shrikar/anaconda3/envs/vgn_env/bin/python3"),
        Path("/home/shrikar/anaconda3/envs/vgn_env/bin/python"),
        Path(sys.executable),
    ]

    for candidate in search_candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    which_py = shutil.which("python3") or shutil.which("python")
    return Path(which_py) if which_py else search_candidates[1]


_VGN_ENV_PYTHON = _resolve_vgn_python()

DEFAULT_MODEL_PATH = _VGN_DIR / "data" / "models" / "vgn_conv.pth"
DEFAULT_TSDF_SIZE = 0.30        # metres
DEFAULT_TSDF_RESOLUTION = 40    # voxels per side -> ~7.5 mm voxel
MAX_DEPTH = 2.0                 # clip depth beyond this value


def _vgn_cmd(*args: str) -> list:
    """vgn_env interpreter, isolated (-s: no ~/.local site-packages) — a user-site torch
    or numpy (e.g. a broken `pip install --user torch`) would otherwise shadow vgn_env's."""
    return [str(_VGN_ENV_PYTHON), "-s", *args]


def _vgn_env() -> dict:
    """Environment for vgn_env subprocesses: without the ROS 2 / workspace PYTHONPATH,
    whose Python 3.10 packages would be imported in place of vgn_env's."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env

# ── Availability check ────────────────────────────────────────────────────────
VGN_OK = _VGN_ENV_PYTHON.exists() and _WORKER_PATH.exists()
if not VGN_OK:
    log.warning(
        f"VGN unavailable: "
        f"vgn_env Python exists={_VGN_ENV_PYTHON.exists()}, "
        f"vgn_worker.py exists={_WORKER_PATH.exists()}. "
        "Falling back to depth-unproject."
    )


# ==============================================================================
class VGNBridge:
    """
    Manages TSDF depth-view collection and VGN inference via subprocess.

    Parameters
    ----------
    model_path       : Path to vgn_conv.pth weights.
    tsdf_size        : Cube side length (m) for the TSDF volume.
    tsdf_resolution  : Voxels per side (40 -> 7.5 mm voxels).
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        tsdf_size: float = DEFAULT_TSDF_SIZE,
        tsdf_resolution: int = DEFAULT_TSDF_RESOLUTION,
    ):
        self.model_path = Path(model_path)
        self.tsdf_size = tsdf_size
        self.tsdf_resolution = tsdf_resolution
        self.available = VGN_OK and self.model_path.exists()

        self._views: list = []
        self._tsdf_origin: Optional[np.ndarray] = None
        self._grasp_pos_tsdf: Optional[np.ndarray] = None
        self._grasp_rot: Optional[np.ndarray] = None
        self._grasp_width: float = 0.0
        self._grasp_score: float = 0.0
        self._n_candidates: int = 0

        if not VGN_OK:
            print("  [VGN] vgn_env Python not found — fallback: depth-unproject")
        elif not self.model_path.exists():
            print(f"  [VGN] Model not found: {self.model_path} — fallback active")
            self.available = False
        else:
            print(
                f"  [VGN] Ready  model={self.model_path.name}  "
                f"Python={_VGN_ENV_PYTHON}  worker={_WORKER_PATH.name}"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def init_tsdf(self, box_approx_pos: np.ndarray) -> None:
        """Initialise a fresh TSDF centred on box_approx_pos."""
        half = self.tsdf_size / 2.0
        self._tsdf_origin = box_approx_pos.copy() - np.array([half, half, half])
        self._views = []
        self._grasp_pos_tsdf = None
        self._grasp_rot = None
        self._grasp_width = 0.0
        self._grasp_score = 0.0
        self._n_candidates = 0

    def collect_view(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        renderer: mujoco.Renderer,
        cam_name: str,
        box_approx_pos: np.ndarray,
        obj_shift: Optional[np.ndarray] = None,
    ) -> int:
        """
        Render one depth frame from the current arm pose and store it.
        Initialises the TSDF on the first call.
        obj_shift: how far the object has moved since the TSDF was initialised; the camera
        pose is shifted back by it, so a moving object is static in the TSDF (the table only
        slides along its own plane).
        Returns the number of views collected so far (0 if unavailable).
        """
        if not self.available:
            return 0

        if self._tsdf_origin is None:
            self.init_tsdf(box_approx_pos)

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        fovy = float(model.cam_fovy[cam_id])
        cam_pos = data.cam_xpos[cam_id].copy()
        if obj_shift is not None:
            cam_pos -= np.asarray(obj_shift, float)
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()  # world<-cam

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam_name)
        depth = renderer.render().copy().astype(np.float32)
        depth[depth > MAX_DEPTH] = 0.0
        depth[depth < 1e-6] = 0.0

        self._views.append(
            {
                "depth": depth,
                "cam_pos": cam_pos,
                "cam_mat": cam_mat.flatten(),  # flat 9-elem array
                "img_h": renderer.height,
                "img_w": renderer.width,
                "fovy": fovy,
            }
        )
        return len(self._views)

    # ── Environment health check (runs in the background) ────────────────────
    # vgn_env must import torch + open3d and exchange arrays with NumPy; a broken env
    # (e.g. torch built against NumPy 1.x with NumPy 2 installed, or x86_64 wheels that
    # the macOS loader rejects under Rosetta) otherwise errors out or hangs at inference.
    _HEALTH_CHECK = "import numpy, torch, open3d; torch.from_numpy(numpy.zeros(3)); print('VGN_ENV_OK')"

    def start_health_check(self) -> None:
        self._health, self.health_msg = None, ""
        if not self.available:
            return
        self._health = subprocess.Popen(_vgn_cmd("-c", self._HEALTH_CHECK), env=_vgn_env(),
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self._health_t0 = time.monotonic()

    def health(self, timeout: float = 120.0) -> Optional[bool]:
        """True/False once the check is done, None while it is still running."""
        p = getattr(self, "_health", None)
        if p is None:
            return self.available
        if p.poll() is None:
            if time.monotonic() - self._health_t0 < timeout:
                return None
            p.kill()
            out, err = p.communicate()
            ok, self.health_msg = False, f"environment check timed out after {timeout:.0f} s"
        else:
            out, err = p.communicate()
            ok = p.returncode == 0 and "VGN_ENV_OK" in out
            if not ok:
                lines = [l for l in (err or "").strip().splitlines() if l.strip()]
                self.health_msg = lines[-1][:300] if lines else f"exit code {p.returncode}"
        self._health = None
        if not ok:
            self.available = False
            print(f"  [VGN] vgn_env unusable ({_VGN_ENV_PYTHON}): {self.health_msg}\n"
                  "        -> VGN disabled, falling back to the tracked object position")
        return ok

    # ── Inference (non-blocking) ─────────────────────────────────────────────
    def start_inference(self) -> bool:
        """Write the views and start the vgn_env worker. Returns False if it cannot start."""
        if not self.available or not self._views or self._tsdf_origin is None:
            return False
        npz_data: dict = {"n_views": np.array(len(self._views)), "tsdf_origin": self._tsdf_origin}
        for i, v in enumerate(self._views):
            npz_data[f"depth_{i}"] = v["depth"]
            npz_data[f"cam_pos_{i}"] = v["cam_pos"]
            npz_data[f"cam_mat_{i}"] = v["cam_mat"]
            npz_data[f"fovy_{i}"] = np.array(v["fovy"])
            npz_data[f"img_h_{i}"] = np.array(v["img_h"])
            npz_data[f"img_w_{i}"] = np.array(v["img_w"])
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False, prefix="vgn_views_") as nf:
            self._npz_path = nf.name
        np.savez_compressed(self._npz_path, **npz_data)
        try:
            self._proc = subprocess.Popen(
                _vgn_cmd(str(_WORKER_PATH), self._npz_path, str(self.model_path),
                         str(self.tsdf_size), str(self.tsdf_resolution), str(_VGN_SRC)),
                env=_vgn_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError as exc:
            print(f"  [VGN] cannot start worker: {exc}")
            self._cleanup()
            return False
        self._proc_t0 = time.monotonic()
        return True

    def poll_inference(self, timeout: float = 30.0) -> Optional[bool]:
        """None while running; True if a grasp was found, False on failure/timeout."""
        p = getattr(self, "_proc", None)
        if p is None:
            return False
        if p.poll() is None:
            if time.monotonic() - self._proc_t0 < timeout:
                return None
            p.kill()
            p.communicate()
            print(f"  [VGN] inference timed out after {timeout:.0f} s")
            self._cleanup()
            return False
        out, err = p.communicate()
        self._cleanup()
        return self._parse_result(p.returncode, out, err)

    def run_inference(self, timeout: float = 120.0) -> bool:
        """Blocking convenience wrapper (used by the older scripts)."""
        if not self.start_inference():
            return False
        while True:
            r = self.poll_inference(timeout)
            if r is not None:
                return r
            time.sleep(0.05)

    def _cleanup(self) -> None:
        self._proc = None
        try:
            os.unlink(self._npz_path)
        except (OSError, AttributeError):
            pass

    def _parse_result(self, returncode: int, out: str, err: str) -> bool:
        out = (out or "").strip()
        if returncode != 0:
            print(f"  [VGN worker stderr]:\n{(err or '')[-800:]}")
        if not out:
            print("  [VGN] Worker produced no output")
            return False
        try:
            res = json.loads(out.splitlines()[-1])
        except json.JSONDecodeError as exc:
            print(f"  [VGN] JSON parse error: {exc}  raw: {out[:200]}")
            return False
        if res.get("error"):
            print(f"  [VGN worker] Error:\n{res['error']}")
            return False
        self._n_candidates = res.get("n_candidates", 0)
        if not res.get("found"):
            print("  [VGN] No grasp candidates above quality threshold")
            return False
        self._candidates = res.get("candidates") or [
            {"pos_tsdf": res["pos_tsdf"], "rot_flat": res["rot_flat"], "width_m": res["width_m"],
             "score": res["score"]}]
        self._grasp_pos_tsdf = np.array(res["pos_tsdf"])
        self._grasp_rot = np.array(res["rot_flat"]).reshape(3, 3)
        self._grasp_width = float(res["width_m"])
        self._grasp_score = float(res["score"])
        print(f"  [VGN] {self._n_candidates} candidate(s)  best score={self._grasp_score:.3f}")
        return True

    def best_grasp_world(
        self, near: Optional[np.ndarray] = None, xy_radius: float = np.inf, z_window: float = np.inf,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
        """
        Return (pos_world, rot_mat_3x3, width_m, score) or None.

        near / xy_radius / z_window: only candidates whose grasp centre lies within xy_radius
        (horizontally) and z_window (vertically) of `near` (world, TSDF frame) count — so a
        clutter object in the TSDF volume is never chosen instead of the target.
        pos_world : grasp centre in world frame (m)
        rot_mat   : 6-DOF grasp orientation from VGN (3x3 rotation matrix)
        width_m   : predicted finger opening width (m)
        score     : grasp quality score [0, 1]
        """
        if self._grasp_pos_tsdf is None or self._tsdf_origin is None:
            return None
        cands = getattr(self, "_candidates", [])
        best = None
        for c in cands:
            pos = self._tsdf_origin + np.asarray(c["pos_tsdf"])
            if near is not None:
                d = pos - np.asarray(near, float)
                if np.hypot(d[0], d[1]) > xy_radius or abs(d[2]) > z_window:
                    continue
            if best is None or c["score"] > best[3]:
                best = (pos, np.asarray(c["rot_flat"]).reshape(3, 3), float(c["width_m"]), float(c["score"]))
        if near is not None:
            print(f"  [VGN] {sum(1 for _ in cands)} candidate(s), "
                  f"{'best on the target: score %.3f' % best[3] if best else 'none on the target'}")
        return best

    @property
    def n_views(self) -> int:
        return len(self._views)


# ==============================================================================
# Standalone helpers
# ==============================================================================

def scan_arm_configs(n_views: int = 4) -> list:
    """
    Return n_views shoulder_pan angles (rad) evenly spanning [-0.35, +0.35].
    Focuses directly on object for fast TSDF volume capture.
    """
    return list(np.linspace(-0.35, 0.35, n_views))


def read_lidar(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sensor_name: str,
) -> float:
    """
    Read a MuJoCo rangefinder sensor by name.
    Returns distance in metres, or +inf when no intersection detected
    (MuJoCo reports -1 for out-of-range).
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sid < 0:
        return float("inf")
    adr = model.sensor_adr[sid]
    val = float(data.sensordata[adr])
    return val if val > 0.0 else float("inf")
