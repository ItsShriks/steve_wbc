#!/usr/bin/env python3
"""
vgn_worker.py
=============
Standalone VGN inference worker — must be run under vgn_env Python 3.10.

Usage (called by vgn_mujoco_bridge.py via subprocess):
  python3 vgn_worker.py <data.npz> <model.pth> <tsdf_size> <tsdf_res> <vgn_src>

Input npz keys
--------------
  n_views       : int scalar
  tsdf_origin   : (3,) float64
  depth_<i>     : (H, W) float32  metres
  cam_pos_<i>   : (3,) float64    world position of camera
  cam_mat_<i>   : (9,) float64    cam-to-world rotation (row-major 3×3)
  fovy_<i>      : float64 scalar  vertical fov (degrees)        [MuJoCo views]
  intr_<i>      : (4,) float64    fx, fy, cx, cy; cam_mat is then world<-ROS optical frame [real views]
  img_h_<i>     : int scalar
  img_w_<i>     : int scalar

Output: one JSON line to stdout
--------------------------------
  {"found": bool, "pos_tsdf": [x,y,z], "rot_flat": [9 floats],
   "width_m": float, "score": float, "n_candidates": int, "error": null|str}
"""

import json
import sys
import numpy as np
from pathlib import Path


def main():
    npz_path   = sys.argv[1]
    model_path = sys.argv[2]
    tsdf_size  = float(sys.argv[3])
    tsdf_res   = int(sys.argv[4])
    vgn_src    = sys.argv[5]

    # Add VGN source to path
    if vgn_src not in sys.path:
        sys.path.insert(0, vgn_src)

    # ── Mock ROS modules + VGN ROS helpers ───────────────────────────────────
    # detection.py does `from vgn import vis` which chains:
    #   vis.py → vgn.utils.ros_utils → tf2_ros → tf2_py → libtf2.so (missing)
    # Mocking vgn.vis and vgn.utils.ros_utils directly short-circuits the chain.
    from unittest.mock import MagicMock
    for _mod in (
        "rospy", "ros_numpy",
        "tf2_ros", "tf2_py", "tf2_py._tf2_py",
        "geometry_msgs", "geometry_msgs.msg",
        "sensor_msgs", "sensor_msgs.msg",
        "visualization_msgs", "visualization_msgs.msg",
        "vgn.vis",                   # direct mock — prevents vis.py from executing
        "vgn.utils.ros_utils",       # direct mock — prevents ros_utils.py from executing
    ):
        sys.modules.setdefault(_mod, MagicMock())

    import torch
    from vgn.detection import predict, process, select
    from vgn.perception import TSDFVolume, CameraIntrinsic
    from vgn.grasp import from_voxel_coordinates
    from vgn.networks import load_network
    from vgn.utils.transform import Transform

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net        = load_network(Path(model_path), device)
    voxel_size = tsdf_size / tsdf_res

    data       = np.load(npz_path, allow_pickle=False)
    n_views    = int(data["n_views"])
    tsdf_origin = data["tsdf_origin"]   # (3,)

    tsdf = TSDFVolume(tsdf_size, tsdf_res)

    for i in range(n_views):
        depth   = data[f"depth_{i}"].astype(np.float32)
        cam_pos = data[f"cam_pos_{i}"]                      # (3,)
        cam_mat = data[f"cam_mat_{i}"].reshape(3, 3)        # world←cam
        img_h   = int(data[f"img_h_{i}"])
        img_w   = int(data[f"img_w_{i}"])
        if f"intr_{i}" in data:
            # real camera: ROS optical frame (+X right, +Y down, +Z forward) = Open3D's,
            # intrinsics [fx, fy, cx, cy] from camera_info
            fx, fy, cx, cy = (float(v) for v in data[f"intr_{i}"])
        else:
            # MuJoCo camera frame (+X right, +Y up, -Z forward) -> Open3D camera frame
            cam_mat = cam_mat @ np.diag([1.0, -1.0, -1.0])
            fovy = float(data[f"fovy_{i}"])
            fy = 0.5 * img_h / np.tan(np.deg2rad(fovy) / 2.0)
            fx = fy
            cx = img_w / 2.0
            cy = img_h / 2.0
        intrinsic = CameraIntrinsic(img_w, img_h, fx, fy, cx, cy)

        # Build T_cam_tsdf = T_cam_world @ T_world_tsdf
        R_cw = cam_mat.T
        T_cw = np.eye(4)
        T_cw[:3, :3] = R_cw
        T_cw[:3,  3] = -R_cw @ cam_pos

        T_wt = np.eye(4)
        T_wt[:3, 3] = tsdf_origin

        T_ct = T_cw @ T_wt
        extrinsic = Transform.from_matrix(T_ct)

        tsdf.integrate(depth, intrinsic, extrinsic)

    # VGN inference
    tsdf_grid        = tsdf.get_grid()          # (1, 40, 40, 40)
    qual, rot, width = predict(tsdf_grid, net, device)
    qual, rot, width = process(tsdf_grid, qual, rot, width)
    grasps, scores   = select(qual.copy(), rot, width)

    if len(grasps) == 0:
        print(json.dumps({
            "found": False, "n_candidates": 0, "error": None,
        }))
        return

    grasps_m = [from_voxel_coordinates(g, voxel_size) for g in grasps]
    best     = int(np.argmax(scores))
    g        = grasps_m[best]

    print(json.dumps({
        "found":        True,
        "pos_tsdf":     g.pose.translation.tolist(),
        "rot_flat":     g.pose.rotation.as_matrix().flatten().tolist(),
        "width_m":      float(g.width),
        "score":        float(scores[best]),
        "n_candidates": len(grasps),
        # every candidate, so the caller can keep only grasps on its target object
        "candidates":   [{"pos_tsdf": gm.pose.translation.tolist(),
                          "rot_flat": gm.pose.rotation.as_matrix().flatten().tolist(),
                          "width_m": float(gm.width), "score": float(sc)}
                         for gm, sc in zip(grasps_m, scores)],
        "error":        None,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print(json.dumps({
            "found": False, "n_candidates": 0,
            "error": traceback.format_exc(),
        }))
        sys.exit(1)
