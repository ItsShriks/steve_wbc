"""Where the gripper camera (the D415) sits on the flange, per mount: the calibration file
gripper_camera_calibrator writes and gripper_camera.launch.py publishes, and the mount's design pose
(steve_grasping tool config `camera:`) used until it is calibrated.

    ability_hand   the D415 on the Ability Hand's wrist collar beside the palm -> config/gripper_camera_ability_hand.yaml
    camera_only    the D415 on a post under the flange (2026-10-14 .. 11-04)  -> config/gripper_camera.yaml
"""

from pathlib import Path

MOUNTS = ("ability_hand", "camera_only")
DEFAULT_MOUNT = "ability_hand"
PARENT, CHILD = "ur5etool0", "gripper_camera_color_optical_frame"


def calibration_name(mount):
    if mount not in MOUNTS:
        raise ValueError(f"camera mount must be one of {MOUNTS}, got '{mount}'")
    return "gripper_camera" if mount == "camera_only" else f"gripper_camera_{mount}"


def calibration_files(mount, ws_src=None, share=None):
    """Candidate calibration files of `mount`: the package SOURCE folder (where the calibrator writes it, as
    table models), then the installed share folder."""
    name = f"{calibration_name(mount)}.yaml"
    return [Path(d) / "config" / name for d in (ws_src, share) if d is not None]


def nominal(mount):
    """The mount's design pose from its steve_grasping tool config: dict(parent, child, xyz, quat_xyzw), or None."""
    try:
        from steve_grasping.tools import load_tool
        cam = (load_tool(mount).config or {}).get("camera")
    except Exception:
        return None
    if not cam or "xyz" not in cam or "quat_xyzw" not in cam:
        return None
    return {"parent": PARENT, "child": CHILD, "xyz": [float(v) for v in cam["xyz"]],
            "quat_xyzw": [float(v) for v in cam["quat_xyzw"]]}
