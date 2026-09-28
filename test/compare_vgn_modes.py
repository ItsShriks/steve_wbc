#!/usr/bin/env python3
"""
Compare the VGN timing modes of active_perception_wbc in headless MuJoCo.

    python3 src/steve_wbc/test/compare_vgn_modes.py                       # all modes x scenes
    python3 src/steve_wbc/test/compare_vgn_modes.py --modes scan drive --scenes single

  scan   original: stop at the table, 4-view scan, wait for the inference
  async  same scan, inference in the background while the gripper approaches
  drive  views while driving up, inference in the background

Runs are sequential on purpose: the measured VGN inference time is charged to simulation
time, so parallel runs competing for the CPU/GPU would distort the comparison.
Needs the vgn_env conda environment (see vgn_mujoco_bridge.py).
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "steve_wbc" / "active_perception_wbc.py"
MODES = ("scan", "async", "drive")
SCENES = ("single", "clutter", "far")


def run(mode, scene):
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    out = subprocess.run([sys.executable, str(SCRIPT), "--mode", "sim", "--headless",
                          "--scene", scene, "--vgn_mode", mode],
                         capture_output=True, text=True, timeout=900, env=env).stdout
    g = lambda pat, conv=str: (lambda m: conv(m.group(1)) if m else None)(re.search(pat, out, re.M))
    return dict(
        result=g(r"Result: (.+?)  \(final state"),
        t_total=g(r"final state \w+, t=([\d.]+) s\)", float),
        waited=g(r"robot waited ([\d.]+) s", float),
        views=g(r"VGN \[\w+\]: views (.+?)(?: \||$)"),
        vgn=g(r"\| result (.+?)$", lambda s: s.strip()),
        t_ready=g(r"inference t=[\d.]+->([\d.]+)s", float),
        base=g(r"Min base clearance: ([\d.]+) cm", float),
        contacts="[COLLISION]" in out,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--scenes", nargs="+", choices=SCENES, default=["single", "clutter"],
                   help="default: single clutter (far is paused)")
    a = p.parse_args()

    rows = []
    for scene in a.scenes:
        for mode in a.modes:
            print(f"running {scene:8s} {mode:6s} ...", flush=True)
            rows.append((scene, mode, run(mode, scene)))

    fmt = lambda v, f="{:.1f}": "-" if v is None else f.format(v)
    print(f"\n{'scene':8s} {'mode':6s} {'result':30s} {'total':>7s} {'waited':>7s} {'VGN ready':>9s} "
          f"{'base clr':>8s}  {'views':14s} VGN result")
    for scene, mode, r in rows:
        res = (r["result"] or "?") + (" + CONTACT" if r["contacts"] else "")
        print(f"{scene:8s} {mode:6s} {res:30s} {fmt(r['t_total']):>6s}s {fmt(r['waited']):>6s}s "
              f"{fmt(r['t_ready']):>8s}s {fmt(r['base']):>6s}cm  {(r['views'] or '-'):14s} {r['vgn'] or '-'}")
    print("\ntotal: time to finish the task | waited: robot held at the table for VGN (scan, "
          "wait for inference, hover) | VGN ready: sim time the grasp result became usable")


if __name__ == "__main__":
    main()
