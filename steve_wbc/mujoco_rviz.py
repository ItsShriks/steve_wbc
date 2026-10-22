"""
mujoco_rviz.py — show the MuJoCo simulation in RViz (active_perception_wbc --rviz, launch rviz:=true).

The MuJoCo mode is a plain Python loop without ROS: RViz had nothing to show. This publishes what MuJoCo
simulates, not the URDF — so the tool on the flange (Ability Hand, Robotiq, Dex5, camera only), the table,
the obstacles and the object appear exactly as simulated:

  /tf                       world -> mj/<body> for every body with visible geometry, at --rviz_rate
  /mujoco/scene             MarkerArray (latched, once): every visible geom (MuJoCo groups 0-2) in its
                            body's frame — boxes, spheres, cylinders, capsules, planes, meshes as triangles
  /mujoco/wbc               MarkerArray: tracked object estimate, its predicted path, the grasp goal,
                            the EE reference and the EE, the state as text

Fixed frame in RViz: `world` (= MuJoCo's world = odom). Wall-clock stamps (the sim runs ~real time).
"""

import time

import numpy as np

PREFIX = "mj/"
VISIBLE_GROUPS = (0, 1, 2)   # MuJoCo's viewer default; collision hulls (3), sites (4/5) stay hidden
MAX_MESH_TRIS = 3000         # meshes with more triangles are thinned (the scene stays ~2 MB, RViz smooth)


class MujocoRviz:
    def __init__(self, model, rate=30.0, node_name="mujoco_rviz"):
        import mujoco
        import rclpy
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from tf2_ros import TransformBroadcaster
        from visualization_msgs.msg import MarkerArray
        self.mujoco, self.m = mujoco, model
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(node_name)
        self.tf = TransformBroadcaster(self.node)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_scene = self.node.create_publisher(MarkerArray, "/mujoco/scene", latched)
        self.pub_wbc = self.node.create_publisher(MarkerArray, "/mujoco/wbc", 10)
        self.period, self.last = 1.0 / rate, -np.inf
        self.bodies = sorted({int(model.geom_bodyid[g]) for g in range(model.ngeom) if self._visible(g)} - {0})
        self.bname = lambda b: PREFIX + (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}")
        self.pub_scene.publish(self._scene())
        print(f"[rviz] MuJoCo -> RViz: {len(self.bodies)} body frames on /tf (fixed frame 'world'), scene on "
              "/mujoco/scene, WBC markers on /mujoco/wbc")

    # ── scene (once) ────────────────────────────────────────────────────────
    def _visible(self, g):
        m = self.m
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        return int(m.geom_group[g]) in VISIBLE_GROUPS and rgba[3] > 0.01

    def _scene(self):
        from visualization_msgs.msg import Marker, MarkerArray
        m, T = self.m, self.mujoco.mjtGeom
        out = MarkerArray()
        mid = 0
        for g in range(m.ngeom):
            if not self._visible(g):
                continue
            b = int(m.geom_bodyid[g])
            frame = "world" if b == 0 else self.bname(b)
            rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
            s, t = m.geom_size[g], m.geom_type[g]
            parts = []  # (marker type, local pos, local quat wxyz, scale)
            if t == T.mjGEOM_BOX:
                parts.append((Marker.CUBE, m.geom_pos[g], m.geom_quat[g], 2 * s))
            elif t == T.mjGEOM_SPHERE:
                parts.append((Marker.SPHERE, m.geom_pos[g], m.geom_quat[g], [2 * s[0]] * 3))
            elif t == T.mjGEOM_ELLIPSOID:
                parts.append((Marker.SPHERE, m.geom_pos[g], m.geom_quat[g], 2 * s))
            elif t in (T.mjGEOM_CYLINDER, T.mjGEOM_CAPSULE):
                parts.append((Marker.CYLINDER, m.geom_pos[g], m.geom_quat[g], [2 * s[0], 2 * s[0], 2 * s[1]]))
                if t == T.mjGEOM_CAPSULE:  # + the two half spheres at the ends of the axis (geom z)
                    R = np.zeros(9)
                    self.mujoco.mju_quat2Mat(R, m.geom_quat[g])
                    ax = R.reshape(3, 3)[:, 2] * s[1]
                    for e in (ax, -ax):
                        parts.append((Marker.SPHERE, m.geom_pos[g] + e, m.geom_quat[g], [2 * s[0]] * 3))
            elif t == T.mjGEOM_PLANE:
                sx, sy = (s[0] or 5.0), (s[1] or 5.0)
                parts.append((Marker.CUBE, m.geom_pos[g], m.geom_quat[g], [2 * sx, 2 * sy, 0.001]))
            elif t == T.mjGEOM_MESH:
                parts.append(("mesh", m.geom_pos[g], m.geom_quat[g], int(m.geom_dataid[g])))
            for typ, pos, quat, scale in parts:
                mk = Marker()
                mk.header.frame_id, mk.ns, mk.id = frame, "mujoco", mid
                mid += 1
                mk.action = Marker.ADD
                mk.frame_locked = True
                mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = (float(x) for x in pos)
                mk.pose.orientation.w, mk.pose.orientation.x, mk.pose.orientation.y, mk.pose.orientation.z = \
                    (float(x) for x in quat)
                mk.color.r, mk.color.g, mk.color.b, mk.color.a = (float(x) for x in rgba)
                if typ == "mesh":
                    mk.type = Marker.TRIANGLE_LIST
                    mk.scale.x = mk.scale.y = mk.scale.z = 1.0
                    mk.points = self._mesh_points(scale)
                    if not mk.points:
                        continue
                else:
                    mk.type = typ
                    mk.scale.x, mk.scale.y, mk.scale.z = (float(max(x, 1e-4)) for x in scale)
                out.markers.append(mk)
        return out

    def _mesh_points(self, mesh_id):
        from geometry_msgs.msg import Point
        m = self.m
        va, vn = m.mesh_vertadr[mesh_id], m.mesh_vertnum[mesh_id]
        fa, fn = m.mesh_faceadr[mesh_id], m.mesh_facenum[mesh_id]
        V = m.mesh_vert[va:va + vn]
        F = m.mesh_face[fa:fa + fn]
        if fn > MAX_MESH_TRIS:
            F = F[:: int(np.ceil(fn / MAX_MESH_TRIS))]
        pts = []
        for tri in F:
            for i in tri:
                p = Point()
                p.x, p.y, p.z = (float(x) for x in V[i])
                pts.append(p)
        return pts

    # ── every tick (rate limited) ───────────────────────────────────────────
    def update(self, data, wbc=None):
        now = time.monotonic()
        if now - self.last < self.period:
            return
        self.last = now
        from geometry_msgs.msg import TransformStamped
        stamp = self.node.get_clock().now().to_msg()
        tfs = []
        for b in self.bodies:
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, "world", self.bname(b)
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = \
                (float(x) for x in data.xpos[b])
            q = data.xquat[b]
            t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = \
                (float(x) for x in q)
            tfs.append(t)
        self.tf.sendTransform(tfs)
        if wbc:
            self.pub_wbc.publish(self._wbc_markers(wbc, stamp))

    def _wbc_markers(self, w, stamp):
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker, MarkerArray
        out = MarkerArray()

        def mk(i, typ, rgba, scale):
            k = Marker()
            k.header.frame_id, k.header.stamp, k.ns, k.id, k.type, k.action = "world", stamp, "wbc", i, typ, Marker.ADD
            k.pose.orientation.w = 1.0
            k.color.r, k.color.g, k.color.b, k.color.a = rgba
            k.scale.x, k.scale.y, k.scale.z = scale
            return k

        def pt(p):
            q = Point()
            q.x, q.y, q.z = (float(x) for x in p)
            return q

        items = [("object", 0, (1.0, 0.6, 0.0, 0.6), 0.07), ("goal", 1, (0.0, 0.9, 0.2, 0.9), 0.025),
                 ("ee_ref", 2, (0.2, 0.4, 1.0, 0.9), 0.02), ("ee", 3, (1.0, 1.0, 1.0, 0.9), 0.015)]
        for key, i, rgba, d in items:
            k = mk(i, Marker.SPHERE, rgba, (d, d, d))
            if w.get(key) is None:
                k.action = Marker.DELETE
            else:
                k.pose.position = pt(w[key])
            out.markers.append(k)
        path = w.get("path")
        k = mk(4, Marker.LINE_STRIP, (1.0, 0.6, 0.0, 0.8), (0.006, 0.0, 0.0))
        if path is not None and len(path) > 1:
            k.points = [pt(p) for p in path]
        else:
            k.action = Marker.DELETE
        out.markers.append(k)
        k = mk(5, Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.08))
        k.text = w.get("text", "")
        k.pose.position = pt(w.get("text_at", (0.0, 0.0, 1.8)))
        out.markers.append(k)
        return out

    def close(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass
