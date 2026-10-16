"""Human-retargeted play: every motion is a human LIBERO demo segment, rigidly
re-anchored onto objects / fixtures of a new random scene and blended into the
robot's current pose (see retarget.py). Exploration comes from the choice of
what to interact with, which human segment to borrow (including segments from
similar object categories and segments in which the human failed), perturbed
targets, and retries."""
import json
import os

import numpy as np

from retarget import plan, execute, choose_transform, wrap, correct_to, match_start, Transform
from play_controller import EpisodeTooLong
from transit import get_db, _sym_angle, task_base

HERE = os.path.dirname(os.path.abspath(__file__))
GEOM = json.load(open(os.path.join(HERE, "object_geometry.json")))

GROUPS = {
    "bowl": ["akita_black_bowl", "white_bowl", "glazed_rim_porcelain_ramekin"],
    "plate": ["plate"],
    "box": ["butter", "cream_cheese", "chocolate_pudding", "cookies", "popcorn", "macaroni_and_cheese"],
    "can": ["alphabet_soup", "tomato_sauce"],
    "bottle": ["ketchup", "bbq_sauce", "salad_dressing", "new_salad_dressing", "milk",
               "orange_juice", "wine_bottle"],
    "mug": ["porcelain_mug", "white_yellow_mug", "red_coffee_mug"],
    "book": ["black_book", "yellow_book"],
    "pot": ["moka_pot"],
    "pan": ["chefmate_8_frypan", "chefmate8_frypan"],
    "basket": ["basket"],
    "container": ["bowl_drainer", "wooden_tray"],
}
GROUP_OF = {c: g for g, cs in GROUPS.items() for c in cs}
SYM = {"bowl": "inf", "plate": "inf", "can": "inf", "box": 2, "bottle": 2, "book": 2,
       "mug": None, "pot": None, "pan": None, "basket": 2, "container": 2}
RECEPTACLE_GROUPS = {"plate", "bowl", "basket", "container"}
MAX_YAW = {"mug": np.radians(70), "pot": np.radians(40), "pan": np.radians(40)}


def quat_to_mat(q):  # mujoco wxyz
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rel_yaw(cat, quat):
    """Yaw of the object relative to its canonical spawn orientation."""
    if cat not in GEOM:
        return 0.0
    Rr = quat_to_mat(quat) @ quat_to_mat(GEOM[cat]["quat"]).T
    return float(np.arctan2(Rr[1, 0], Rr[0, 0]))


def top_center(cat, pos, quat):
    """World top-center of the object's bbox, from canonical geometry."""
    if cat not in GEOM:
        return np.asarray(pos, float)
    g = GEOM[cat]
    lo, hi = np.array(g["lo"]), np.array(g["hi"])
    Rr = quat_to_mat(quat) @ quat_to_mat(g["quat"]).T
    c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]])
    return np.asarray(pos, float) + Rr @ c


def rotz_np(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])


def bottom_offset(cat):
    return -GEOM[cat]["lo"][2] if cat in GEOM else 0.0


class HumanPlay:
    def __init__(self, pc, tasks, segs, rng, p_explore=0.25, clip_topk=3, quality=None):
        self.pc, self.tasks, self.rng = pc, tasks, rng
        self.p_explore = p_explore
        self.clip_topk = clip_topk
        self.quality = quality or {}
        self.approach = [s for s in segs if s["type"] == "grasp_approach"]
        self.transport = {(s["task"], s["demo"], s["t0"]): s for s in segs if s["type"] == "transport"}
        self.transport_list = list(self.transport.values())
        self.artic = [s for s in segs if s["type"] == "articulation"]
        self.segments = []
        pc.drawers = pc._find_drawers(("top", "middle", "bottom"))
        self.db = get_db(tasks)
        self.base = np.array(pc.robot.base_pos, dtype=float)
        self.n_transit = 0
        self.last_entry = {}
        self._bases = {}
        self.fixture_boxes = self._fixture_boxes()

    # ----------------------------------------------------------- scene info
    def cat(self, name):
        return self.pc.obj_category(name)

    def obj_pose(self, name):
        m, d = self.pc.m, self.pc.d
        b = m.body(self.pc.e.objects_dict[name].root_body).id
        return d.xpos[b].copy(), d.xquat[b].copy()

    def upright(self, name):
        cat = self.cat(name)
        if cat not in GEOM:
            return True
        _, q = self.obj_pose(name)
        Rr = quat_to_mat(q) @ quat_to_mat(GEOM[cat]["quat"]).T
        return Rr[2, 2] > np.cos(np.radians(30))

    def _fixture_boxes(self):
        m, d, e = self.pc.m, self.pc.d, self.pc.e
        boxes = []
        for fname, fx in e.fixtures_dict.items():
            if fname.endswith("table") or fname == "floor":
                continue
            bid = m.body(fx.root_body).id
            pts = []
            for g in range(m.ngeom):
                if m.body_rootid[m.geom_bodyid[g]] == bid and m.geom_contype[g] != 0:
                    c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                    cs = c + h * np.array([[a, b, cc] for a in (-1, 1) for b in (-1, 1) for cc in (-1, 1)])
                    pts.append(cs @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
            if pts:
                pts = np.concatenate(pts)
                boxes.append((fname, pts.min(0), pts.max(0)))
        return boxes

    def feasible(self, P, allow_fixture=None, margin=0.01):
        tz = self.pc.table_z
        base = self.pc.robot.base_pos if hasattr(self.pc.robot, "base_pos") else np.array([-0.66, 0, tz])
        if (P[:, 2] < tz + 0.003).any():
            return False
        if (np.linalg.norm(P[:, :2] - np.asarray(base)[:2], axis=1) > 0.95).any():
            return False
        if (P[:, 2] > tz + 0.75).any():
            return False
        for fname, lo, hi in self.fixture_boxes:
            if allow_fixture and fname == allow_fixture:
                continue
            inside = np.all((P > lo + margin) & (P < hi - margin), axis=1)
            if inside.any():
                return False
        return True

    def seg_data(self, s, t0=None, t1=None):
        d = self.tasks[s["task"]]["demos"][s["demo"]]
        t0 = s["t0"] if t0 is None else t0
        t1 = s["t1"] if t1 is None else t1
        return {k: d[k][t0:t1] for k in ["eef_pos", "eef_mat", "actions"]}, d

    # ------------------------------------------------------------ primitives
    # ------------------------------------------------------------ selection
    def _donor_cats(self, cat):
        g = GROUP_OF.get(cat)
        return [cat], [c for c in GROUPS.get(g, []) if c != cat]

    def _cur_state(self):
        pc = self.pc
        return pc.eef_pos, getattr(pc, "eef_vel", np.zeros(3)), pc.eef_mat

    def _pick_approach(self, name, want_lifted=True, explore=False, n_cands=100, dest=None):
        """Select (segment, transform, start index, landing) for grasping `name`
        via motion matching. Candidates are scored on: start pose/velocity
        match, warp rate (start offset / remaining path), and -- for full
        pick-and-place -- how well the same demo's transport lands where we
        want (dest = ("table", None) or (kind, goal_xyz))."""
        cat = self.cat(name)
        pos, quat = self.obj_pose(name)
        anchor_dst = top_center(cat, pos, quat)
        yaw_dst = rel_yaw(cat, quat)
        same, group = self._donor_cats(cat)
        pool = same if (self.rng.random() > 0.3 or not group) else group
        need_tr = want_lifted and dest is not None
        def ok(s):
            if s["lifted"] != want_lifted:
                return False
            if need_tr:
                tr = self.transport.get((s["task"], s["demo"], s["t1"]))
                if tr is None or not tr["ends_open"]:
                    return False
                dz = tr["settle_obj_pos"][2] - tr["grasp_obj_pos"][2]
                if dest[0] == "table" and abs(dz) > 0.025:
                    return False
            return True
        cands = [s for s in self.approach if s["cat"] in pool and ok(s)]
        if len(cands) < 5:
            cands += [s for s in self.approach if s["cat"] in same + group and ok(s) and s["cat"] not in pool]
        if not cands:
            return None
        grp = GROUP_OF.get(cat)
        sym = SYM.get(grp, None)
        if explore:
            anchor_dst = anchor_dst + np.r_[self.rng.normal(0, 0.025, 2), self.rng.uniform(-0.01, 0.02)]
        p_cur, v_cur, R_cur = self._cur_state()
        scored = []
        for i in self.rng.permutation(len(cands))[:n_cands]:
            s = cands[i]
            a_src = top_center(s["cat"], s["obj_pos"], s["obj_quat"])
            dy = self._wrist_yaw(rel_yaw(s["cat"], s["obj_quat"]), yaw_dst, sym, grp)
            if dy is None:
                continue
            G = Transform(a_src, anchor_dst, 0.0)
            sd, _ = self.seg_data(s)
            P, Rm = G.pos(sd["eef_pos"]), G.mat(sd["eef_mat"])
            key = len(P) - 1
            ts, cost = match_start(P, Rm, p_cur, v_cur, R_cur, key - 20)
            arc = np.linalg.norm(np.diff(P[ts:], axis=0), axis=1).sum()
            d0 = np.linalg.norm(P[ts] - p_cur)
            cost += 1.5 * d0 / max(arc, 0.05)
            if not self.feasible(P[ts + (key - ts) // 3:]):
                continue
            land = None
            if need_tr:
                tr = self.transport[(s["task"], s["demo"], s["t1"])]
                land = pos + G.L @ (tr["settle_obj_pos"] - tr["grasp_obj_pos"])
                if dest[0] == "table":
                    if not self._valid_landing(name, land):
                        alt = self._nearest_valid(name, land)
                        if alt is None:
                            continue
                        cost += 2.0 * np.linalg.norm(alt[:2] - land[:2])
                        land = alt
                else:
                    cost += 2.0 * np.linalg.norm(dest[1][:2] - land[:2])
                Pt = G.pos(self.seg_data(tr)[0]["eef_pos"])
                if not self.feasible(Pt[3:], allow_fixture=None):
                    cost += 0.5
            q, skip = self._quality("pick", s) if want_lifted else (0.5, False)
            if skip:
                continue
            shift = self._shift(a_src, anchor_dst, s)
            scored.append((cost + 0.03 * abs(dy) + 0.6 * (1 - q) + 0.8 * shift, s, G, ts, land, dy))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0])
        return scored[self.rng.integers(min(len(scored), self.clip_topk))][1:]

    def src_base(self, s):
        b = self._bases.get(s["task"])
        if b is None:
            b = self._bases[s["task"]] = task_base(self.tasks[s["task"]])
        return b

    def _shift(self, a_src, a_dst, s):
        """How far the clip moves relative to the robot base (what matters for
        arm configuration), not in world coordinates."""
        return float(np.linalg.norm((a_dst - self.base) - (a_src - self.src_base(s))))

    def _quality(self, kind, s):
        """Smoothed empirical transfer success of a human clip (task prior +
        clip evidence). Returns (p, skip)."""
        T = self.quality.get("tasks", {}).get(f"{kind}|{s['task']}")
        if T:
            p_t = (T[0] + 1) / (T[1] + 2)
            if T[1] >= 10 and p_t < 0.12:
                return p_t, True
        else:
            p_t = 0.4
        C = self.quality.get("clips", {}).get(f"{kind}|{s['task']}#{s['demo']}")
        p = (C[0] + 4 * p_t) / (C[1] + 4) if C else p_t
        return p, False

    def _wrist_yaw(self, yaw_src, yaw_dst, sym, grp):
        """Wrist yaw offset (about world z) that maps the human's grasp onto the
        rotated object, using the object's rotational symmetry."""
        if sym == "inf":
            return 0.0
        base = wrap(yaw_dst - yaw_src)
        if sym:
            per = 2 * np.pi / sym
            base = min((wrap(base + k * per) for k in range(sym)), key=abs)
        lim = MAX_YAW.get(grp, np.radians(95))
        return float(base) if abs(base) <= lim else None

    def _rotated_grasp_goal(self, s, G, sd, dy):
        """For off-center grasps (handles, rims) rotate the human's grasp offset
        about the object by the wrist yaw; None if the change is negligible."""
        if abs(dy) < 1e-3:
            return None
        a_src = top_center(s["cat"], s["obj_pos"], s["obj_quat"])
        off = sd["eef_pos"][-1] - a_src
        new = G.a_dst + rotz_np(dy) @ off
        old = G.pos(sd["eef_pos"][-1])
        return new if np.linalg.norm(new - old) > 0.004 else None

    def _nearest_valid(self, name, land, tries=40):
        for r in np.linspace(0.03, 0.2, tries // 4):
            for _ in range(4):
                a = self.rng.uniform(-np.pi, np.pi)
                c = land + np.r_[r * np.cos(a), r * np.sin(a), 0]
                if self._valid_landing(name, c):
                    return c
        return None

    def _valid_landing(self, name, land):
        pc = self.pc
        x0, y0 = pc.e.workspace_offset[:2]
        from scene_gen import OBJ_XY
        sc = {"kitchen_table": "kitchen", "living_room_table": "living_room", "study_table": "study"}
        tname = next((f for f in pc.e.fixtures_dict if f.endswith("table")), "kitchen_table")
        (xl, xh), (yl, yh) = OBJ_XY.get(sc.get(tname, "kitchen"))
        if not (xl - 0.02 < land[0] - x0 < xh + 0.02 and yl - 0.02 < land[1] - y0 < yh + 0.02):
            return False
        if self._near_fixture_xy(land[:2], 0.06):
            return False
        for n in pc.names:
            if n == name:
                continue
            p, _ = self.obj_pose(n)
            if np.linalg.norm(p[:2] - land[:2]) < 0.08:
                return False
        return True

    def _near_fixture_xy(self, xy, margin):
        for fname, lo, hi in self.fixture_boxes:
            if np.all(xy > lo[:2] - margin) and np.all(xy < hi[:2] + margin):
                return True
        return False

    def _place_goal(self, name):
        """Exploratory destination: natural table placement (most often), onto
        a receptacle / another object, or into an open drawer."""
        pc = self.pc
        cat = self.cat(name)
        recs = [n for n in pc.names if n != name and GROUP_OF.get(self.cat(n)) in RECEPTACLE_GROUPS
                and self.upright(n)]
        r = self.rng.random()
        if recs and r < 0.3:
            t = recs[self.rng.integers(len(recs))]
            p, q = self.obj_pose(t)
            tc = top_center(self.cat(t), p, q)
            goal = np.r_[tc[:2] + self.rng.normal(0, 0.01, 2), tc[2] + bottom_offset(cat) - 0.01]
            return "on", goal, t
        open_drawers = [dr for dr in pc.drawers if pc.d.qpos[pc.m.jnt_qposadr[dr["jid"]]] < -0.1]
        if open_drawers and r < 0.5:
            dr = open_drawers[self.rng.integers(len(open_drawers))]
            R = pc.d.xmat[dr["bid"]].reshape(3, 3)
            hp = pc.d.geom_xpos[dr["handle"]]
            inward = R[:, 1]
            xy = hp[:2] + inward[:2] * 0.1
            goal = np.r_[xy, hp[2] - 0.03 + bottom_offset(cat)]
            return "in_drawer", goal, f"{dr['fixture']}:{dr['level']}"
        return "table", None, "table"

    def _enter_clip(self, s, G, ts, key, t0_field="t0"):
        """If the robot's state is far from the clip's matched frame, travel to
        the clip's entry with a human free-space transit. Returns start index."""
        pc = self.pc
        sd, _ = self.seg_data(s)
        P, Rm = G.pos(sd["eef_pos"]), G.mat(sd["eef_mat"])
        p_cur, v_cur, R_cur = self._cur_state()
        d0 = np.linalg.norm(P[ts] - p_cur)
        ang = float(_sym_angle(Rm[ts][None], R_cur)[0])
        self.last_entry = dict(d0_direct=float(d0), ang_direct=ang, transit=False)
        if d0 < 0.05 and ang < 0.35:
            return ts
        # entry: first moving frame of the clip (or 0), well before the key frame
        V = np.gradient(P, axis=0) * 20
        te = int(np.argmax(np.linalg.norm(V[:max(1, key - 25)], axis=1) > 0.03)) if key > 30 else 0
        res = self.db.search(p_cur - self.base, v_cur, R_cur, P[te] - self.base, V[te], Rm[te])
        if res is None:
            self.last_entry["why"] = "no_transit"
            return ts
        cost, ci, a, b, ca, cb = res
        direct = d0 + 0.06 * ang
        self.last_entry.update(transit_cost=cost, ca=ca, cb=cb)
        if cost > direct + 0.05 and not (d0 > 0.10 and cb < 0.08):
            self.last_entry["why"] = "transit_worse"
            return ts
        pc.phase = "transit"
        tsd, mirror, hb = self.db.clip_data(ci, a, b)
        Gt = Transform(hb, self.base, 0.0, mirror=mirror)
        pl = plan(tsd, Gt, pc.eef_pos, pc.eef_mat, blend_frac=0.6, end_goal=P[te], end_blend_start=0.2)
        execute(pc, pl)
        self.n_transit += 1
        self.last_entry.update(transit=True, len=int(b - a),
                               entry_err=float(np.linalg.norm(pc.eef_pos - P[te])))
        return te

    # ------------------------------------------------------------ primitives
    def _approach(self, name, want_lifted=True, explore=False, truncate=None):
        pc = self.pc
        sel = self._pick_approach(name, want_lifted=want_lifted, explore=explore)
        if sel is None:
            return None
        s, G, ts, _, dy = sel
        ts = self._enter_clip(s, G, ts, key=s["t1"] - s["t0"])
        sd, d = self.seg_data(s, t0=s["t0"] + ts)
        if truncate is not None:
            n = max(25, int(len(sd["actions"]) * truncate))
            sd = {k: v[:n] for k, v in sd.items()}
        p_start = pc.eef_pos
        pl = plan(sd, G, pc.eef_pos, pc.eef_mat, blend_frac=0.7, key_t=len(sd["actions"]) - 1,
                  wrist_yaw=dy)
        obj_before, q_before = self.obj_pose(name)
        pc.phase = "approach"
        execute(pc, pl)
        n_corr = 0
        if truncate is None:
            n_corr = correct_to(pc, pl["P"][-1], pl["R"][-1], rng=self.rng)
        diag = dict(n_corr=int(n_corr), donor=s["cat"], t_start=int(ts), dyaw=float(dy),
                    d0=float(np.linalg.norm(pl["P"][0] - G.pos(sd["eef_pos"][0]))),
                    key_err=pl.get("key_err", -1))
        return s, G, diag, obj_before

    def p_pick_place(self, name, explore=False, max_regrasps=2):
        pc = self.pc
        nice = name.rsplit("_", 1)[0].replace("_", " ")
        fumble = explore and self.rng.random() < 0.35
        kind, goal, label = self._place_goal(name)
        dest = None if fumble else (("table", None) if kind == "table" else (kind, goal))
        attempts, srcs, diag = 0, [], {}
        obj_start, _ = self.obj_pose(name)
        state = dict(grasped=False)
        target = obj_start
        while True:
            sel = self._pick_approach(name, want_lifted=not fumble, explore=explore and attempts == 0,
                                      dest=dest)
            if sel is None and dest is not None and kind != "table":
                kind, goal, label = "table", None, "table"
                dest = ("table", None)
                sel = self._pick_approach(name, want_lifted=True, dest=dest)
            if sel is None:
                if attempts == 0:
                    return None
                break
            attempts += 1
            s, G, ts, land, dy = sel
            src = f"{s['task']}#{s['demo']}"
            srcs.append(src)
            # scene-context diagnostics for this pick
            _p, _q = self.obj_pose(name)
            _cat = self.cat(name)
            _Rr = quat_to_mat(_q) @ quat_to_mat(GEOM[_cat]["quat"]).T if _cat in GEOM else np.eye(3)
            ctx = dict(tilt=float(np.degrees(np.arccos(np.clip(_Rr[2, 2], -1, 1)))),
                       nearest=float(min([np.linalg.norm(self.obj_pose(n)[0][:2] - _p[:2]) for n in pc.names if n != name] + [1.0])),
                       touched=bool(any(name in (x.get("target") or "") for x in self.segments)))
            ts = self._enter_clip(s, G, ts, key=s["t1"] - s["t0"])
            sd, d = self.seg_data(s, t0=s["t0"] + ts)
            end_goal = self._rotated_grasp_goal(s, G, sd, dy)
            pl = plan(sd, G, pc.eef_pos, pc.eef_mat, blend_frac=0.7, key_t=len(sd["actions"]) - 1,
                      wrist_yaw=dy, end_goal=end_goal, end_blend_start=0.3)
            obj_before, _ = self.obj_pose(name)
            pc.phase = "approach" if attempts == 1 else "regrasp"
            execute(pc, pl)
            n_corr = correct_to(pc, pl["P"][-1], pl["R"][-1], rng=self.rng)
            if attempts == 1:
                diag = dict(n_corr=int(n_corr), donor=s["cat"], t_start=int(ts),
                            shift=self._shift(G.a_src, G.a_dst, s), ctx=ctx, dyaw=float(dy),
                            key_err=pl.get("key_err", -1), key_rot=float(pl.get("key_rot_err", -1)),
                            T=int(pl["T"]), n_transit=self.n_transit, entry=dict(self.last_entry),
                            obj_xy=(obj_before[:2] - pc.e.workspace_offset[:2]).round(3).tolist())
            if fumble:
                t1 = min(s["t1"] + 15, len(d["actions"]))
                sdf = {k: d[k][s["t1"]:t1] for k in ["eef_pos", "eef_mat", "actions"]}
                if len(sdf["actions"]) > 3:
                    pc.phase = "fumble"
                    execute(pc, plan(sdf, G, pc.eef_pos, pc.eef_mat, blend_frac=0.3, wrist_yaw=dy,
                                     wrist_const=True))
                grasped = pc.is_grasped(name)
                if pc.grip > 0:
                    pc.gripper_action(-1.0, 5)
                return dict(primitive="pick_fumble", target=name, success=False, grasped=bool(grasped),
                            language=f"try to pick up the {nice}", source=src, diag=diag)
            tr = self.transport[(s["task"], s["demo"], s["t1"])]
            sd_t, _ = self.seg_data(tr)
            natural = obj_before + G.L @ (tr["settle_obj_pos"] - tr["grasp_obj_pos"])
            target = goal if goal is not None else land
            P_end = G.pos(sd_t["eef_pos"][-1]) + (pc.eef_pos - G.pos(sd_t["eef_pos"][0]))
            end_goal = P_end + (target - natural)
            if goal is None:
                end_goal[2] = P_end[2]  # natural release height on the table
            pl_t = plan(sd_t, G, pc.eef_pos, pc.eef_mat, blend_frac=0.15, end_goal=end_goal,
                        end_blend_start=0.1, wrist_yaw=dy, wrist_const=True)
            z0 = obj_before[2]
            ez0 = pc.eef_pos[2]
            state = dict(grasped=False, abort=False, k=0)

            def check():
                state["k"] += 1
                if not state["grasped"] and pc.is_grasped(name) and self.obj_pose(name)[0][2] > z0 + 0.03:
                    state["grasped"] = True
                if state["grasped"] or attempts > max_regrasps:
                    return False
                width = float(pc.obs["robot0_gripper_qpos"][0] - pc.obs["robot0_gripper_qpos"][1])
                lifted = self.obj_pose(name)[0][2] - z0
                if state["k"] == 14 and width < 0.006:
                    state["abort"] = True  # closed on nothing
                elif state["k"] >= 24 and pc.eef_pos[2] - ez0 > 0.04 and lifted < 0.01:
                    state["abort"] = True  # hand rising without the object
                return state["abort"]

            pc.phase = "transport"
            execute(pc, pl_t, stop_fn=check)
            if state["abort"]:
                # human-like regrasp: let go and go again with a fresh clip
                pc.gripper_action(-1.0, 5)
                continue
            if pc.grip > 0:
                pc.gripper_action(-1.0, 4)
            break
        p_end, _ = self.obj_pose(name)
        grasped = state["grasped"]
        placed = grasped and np.linalg.norm(p_end[:2] - target[:2]) < 0.06
        if kind == "on":
            lang = f"put the {nice} on the {label.rsplit('_', 1)[0].replace('_', ' ')}"
            prim = "pick_place_on"
        elif kind == "in_drawer":
            lang = f"put the {nice} in the {label.split(':')[1]} drawer"
            prim = "pick_place_drawer"
        else:
            lang = f"pick up the {nice} and put it down somewhere else"
            prim = "pick_place"
        diag["attempts"] = attempts
        return dict(primitive=prim, target=f"{name}->{label}", success=bool(placed),
                    grasped=bool(grasped), language=lang, source=srcs[0], sources=srcs,
                    attempts=attempts, diag=diag)

    def p_hover(self, name):
        """Reach toward an object and change one's mind (partial human approach)."""
        r = self._approach(name, want_lifted=True, truncate=self.rng.uniform(0.5, 0.85))
        if r is None:
            return None
        nice = name.rsplit("_", 1)[0].replace("_", " ")
        return dict(primitive="reach_toward", target=name, success=True,
                    language=f"reach toward the {nice}", source=f"{r[0]['task']}#{r[0]['demo']}",
                    diag=r[2])

    def p_articulate(self, kind, want_open, level=None, explore=False):
        pc = self.pc
        m, d = pc.m, pc.d
        if kind == "drawer":
            drs = [dr for dr in pc.drawers if level is None or dr["level"] == level]
            if not drs:
                return None
            dr = drs[self.rng.integers(len(drs))]
            adr = m.jnt_qposadr[dr["jid"]]
            anchor = d.geom_xpos[dr["handle"]].copy()
            Rb = d.xmat[dr["bid"]].reshape(3, 3)
            face = np.arctan2(Rb[1, 0], Rb[0, 0]) - np.pi / 2
            fcat = pc.e.fixtures_dict[dr["fixture"]].category_name
            lv = dr["level"]
            cands = [s for s in self.artic if s["kind"] == "drawer"
                     and ((s["q_end"] < s["q_start"] - 0.05) if want_open else (s["q_end"] > s["q_start"] + 0.05))]
            same = [s for s in cands if s["level"] == lv and s["fixture_cat"] == fcat]
            near = [s for s in cands if s["level"] == lv]
            cands = same if same and self.rng.random() < 0.8 else (near or cands)
            label = f"{lv} drawer"
            src_anchor = lambda s: s["anchor_pos"]
            src_face = lambda s: s["anchor_yaw"] - np.pi / 2
        else:
            fx = [f for f in pc.e.fixtures_dict if kind in f]
            if not fx:
                return None
            fname = fx[0]
            bid = m.body(pc.e.fixtures_dict[fname].root_body).id
            anchor = d.xpos[bid].copy()
            q = d.xquat[bid]
            face = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
            js = [j for j in range(m.njnt) if m.joint(j).name.startswith(fname + "_") and m.jnt_type[j] in (2, 3)]
            if not js:
                return None
            adr = m.jnt_qposadr[js[0]]
            cands = [s for s in self.artic if s["kind"] == kind and s["fix_pos"] is not None]
            if kind == "microwave":
                cands = [s for s in cands if ((s["q_end"] < s["q_start"]) if want_open else (s["q_end"] > s["q_start"]))]
            label = kind
            src_anchor = lambda s: s["fix_pos"]
            src_face = lambda s: float(np.arctan2(2 * (s["fix_quat"][0] * s["fix_quat"][3] + s["fix_quat"][1] * s["fix_quat"][2]),
                                                  1 - 2 * (s["fix_quat"][2] ** 2 + s["fix_quat"][3] ** 2)))
        if not cands:
            return None
        q0 = d.qpos[adr]
        p_cur, v_cur, R_cur = self._cur_state()
        scored = []
        for i in self.rng.permutation(len(cands))[:80]:
            s = cands[i]
            a_dst = anchor + (np.r_[self.rng.normal(0, 0.015, 2), 0] if explore else 0)
            if abs(wrap(face - src_face(s))) > np.radians(12):
                continue
            G = Transform(src_anchor(s), a_dst, 0.0)
            sd, _ = self.seg_data(s)
            P, Rm = G.pos(sd["eef_pos"]), G.mat(sd["eef_mat"])
            if (P[:, 2] < pc.table_z + 0.003).any():
                continue
            key = s["move_t0"] - s["t0"]
            ts, cost = match_start(P, Rm, p_cur, v_cur, R_cur, key - 15)
            q, skip = self._quality(f"{kind}_{'open' if want_open else 'close'}", s)
            if skip:
                continue
            scored.append((cost + 0.6 * (1 - q) + 0.8 * self._shift(src_anchor(s), a_dst, s), s, G, ts))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0])
        _, s, G, ts = scored[self.rng.integers(min(max(4, self.clip_topk), len(scored)))]
        ts = self._enter_clip(s, G, ts, key=s["move_t0"] - s["t0"])
        t1 = s["t1"]
        if explore and self.rng.random() < 0.5:
            # stop partway through the motion (partially opened/closed)
            t1 = int(s["move_t0"] + self.rng.uniform(0.3, 0.8) * (s["move_t1"] - s["move_t0"]))
        sd, _ = self.seg_data(s, t0=s["t0"] + ts, t1=t1)
        pl = plan(sd, G, pc.eef_pos, pc.eef_mat, blend_frac=0.7, key_t=s["move_t0"] - s["t0"] - ts)
        pc.phase = "articulate"
        execute(pc, pl)
        if pc.grip > 0:
            pc.gripper_action(-1.0, 4)
        dq = d.qpos[adr] - q0
        if kind == "drawer":
            ok = dq < -0.06 if want_open else dq > 0.04
            lang = f"{'open' if want_open else 'close'} the {label} of the cabinet"
        elif kind == "microwave":
            ok = abs(dq) > 0.3
            lang = f"{'open' if want_open else 'close'} the microwave"
        else:
            ok = abs(dq) > 0.2
            lang = "turn the stove knob"
        return dict(primitive=f"{kind}_{'open' if want_open else 'close'}", target=label,
                    success=bool(ok), language=lang, source=f"{s['task']}#{s['demo']}",
                    explore=bool(explore),
                    diag=dict(t_start=int(ts), key_err=pl.get("key_err", -1), n_transit=self.n_transit))

    # ---------------------------------------------------------------- driver
    def choose(self):
        pc = self.pc
        explore = self.rng.random() < self.p_explore
        opts = []
        movable = [n for n in pc.names if GROUP_OF.get(self.cat(n)) not in (None, "basket")
                   and self.upright(n)]
        last = self.segments[-1] if self.segments else None
        if movable:
            if (last is not None and last["primitive"] in ("pick_place", "pick_fumble")
                    and not last.get("grasped", True) and self.rng.random() < 0.6):
                tgt = last["target"].split("->")[0]
                if tgt in movable:
                    return lambda: self.p_pick_place(tgt, explore=False)  # retry
            # prefer objects that haven't been played with recently
            recent = [x.get("target", "").split("->")[0] for x in self.segments[-4:]]
            w = np.array([1.0 / (1 + 2 * recent.count(m)) for m in movable])
            n = movable[self.rng.choice(len(movable), p=w / w.sum())]
            opts.append((0.55, lambda: self.p_pick_place(n, explore=explore)))
            n2 = movable[self.rng.integers(len(movable))]
            opts.append((0.06, lambda: self.p_hover(n2)))
        if pc.drawers:
            dr = pc.drawers[self.rng.integers(len(pc.drawers))]
            is_open = pc.d.qpos[pc.m.jnt_qposadr[dr["jid"]]] < -0.04
            opts.append((0.3, lambda: self.p_articulate("drawer", not is_open, level=dr["level"],
                                                        explore=explore)))
        for k in ("microwave", "stove"):
            if any(k in f for f in pc.e.fixtures_dict):
                opts.append((0.2, lambda k=k: self.p_articulate(k, bool(self.rng.random() < 0.5),
                                                                explore=explore)))
        if not opts:
            return None
        w = np.array([o[0] for o in opts])
        return opts[self.rng.choice(len(opts), p=w / w.sum())][1]

    def run_episode(self, n_primitives):
        fails = 0
        while len(self.segments) < n_primitives:
            fn = self.choose()
            if fn is None:
                break
            t0 = self.pc.t
            try:
                info = fn()
            except EpisodeTooLong:
                break
            if info is None:
                fails += 1
                if fails > 15:
                    break
                continue
            info.update(start=t0, end=self.pc.t)
            self.segments.append(info)
        return self.segments
