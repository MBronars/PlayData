"""More inventive play with the objects: lining things up in a row, building
small towers, using one object to push another, and setting things down
snugly against a neighbour (so they nudge it). Mixin for PlayfulPlay."""
import numpy as np

from human_play import GEOM, GROUP_OF, bottom_offset
from scene_gen import REACH


def _nice(n):
    return n.rsplit("_", 1)[0].replace("_", " ")


class CreativePlay:
    # ------------------------------------------------------------ spots
    def _half_along(self, n, u):
        bb = self.pc.obj_bbox(n)
        return 0.5 * float(np.abs((bb["hi"] - bb["lo"])[:2]) @ np.abs(u[:2]))

    def _spot_ok(self, n, xy, exclude=(), clearance=0.01):
        """Is xy a reachable table spot where n's footprint clears everything
        (except `exclude`) by `clearance`?"""
        pc = self.pc
        base = np.array(pc.robot.base_pos)[:2]
        if not (REACH[0] < np.linalg.norm(xy - base) < REACH[1]):
            return False
        x0, y0 = pc.e.workspace_offset[:2]
        if not (-0.3 < xy[0] - x0 < 0.18 and -0.33 < xy[1] - y0 < 0.33):
            return False
        if any(np.all(xy > lo[:2] - 0.05) and np.all(xy < hi[:2] + 0.05) for _, lo, hi in self.fixture_boxes):
            return False
        for m in pc.names:
            if m == n or m in exclude:
                continue
            pm = self.obj_pose(m)[0][:2]
            if np.linalg.norm(pm - xy) < self._radius(n) + self._radius(m) + clearance:
                return False
        return True

    def _dest_snug(self, n):
        """A spot right up against another object (touching or a hair away):
        setting n down there nudges it a little, as happens to people."""
        pc = self.pc
        others = [m for m in pc.names if m != n and self.support_of(m) is None
                  and GROUP_OF.get(self.cat(m)) is not None]
        for _ in range(12):
            if not others:
                return None
            m = others[int(self.rng.integers(len(others)))]
            pm = 0.5 * (pc.obj_bbox(m)["lo"] + pc.obj_bbox(m)["hi"])
            a = self.rng.uniform(-np.pi, np.pi)
            u = np.r_[np.cos(a), np.sin(a), 0.0]
            gap = self.rng.uniform(-0.01, 0.004)   # touching, or just about
            xy = pm[:2] + u[:2] * (self._half_along(n, u) + self._half_along(m, u) + gap)
            if np.linalg.norm(xy - self.obj_pose(n)[0][:2]) < 0.06:
                continue
            if self._spot_ok(n, xy, exclude=(m,), clearance=0.02):
                return ("spot", np.r_[xy, pc.table_z + bottom_offset(self.cat(n)) + 0.003], "table"), m
        return None

    # ------------------------------------------------------------ line up
    def p_line_up(self, anchor, items):
        """Put a few things in a row next to `anchor`."""
        pc = self.pc
        pa = 0.5 * (pc.obj_bbox(anchor)["lo"] + pc.obj_bbox(anchor)["hi"])
        base = np.array(pc.robot.base_pos)[:2]
        # a row across the table or away from the robot, whichever fits
        radial = (pa[:2] - base) / np.linalg.norm(pa[:2] - base)
        dirs = [np.r_[-radial[1], radial[0], 0], np.r_[radial[1], -radial[0], 0], np.r_[radial, 0], np.r_[-radial, 0]]
        self.rng.shuffle(dirs)
        gap = self.rng.uniform(0.02, 0.04)
        for u in dirs:
            spots, prev, prev_half, ok = [], pa[:2], self._half_along(anchor, u), True
            for n in items:
                h = self._half_along(n, u)
                xy = prev + u[:2] * (prev_half + h + gap) + self.rng.normal(0, 0.004, 2)
                if not self._spot_ok(n, xy, exclude=(anchor,) + tuple(items), clearance=0.015):
                    ok = False
                    break
                spots.append(xy)
                prev, prev_half = xy, h
            if ok:
                break
        else:
            self.plan_log.append(("line_up", "no room"))
            return None
        placed, infos = 0, []
        for n, xy in zip(items, spots):
            dest = ("spot", np.r_[xy, pc.table_z + bottom_offset(self.cat(n)) + 0.003], "table")
            info = self.p_pick_place(n, dest=dest, prim="line_up", lang=f"line up the {_nice(n)} next to the {_nice(anchor)}")
            if info is None:
                break
            infos.append(info)
            placed += bool(info.get("success"))
            if not info.get("success"):
                break
        if not infos:
            return None
        names = [_nice(x) for x in [anchor] + list(items)]
        out = infos[-1]
        out.update(primitive="line_up", target="->".join([items[0], anchor]), interact=[anchor] + list(items[1:]),
                   success=bool(placed == len(items)),
                   language="line up the " + ", the ".join(names[:-1]) + " and the " + names[-1] + " in a row",
                   diag=dict(out.get("diag", {}), placed=placed, of=len(items)))
        return out

    # ------------------------------------------------------------ towers
    def p_tower(self, base_obj, items):
        """Stack a couple of things into a little tower."""
        below, done = base_obj, 0
        last = None
        for n in items:
            if not self.can_stack(n, below):
                break
            info = self.run_task("stack", n, below)
            if info is None:
                break
            last = info
            if not info.get("success"):
                break
            done += 1
            below = n
        if last is None:
            return None
        last.update(primitive="tower", target=f"{items[0]}->{base_obj}", interact=[base_obj] + list(items[1:]),
                    success=bool(done == len(items)),
                    language=f"build a tower on the {_nice(base_obj)} with the " + " and the ".join(_nice(x) for x in items),
                    diag=dict(last.get("diag", {}), levels=done))
        return last

    # ------------------------------------------------------------ relations
    # direction of each relation in the world frame (robot at -x facing +x): "behind" is further from the robot,
    # "left" the robot's left. "next_to" takes any side.
    REL_DIR = {"behind": (1, 0), "in_front": (-1, 0), "left": (0, 1), "right": (0, -1)}
    REL_WORD = {"behind": "behind", "in_front": "in front of", "left": "to the left of", "right": "to the right of",
                "next_to": "next to"}

    def p_place_rel(self, n, m, rel):
        """Put n down on a given side of m, 2-8 cm from it (face to face). Counts when n ends up on that side (centre
        offset along the side's direction at least the two half-widths, at most 5 cm off sideways, upright) and m
        stayed within 3 cm of where it was (bumping the reference along doesn't count)."""
        pc = self.pc
        pm = 0.5 * (pc.obj_bbox(m)["lo"] + pc.obj_bbox(m)["hi"])
        m0 = self.obj_pose(m)[0].copy()
        if rel == "next_to":
            angs = self.rng.permutation(np.linspace(-np.pi, np.pi, 8, endpoint=False))
        else:
            a0 = np.arctan2(*self.REL_DIR[rel][::-1])
            angs = a0 + self.rng.uniform(-0.25, 0.25, 4)
        dest = None
        for a in angs:
            u = np.r_[np.cos(a), np.sin(a), 0.0]
            gap = self.rng.uniform(0.02, 0.08)
            xy = pm[:2] + u[:2] * (self._half_along(n, u) + self._half_along(m, u) + gap)
            if np.linalg.norm(xy - self.obj_pose(n)[0][:2]) < 0.05:
                continue
            if self._spot_ok(n, xy, exclude=(m,), clearance=0.015):
                dest = ("spot", np.r_[xy, pc.table_z + bottom_offset(self.cat(n)) + 0.003], "table")
                break
        if dest is None:
            self.plan_log.append(("place_rel", "no room", rel))
            return None
        info = self.p_pick_place(n, dest=dest, lang=f"put the {_nice(n)} {self.REL_WORD[rel]} the {_nice(m)}", prim="place_rel")
        if info is not None:
            on = self.obj_pose(n)[0]; om = self.obj_pose(m)[0]
            d = on[:2] - om[:2]
            if rel == "next_to":
                u = d / (np.linalg.norm(d) + 1e-9); side_off = 0.0
            else:
                u = np.asarray(self.REL_DIR[rel], float); side_off = abs(float(np.cross(u, d)))
            u3 = np.r_[u, 0.0]
            along = float(d @ u); need = self._half_along(n, u3) + self._half_along(m, u3)
            m_moved = float(np.linalg.norm(om[:2] - m0[:2]))
            ok = need - 0.01 <= along <= need + 0.12 and side_off < 0.05 and m_moved < 0.03 and self.upright(n)
            info.update(interact=[m], relation=rel)
            info.setdefault("diag", {}).update(rel_along=along - need, rel_side=side_off, ref_moved=m_moved)
            info["success"] = bool(info.get("grasped", True) and ok)
        return info

    # ------------------------------------------------------------ tool use
    def p_push_with(self, tool, m):
        """Hold one object and use it to shove another along the table."""
        self._bump_mode = "push"
        try:
            info = self.p_bump_with(tool, m)
        finally:
            self._bump_mode = None
        if info is not None:
            info.update(primitive="push_with", language=f"use the {_nice(tool)} to push the {_nice(m)}")
            info["success"] = bool(info.get("grasped")) and info["diag"].get("shoved", 0) > 0.03
        return info
