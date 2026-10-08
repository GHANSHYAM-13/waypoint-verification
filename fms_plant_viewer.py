#!/usr/bin/env python3
"""
tcs_viewer.py - inspect and validate an OpenTCS plant-model XML.

OpenTCS stores NO heading on a path and only a free-form `theta` property on each point, and its editor
never checks them against each other. This tool draws the model with directions and checks them:

  * point headings (theta) as arrows, paths as directed arrows, in-place rotations as arcs
  * pick / drop / nop locations (from their location type's allowed operations)
  * checks: reverse travel, heading vs. path direction, length vs. distance, rotation edges,
    dead ends, missing / bad theta, broken links, unreachable pick/drop locations
  * route finder (shortest by path length) and an "all pick -> drop" report
  * export: issues CSV, points as wp_plotter .txt, map image

Usage
-----
    python3 tcs_viewer.py                       # GUI, then Open XML
    python3 tcs_viewer.py model.xml             # GUI with the model loaded
    python3 tcs_viewer.py model.xml --report    # text report in the terminal (no GUI)
    python3 tcs_viewer.py model.xml --csv issues.csv --png map.png [--theme dark] [--tol 15]

Needs matplotlib, and wp_plotter.py in the same folder (shared look and feel).
Theta convention: degrees, CCW from +X (0 = +x, 90 = +y), as in wp_plotter.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import wp_plotter as wp
except ImportError:  # pragma: no cover
    sys.exit("tcs_viewer.py needs wp_plotter.py in the same folder.")

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

Obj = Tuple[str, str]  # ('P' | 'A' | 'L', name)  point / path (arc) / location


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@dataclass
class Pt:
    name: str
    x: float
    y: float
    theta: Optional[float]
    ptype: str = ""
    props: Dict[str, str] = field(default_factory=dict)
    layout: Optional[Tuple[float, float]] = None


@dataclass
class Pth:
    name: str
    src: str
    dst: str
    length: float
    vmax: float
    vrev: float
    locked: bool


@dataclass
class LType:
    name: str
    ops: List[str]
    props: Dict[str, str] = field(default_factory=dict)


@dataclass
class Loc:
    name: str
    ltype: str
    x: float
    y: float
    links: List[str]
    props: Dict[str, str] = field(default_factory=dict)
    layout: Optional[Tuple[float, float]] = None


@dataclass
class Veh:
    name: str
    props: Dict[str, str]
    length: float = 0.0


@dataclass
class Model:
    name: str = ""
    points: Dict[str, Pt] = field(default_factory=dict)
    paths: List[Pth] = field(default_factory=list)
    locs: Dict[str, Loc] = field(default_factory=dict)
    ltypes: Dict[str, LType] = field(default_factory=dict)
    vehicles: List[Veh] = field(default_factory=list)
    pinfo: Dict[str, dict] = field(default_factory=dict)  # per-path geometry, filled by analyze()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _f(v: Optional[str], default: float = float("nan")) -> float:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def load_model(path: str) -> Model:
    root = ET.parse(path).getroot()
    for e in root.iter():
        e.tag = _local(e.tag)
    m = Model(name=root.get("name", os.path.basename(path)))
    layout: Dict[str, Dict[str, str]] = {}
    vl = root.find("visualLayout")
    if vl is not None:
        for e in vl.findall("modelLayoutElement"):
            layout[e.get("visualizedObjectName", "")] = {p.get("name", ""): p.get("value", "") for p in e.findall("property")}

    def lay(name: str) -> Optional[Tuple[float, float]]:
        d = layout.get(name)
        if d and "POSITION_X" in d and "POSITION_Y" in d:
            return (_f(d["POSITION_X"]), _f(d["POSITION_Y"]))
        return None

    for e in root.findall("point"):
        props = {p.get("name", ""): p.get("value", "") for p in e.findall("property")}
        th = _f(props.get("theta"))
        m.points[e.get("name", "")] = Pt(e.get("name", ""), _f(e.get("xPosition"), 0.0), _f(e.get("yPosition"), 0.0),
                                         None if math.isnan(th) else th, e.get("type", ""), props, lay(e.get("name", "")))
    for e in root.findall("path"):
        m.paths.append(Pth(e.get("name", ""), e.get("sourcePoint", ""), e.get("destinationPoint", ""),
                           _f(e.get("length"), 0.0), _f(e.get("maxVelocity"), 0.0), _f(e.get("maxReverseVelocity"), 0.0),
                           (e.get("locked", "false").lower() == "true")))
    for e in root.findall("locationType"):
        m.ltypes[e.get("name", "")] = LType(e.get("name", ""), [o.get("name", "") for o in e.findall("allowedOperation")],
                                            {p.get("name", ""): p.get("value", "") for p in e.findall("property")})
    for e in root.findall("location"):
        m.locs[e.get("name", "")] = Loc(e.get("name", ""), e.get("type", ""), _f(e.get("xPosition"), 0.0),
                                        _f(e.get("yPosition"), 0.0), [k.get("point", "") for k in e.findall("link")],
                                        {p.get("name", ""): p.get("value", "") for p in e.findall("property")}, lay(e.get("name", "")))
    for e in root.findall("vehicle"):
        m.vehicles.append(Veh(e.get("name", ""), {p.get("name", ""): p.get("value", "") for p in e.findall("property")},
                              _f(e.get("length"), 0.0)))
    analyze(m)
    return m


def angdiff(a: float, b: float) -> float:
    """a - b wrapped to (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


MOVE_EPS = 50.0  # map units: shorter than this = rotation in place


def analyze(m: Model) -> None:
    """Per path: distance, travel heading, and heading difference to the end points' theta."""
    m.pinfo = {}
    for p in m.paths:
        a, b = m.points.get(p.src), m.points.get(p.dst)
        if a is None or b is None:
            m.pinfo[p.name] = dict(kind="broken", dist=float("nan"), heading=None, ds=None, dt=None, dth=None)
            continue
        dist = math.hypot(b.x - a.x, b.y - a.y)
        kind = "move" if dist > MOVE_EPS else "rotate"
        heading = math.degrees(math.atan2(b.y - a.y, b.x - a.x)) % 360.0 if kind == "move" else None
        ds = angdiff(heading, a.theta) if heading is not None and a.theta is not None else None
        dt = angdiff(heading, b.theta) if heading is not None and b.theta is not None else None
        dth = angdiff(b.theta, a.theta) if a.theta is not None and b.theta is not None else None
        m.pinfo[p.name] = dict(kind=kind, dist=dist, heading=heading, ds=ds, dt=dt, dth=dth)


def loc_kind(m: Model, loc: Loc) -> str:
    ops = " ".join(m.ltypes[loc.ltype].ops).lower() if loc.ltype in m.ltypes else loc.ltype.lower()
    return "pick" if "pick" in ops else "drop" if "drop" in ops else "nop"


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
@dataclass
class Issue:
    sev: str  # ERROR | WARN | INFO
    code: str
    msg: str
    objs: List[Obj]

    @property
    def obj_text(self) -> str:
        return ", ".join(n for _, n in self.objs[:3]) + (" ..." if len(self.objs) > 3 else "")


def adjacency(m: Model) -> Dict[str, List[Pth]]:
    adj: Dict[str, List[Pth]] = defaultdict(list)
    for p in m.paths:
        if p.src in m.points and p.dst in m.points and not p.locked:
            adj[p.src].append(p)
    return adj


def reachable(m: Model, start: str, adj: Dict[str, List[Pth]]) -> set:
    seen, dq = {start}, deque([start])
    while dq:
        for p in adj.get(dq.popleft(), ()):
            if p.dst not in seen:
                seen.add(p.dst)
                dq.append(p.dst)
    return seen


def loc_points(m: Model) -> set:
    return {lk for l in m.locs.values() for lk in l.links}


def loc_names_at(m: Model, point: str) -> List[str]:
    return [l.name for l in m.locs.values() if point in l.links]


def reverse_kind(m: Model, p: Pth) -> Optional[str]:
    """None = drives forward. 'ok' = reverse allowed, 'dock' = backs into a location point, 'error' = backwards elsewhere."""
    i = m.pinfo.get(p.name)
    if not i or i["kind"] != "move" or i["ds"] is None or abs(i["ds"]) <= 150.0:
        return None
    if p.vrev > 0:
        return "ok"
    return "dock" if p.dst in loc_points(m) else "error"


def validate(m: Model, tol: float = 15.0, len_tol: float = 0.10) -> List[Issue]:
    out: List[Issue] = []
    add = lambda sev, code, msg, objs: out.append(Issue(sev, code, msg, objs))
    inc, outc = defaultdict(int), defaultdict(int)

    for pt in m.points.values():
        if pt.theta is None:
            add("ERROR", "BAD_THETA", f"{pt.name}: theta property missing or not a number", [("P", pt.name)])
        elif not (0.0 <= pt.theta < 360.0):
            add("WARN", "THETA_RANGE", f"{pt.name}: theta {pt.theta:.2f} outside [0, 360)", [("P", pt.name)])

    seen_pairs: Dict[Tuple[str, str], str] = {}
    for p in m.paths:
        o: List[Obj] = [("A", p.name)]
        if p.src not in m.points or p.dst not in m.points:
            add("ERROR", "BROKEN_PATH", f"{p.name}: endpoint point does not exist", o)
            continue
        outc[p.src] += 1
        inc[p.dst] += 1
        nm = re.fullmatch(r"\s*(.+?)\s*---\s*(.+?)\s*", p.name)
        if nm and (nm.group(1), nm.group(2)) != (p.src, p.dst):
            add("WARN", "NAME_MISMATCH", f"{p.name}: name says {nm.group(1)} -> {nm.group(2)} but the path really goes "
                f"{p.src} -> {p.dst} (stale name after editing?)", o + [("P", p.src), ("P", p.dst)])
        if (p.src, p.dst) in seen_pairs:
            add("WARN", "DUP_PATH", f"{p.name}: duplicates {seen_pairs[(p.src, p.dst)]}", o)
        seen_pairs[(p.src, p.dst)] = p.name
        if p.locked:
            add("INFO", "LOCKED", f"{p.name}: path is locked (routing ignores it)", o)
        i = m.pinfo[p.name]
        a, b = m.points[p.src], m.points[p.dst]
        if i["kind"] == "move":
            if i["ds"] is not None and abs(i["ds"]) > 150.0:
                rk = reverse_kind(m, p)
                if rk == "error":
                    add("ERROR", "REVERSE", f"{p.name}: drives BACKWARDS - travel heading {i['heading']:.1f}, but {p.src} theta is "
                        f"{a.theta:.1f} (diff {i['ds']:+.1f}) and maxReverseVelocity is {p.vrev:g}", o + [("P", p.src), ("P", p.dst)])
                elif rk == "dock":
                    add("WARN", "REVERSE_DOCK", f"{p.name}: backs into {', '.join(loc_names_at(m, p.dst))} at {p.dst} (heading "
                        f"{i['heading']:.1f}, {p.src} theta {a.theta:.1f}) - vehicle must reverse, maxReverseVelocity is {p.vrev:g}",
                        o + [("P", p.src), ("P", p.dst)])
                else:
                    add("INFO", "REVERSE_OK", f"{p.name}: reverse travel (heading {i['heading']:.1f}, theta {a.theta:.1f})", o)
            else:
                if i["ds"] is not None and abs(i["ds"]) > tol:
                    add("WARN", "HEADING_SRC", f"{p.name}: travel heading {i['heading']:.1f} differs {i['ds']:+.1f} deg from "
                        f"{p.src} theta {a.theta:.1f}", o + [("P", p.src)])
                if i["dt"] is not None and abs(i["dt"]) > tol:
                    add("WARN", "HEADING_DST", f"{p.name}: travel heading {i['heading']:.1f} differs {i['dt']:+.1f} deg from "
                        f"{p.dst} theta {b.theta:.1f} (arrives with another heading)", o + [("P", p.dst)])
            if abs(p.length - i["dist"]) > max(len_tol * i["dist"], MOVE_EPS):
                add("WARN", "LENGTH", f"{p.name}: length {p.length:g} vs {i['dist']:.0f} between points "
                    f"({(p.length / i['dist'] - 1) * 100:+.0f}%)", o)
        else:
            if i["dth"] is not None and abs(i["dth"]) < 2.0:
                add("WARN", "ROT_NOOP", f"{p.name}: in-place path but theta does not change ({a.theta:.1f} -> {b.theta:.1f})", o)
            if p.length > 10000:
                add("INFO", "ROT_COST", f"{p.name}: in-place rotation with very high length/cost {p.length:g} (route penalty?)", o)
        if p.vmax <= 0:
            add("WARN", "NO_SPEED", f"{p.name}: maxVelocity is {p.vmax:g}", o)

    for n in m.points:
        if outc[n] == 0 and inc[n] == 0:
            add("WARN", "ISOLATED", f"{n}: point has no paths at all", [("P", n)])
        elif outc[n] == 0:
            add("WARN", "DEAD_END", f"{n}: no outgoing path - a vehicle arriving here is stuck", [("P", n)])
        elif inc[n] == 0:
            add("INFO", "NO_INCOMING", f"{n}: nothing leads to this point (start / home only)", [("P", n)])

    groups: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for pt in m.points.values():
        groups[(round(pt.x / 5), round(pt.y / 5))].append(pt.name)
    for names in groups.values():
        if len(names) > 1:
            add("INFO", "COINCIDENT", f"{len(names)} points share one position (rotation node): {', '.join(names)}",
                [("P", n) for n in names])

    adj = adjacency(m)
    pick_pts: Dict[str, str] = {}
    drop_pts: Dict[str, str] = {}
    for loc in m.locs.values():
        o = [("L", loc.name)]
        if loc.ltype not in m.ltypes:
            add("ERROR", "LOC_TYPE", f"{loc.name}: location type '{loc.ltype}' does not exist", o)
        if not loc.links:
            add("ERROR", "LOC_LINK", f"{loc.name}: location is not linked to any point", o)
            continue
        for lk in loc.links:
            if lk not in m.points:
                add("ERROR", "LOC_LINK", f"{loc.name}: linked point '{lk}' does not exist", o)
        valid = [lk for lk in loc.links if lk in m.points]
        if not valid:
            continue
        k = loc_kind(m, loc)
        if k == "nop":
            add("INFO", "LOC_NOP", f"{loc.name}: no pick/drop operation (type {loc.ltype})", o + [("P", valid[0])])
        (pick_pts if k == "pick" else drop_pts if k == "drop" else {})[loc.name] = valid[0]
        if m.points[valid[0]].theta is None:
            add("ERROR", "LOC_THETA", f"{loc.name}: linked point {valid[0]} has no heading, approach direction is undefined",
                o + [("P", valid[0])])
    reach = {n: reachable(m, p, adj) for n, p in {**pick_pts, **drop_pts}.items()}
    for n, p in pick_pts.items():
        if not any(dp in reach[n] for dp in drop_pts.values()):
            add("WARN", "NO_DROP_ROUTE", f"{n} (pick at {p}): no drop location can be reached from here", [("L", n), ("P", p)])
    for n, p in drop_pts.items():
        if not any(p in reach[pn] for pn in pick_pts):
            add("WARN", "NO_PICK_ROUTE", f"{n} (drop at {p}): not reachable from any pick location", [("L", n), ("P", p)])
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    out.sort(key=lambda i: (order[i.sev], i.code))
    return out


# ----------------------------------------------------------------------------
# Routing (OpenTCS default cost = path length)
# ----------------------------------------------------------------------------
def point_of(m: Model, name: str) -> Optional[str]:
    if name in m.points:
        return name
    loc = m.locs.get(name)
    return next((lk for lk in (loc.links if loc else []) if lk in m.points), None)


def shortest_route(m: Model, a: str, b: str) -> Optional[Tuple[float, List[Pth]]]:
    s, t = point_of(m, a), point_of(m, b)
    if s is None or t is None:
        return None
    adj = adjacency(m)
    dist, prev, heap = {s: 0.0}, {}, [(0.0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == t:
            break
        if d > dist.get(u, 1e30):
            continue
        for p in adj.get(u, ()):
            nd = d + max(p.length, 0.0)
            if nd < dist.get(p.dst, 1e30):
                dist[p.dst], prev[p.dst] = nd, p
                heapq.heappush(heap, (nd, p.dst))
    if t not in dist:
        return None
    steps: List[Pth] = []
    cur = t
    while cur != s:
        steps.append(prev[cur])
        cur = prev[cur].src
    return dist[t], steps[::-1]


def route_text(m: Model, a: str, b: str, tol: float) -> Tuple[str, List[str], int]:
    r = shortest_route(m, a, b)
    if r is None:
        return f"No route from {a} to {b}.", [], 0
    total, steps = r
    lines = [f"{a} -> {b}   {len(steps)} paths   length {total:,.0f}", ""]
    bad = 0
    for k, p in enumerate(steps, 1):
        i = m.pinfo[p.name]
        flag = ""
        if i["kind"] == "move":
            s, d = m.points[p.src], m.points[p.dst]
            txt = f"drive  heading {i['heading']:.1f}  (theta {s.theta if s.theta is not None else float('nan'):.1f})"
            rk = reverse_kind(m, p)
            if rk == "error":
                flag = "  !! BACKWARDS"
            elif rk == "dock":
                flag = "  ~ backs into location"
            elif (i["ds"] is not None and abs(i["ds"]) > tol) or (i["dt"] is not None and abs(i["dt"]) > tol):
                flag = "  ! heading"
        else:
            txt = f"rotate {m.points[p.src].theta:.1f} -> {m.points[p.dst].theta:.1f}" + (
                f"  ({i['dth']:+.1f})" if i["dth"] is not None else "") if i["dth"] is not None else "rotate"
        bad += bool(flag)
        lines.append(f"{k:>2}. {p.name:<18} {txt}  len {p.length:g}{flag}")
    return "\n".join(lines), [p.name for p in steps], bad


def route_stats(m: Model, a: str, b: str, tol: float) -> Optional[dict]:
    r = shortest_route(m, a, b)
    if r is None:
        return None
    total, steps = r
    back = sum(1 for p in steps if reverse_kind(m, p) == "error")
    dock = sum(1 for p in steps if reverse_kind(m, p) == "dock")
    hd = sum(1 for p in steps if reverse_kind(m, p) is None and m.pinfo[p.name]["kind"] == "move" and
             any(v is not None and abs(v) > tol for v in (m.pinfo[p.name]["ds"], m.pinfo[p.name]["dt"])))
    return dict(length=total, n=len(steps), back=back, dock=dock, hd=hd)


def pick_drop_report(m: Model, tol: float) -> str:
    picks = [l for l in m.locs.values() if loc_kind(m, l) == "pick"]
    drops = [l for l in m.locs.values() if loc_kind(m, l) == "drop"]
    lines = [f"{len(picks)} pick x {len(drops)} drop locations", "", f"{'from':<8}{'to':<8}{'length':>10}{'paths':>7}  note"]
    for a in sorted(picks, key=lambda l: l.name):
        for b in sorted(drops, key=lambda l: l.name):
            r = shortest_route(m, a.name, b.name)
            if r is None:
                lines.append(f"{a.name:<8}{b.name:<8}{'-':>10}{'-':>7}  UNREACHABLE")
                continue
            _t, steps = r
            back = sum(1 for p in steps if reverse_kind(m, p) == "error")
            dock = sum(1 for p in steps if reverse_kind(m, p) == "dock")
            hd = sum(1 for p in steps if reverse_kind(m, p) is None and m.pinfo[p.name]["kind"] == "move" and
                     ((m.pinfo[p.name]["ds"] is not None and abs(m.pinfo[p.name]["ds"]) > tol) or
                      (m.pinfo[p.name]["dt"] is not None and abs(m.pinfo[p.name]["dt"]) > tol)))
            note = ("BACKWARDS x%d " % back if back else "") + ("backs-in x%d " % dock if dock else "") + ("heading x%d" % hd if hd else "")
            lines.append(f"{a.name:<8}{b.name:<8}{_t:>10,.0f}{len(steps):>7}  {note.strip() or 'ok'}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Drawing (shared by GUI and --png)
# ----------------------------------------------------------------------------
ARROW_PT = 24.0  # point heading arrow length in typographic points (screen constant, independent of zoom)
RED = {"dark": "#ff5d5d", "light": "#d32f2f"}
WARNC = {"dark": "#f5a623", "light": "#e08a00"}  # warnings
SELC = {"dark": "#f5a623", "light": "#0b6fb8"}  # selection glow
NOPC = {"dark": "#b4bcc6", "light": "#59606a"}  # nop / home locations
ROUTEC = {"dark": "#ff4dd2", "light": "#c2188f"}  # highlighted route
DIM = 0.10  # alpha of everything that is not on the active route


@dataclass
class Opts:
    points: bool = True
    paths: bool = True
    locs: bool = True
    labels: bool = True
    theta: bool = True
    only_issues: bool = False
    layout: bool = False  # draw in OpenTCS's editor drawing frame instead of plant coordinates
    tol: float = 15.0
    focus: bool = True  # dim everything that is not on the active route


def route_sets(m: Model, route: Optional[List[str]]):
    rp = set(route or [])
    pts = set()
    for p in m.paths:
        if p.name in rp:
            pts |= {p.src, p.dst}
    locs = {l.name for l in m.locs.values() if any(lk in pts for lk in l.links)}
    return rp, pts, locs


def ppos(m: Model, o: Opts, name: str) -> Tuple[float, float]:
    pt = m.points[name]
    return pt.layout if (o.layout and pt.layout) else (pt.x, pt.y)


def lpos(m: Model, o: Opts, loc: Loc) -> Tuple[float, float]:
    if o.layout:
        return loc.layout or (loc.x, loc.y)
    for lk in loc.links:  # plant frame: a location sits on the point it is linked to
        if lk in m.points:
            return ppos(m, o, lk)
    return (loc.x, loc.y)


def data_bounds(m: Model, o: Opts) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs = [ppos(m, o, n)[0] for n in m.points] + [lpos(m, o, l)[0] for l in m.locs.values()]
    ys = [ppos(m, o, n)[1] for n in m.points] + [lpos(m, o, l)[1] for l in m.locs.values()]
    xs, ys = xs or [0.0, 1.0], ys or [0.0, 1.0]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    pad = 0.06 * span
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


def units_per_pt(ax) -> float:
    ax.apply_aspect()
    w_px = max(ax.get_window_extent().width, 1.0)
    x0, x1 = ax.get_xlim()
    return (x1 - x0) / (w_px * 72.0 / ax.figure.dpi)


def point_kind(m: Model, name: str) -> str:
    for l in m.locs.values():
        if name in l.links:
            return loc_kind(m, l)
    return "other"


def kind_color(pal: dict, kind: str) -> str:
    nop = NOPC["light" if pal["bg"] == "#ffffff" else "dark"]
    return {"pick": pal["pick"], "drop": pal["drop"], "nop": nop}.get(kind, pal["other"])


def severity_map(issues: List[Issue]) -> Dict[Obj, str]:
    rank = {"INFO": 0, "WARN": 1, "ERROR": 2}
    out: Dict[Obj, str] = {}
    for i in issues:
        for ob in i.objs:
            if rank[i.sev] > rank.get(out.get(ob, "INFO"), -1) or ob not in out:
                out[ob] = i.sev if rank[i.sev] >= rank.get(out.get(ob, "INFO"), 0) else out[ob]
    return {k: v for k, v in out.items() if v in ("WARN", "ERROR")}


def draw_static(ax, m: Model, o: Opts, pal: dict, theme: str, sev: Dict[Obj, str], route: Optional[List[str]] = None) -> None:
    """Everything whose size does not depend on the zoom level."""
    ax.clear()
    ax.figure.set_facecolor(pal["bg"])
    ax.set_facecolor(pal["bg"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, lw=0.25, color=pal["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6, length=2, width=0.4, colors=pal["muted"])
    for s in ax.spines.values():
        s.set_linewidth(0.4)
        s.set_edgecolor(pal["spine"])
    red, warnc = RED[theme], WARNC[theme]
    dim = o.only_issues
    pairs = {(p.src, p.dst) for p in m.paths}

    rp, rpts, rlocs = route_sets(m, route)
    focus = bool(rp) and o.focus

    if o.paths:
        for p in m.paths:
            if p.src not in m.points or p.dst not in m.points:
                continue
            kind = m.pinfo[p.name]["kind"]
            inr = p.name in rp
            if kind == "rotate" and not o.layout:
                continue  # drawn as arcs in the zoom-dependent layer
            sv = sev.get(("A", p.name))
            col = red if sv == "ERROR" else warnc if sv == "WARN" else (pal["other"] if kind == "rotate" else pal["muted"])
            alpha = 0.15 if dim and not sv else 0.95
            lw, ms, z = (1.1 if sv else 0.8), 8, 2
            if inr:
                col, alpha, lw, ms, z = ROUTEC[theme], 1.0, 2.6, 12, 7
            elif focus:
                alpha = DIM
            rad = 0.14 if (p.dst, p.src) in pairs and kind == "move" else 0.0
            ann = ax.annotate("", xy=ppos(m, o, p.dst), xytext=ppos(m, o, p.src), zorder=z,
                              arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, alpha=alpha, mutation_scale=ms,
                                              shrinkA=4, shrinkB=4, ls=":" if kind == "rotate" else "-",
                                              connectionstyle=f"arc3,rad={rad}"))
            if ann.arrow_patch is not None:
                ann.arrow_patch.set_clip_path(ax.patch)  # annotation arrows are not clipped to the axes by default
    if o.points:
        for pt in m.points.values():
            x, y = ppos(m, o, pt.name)
            col = kind_color(pal, point_kind(m, pt.name))
            sv = sev.get(("P", pt.name))
            inr = pt.name in rpts
            alpha = DIM if (focus and not inr) else (0.2 if dim and not sv else 1.0)
            ax.plot(x, y, "o", ms=4.8 if inr else 3.4, color=col, alpha=alpha, zorder=6 if inr else 5)
            if sv and not (focus and not inr):
                ax.plot(x, y, "o", ms=9, mfc="none", mec=red if sv == "ERROR" else warnc, mew=0.9, alpha=0.9, zorder=4)
            if o.theta and pt.theta is not None:
                r = math.radians(pt.theta)
                ax.annotate("", xy=(x, y), xytext=(ARROW_PT * math.cos(r), ARROW_PT * math.sin(r)), textcoords="offset points",
                            zorder=4, arrowprops=dict(arrowstyle="<|-", color=col, lw=1.4 if (inr and rp) else 1.0, alpha=alpha,
                                                      mutation_scale=7, shrinkA=0, shrinkB=0))
    if o.locs:
        for loc in m.locs.values():
            x, y = lpos(m, o, loc)
            col = kind_color(pal, loc_kind(m, loc))
            sv = sev.get(("L", loc.name))
            alpha = DIM if (focus and loc.name not in rlocs) else (0.2 if dim and not sv else 1.0)
            ax.plot(x, y, "s", ms=11 if not o.layout else 7, mfc="none" if not o.layout else col, mec=red if sv == "ERROR" else col,
                    mew=1.5, zorder=6, alpha=alpha)
            if o.layout:
                for lk in loc.links:
                    if lk in m.points:
                        px, py = ppos(m, o, lk)
                        ax.plot([x, px], [y, py], ":", color=col, lw=0.8, zorder=1, alpha=0.7 * alpha)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()


def draw_view(ax, m: Model, o: Opts, pal: dict, theme: str, sev: Dict[Obj, str], fs: float = 6.5, route: Optional[List[str]] = None):
    """Zoom-dependent layer: in-place rotation arcs and decluttered labels. Returns (artists, arc polylines)."""
    arts: list = []
    arcs: Dict[str, Tuple[List[float], List[float]]] = {}
    rp, rpts, rlocs = route_sets(m, route)
    focus = bool(rp) and o.focus
    u = units_per_pt(ax)
    red = RED[theme]
    dpi = ax.figure.dpi
    k_at: Dict[Tuple[int, int], int] = defaultdict(int)
    if o.paths and not o.layout:
        for p in m.paths:
            if m.pinfo[p.name]["kind"] != "rotate":
                continue
            a, b = m.points[p.src], m.points[p.dst]
            if a.theta is None or b.theta is None:
                continue
            d = angdiff(b.theta, a.theta)
            if abs(d) < 1.0:
                continue
            key = (round(a.x / 5), round(a.y / 5))
            R = (13.0 + 5.0 * k_at[key]) * u
            k_at[key] += 1
            n = max(6, int(abs(d) / 5))
            xs = [a.x + R * math.cos(math.radians(a.theta + d * i / n)) for i in range(n + 1)]
            ys = [a.y + R * math.sin(math.radians(a.theta + d * i / n)) for i in range(n + 1)]
            sv = sev.get(("A", p.name))
            col = red if sv == "ERROR" else WARNC[theme] if sv == "WARN" else pal["other"]
            alpha = 0.2 if o.only_issues and not sv else 0.95
            lw, ms, z = 1.0, 7, 3
            if p.name in rp:
                col, alpha, lw, ms, z = ROUTEC[theme], 1.0, 2.6, 11, 7
            elif focus:
                alpha = DIM
            arts += ax.plot(xs, ys, "-", color=col, lw=lw, alpha=alpha, zorder=z)
            arts.append(ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]), zorder=z,
                                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, alpha=alpha, mutation_scale=ms, shrinkA=0, shrinkB=0)))
            arcs[p.name] = (xs, ys)
    if o.labels and (o.points or o.locs):
        arts += _labels(ax, m, o, pal, fs, sev, rpts if focus else None, rlocs if focus else None)
    return arts, arcs


def _labels(ax, m: Model, o: Opts, pal: dict, fs: float, sev: Dict[Obj, str], show_pts=None, show_locs=None) -> list:
    dpi = ax.figure.dpi
    ppx = dpi / 72.0
    tr = ax.transData
    bb = ax.get_window_extent()
    fpx, Lpx = fs * ppx, ARROW_PT * ppx
    cw, lh = 0.6 * fpx, 1.3 * fpx
    obst: List[Tuple[float, float, float, float]] = []
    anchors: List[dict] = []  # clusters of items sharing one screen position
    for name, pt in m.points.items():
        x, y = ppos(m, o, name)
        px, py = tr.transform((x, y))
        obst.append((px - 4, py - 4, px + 4, py + 4))
        dirs = []
        if o.theta and pt.theta is not None and o.points:
            r = math.radians(pt.theta)
            tx, ty = px + Lpx * math.cos(r), py + Lpx * math.sin(r)
            obst.append((min(px, tx) - 2, min(py, ty) - 2, max(px, tx) + 2, max(py, ty) + 2))
            dirs.append(r)
        if not o.points or (show_pts is not None and name not in show_pts):
            continue
        for a in anchors:
            if math.hypot(a["px"] - px, a["py"] - py) < 5:
                a["lines"].append((name, pal["fg"])); a["dirs"] += dirs
                break
        else:
            anchors.append(dict(px=px, py=py, x=x, y=y, lines=[(name, pal["fg"])], dirs=dirs))
    if o.locs:
        for loc in m.locs.values():
            x, y = lpos(m, o, loc)
            px, py = tr.transform((x, y))
            col = kind_color(pal, loc_kind(m, loc))
            obst.append((px - 7, py - 7, px + 7, py + 7))
            if show_locs is not None and loc.name not in show_locs:
                continue
            for a in anchors:
                if math.hypot(a["px"] - px, a["py"] - py) < 9:
                    a["lines"].insert(0, (loc.name + " " + loc_kind(m, loc), col))
                    break
            else:
                anchors.append(dict(px=px, py=py, x=x, y=y, lines=[(loc.name + " " + loc_kind(m, loc), col)], dirs=[]))
    placed: List[Tuple[float, float, float, float]] = []
    out: list = []

    def ov(a, b):
        w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
        return w * h if w > 0 and h > 0 else 0.0

    for a in sorted(anchors, key=lambda a: -len(a["lines"])):
        px, py = a["px"], a["py"]
        if not (bb.x0 - 10 <= px <= bb.x1 + 10 and bb.y0 - 10 <= py <= bb.y1 + 10):
            continue
        lines = a["lines"][:5]
        extra = len(a["lines"]) - len(lines)
        if extra > 0:
            lines.append((f"+{extra} more", pal["muted"]))
        w, h = max(len(t) for t, _ in lines) * cw, len(lines) * lh
        best = None
        for rank, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
            hit = any(math.cos(r) * dx + math.sin(r) * dy > 0.3 for r in a["dirs"])
            off = 9 + (Lpx if hit else 0)
            if dx == 1:
                rect = (px + off, py - h / 2, px + off + w, py + h / 2)
            elif dx == -1:
                rect = (px - off - w, py - h / 2, px - off, py + h / 2)
            elif dy == 1:
                rect = (px - w / 2, py + off, px + w / 2, py + off + h)
            else:
                rect = (px - w / 2, py - off - h, px + w / 2, py - off)
            score = sum(ov(rect, q) for q in obst) + sum(ov(rect, q) for q in placed) + 5 * (w * h - ov(rect, (bb.x0, bb.y0, bb.x1, bb.y1))) + rank * 0.01
            if best is None or score < best[0]:
                best = (score, rect, (dx, dy))
        score, rect, (dx, dy) = best
        if score > 0.4 * w * h:
            continue  # too crowded at this zoom level: zoom in to read it
        placed.append(rect)
        ha = "left" if dx == 1 else "right" if dx == -1 else "center"
        ax_px = rect[0] if dx == 1 else rect[2] if dx == -1 else (rect[0] + rect[2]) / 2
        for i, (text, col) in enumerate(lines):
            ty = rect[3] - (i + 0.5) * lh
            out.append(ax.annotate(text, xy=(a["x"], a["y"]), xytext=((ax_px - px) / ppx, (ty - py) / ppx),
                                   textcoords="offset points", fontsize=fs, ha=ha, va="center", color=col, zorder=8))
    return out


def legend_handles(pal: dict, theme: str) -> list:
    red = RED[theme]
    return [Line2D([], [], color=pal["muted"], lw=1, label="path"), Line2D([], [], color=WARNC[theme], lw=1.2, label="path: heading warning"),
            Line2D([], [], color=red, lw=1.2, label="path: backwards / error"), Line2D([], [], color=pal["other"], lw=1, ls=":", label="rotate in place"),
            Line2D([], [], marker="s", mfc="none", mec=pal["pick"], ls="", label="pick"), Line2D([], [], marker="s", mfc="none", mec=pal["drop"], ls="", label="drop"),
            Line2D([], [], marker="s", mfc="none", mec=kind_color(pal, "nop"), ls="", label="nop / home")]


def render_png(m: Model, path: str, theme: str = "light", opts: Optional[Opts] = None, dpi: int = 200) -> None:
    o = opts or Opts()
    pal = wp.PLOT_THEMES[theme]
    sev = severity_map(validate(m, o.tol))
    (x0, x1), (y0, y1) = data_bounds(m, o)
    w = 11.0
    fig = Figure(figsize=(w, min(max(w * (y1 - y0) / (x1 - x0), 5), 16)), dpi=dpi)
    ax = fig.add_axes([0.06, 0.06, 0.92, 0.92])
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    draw_static(ax, m, o, pal, theme, sev)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    draw_view(ax, m, o, pal, theme, sev)
    leg = ax.legend(handles=legend_handles(pal, theme), fontsize=6, loc="upper left", frameon=True, facecolor=pal["bg"], edgecolor=pal["spine"])
    for t in leg.get_texts():
        t.set_color(pal["muted"])
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.05)


# ----------------------------------------------------------------------------
# PDF report (map, points, routes) - print-friendly, no issues / warnings
# ----------------------------------------------------------------------------
def _natkey(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def export_pdf(m: Model, path: str, tol: float = 15.0) -> None:
    """A4 report: page 1 = plotted map, then points + locations, then routes. Deliberately contains no checks."""
    import datetime
    import textwrap
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    W, H = 8.27, 11.69
    MX, TOP, BOT = 0.55, 0.78, 0.62
    INK, BAND, AMB, ZEB, RULE, MUT = "#1c2228", "#1c2026", "#f5a623", "#f3f5f7", "#d5dbe1", "#667280"
    pal, theme = wp.PLOT_THEMES["light"], "light"
    when = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
    pages: List[Figure] = []

    def new_page(title: str, sub: str = ""):
        fig = Figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), W, TOP, fc=BAND, ec="none"))
        ax.add_patch(Rectangle((0, 0), 0.1, TOP, fc=AMB, ec="none"))
        ax.text(MX - 0.1, 0.30, title, color="white", fontsize=15, fontweight="bold", va="center")
        ax.text(MX - 0.1, 0.57, sub, color="#aab4c0", fontsize=7.5, va="center")
        ax.text(W - MX + 0.1, 0.30, "OpenTCS model report", color="#aab4c0", fontsize=7.5, va="center", ha="right")
        ax.text(W - MX + 0.1, 0.57, when, color="#aab4c0", fontsize=7.5, va="center", ha="right")
        pages.append(fig)
        return fig, ax

    picks = [l for l in m.locs.values() if loc_kind(m, l) == "pick"]
    drops = [l for l in m.locs.values() if loc_kind(m, l) == "drop"]
    nops = [l for l in m.locs.values() if loc_kind(m, l) == "nop"]
    sub = f"{m.name}"

    # ---------------- page 1: the map
    fig, ax = new_page("Plant map", sub)
    stats = (("POINTS", len(m.points)), ("PATHS", len(m.paths)), ("LOCATIONS", len(m.locs)), ("PICK", len(picks)),
             ("DROP", len(drops)), ("HOME / NOP", len(nops)), ("VEHICLES", len(m.vehicles)))
    bw = (W - 2 * MX) / len(stats)
    for k, (lab, val) in enumerate(stats):
        x = MX + k * bw
        ax.add_patch(Rectangle((x + 0.03, TOP + 0.2), bw - 0.06, 0.62, fc=ZEB, ec=RULE, lw=0.6))
        ax.text(x + bw / 2, TOP + 0.43, str(val), ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
        ax.text(x + bw / 2, TOP + 0.68, lab, ha="center", va="center", fontsize=6, color=MUT)
    y0, y1 = TOP + 1.05, H - 1.45
    o = Opts(tol=tol)
    (xa, xb), (ya, yb) = data_bounds(m, o)
    mp = fig.add_axes([MX / W, (H - y1) / H, (W - 2 * MX) / W, (y1 - y0) / H])
    mp.set_xlim(xa, xb)
    mp.set_ylim(ya, yb)
    draw_static(mp, m, o, pal, theme, {}, None)
    mp.set_xlim(xa, xb)
    mp.set_ylim(ya, yb)
    draw_view(mp, m, o, pal, theme, {}, fs=4.6)
    mp.tick_params(labelsize=5)
    hnd = [Line2D([], [], color=pal["muted"], lw=1, label="path (arrow = direction of travel)"),
           Line2D([], [], color=pal["other"], lw=1, ls=":", label="rotation in place"),
           Line2D([], [], marker="s", mfc="none", mec=pal["pick"], ls="", label="pick location"),
           Line2D([], [], marker="s", mfc="none", mec=pal["drop"], ls="", label="drop location"),
           Line2D([], [], marker="s", mfc="none", mec=kind_color(pal, "nop"), ls="", label="home / nop"),
           Line2D([], [], marker=">", color=pal["other"], ls="", label="point heading (theta)")]
    mp.legend(handles=hnd, loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=3, frameon=False, fontsize=6.3, handlelength=2.2)
    ax.text(MX, H - 0.74, "Plant coordinates (OpenTCS model units, mm). Arrows at points show the heading (theta property); locations are drawn on "
            "the point they are linked to.", fontsize=6.2, color=MUT, va="center")

    # ---------------- generic table engine
    state = {"fig": None, "ax": None, "y": 0.0, "cols": None, "title": ""}

    def start(title: str, sub2: str = ""):
        state["fig"], state["ax"] = new_page(title, sub2 or sub)
        state["y"] = TOP + 0.35

    def header(cols):
        ax = state["ax"]
        y = state["y"]
        ax.add_patch(Rectangle((MX, y), W - 2 * MX, 0.27, fc=BAND, ec="none"))
        x = MX
        for title, w, al, *_ in cols:
            tx = x + 0.07 if al == "left" else x + w - 0.07
            ax.text(tx, y + 0.135, title, color="white", fontsize=6.6, fontweight="bold", va="center", ha=al)
            x += w
        state["y"] = y + 0.27

    def section(text: str, note: str = ""):
        if state["y"] > H - BOT - 1.4:
            start(state["title"])
        ax = state["ax"]
        ax.text(MX, state["y"] + 0.14, text, fontsize=10.5, fontweight="bold", color=INK, va="center")
        ax.plot([MX, W - MX], [state["y"] + 0.3, state["y"] + 0.3], color=AMB, lw=1.6)
        state["y"] += 0.42
        if note:
            ax.text(MX, state["y"] + 0.02, note, fontsize=6.6, color=MUT, va="center")
            state["y"] += 0.22

    def table(cols, rows):
        header(cols)
        state["cols"] = cols
        for r_i, cells in enumerate(rows):
            lines = []
            for (title, w, al, mono, *_), cell in zip(cols, cells):
                chars = max(int((w - 0.14) * 72 / (6.7 * 0.56)), 4)
                lines.append(textwrap.wrap(str(cell), chars, break_long_words=True) or [""])
            n = max(len(l) for l in lines)
            h = 0.075 + 0.125 * n
            if state["y"] + h > H - BOT:
                start(state["title"])
                header(cols)
            ax, y = state["ax"], state["y"]
            if r_i % 2 == 0:
                ax.add_patch(Rectangle((MX, y), W - 2 * MX, h, fc=ZEB, ec="none"))
            x = MX
            for (title, w, al, mono, *_), ln in zip(cols, lines):
                for k, t in enumerate(ln):
                    tx = x + 0.07 if al == "left" else x + w - 0.07
                    ax.text(tx, y + 0.04 + 0.125 * (k + 0.5), t, fontsize=6.7, va="center", ha=al, color=INK,
                            family="DejaVu Sans Mono" if mono else "DejaVu Sans")
                x += w
            state["y"] = y + h
        ax = state["ax"]
        ax.plot([MX, W - MX], [state["y"], state["y"]], color=RULE, lw=0.6)
        state["y"] += 0.3

    # ---------------- points + locations
    state["title"] = "Points and locations"
    start("Points and locations", sub)
    section(f"Points ({len(m.points)})", "Position and heading of every point. Theta is degrees, counter-clockwise from +X.")
    rows = []
    for k, n in enumerate(sorted(m.points, key=_natkey), 1):
        p = m.points[n]
        locs = ", ".join(f"{l} ({loc_kind(m, m.locs[l])})" for l in loc_names_at(m, n))
        rows.append([k, n, f"{p.x:,.1f}", f"{p.y:,.1f}", "-" if p.theta is None else f"{p.theta:.2f}", locs or "-",
                     sum(1 for q in m.paths if q.src == n), sum(1 for q in m.paths if q.dst == n)])
    table([("#", 0.4, "right", True), ("Point", 0.85, "left", False), ("X", 1.05, "right", True), ("Y", 1.05, "right", True),
           ("Theta", 0.8, "right", True), ("Location", 1.9, "left", False), ("Out", 0.55, "right", True), ("In", 0.55, "right", True)], rows)
    section(f"Locations ({len(m.locs)})", "Pick / drop behaviour comes from the location type's allowed operations.")
    rows = []
    for l in sorted(m.locs.values(), key=lambda l: _natkey(l.name)):
        lk = l.links[0] if l.links else "-"
        th = m.points[lk].theta if lk in m.points else None
        rows.append([l.name, loc_kind(m, l), l.ltype, lk, "-" if th is None else f"{th:.2f}",
                     ", ".join(m.ltypes[l.ltype].ops) if l.ltype in m.ltypes else "-"])
    table([("Location", 1.1, "left", False), ("Kind", 0.7, "left", False), ("Type", 1.5, "left", False), ("Point", 0.8, "left", False),
           ("Theta", 0.8, "right", True), ("Allowed operations", 2.37, "left", False)], rows)

    # ---------------- routes (reachable only)
    state["title"] = "Routes"
    start("Routes", sub)
    lens_unit = 1000.0
    for label, a_kind, b_kind in (("Pick to drop", "pick", "drop"), ("Drop to pick", "drop", "pick"), ("Home / nop to pick", "nop", "pick")):
        rows = []
        for a in sorted((l for l in m.locs.values() if loc_kind(m, l) == a_kind), key=lambda l: _natkey(l.name)):
            for b in sorted((l for l in m.locs.values() if loc_kind(m, l) == b_kind), key=lambda l: _natkey(l.name)):
                r = shortest_route(m, a.name, b.name)
                if r is None:
                    continue
                total, steps = r
                nodes = [steps[0].src] + [p.dst for p in steps] if steps else [point_of(m, a.name)]
                rows.append([a.name, b.name, f"{total / lens_unit:,.2f}", len(steps), " > ".join(nodes)])
        if not rows:
            continue
        section(f"{label} routes ({len(rows)})", "Shortest route by path length, as OpenTCS routes it. Via lists every point the vehicle passes.")
        table([("From", 0.75, "left", False), ("To", 0.75, "left", False), ("Length (m)", 0.9, "right", True), ("Paths", 0.6, "right", True),
               ("Via points", 4.22, "left", True)], rows)

    # ---------------- footers + write
    n = len(pages)
    for i, f in enumerate(pages, 1):
        a = f.axes[0]
        a.plot([MX, W - MX], [H - 0.5, H - 0.5], color=RULE, lw=0.6)
        a.text(MX, H - 0.33, f"{m.name}", fontsize=6.6, color=MUT, va="center")
        a.text(W - MX, H - 0.33, f"Page {i} of {n}", fontsize=6.6, color=MUT, va="center", ha="right")
    with PdfPages(path) as pdf:
        pdf.infodict().update(Title=f"OpenTCS model report - {m.name}", Creator="tcs_viewer.py")
        for f in pages:
            pdf.savefig(f)


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    _TK_ERR: Optional[Exception] = None
except Exception as _e:  # headless / tkinter missing
    tk = ttk = filedialog = messagebox = FigureCanvasTkAgg = NavigationToolbar2Tk = None  # type: ignore
    _TK_ERR = _e

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".tcs_viewer.json")
HELP_TEXT = """\
Issues     every problem found, worst first. Click one to select + show the objects, double-click to zoom to them.
Objects    all points / paths / locations, filterable.
Route      shortest route (by path length, like OpenTCS) between two locations or points, with a step list that
           flags backwards driving and heading mismatches. "All pick -> drop" checks every pair.
Map        hover = details    click = select (Shift adds)    wheel = zoom at cursor
           arrow on a point = its theta (heading)      dotted arc = rotation in place
           square = location (teal pick, orange drop, amber nop)      red / amber path = problem
View       "Layout frame" shows the drawing OpenTCS's editor uses (points spread out); the checks always use
           the real plant coordinates.
Checks     REVERSE (backwards, no reverse speed)   REVERSE_DOCK (backs into a location)   HEADING_SRC / DST
           (path direction vs point theta, tolerance in Display)   LENGTH   ROT_NOOP   DEAD_END   NAME_MISMATCH
           NO_DROP_ROUTE / NO_PICK_ROUTE   LOC_LINK / LOC_TYPE   BAD_THETA   DUP_PATH   BROKEN_PATH
Shortcuts  Ctrl+O open   Ctrl+R reload   Ctrl+T theme   Esc clear selection
"""
SEV_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}


def _load_cfg() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


class TcsApp:
    DPI = 110
    _fonts = wp.App._fonts
    _style = wp.App._style
    _menu = wp.App._menu
    _card = wp.App._card
    _led = wp.App._led

    def __init__(self, root, path: Optional[str] = None) -> None:
        self.root = root
        self.cfg = _load_cfg()
        c = self.cfg
        self.theme = c.get("theme", "dark") if c.get("theme") in wp.UI_THEMES else "dark"
        self.m: Optional[Model] = None
        self.path: Optional[str] = None
        self.issues: List[Issue] = []
        self.sev: Dict[Obj, str] = {}
        self.obj_issues: Dict[Obj, List[Issue]] = defaultdict(list)
        self.sel: List[Obj] = []
        self.hover: List[Obj] = []
        self.route: List[str] = []
        self.route_ends: Tuple[Optional[str], Optional[str]] = (None, None)
        self.arcs: Dict[str, Tuple[List[float], List[float]]] = {}
        self._view_arts: list = []
        self._ovl: list = []
        self._view_job = self._tol_job = None
        self._has_view = False
        self._view_key = None
        self.init_lims = ((0, 1), (0, 1))
        self._syncing = False

        bv = lambda k, d: tk.BooleanVar(value=c.get(k, d))
        self.v_points, self.v_paths, self.v_locs = bv("points", True), bv("paths", True), bv("locs", True)
        self.v_labels, self.v_theta = bv("labels", True), bv("theta", True)
        self.v_only, self.v_layout = bv("only", False), bv("layout", False)
        self.v_focus = bv("focus", True)
        self.v_err, self.v_warn, self.v_info = bv("f_err", True), bv("f_warn", True), bv("f_info", False)
        self.tol = tk.StringVar(value=str(c.get("tol", 15)))
        self.filter, self.kindf = tk.StringVar(), tk.StringVar(value="All")
        self.r_from, self.r_to = tk.StringVar(), tk.StringVar()
        self.r_mode = tk.StringVar(value="Pick > Drop")
        self.status, self.info = tk.StringVar(value="Open an OpenTCS model XML to begin."), tk.StringVar()
        self.hud = tk.StringVar(value="X  -          Y  -")

        self._fonts()
        self._build_ui()
        for v in (self.v_points, self.v_paths, self.v_locs, self.v_labels, self.v_theta, self.v_only, self.v_focus):
            v.trace_add("write", lambda *_: self.redraw())
        self.v_layout.trace_add("write", lambda *_: self.redraw(reset=True))
        for v in (self.v_err, self.v_warn, self.v_info):
            v.trace_add("write", lambda *_: self._fill_issues())
        self.tol.trace_add("write", self._on_tol)
        self.filter.trace_add("write", lambda *_: self._fill_objects())
        self.kindf.trace_add("write", lambda *_: self._fill_objects())
        self.r_mode.trace_add("write", lambda *_: self._fill_routes())
        root.update_idletasks()
        target = path or (c.get("last_file") if c.get("last_file") and os.path.exists(c.get("last_file")) else None)
        if target:
            root.after(150, lambda: self.load(target))

    # ----------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        r, P = self.root, wp.UI_THEMES[self.theme]
        self.P, self.pal = P, wp.PLOT_THEMES[self.theme]
        r.title("OpenTCS model viewer")
        r.minsize(1000, 620)
        if not getattr(self, "_geo", False):
            sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
            r.geometry(self.cfg.get("geometry") or f"{min(1480, sw - 60)}x{min(920, sh - 110)}+20+20")
            self._geo = True
        r.protocol("WM_DELETE_WINDOW", self.close)
        self._style()

        hdr = tk.Frame(r, bg=P["panel"], height=58)
        hdr.pack(side="top", fill="x")
        hdr.pack_propagate(False)
        tk.Frame(hdr, bg=P["accent"], width=5).pack(side="left", fill="y")
        tb = tk.Frame(hdr, bg=P["panel"])
        tb.pack(side="left", padx=(14, 0))
        tk.Label(tb, text="OpenTCS model viewer", bg=P["panel"], fg=P["fg"], font=self.f_title).pack(anchor="w", pady=(9, 0))
        tk.Label(tb, textvariable=self.info, bg=P["panel"], fg=P["muted"], font=self.f_small).pack(anchor="w")
        rb = tk.Frame(hdr, bg=P["panel"])
        rb.pack(side="right", padx=14)
        ttk.Button(rb, text="Open XML", style="Accent.TButton", command=self.open_file).pack(side="right", padx=(8, 0))
        ttk.Button(rb, text="Help", style="Hdr.TButton", command=lambda: messagebox.showinfo("Help", HELP_TEXT)).pack(side="right", padx=(8, 0))
        ttk.Button(rb, text="Dark" if self.theme == "light" else "Light", style="Hdr.TButton", command=self.toggle_theme).pack(side="right", padx=(8, 0))
        eb = ttk.Menubutton(rb, text="Export", style="Hdr.TMenubutton")
        em = self._menu(eb)
        eb["menu"] = em
        em.add_command(label="Issues as CSV...", command=self.export_issues)
        em.add_command(label="Full report (.txt)...", command=self.export_report)
        em.add_command(label="Points as wp_plotter .txt...", command=self.export_points)
        em.add_command(label="PDF report (map, points, routes)...", command=self.export_pdf_report)
        em.add_command(label="Map image (PNG / SVG / PDF)...", command=self.export_image)
        eb.pack(side="right", padx=(8, 0))
        ttk.Button(rb, text="Reload", style="Hdr.TButton", command=self.reload).pack(side="right", padx=(8, 0))
        tk.Frame(r, bg=P["border"], height=1).pack(side="top", fill="x")

        sb = tk.Frame(r, bg=P["panel"], height=30)
        sb.pack(side="bottom", fill="x")
        sb.pack_propagate(False)
        tk.Frame(r, bg=P["border"], height=1).pack(side="bottom", fill="x")
        self.led = tk.Canvas(sb, width=14, height=14, bg=P["panel"], highlightthickness=0)
        self.led.pack(side="left", padx=(12, 6))
        self._led_item = self.led.create_oval(2, 2, 12, 12, fill=P["muted"], outline="")
        self.status_lbl = tk.Label(sb, textvariable=self.status, bg=P["panel"], fg=P["fg"], font=self.f_small, anchor="w")
        self.status_lbl.pack(side="left", fill="x", expand=True)

        paned = tk.PanedWindow(r, orient="horizontal", bg=P["bg"], sashwidth=6, sashrelief="flat", bd=0)
        paned.pack(fill="both", expand=True)
        left, right = tk.Frame(paned, bg=P["bg"], padx=10, pady=10), tk.Frame(paned, bg=P["bg"], padx=4, pady=10)
        paned.add(left, minsize=400, width=500, stretch="never")
        paned.add(right, minsize=520, stretch="always")

        # --- bottom cards (pinned) then notebook
        stack = tk.Frame(left, bg=P["bg"])
        stack.pack(side="bottom", fill="x")
        _c, ihead, ibody, _t = self._card(stack, "INSPECTOR", pady=(8, 0))
        ttk.Button(ihead, text="Copy", style="Tool.TButton", command=self.copy_inspector).pack(side="right")
        self.insp = tk.Text(ibody, height=7, wrap="word", bg=P["field"], fg=P["fg"], font=self.f_mono_s, bd=0, highlightthickness=0,
                            padx=8, pady=6, state="disabled", cursor="arrow", selectbackground=P["sel"])
        self.insp.pack(fill="x")
        self.insp.tag_configure("h", foreground=P["accent"], font=self.f_bold)
        self.insp.tag_configure("bad", foreground=RED[self.theme])
        self.insp.tag_configure("warn", foreground=WARNC[self.theme])
        self.insp.tag_configure("dim", foreground=P["muted"])
        _c, _h, dbody, _t = self._card(stack, "DISPLAY", pady=(8, 0))
        g = tk.Frame(dbody, bg=P["panel"])
        g.pack(fill="x")
        items = (("Points", self.v_points), ("Paths", self.v_paths), ("Locations", self.v_locs), ("Labels", self.v_labels),
                 ("Headings", self.v_theta), ("Only issues", self.v_only), ("Layout frame", self.v_layout), ("Focus route", self.v_focus))
        for i, (t, v) in enumerate(items):
            ttk.Checkbutton(g, text=t, variable=v).grid(row=i // 4, column=i % 4, sticky="w", padx=(0, 14), pady=2)

        self.nb = ttk.Notebook(left)
        self.nb.pack(fill="both", expand=True)

        t_iss = tk.Frame(self.nb, bg=P["panel"])
        self.nb.add(t_iss, text="Issues")
        chips = tk.Frame(t_iss, bg=P["panel"])
        chips.pack(fill="x", padx=10, pady=(10, 6))
        self.chip = {}
        for k, v in (("ERROR", self.v_err), ("WARN", self.v_warn), ("INFO", self.v_info)):
            cb = ttk.Checkbutton(chips, text=k.title(), variable=v)
            cb.pack(side="left", padx=(0, 14))
            self.chip[k] = cb
        ttk.Spinbox(chips, from_=1, to=90, increment=1, textvariable=self.tol, width=4).pack(side="right")
        tk.Label(chips, text="HEADING TOL (deg)", bg=P["panel"], fg=P["muted"], font=self.f_sec).pack(side="right", padx=(0, 6))
        self.t_iss = self._tree(t_iss, (("sev", "SEV", 58, "w"), ("code", "CHECK", 138, "w"), ("obj", "OBJECT", 150, "w"), ("msg", "MESSAGE", 600, "w")))
        self.t_iss.bind("<<TreeviewSelect>>", self._on_issue_select)
        self.t_iss.bind("<Double-1>", self._on_issue_dbl)
        self._sev_tags(self.t_iss)

        t_obj = tk.Frame(self.nb, bg=P["panel"])
        self.nb.add(t_obj, text="Objects")
        fr = tk.Frame(t_obj, bg=P["panel"])
        fr.pack(fill="x", padx=10, pady=10)
        ttk.Entry(fr, textvariable=self.filter).pack(side="left", fill="x", expand=True)
        ttk.Combobox(fr, textvariable=self.kindf, values=("All", "Points", "Paths", "Locations"), width=9, state="readonly").pack(side="left", padx=(8, 0))
        self.t_obj = self._tree(t_obj, (("kind", "KIND", 70, "w"), ("name", "NAME", 150, "w"), ("info", "INFO", 260, "w")), selectmode="extended")
        self.t_obj.bind("<<TreeviewSelect>>", self._on_obj_select)
        self.t_obj.bind("<Double-1>", lambda e: self._focus(self.sel))

        t_rt = tk.Frame(self.nb, bg=P["panel"])
        self.nb.add(t_rt, text="Route")
        rr = tk.Frame(t_rt, bg=P["panel"])
        rr.pack(fill="x", padx=10, pady=(10, 4))
        for i, (lab, var) in enumerate((("FROM", self.r_from), ("TO", self.r_to))):
            tk.Label(rr, text=lab, bg=P["panel"], fg=P["muted"], font=self.f_sec).grid(row=i, column=0, sticky="w", pady=3)
            cb = ttk.Combobox(rr, textvariable=var, width=14)
            cb.grid(row=i, column=1, sticky="we", padx=8, pady=3)
            cb.bind("<<ComboboxSelected>>", lambda e: self._auto_route())
            cb.bind("<Return>", lambda e: self.find_route())
            setattr(self, "cb_from" if i == 0 else "cb_to", cb)
        rr.columnconfigure(1, weight=1)
        bt = tk.Frame(rr, bg=P["panel"])
        bt.grid(row=2, column=0, columnspan=2, sticky="we", pady=(6, 0))
        ttk.Button(bt, text="Find route", style="AccentSm.TButton", command=self.find_route).pack(side="left")
        ttk.Button(bt, text="Swap", style="Tool.TButton", command=self.swap_route).pack(side="left", padx=6)
        ttk.Button(bt, text="Clear", style="Tool.TButton", command=self.clear_route).pack(side="left")
        ttk.Button(bt, text="Report", style="Tool.TButton", command=self.all_routes).pack(side="right")
        mr = tk.Frame(t_rt, bg=P["panel"])
        mr.pack(fill="x", padx=10, pady=(6, 4))
        tk.Label(mr, text="ROUTES", bg=P["panel"], fg=P["muted"], font=self.f_sec).pack(side="left")
        ttk.Combobox(mr, textvariable=self.r_mode, values=("Pick > Drop", "Drop > Pick", "Home / nop > Pick", "All locations"),
                     width=18, state="readonly").pack(side="left", padx=8)
        self.t_rt = self._tree(t_rt, (("from", "FROM", 62, "w"), ("to", "TO", 62, "w"), ("len", "LENGTH", 78, "e"), ("n", "PATHS", 52, "e"),
                                      ("st", "STATUS", 220, "w")), height=5)
        self._sev_tags(self.t_rt)
        self.t_rt.bind("<<TreeviewSelect>>", self._on_route_select)
        self.t_rt.bind("<Double-1>", lambda e: self._focus([("A", n) for n in self.route]))
        self.rtxt = tk.Text(t_rt, wrap="none", bg=P["field"], fg=P["fg"], font=self.f_mono_s, bd=0, highlightthickness=0, padx=8, pady=6,
                            state="disabled", selectbackground=P["sel"], height=7)
        ys = ttk.Scrollbar(t_rt, orient="vertical", command=self.rtxt.yview)
        xs = ttk.Scrollbar(t_rt, orient="horizontal", command=self.rtxt.xview)
        self.rtxt.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y", pady=(0, 10), padx=(0, 10))
        xs.pack(side="bottom", fill="x", padx=(10, 0), pady=(0, 6))
        self.rtxt.pack(fill="both", expand=True, padx=(10, 0), pady=(4, 2))
        self.rtxt.tag_configure("bad", foreground=RED[self.theme])
        self.rtxt.tag_configure("warn", foreground=WARNC[self.theme])
        self.rtxt.tag_configure("h", foreground=P["accent"], font=self.f_bold)
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab)

        # --- map
        mh = tk.Frame(right, bg=P["panel"], height=46, highlightbackground=P["border"], highlightthickness=1)
        mh.pack(fill="x")
        mh.pack_propagate(False)
        tk.Label(mh, text="MAP", bg=P["panel"], fg=P["muted"], font=self.f_sec).pack(side="left", padx=(14, 12))
        self.tool_btns = {}
        for key, text, cmd in (("reset", "Reset", self.reset_view), ("pan", "Pan", lambda: self._tool("pan")),
                               ("zoom", "Box zoom", lambda: self._tool("zoom")), ("in", "+", lambda: self._zoom(0.8)), ("out", "-", lambda: self._zoom(1.25))):
            b = ttk.Button(mh, text=text, style="Tool.TButton", command=cmd, width=9 if len(text) > 1 else 3)
            b.pack(side="left", padx=3, pady=8)
            self.tool_btns[key] = b
        tk.Label(mh, textvariable=self.hud, bg=P["panel"], fg=P["accent"], font=self.f_mono).pack(side="right", padx=14)
        box = tk.Frame(right, bg=self.pal["bg"], highlightbackground=P["border"], highlightthickness=1)
        box.pack(fill="both", expand=True)
        self.fig = Figure(dpi=self.DPI, facecolor=self.pal["bg"])
        self.ax = self.fig.add_axes([0.075, 0.05, 0.905, 0.93])
        self.canvas = FigureCanvasTkAgg(self.fig, master=box)
        self.toolbar = NavigationToolbar2Tk(self.canvas, box, pack_toolbar=False)
        self.canvas.get_tk_widget().configure(bg=self.pal["bg"], highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.get_tk_widget().bind("<Map>", lambda e: self.canvas.draw_idle())
        for ev, fn in (("motion_notify_event", self._on_motion), ("button_press_event", self._on_click), ("scroll_event", self._on_scroll),
                       ("button_release_event", lambda e: self._sched_view()), ("resize_event", lambda e: self._sched_view()),
                       ("axes_leave_event", lambda e: self._on_leave())):
            self.canvas.mpl_connect(ev, fn)
        self.tip = None
        r.bind("<Control-o>", lambda _e: self.open_file())
        r.bind("<Control-r>", lambda _e: self.reload())
        r.bind("<Control-t>", lambda _e: self.toggle_theme())
        r.bind("<Escape>", lambda _e: self._set_sel([]))
        self._show_empty()
        self._set_inspector([("Open an OpenTCS XML to see details here.", "dim")])

    def _tree(self, parent, cols, selectmode="browse", height=None):
        P = self.P
        holder = tk.Frame(parent, bg=P["panel"])
        if height:
            holder.pack(fill="x")
        else:
            holder.pack(fill="both", expand=True)
        tv = ttk.Treeview(holder, columns=[c[0] for c in cols], show="headings", selectmode=selectmode)
        if height:
            tv.configure(height=height)
        for cid, text, w, anc in cols:
            tv.heading(cid, text=text)
            tv.column(cid, width=w, anchor=anc, stretch=(cid == cols[-1][0]))
        ys = ttk.Scrollbar(holder, orient="vertical", command=tv.yview)
        xs = ttk.Scrollbar(holder, orient="horizontal", command=tv.xview)
        tv.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y", pady=(0, 10 if not height else 2), padx=(0, 10))
        xs.pack(side="bottom", fill="x", padx=(10, 0))
        tv.pack(fill="both", expand=not height, padx=(10, 0), pady=(0, 2))
        return tv

    def _sev_tags(self, tv) -> None:
        tv.tag_configure("ERROR", foreground=RED[self.theme])
        tv.tag_configure("WARN", foreground=WARNC[self.theme])
        tv.tag_configure("INFO", foreground=self.P["muted"])

    def toggle_theme(self) -> None:
        for job in (self._view_job, self._tol_job):
            if job:
                self.root.after_cancel(job)
        self._view_job = self._tol_job = None
        self.theme = "light" if self.theme == "dark" else "dark"
        self.cfg["theme"] = self.theme
        for w in self.root.winfo_children():
            w.destroy()
        self._view_arts, self._ovl, self._has_view = [], [], False
        self._build_ui()
        if self.m:
            self._fill_all()
            self.redraw(reset=True)
            self._set_sel(list(self.sel))

    # ------------------------------------------------------------ helpers
    def _status(self, msg: str, error: bool = False) -> None:
        self.status.set(msg)
        self.status_lbl.configure(fg=RED[self.theme] if error else self.P["fg"])
        self._led("error" if error else "ok")

    def opts(self) -> Opts:
        try:
            tol = float(self.tol.get())
        except ValueError:
            tol = 15.0
        return Opts(self.v_points.get(), self.v_paths.get(), self.v_locs.get(), self.v_labels.get(), self.v_theta.get(),
                    self.v_only.get(), self.v_layout.get(), tol, self.v_focus.get())

    def _show_empty(self) -> None:
        ax = self.ax
        ax.clear()
        ax.set_facecolor(self.pal["bg"])
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Open an OpenTCS model XML\n(Open XML, top right)", ha="center", va="center", transform=ax.transAxes,
                color=self.pal["muted"], fontsize=11)
        self.canvas.draw_idle()

    # ------------------------------------------------------------ loading
    def open_file(self) -> None:
        p = filedialog.askopenfilename(parent=self.root, initialdir=self.cfg.get("last_dir") or None,
                                       filetypes=[("OpenTCS model", "*.xml"), ("All files", "*.*")])
        if p:
            self.load(p)

    def reload(self) -> None:
        if self.path:
            self.load(self.path, keep_view=True)

    def load(self, path: str, keep_view: bool = False) -> None:
        try:
            m = load_model(path)
        except (ET.ParseError, OSError) as e:
            self._status(f"Could not read {path}: {e}", error=True)
            return
        self.m, self.path = m, path
        self.cfg.update(last_file=path, last_dir=os.path.dirname(path))
        self.sel, self.hover, self.route = [], [], []
        self.revalidate(redraw=False)
        names = sorted(m.locs, key=lambda n: (len(n), n)) + sorted(m.points, key=lambda n: (len(n), n))
        self.cb_from["values"] = names
        self.cb_to["values"] = names
        self._fill_all()
        self.redraw(reset=not keep_view)
        self.root.title(f"OpenTCS model viewer - {m.name}")
        self._set_inspector([("Click an object on the map, or an issue / object in the lists.", "dim")])
        e, w = sum(i.sev == "ERROR" for i in self.issues), sum(i.sev == "WARN" for i in self.issues)
        self._status(f"Loaded {os.path.basename(path)}: {e} errors, {w} warnings", error=False)
        self._led("error" if e else "busy" if w else "ok")

    def _on_tol(self, *_a) -> None:
        if self._tol_job:
            self.root.after_cancel(self._tol_job)
        self._tol_job = self.root.after(400, lambda: (self.revalidate(), None))

    def revalidate(self, redraw: bool = True) -> None:
        if self.m is None:
            return
        self.issues = validate(self.m, self.opts().tol)
        self.sev = severity_map(self.issues)
        self.obj_issues = defaultdict(list)
        for i in self.issues:
            for ob in i.objs:
                self.obj_issues[ob].append(i)
        cnt = {k: sum(i.sev == k for i in self.issues) for k in SEV_ORDER}
        for k, cb in self.chip.items():
            cb.configure(text=f"{k.title()} ({cnt[k]})")
        self.nb.tab(0, text=f"Issues ({cnt['ERROR'] + cnt['WARN']})")
        m = self.m
        self.info.set(f"{m.name}   |   {len(m.points)} points   {len(m.paths)} paths   {len(m.locs)} locations   {len(m.vehicles)} vehicles"
                      f"   |   {cnt['ERROR']} errors   {cnt['WARN']} warnings")
        if redraw:
            self._fill_issues()
            self._fill_routes()
            self.redraw()

    def _fill_all(self) -> None:
        self._fill_issues()
        self._fill_objects()
        self._fill_routes()

    def _fill_issues(self) -> None:
        ch = self.t_iss.get_children()
        if ch:
            self.t_iss.delete(*ch)
        want = {"ERROR": self.v_err.get(), "WARN": self.v_warn.get(), "INFO": self.v_info.get()}
        for k, i in enumerate(self.issues):
            if want[i.sev]:
                self.t_iss.insert("", "end", iid=str(k), tags=(i.sev,), values=(i.sev, i.code, i.obj_text, i.msg))

    def _fill_objects(self) -> None:
        ch = self.t_obj.get_children()
        if ch:
            self.t_obj.delete(*ch)
        if self.m is None:
            return
        q, kf = self.filter.get().strip().lower(), self.kindf.get()
        m = self.m
        rows = []
        if kf in ("All", "Locations"):
            rows += [("L", "Location", n, f"{l.ltype}   {loc_kind(m, l)}   at {', '.join(l.links)}") for n, l in m.locs.items()]
        if kf in ("All", "Points"):
            rows += [("P", "Point", n, f"x {p.x:,.0f}  y {p.y:,.0f}  theta {'-' if p.theta is None else format(p.theta, '.1f')}") for n, p in m.points.items()]
        if kf in ("All", "Paths"):
            rows += [("A", "Path", p.name, f"{p.src} > {p.dst}   len {p.length:g}   {m.pinfo[p.name]['kind']}") for p in m.paths]
        for ob, kind, name, info in rows:
            if q and q not in name.lower() and q not in info.lower():
                continue
            sv = self.sev.get((ob, name))
            self.t_obj.insert("", "end", iid=f"{ob}|{name}", tags=(sv,) if sv else (), values=(kind, name, info))
        self._sev_tags(self.t_obj)

    # ---------------------------------------------------------- selection
    def _on_issue_select(self, _e=None) -> None:
        sel = self.t_iss.selection()
        if sel and not self._syncing:
            i = self.issues[int(sel[0])]
            self._set_sel(i.objs, src="issue", issue=i)
            self._focus(i.objs, only_if_hidden=True)

    def _on_issue_dbl(self, _e=None) -> None:
        sel = self.t_iss.selection()
        if sel:
            self._focus(self.issues[int(sel[0])].objs)

    def _on_obj_select(self, _e=None) -> None:
        if self._syncing:
            return
        objs = [tuple(i.split("|", 1)) for i in self.t_obj.selection()]
        self._set_sel(objs, src="obj")  # type: ignore[arg-type]
        self._focus(self.sel, only_if_hidden=True)

    def _set_sel(self, objs: List[Obj], src: Optional[str] = None, issue: Optional[Issue] = None) -> None:
        self.sel = list(objs)
        if src != "obj":
            self._syncing = True
            ids = [f"{k}|{n}" for k, n in self.sel if self.t_obj.exists(f"{k}|{n}")]
            self.t_obj.selection_set(ids)
            if ids:
                self.t_obj.see(ids[0])
            self._syncing = False
        self._refresh_overlay()
        lines: List[Tuple[str, str]] = []
        if issue is not None:
            lines += [(f"{issue.sev}  {issue.code}\n", "bad" if issue.sev == "ERROR" else "warn" if issue.sev == "WARN" else "dim"), (issue.msg + "\n\n", "")]
        if self.sel:
            lines += self.describe(*self.sel[0])
            if len(self.sel) > 1:
                lines.append((f"\n+ {len(self.sel) - 1} more selected: " + ", ".join(n for _, n in self.sel[1:6]), "dim"))
        elif issue is None:
            lines = [("Nothing selected.", "dim")]
        self._set_inspector(lines)

    def _set_inspector(self, parts: List[Tuple[str, str]]) -> None:
        self.insp.configure(state="normal")
        self.insp.delete("1.0", "end")
        for text, tag in parts:
            self.insp.insert("end", text, tag or ())
        self.insp.configure(state="disabled")

    def copy_inspector(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.insp.get("1.0", "end").strip())
        self._status("Inspector text copied.")

    def describe(self, kind: str, name: str) -> List[Tuple[str, str]]:
        m = self.m
        out: List[Tuple[str, str]] = []
        if m is None:
            return out
        if kind == "P" and name in m.points:
            p = m.points[name]
            th = "missing" if p.theta is None else f"{p.theta:.2f} deg"
            outs = [x.name for x in m.paths if x.src == name]
            ins = [x.name for x in m.paths if x.dst == name]
            out += [(f"POINT {name}   [{p.ptype}]\n", "h"), (f"x {p.x:,.1f}   y {p.y:,.1f}   theta {th}\n", ""),
                    (f"out: {', '.join(outs) or '-'}\nin : {', '.join(ins) or '-'}\n", "")]
            locs = loc_names_at(m, name)
            if locs:
                out.append((f"location: {', '.join(f'{l} ({loc_kind(m, m.locs[l])})' for l in locs)}\n", ""))
            out.append(("props: " + "  ".join(f"{k}={v.strip()}" for k, v in p.props.items() if k != "theta") + "\n", "dim"))
        elif kind == "A":
            p = next((x for x in m.paths if x.name == name), None)
            if p:
                i = m.pinfo[name]
                out += [(f"PATH {name}\n", "h"), (f"{p.src} > {p.dst}   length {p.length:g}   v {p.vmax:g} / reverse {p.vrev:g}"
                                                  f"{'   LOCKED' if p.locked else ''}\n", "")]
                if i["kind"] == "move":
                    s, d = m.points.get(p.src), m.points.get(p.dst)
                    out.append((f"drive {i['dist']:.0f}   heading {i['heading']:.1f}\n", ""))
                    if s and s.theta is not None and i["ds"] is not None:
                        out.append((f"{p.src} theta {s.theta:.1f}  (heading diff {i['ds']:+.1f})\n", ""))
                    if d and d.theta is not None and i["dt"] is not None:
                        out.append((f"{p.dst} theta {d.theta:.1f}  (heading diff {i['dt']:+.1f})\n", ""))
                else:
                    out.append((f"rotate in place: {m.points[p.src].theta} > {m.points[p.dst].theta}"
                                f"{'' if i['dth'] is None else f'   ({i[chr(100) + chr(116) + chr(104)]:+.1f} deg)'}\n", ""))
        elif kind == "L" and name in m.locs:
            l = m.locs[name]
            lt = m.ltypes.get(l.ltype)
            out += [(f"LOCATION {name}   [{loc_kind(m, l)}]\n", "h"), (f"type {l.ltype}   ops {', '.join(lt.ops) if lt else '?'}\n", ""),
                    (f"linked point: {', '.join(l.links)}", "")]
            for lk in l.links:
                if lk in m.points and m.points[lk].theta is not None:
                    out.append((f"  (theta {m.points[lk].theta:.1f})", ""))
            out.append((f"\nstored position {l.x:,.0f}, {l.y:,.0f}   (editor drawing frame)\n", "dim"))
            out.append(("props: " + "  ".join(f"{k}={v}" for k, v in l.props.items()) + "\n", "dim"))
        for i in self.obj_issues.get((kind, name), [])[:4]:
            out.append((f"\n{i.sev} {i.code}: {i.msg}", "bad" if i.sev == "ERROR" else "warn" if i.sev == "WARN" else "dim"))
        return out

    # ------------------------------------------------------------- drawing
    def redraw(self, reset: bool = False) -> None:
        if self.m is None:
            return
        o, m = self.opts(), self.m
        prev = (self.ax.get_xlim(), self.ax.get_ylim()) if self._has_view and not reset else None
        self.ax.set_axis_on()
        draw_static(self.ax, m, o, self.pal, self.theme, self.sev, self.route)
        self.init_lims = data_bounds(m, o)
        key = (o.layout, self.path)
        lims = prev if (prev is not None and self._view_key == key) else self.init_lims
        self.ax.set_xlim(*lims[0])
        self.ax.set_ylim(*lims[1])
        self._has_view, self._view_key = True, key
        leg = self.ax.legend(handles=legend_handles(self.pal, self.theme), fontsize=6, loc="upper left", frameon=True,
                             facecolor=self.pal["tip_bg"], edgecolor=self.pal["tip_edge"], framealpha=0.9)
        for t in leg.get_texts():
            t.set_color(self.pal["muted"])
        self._view_arts, self._ovl, self.tip = [], [], None
        self._refresh_view()

    def _sched_view(self) -> None:
        if self.m is None:
            return
        if self._view_job:
            self.root.after_cancel(self._view_job)
        self._view_job = self.root.after(140, self._refresh_view)

    def _refresh_view(self) -> None:
        self._view_job = None
        if self.m is None:
            return
        for a in self._view_arts:
            try:
                a.remove()
            except (ValueError, NotImplementedError):
                pass
        self._view_arts, self.arcs = draw_view(self.ax, self.m, self.opts(), self.pal, self.theme, self.sev, route=self.route)
        self._refresh_overlay()

    def _glow(self, ob: Obj, color: str, lw: float, alpha: float) -> list:
        m, o, ax = self.m, self.opts(), self.ax
        k, n = ob
        arts: list = []
        u = units_per_pt(ax)
        if k == "P" and n in m.points and o.points:
            x, y = ppos(m, o, n)
            arts += ax.plot([x], [y], "o", ms=15, mfc="none", mec=color, mew=1.6, alpha=min(alpha + 0.3, 1), zorder=9)
            th = m.points[n].theta
            if th is not None and o.theta:
                r = math.radians(th)
                arts += ax.plot([x, x + ARROW_PT * u * math.cos(r)], [y, y + ARROW_PT * u * math.sin(r)], "-", color=color, lw=lw,
                                alpha=alpha, solid_capstyle="round", zorder=3)
        elif k == "A" and o.paths:
            p = next((x for x in m.paths if x.name == n), None)
            if p and p.src in m.points and p.dst in m.points:
                if n in self.arcs and not o.layout:
                    arts += ax.plot(*self.arcs[n], "-", color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=3)
                else:
                    (x0, y0), (x1, y1) = ppos(m, o, p.src), ppos(m, o, p.dst)
                    arts += ax.plot([x0, x1], [y0, y1], "-", color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=3)
        elif k == "L" and n in m.locs and o.locs:
            x, y = lpos(m, o, m.locs[n])
            arts += ax.plot([x], [y], "s", ms=18, mfc="none", mec=color, mew=2.0, alpha=min(alpha + 0.3, 1), zorder=9)
        return arts

    def _refresh_overlay(self) -> None:
        for a in self._ovl:
            try:
                a.remove()
            except (ValueError, NotImplementedError):
                pass
        self._ovl, self.tip = [], None
        if self.m is None:
            return
        o, m, ax, pal = self.opts(), self.m, self.ax, self.pal
        rc = ROUTEC[self.theme]
        if 0 < len(self.route) <= 80:
            for k, n in enumerate(self.route, 1):
                p = next((x for x in m.paths if x.name == n), None)
                if p is None or p.src not in m.points or p.dst not in m.points:
                    continue
                if n in self.arcs and not o.layout:
                    xs, ys = self.arcs[n]
                    mx, my = xs[len(xs) // 2], ys[len(ys) // 2]
                else:
                    (x0, y0), (x1, y1) = ppos(m, o, p.src), ppos(m, o, p.dst)
                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                self._ovl.append(ax.annotate(str(k), (mx, my), ha="center", va="center", fontsize=5.5, color="white", fontweight="bold",
                                             zorder=16, bbox=dict(boxstyle="circle,pad=0.18", fc=rc, ec="none")))
        for tag, name in zip(("START", "GOAL"), self.route_ends):
            if name and name in m.points and self.route:
                x, y = ppos(m, o, name)
                self._ovl.append(ax.annotate(tag, (x, y), xytext=(-14, 14), textcoords="offset points", color="white", fontsize=7,
                                             fontweight="bold", zorder=20, bbox=dict(boxstyle="round,pad=0.25", fc=rc, ec="none"),
                                             arrowprops=dict(arrowstyle="-", color=rc, lw=1.0)))
        for ob in self.hover:
            self._ovl += self._glow(ob, pal["fg"], 3.5, 0.35)
        for ob in self.sel:
            self._ovl += self._glow(ob, SELC[self.theme], 4.0, 0.6)
        self.canvas.draw_idle()

    # --------------------------------------------------------- map events
    def _hit(self, event) -> List[Obj]:
        import numpy as np
        if self.m is None or event.inaxes is not self.ax:
            return []
        m, o, tr = self.m, self.opts(), self.ax.transData
        c = np.array([event.x, event.y], dtype=float)
        Lpx = ARROW_PT * self.fig.dpi / 72.0
        cands: List[Tuple[float, Obj]] = []

        def seg(a, b):
            ab = b - a
            den = float((ab ** 2).sum()) or 1.0
            t = min(max(float(((c - a) * ab).sum()) / den, 0.0), 1.0)
            return float(np.hypot(*(a + ab * t - c)))
        if o.points:
            for n in m.points:
                p = np.array(tr.transform(ppos(m, o, n)))
                d = float(np.hypot(*(p - c)))
                th = m.points[n].theta
                if th is not None and o.theta:
                    r = math.radians(th)
                    d = min(d, seg(p, p + Lpx * np.array([math.cos(r), math.sin(r)])))
                if d <= 8:
                    cands.append((d, ("P", n)))
        if o.locs:
            for n, l in m.locs.items():
                p = np.array(tr.transform(lpos(m, o, l)))
                d = float(np.hypot(*(p - c)))
                if d <= 11:
                    cands.append((d - 4.0, ("L", n)))
        if o.paths:
            for p in m.paths:
                if p.src not in m.points or p.dst not in m.points:
                    continue
                if m.pinfo[p.name]["kind"] == "rotate" and not o.layout:
                    xs, ys = self.arcs.get(p.name, ([], []))
                    if xs:
                        pts = tr.transform(np.column_stack([xs, ys]))
                        d = float(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]).min())
                        if d <= 5:
                            cands.append((d + 3.0, ("A", p.name)))
                    continue
                d = seg(np.array(tr.transform(ppos(m, o, p.src))), np.array(tr.transform(ppos(m, o, p.dst))))
                if d <= 5:
                    cands.append((d + 3.0, ("A", p.name)))
        if not cands:
            return []
        cands.sort(key=lambda t: t[0])
        return [ob for d, ob in cands if d <= cands[0][0] + 2.0][:6]

    def _on_leave(self) -> None:
        self.hud.set("X  -          Y  -")
        if self.hover:
            self.hover = []
            self._refresh_overlay()

    def _on_motion(self, event) -> None:
        if self.m is None:
            return
        if event.inaxes is not self.ax or event.xdata is None:
            self._on_leave()
            return
        self.hud.set(f"X {event.xdata:>12,.1f}    Y {event.ydata:>12,.1f}")
        hit = self._hit(event)
        if hit == self.hover:
            return
        self.hover = hit
        self._refresh_overlay()
        if hit:
            text = "\n".join(self._short(*ob) for ob in hit[:4])
            fw, fh = self.canvas.get_width_height()
            right_half, top_half = event.x > fw * 0.55, event.y > fh * 0.55
            self.tip = self.ax.annotate(text, xy=(event.xdata, event.ydata), xytext=(-12 if right_half else 12, -12 if top_half else 12),
                                        textcoords="offset points", fontsize=7, color=self.pal["fg"], zorder=30, annotation_clip=False,
                                        ha="right" if right_half else "left", va="top" if top_half else "bottom", multialignment="left",
                                        bbox=dict(boxstyle="round,pad=0.4", fc=self.pal["tip_bg"], ec=self.pal["tip_edge"], lw=0.6))
            self._ovl.append(self.tip)
            self.canvas.draw_idle()

    def _short(self, kind: str, name: str) -> str:
        m = self.m
        if kind == "P":
            p = m.points[name]
            return f"{name}  theta {'-' if p.theta is None else format(p.theta, '.1f')}\nx {p.x:,.0f}  y {p.y:,.0f}"
        if kind == "A":
            p = next(x for x in m.paths if x.name == name)
            i = m.pinfo[name]
            t = f"{name}\nlen {p.length:g}   " + (f"heading {i['heading']:.1f}" if i["kind"] == "move" else "rotate in place")
            return t + ("".join(f"\n{x.code}" for x in self.obj_issues.get(("A", name), [])[:2]))
        l = m.locs[name]
        return f"{name}  {loc_kind(m, l)}\n{l.ltype}  at {', '.join(l.links)}"

    def _on_click(self, event) -> None:
        if self.m is None or event.inaxes is not self.ax or event.button != 1 or str(self.toolbar.mode):
            return
        hit = self._hit(event)
        if event.key == "shift":
            hit = self.sel + [h for h in hit if h not in self.sel]
        if self.nb.index(self.nb.select()) == 2:
            self._route_click(hit)
        self._set_sel(hit)

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        f = 0.8 if event.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(event.xdata - (event.xdata - x0) * f, event.xdata + (x1 - event.xdata) * f)
        self.ax.set_ylim(event.ydata - (event.ydata - y0) * f, event.ydata + (y1 - event.ydata) * f)
        self.canvas.draw_idle()
        self._sched_view()

    def _zoom(self, f: float) -> None:
        if self.m is None:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.ax.set_xlim(cx - (cx - x0) * f, cx + (x1 - cx) * f)
        self.ax.set_ylim(cy - (cy - y0) * f, cy + (y1 - cy) * f)
        self.canvas.draw_idle()
        self._sched_view()

    def reset_view(self) -> None:
        if self.m is not None:
            self.ax.set_xlim(*self.init_lims[0])
            self.ax.set_ylim(*self.init_lims[1])
            self.canvas.draw_idle()
            self._sched_view()

    def _tool(self, which: str) -> None:
        (self.toolbar.pan if which == "pan" else self.toolbar.zoom)()
        mode = str(self.toolbar.mode)
        for key in ("pan", "zoom"):
            self.tool_btns[key].configure(style="ToolOn.TButton" if key in mode else "Tool.TButton")

    def _focus(self, objs: List[Obj], only_if_hidden: bool = False) -> None:
        """Pan / zoom so the given objects are visible."""
        if self.m is None or not objs:
            return
        m, o = self.m, self.opts()
        pts: List[Tuple[float, float]] = []
        for k, n in objs:
            if k == "P" and n in m.points:
                pts.append(ppos(m, o, n))
            elif k == "L" and n in m.locs:
                pts.append(lpos(m, o, m.locs[n]))
            elif k == "A":
                p = next((x for x in m.paths if x.name == n), None)
                if p and p.src in m.points and p.dst in m.points:
                    pts += [ppos(m, o, p.src), ppos(m, o, p.dst)]
        if not pts:
            return
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        if only_if_hidden and all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in pts):
            return
        w = max(max(xs) - min(xs), 2500.0) * 1.8
        h = max(max(ys) - min(ys), 2500.0) * 1.8
        asp = (y1 - y0) / (x1 - x0)
        if h / w < asp:
            h = w * asp
        else:
            w = h / asp
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        self.ax.set_xlim(cx - w / 2, cx + w / 2)
        self.ax.set_ylim(cy - h / 2, cy + h / 2)
        self.canvas.draw_idle()
        self._sched_view()

    # -------------------------------------------------------------- routes
    def _route_pairs(self) -> List[Tuple[Loc, Loc]]:
        m = self.m
        locs = sorted(m.locs.values(), key=lambda l: (len(l.name), l.name))
        src, dst = {"Pick > Drop": ("pick", "drop"), "Drop > Pick": ("drop", "pick"), "Home / nop > Pick": ("nop", "pick")}.get(self.r_mode.get(), (None, None))
        return [(a, b) for a in locs for b in locs if a is not b and (src is None or (loc_kind(m, a) == src and loc_kind(m, b) == dst))]

    def _fill_routes(self) -> None:
        ch = self.t_rt.get_children()
        if ch:
            self.t_rt.delete(*ch)
        if self.m is None:
            return
        tol, un = self.opts().tol, 0
        for a, b in self._route_pairs():
            st = route_stats(self.m, a.name, b.name, tol)
            if st is None:
                un += 1
                self.t_rt.insert("", "end", iid=f"{a.name}|{b.name}", tags=("ERROR",), values=(a.name, b.name, "-", "-", "UNREACHABLE"))
                continue
            notes = [t for t in (f"backwards x{st['back']}" if st["back"] else "", f"backs into location x{st['dock']}" if st["dock"] else "",
                                 f"heading x{st['hd']}" if st["hd"] else "") if t]
            tag = ("ERROR",) if st["back"] else ("WARN",) if (st["dock"] or st["hd"]) else ()
            self.t_rt.insert("", "end", iid=f"{a.name}|{b.name}", tags=tag,
                             values=(a.name, b.name, f"{st['length']:,.0f}", st["n"], ", ".join(notes) or "ok"))
        self.nb.tab(2, text=f"Route ({un} unreachable)" if un else "Route")

    def _on_route_select(self, _e=None) -> None:
        sel = self.t_rt.selection()
        if sel:
            a, b = sel[0].split("|", 1)
            self.r_from.set(a)
            self.r_to.set(b)
            self.find_route()

    def _on_tab(self, _e=None) -> None:
        if self.m is not None and self.nb.index(self.nb.select()) == 2 and not self.route:
            kids = self.t_rt.get_children()
            ok = [k for k in kids if "ERROR" not in self.t_rt.item(k, "tags")]
            if kids:
                self.t_rt.selection_set((ok or kids)[0])
        if self.nb.index(self.nb.select()) == 2:
            self._status("Route tab: pick a row, or click a location on the map for FROM, then another for TO.")

    def _route_click(self, hit: List[Obj]) -> None:
        """Route tab open: map clicks pick FROM, then TO."""
        pick = next((n for k, n in hit if k == "L"), None) or next((n for k, n in hit if k == "P"), None)
        if not pick:
            return
        if not self.r_from.get() or (self.r_from.get() and self.r_to.get() and not getattr(self, "_rt_half", False)) or not getattr(self, "_rt_half", False):
            self.r_from.set(pick)
            self.r_to.set("")
            self._rt_half = True
            self._status(f"Route FROM {pick} - now click the TO location.")
        else:
            self.r_to.set(pick)
            self._rt_half = False
            self.find_route(focus=False)

    def _route_write(self, text: str) -> None:
        self.rtxt.configure(state="normal")
        self.rtxt.delete("1.0", "end")
        for line in text.splitlines():
            tag = "bad" if "!!" in line or "UNREACHABLE" in line or "BACKWARDS x" in line else "warn" if (" ! " in line or "~" in line or "heading x" in line or "backs-in" in line) else ""
            self.rtxt.insert("end", line + "\n", tag or ())
        self.rtxt.configure(state="disabled")

    def find_route(self, focus: bool = True) -> None:
        if self.m is None:
            return
        a, b = self.r_from.get().strip(), self.r_to.get().strip()
        if point_of(self.m, a) is None or point_of(self.m, b) is None:
            self._status("Pick a valid FROM and TO (location or point name).", error=True)
            return
        text, steps, bad = route_text(self.m, a, b, self.opts().tol)
        self.route, self.route_ends = steps, (point_of(self.m, a), point_of(self.m, b))
        self._route_write(text)
        self.redraw()
        if steps and focus:
            self._focus([("A", s) for s in steps])
        self._status(f"Route {a} > {b}: {len(steps)} paths" + (f", {bad} flagged" if bad else ", no direction problems") if steps else text, error=not steps)

    def _auto_route(self) -> None:
        if self.m is not None and point_of(self.m, self.r_from.get().strip()) and point_of(self.m, self.r_to.get().strip()):
            self.find_route()

    def swap_route(self) -> None:
        a, b = self.r_from.get(), self.r_to.get()
        self.r_from.set(b)
        self.r_to.set(a)

    def clear_route(self) -> None:
        self.route, self.route_ends = [], (None, None)
        self._route_write("")
        self.redraw()

    def all_routes(self) -> None:
        if self.m is not None:
            self._route_write(pick_drop_report(self.m, self.opts().tol))

    # -------------------------------------------------------------- export
    def _ask_save(self, name: str, types) -> Optional[str]:
        return filedialog.asksaveasfilename(parent=self.root, initialfile=name, initialdir=self.cfg.get("last_dir") or None,
                                            defaultextension=os.path.splitext(name)[1], filetypes=types) or None

    def export_issues(self) -> None:
        if self.m is None:
            return
        p = self._ask_save("tcs_issues.csv", [("CSV", "*.csv")])
        if p:
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["severity", "check", "objects", "message"])
                for i in self.issues:
                    w.writerow([i.sev, i.code, " | ".join(n for _, n in i.objs), i.msg])
            self._status(f"Saved {len(self.issues)} issues to {p}")

    def export_report(self) -> None:
        if self.m is None:
            return
        p = self._ask_save("tcs_report.txt", [("Text", "*.txt")])
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(full_report(self.m, self.opts().tol))
            self._status(f"Saved report to {p}")

    def export_points(self) -> None:
        if self.m is None:
            return
        p = self._ask_save("tcs_points.txt", [("wp_plotter text", "*.txt")])
        if p:
            wps = [wp.Waypoint(n, pt.x, pt.y, pt.theta if pt.theta is not None else 0.0) for n, pt in self.m.points.items()]
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(wp.serialize_waypoints(wps, "txt"))
            self._status(f"Saved {len(wps)} points (name / x / y / theta) to {p} - open it in wp_plotter")

    def export_pdf_report(self) -> None:
        if self.m is None:
            return
        p = self._ask_save("tcs_report.pdf", [("PDF", "*.pdf")])
        if p:
            try:
                export_pdf(self.m, p, self.opts().tol)
            except OSError as e:
                self._status(f"Could not write PDF: {e}", error=True)
                return
            self._status(f"Saved PDF report to {p}")

    def export_image(self) -> None:
        if self.m is None:
            return
        p = self._ask_save("tcs_map.png", [("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")])
        if p:
            render_png(self.m, p, self.theme, self.opts())
            self._status(f"Saved map image to {p}")

    def close(self) -> None:
        self.cfg.update(geometry=self.root.geometry(), theme=self.theme, tol=self.tol.get(), points=self.v_points.get(), paths=self.v_paths.get(),
                        locs=self.v_locs.get(), labels=self.v_labels.get(), theta=self.v_theta.get(), only=self.v_only.get(),
                        layout=self.v_layout.get(), focus=self.v_focus.get(), f_err=self.v_err.get(), f_warn=self.v_warn.get(), f_info=self.v_info.get())
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=1)
        except OSError:
            pass
        self.root.destroy()


def full_report(m: Model, tol: float) -> str:
    iss = validate(m, tol)
    cnt = {k: sum(i.sev == k for i in iss) for k in SEV_ORDER}
    out = [f"OpenTCS model '{m.name}': {len(m.points)} points, {len(m.paths)} paths, {len(m.locs)} locations, {len(m.vehicles)} vehicles",
           f"heading tolerance {tol:g} deg   |   {cnt['ERROR']} errors, {cnt['WARN']} warnings, {cnt['INFO']} info", ""]
    for sev in ("ERROR", "WARN", "INFO"):
        rows = [i for i in iss if i.sev == sev]
        if rows:
            out.append(f"== {sev} ({len(rows)}) ==")
            out += [f"  [{i.code}] {i.msg}" for i in rows]
            out.append("")
    out += ["== LOCATIONS ==", f"  {'name':<10}{'kind':<6}{'point':<8}{'theta':>8}  type"]
    for l in sorted(m.locs.values(), key=lambda l: (len(l.name), l.name)):
        lk = l.links[0] if l.links else "-"
        th = m.points[lk].theta if lk in m.points else None
        out.append(f"  {l.name:<10}{loc_kind(m, l):<6}{lk:<8}{'-' if th is None else format(th, '.1f'):>8}  {l.ltype}")
    out += ["", "== PICK -> DROP ROUTES ==", pick_drop_report(m, tol)]
    return "\n".join(out) + "\n"


def run_gui(path: Optional[str] = None) -> None:
    if _TK_ERR is not None:
        raise RuntimeError(f"tkinter / TkAgg unavailable: {_TK_ERR}")
    try:
        root = tk.Tk()
    except tk.TclError as e:
        raise RuntimeError(f"cannot open a display: {e}") from e
    TcsApp(root, path)
    root.mainloop()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect and validate an OpenTCS model XML (headings, directions, routes, locations).")
    ap.add_argument("xml", nargs="?", help="OpenTCS model .xml (omit to open the GUI empty)")
    ap.add_argument("--report", action="store_true", help="print a full text report instead of opening the GUI")
    ap.add_argument("--csv", help="write issues to this CSV")
    ap.add_argument("--png", help="write the map image (PNG / SVG / PDF)")
    ap.add_argument("--pdf", help="write the PDF report (map, points, routes)")
    ap.add_argument("--theme", choices=("light", "dark"), default="light", help="theme for --png")
    ap.add_argument("--tol", type=float, default=15.0, help="heading tolerance in degrees (default 15)")
    ap.add_argument("--layout", action="store_true", help="--png in the OpenTCS editor drawing frame")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args(argv)
    headless = args.report or args.csv or args.png or args.pdf
    if args.xml and headless:
        m = load_model(args.xml)
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["severity", "check", "objects", "message"])
                for i in validate(m, args.tol):
                    w.writerow([i.sev, i.code, " | ".join(n for _, n in i.objs), i.msg])
        if args.png:
            render_png(m, args.png, args.theme, Opts(layout=args.layout, tol=args.tol))
        if args.pdf:
            export_pdf(m, args.pdf, args.tol)
        if args.report or not (args.csv or args.png or args.pdf):
            print(full_report(m, args.tol))
        return 0
    try:
        run_gui(args.xml)
        return 0
    except RuntimeError as e:
        print(f"GUI unavailable ({e}). Use --report / --csv / --png.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
