#!/usr/bin/env python3
"""
wp_plotter.py - plot named 2D waypoints (x, y, theta) as heading arrows.

Input format (blocks may be separated by '---', blank lines, or nothing at all):

    fg_home
    x: 21519.56
    y: -1451.63
    theta: 90
    ---
    fg_pick1
    x: 17199.42
    ...

Usage
-----
    python3 wp_plotter.py                      # opens the GUI (paste points, click Generate)
    python3 wp_plotter.py --gui -i points.txt  # GUI preloaded with a file
    python3 wp_plotter.py -i points.txt        # file  -> waypoints_map.png
    python3 wp_plotter.py -i points.txt -o out.png --font-size 5
    cat points.txt | python3 wp_plotter.py -o out.png
    python3 wp_plotter.py --cli                # paste in the terminal, finish with Ctrl-D

Conventions
-----------
* theta is CCW from +X (REP-103 style): 0 = +x, 90 = +y. Use --theta-unit rad for radians.
* Names containing 'pick' / 'drop' are coloured teal / coral, everything else purple.
* Names containing 'inplc' are drawn dashed with a hollow marker.

Dependencies: matplotlib (tkinter only for the GUI).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_KV_RE = re.compile(rf"^\s*([A-Za-z_]+)\s*[:=]\s*({_NUM})\s*[A-Za-z°]*\s*$")
_SEP_RE = re.compile(r"^\s*(?:-{3,}|={3,}|_{3,}|\*{3,})\s*$")
_KEYS = {"x": "x", "y": "y", "theta": "theta", "th": "theta", "yaw": "theta", "heading": "theta"}


@dataclass
class Waypoint:
    name: str
    x: float
    y: float
    theta_deg: float
    line: int = 0


def parse_waypoints(text: str, theta_unit: str = "deg") -> Tuple[List[Waypoint], List[str]]:
    """Parse free-form 'name / x: / y: / theta:' blocks. Returns (waypoints, warnings)."""
    wps: List[Waypoint] = []
    warns: List[str] = []
    name: Optional[str] = None
    vals: Dict[str, float] = {}
    start = 0

    def flush() -> None:
        nonlocal name, vals
        if name is None and not vals:
            return
        label = name or f"<unnamed block near line {start}>"
        missing = [k for k in ("x", "y") if k not in vals]
        if missing:
            warns.append(f"line {start}: {label}: missing {', '.join(missing)} - skipped")
        else:
            if "theta" not in vals:
                warns.append(f"line {start}: {label}: no theta - defaulting to 0")
            th = vals.get("theta", 0.0)
            if theta_unit == "rad":
                th = math.degrees(th)
            wps.append(Waypoint(name or f"wp_{len(wps) + 1}", vals["x"], vals["y"], th, start))
        name, vals = None, {}

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if _SEP_RE.match(line):
            flush()
            continue
        if name is None and not vals:
            start = lineno
        m = _KV_RE.match(line)
        if m and m.group(1).lower() in _KEYS:
            key = _KEYS[m.group(1).lower()]
            if key in vals:
                warns.append(f"line {lineno}: duplicate '{key}' in block '{name}', last value wins")
            vals[key] = float(m.group(2))
        elif m:
            warns.append(f"line {lineno}: unknown key '{m.group(1)}' ignored")
        else:
            flush()  # a bare text line starts a new waypoint
            name = re.sub(r"\s+", "_", line.rstrip(":").strip())
            start = lineno
    flush()

    seen = set()
    for w in wps:
        if w.name in seen:
            warns.append(f"line {w.line}: duplicate waypoint name '{w.name}'")
        seen.add(w.name)
    return wps, warns


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
Rect = Tuple[float, float, float, float]  # x0, y0, x1, y1


def _overlap(a: Rect, b: Rect) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _colour(name: str) -> str:
    n = name.lower()
    return "#1D9E75" if "pick" in n else "#D85A30" if "drop" in n else "#7F77DD"


def _is_inplc(name: str) -> bool:
    return "inplc" in name.lower()


def render(
    wps: List[Waypoint],
    font_size: float = 4.2,
    arrow_len: Optional[float] = None,
    show_origin: bool = True,
    dpi: int = 220,
    fig_width: float = 7.0,
    show_labels: bool = True,
    show_grid: bool = True,
    max_size: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Build the figure. Labels are placed automatically to avoid arrows and each other."""
    if not wps:
        raise ValueError("No waypoints to plot.")

    xs = [w.x for w in wps] + ([0.0] if show_origin else [])
    ys = [w.y for w in wps] + ([0.0] if show_origin else [])
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    x_lo, x_hi = min(xs) - 0.08 * span, max(xs) + 0.14 * span
    y_lo, y_hi = min(ys) - 0.08 * span, max(ys) + 0.08 * span
    aspect = (y_hi - y_lo) / (x_hi - x_lo)
    if max_size:  # fit the GUI panel exactly: figure aspect follows the data aspect
        fig_width = max(min(max_size[0], max_size[1] * 0.86 / (0.90 * aspect)), 1.0)
        fig_h = fig_width * 0.90 * aspect / 0.86
    else:
        fig_h = min(max(fig_width * aspect, 3.0), 12.0)

    fig = Figure(figsize=(fig_width, fig_h), dpi=dpi)
    ax = fig.add_axes([0.08, 0.12, 0.90, 0.86])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.apply_aspect()
    pos = ax.get_position()
    u = (x_hi - x_lo) / (pos.width * fig_width * 72.0)  # map units per typographic point

    ax.grid(False)
    if show_grid:
        ax.grid(True, lw=0.25, color="#d8d8d8")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=4.5, length=2, width=0.4)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    ax.set_xlabel("x", fontsize=5)
    ax.set_ylabel("y", fontsize=5)
    if show_origin:
        ax.axhline(0, color="#888", lw=0.5)
        ax.axvline(0, color="#888", lw=0.5)
        ax.plot(0, 0, "k+", ms=5, mew=0.6)
        ax.annotate("origin (0,0)", (0, 0), xytext=(3, 3), textcoords="offset points",
                    fontsize=font_size, color="#444")

    L = arrow_len if arrow_len else 0.035 * span
    obstacles: List[Rect] = []
    for w in wps:
        r = math.radians(w.theta_deg)
        tx, ty = w.x + L * math.cos(r), w.y + L * math.sin(r)
        c, inp = _colour(w.name), _is_inplc(w.name)
        ax.annotate("", xy=(tx, ty), xytext=(w.x, w.y),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9, ls="--" if inp else "-",
                                    mutation_scale=5, shrinkA=0, shrinkB=0))
        ax.plot(w.x, w.y, "o", ms=2.6, mfc="white" if inp else c, mec=c, mew=0.7, zorder=5)
        pad = 1.5 * u
        obstacles.append((min(w.x, tx) - pad, min(w.y, ty) - pad, max(w.x, tx) + pad, max(w.y, ty) + pad))
        obstacles.append((w.x - 2.2 * u, w.y - 2.2 * u, w.x + 2.2 * u, w.y + 2.2 * u))

    # cluster coincident points so their labels stack as one block
    tol = 0.012 * span
    clusters: List[List[Waypoint]] = []
    for w in wps:
        for cl in clusters:
            if math.hypot(cl[0].x - w.x, cl[0].y - w.y) < tol:
                cl.append(w)
                break
        else:
            clusters.append([w])

    cw, lh, gap = 0.58 * font_size * u, 1.3 * font_size * u, 4.0 * u
    view: Rect = (x_lo, y_lo, x_hi, y_hi)
    dirs = {"r": (1, 0), "l": (-1, 0), "u": (0, 1), "d": (0, -1)}

    for cl in clusters:
        px, py = cl[0].x, cl[0].y
        n = len(cl)
        bw, bh = max(len(w.name) for w in cl) * cw, n * lh
        best: Optional[Tuple[float, str, Rect]] = None
        for rank, side in enumerate("rlud"):
            dx, dy = dirs[side]
            hit = any(math.cos(math.radians(w.theta_deg)) * dx + math.sin(math.radians(w.theta_deg)) * dy > 0.3
                      for w in cl)
            off = gap + ((L + 2 * u) if hit else 0.0)
            if side == "r":
                rect = (px + off, py - bh / 2, px + off + bw, py + bh / 2)
            elif side == "l":
                rect = (px - off - bw, py - bh / 2, px - off, py + bh / 2)
            elif side == "u":
                rect = (px - bw / 2, py + off, px + bw / 2, py + off + bh)
            else:
                rect = (px - bw / 2, py - off - bh, px + bw / 2, py - off)
            area = bw * bh
            score = sum(_overlap(rect, o) for o in obstacles)
            score += 5.0 * (area - _overlap(rect, view)) + 1e-3 * area * rank
            if best is None or score < best[0]:
                best = (score, side, rect)
        _, side, rect = best  # type: ignore[misc]
        obstacles.append(rect)
        ha = {"r": "left", "l": "right", "u": "center", "d": "center"}[side]
        ax_x = {"r": rect[0], "l": rect[2], "u": (rect[0] + rect[2]) / 2, "d": (rect[0] + rect[2]) / 2}[side]
        for i, w in enumerate(cl):
            ty = rect[3] - (i + 0.5) * lh
            if show_labels:
                ax.text(ax_x, ty, w.name, fontsize=font_size, ha=ha, va="center", color="#222")

    handles = []
    if any(_colour(w.name) == "#7F77DD" for w in wps):
        handles.append(Line2D([], [], color="#7F77DD", lw=1, label="other (home / gangway)"))
    if any("pick" in w.name.lower() for w in wps):
        handles.append(Line2D([], [], color="#1D9E75", lw=1, label="pick"))
    if any("drop" in w.name.lower() for w in wps):
        handles.append(Line2D([], [], color="#D85A30", lw=1, label="drop"))
    if any(_is_inplc(w.name) for w in wps):
        handles.append(Line2D([], [], color="#666", lw=1, ls="--", label="inplc (dashed, hollow)"))
    ax.legend(handles=handles, fontsize=4.2, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=len(handles), frameon=False, handlelength=1.8)
    ax._wp_init_lims = (ax.get_xlim(), ax.get_ylim())  # type: ignore[attr-defined]
    return fig


def save_png(path: str, wps: List[Waypoint], **kw) -> None:
    kw.setdefault("dpi", 220)
    render(wps, **kw).savefig(path, dpi=kw["dpi"], bbox_inches="tight", pad_inches=0.05)


EXAMPLE = """\
fg_home
x: 21519.560546875
y: -1451.63232421875
theta: 90
---
fg_home_st
x: 21519.560546875
y: 2297.23486328125
theta: 90
---
fg_home_inplc
x: 21552.201171875
y: 2297.23486328125
theta: 180
---
fg_gangway1
x: 17986.73046875
y: 2297.23486328125
theta: 180
---
fg_gangway2
x: 14882.8447265625
y: 2297.23486328125
theta: 180
---
fg_gangway2_inplc
x: 14882.8447265625
y: 2297.23486328125
theta: 90
---
fg_pick1_inplc
x: 14882.8447265625
y: 6579.19580078125
theta: 90.28282014744089
---
fg_pick1_dock
x: 14857.412109375
y: 6579.19580078125
theta: 180.03243474713557
---
fg_pick1
x: 17199.42578125
y: 6579.19580078125
theta: 180.59293373080513
---
fg_drop1_inplc
x: 14882.8447265625
y: 8085.2412109375
theta: 90.28282014744089
fg_drop1_dock
x: 14882.8447265625
y: 8085.2412109375
theta: 180.59293373080513
fg drop1
x: 17199.42578125
y: 8085.2412109375
theta: 180.59293373080513
fg_pick2_inplc
x: 14882.8447265625
y: 9542.5654296875
theta: 90.28282014744089
fg_pick2_dock
x: 14882.8447265625
y: 9542.5654296875
theta: 180.59293373080513
fg_pick2
x: 17199.42578125
y: 9542.5654296875
theta: 180.59293373080513
---
fg_drop2_inplc
x: 14882.8447265625
y: 11017.35546875
theta: 90.28282014744089
fg_drop2_dock
x: 14882.8447265625
y: 11017.35546875
theta: 181.37436201336797
fg_drop2
x: 17199.42578125
y: 11017.35546875
theta: 181.37436201336797
---
fg_gangway3
x: 14443.7060546875
y: 20292.302734375
theta: 357.4539887028987
---
fg_gangway3_inplc
x: 14443.7060546875
y: 20292.302734375
theta: 90
---
rmline_fg_pick
x: 7867.0478515625
y: 3588.7666015625
theta: 0.1
---
rmline_fg_pick_dock
x: 10328.78125
y: 3603.337158203125
theta: 0.2726135537861305
---
rmline_fg_pick_dock_inplc
x: 10306.45703125
y: 3601.192138671875
theta: 90
---
rmline_fg_gangway
x: 10345.671875
y: 2422.69580078125
theta: 179.57830912959975
---
rmline_fg_gangway_inplc
x: 10350.5732421875
y: 2434.61376953125
theta: 92.24656783571125
"""


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter import font as tkfont
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    _TK_ERR: Optional[Exception] = None
except Exception as _e:  # headless / tkinter not installed
    tk = ttk = filedialog = messagebox = tkfont = FigureCanvasTkAgg = NavigationToolbar2Tk = None  # type: ignore
    _TK_ERR = _e

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".wp_plotter.json")

HELP_TEXT = """\
Input tab      paste or type waypoints; the plot updates live (toggle in View).
Waypoints tab  sortable, filterable list. Click a row to highlight it on the map,
               double-click to jump to its line in the input.
Problems       lines the parser could not use; double-click to jump to the line.

Map            hover = name / x / y / theta      click = select (coincident points are all selected)
               mouse wheel = zoom at cursor      toolbar = pan / box-zoom / home
Shortcuts      Ctrl+Enter generate   Ctrl+O open file   Ctrl+S save image (PNG / SVG / PDF)
"""


def _load_cfg() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cfg(cfg: dict) -> None:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
    except OSError:
        pass


class App:
    DPI = 150
    TOOLBAR_PX = 44
    HIT_PX = 12

    def __init__(self, root, initial_text: str = "") -> None:
        self.root = root
        self.cfg = _load_cfg()
        c = self.cfg
        self.wps: List[Waypoint] = []
        self.warns: List[str] = []
        self.sel: List[int] = []
        self.fig = self.ax = self.canvas = self.toolbar = self.hl = self.tip = None
        self.hover_key: Optional[tuple] = None
        self._live_job = self._resize_job = None
        self._last_fit: Optional[Tuple[int, int]] = None
        self._syncing = False
        self.sort: Tuple[Optional[str], bool] = (None, False)

        self.unit = tk.StringVar(value=c.get("unit", "deg"))
        self.font = tk.StringVar(value=str(c.get("font", 4.2)))
        self.arrow = tk.StringVar(value=str(c.get("arrow", "")))
        self.origin = tk.BooleanVar(value=c.get("origin", True))
        self.labels = tk.BooleanVar(value=c.get("labels", True))
        self.grid_on = tk.BooleanVar(value=c.get("grid", True))
        self.live = tk.BooleanVar(value=c.get("live", True))
        self.filter = tk.StringVar()
        self.status = tk.StringVar(value="Ready")

        self._build_ui()
        self.txt.insert("1.0", initial_text or c.get("text", ""))
        self.txt.edit_modified(False)
        for v in (self.unit, self.font, self.arrow):
            v.trace_add("write", self._schedule_live)
        for v in (self.origin, self.labels, self.grid_on):
            v.trace_add("write", self._on_display_toggle)
        self.filter.trace_add("write", lambda *_: self._fill_table())
        root.update_idletasks()
        root.after(120, self.generate)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        r = self.root
        r.title("Waypoint plotter")
        r.geometry(self.cfg.get("geometry", "1280x820"))
        r.minsize(960, 600)
        r.protocol("WM_DELETE_WINDOW", self.close)

        st = ttk.Style()
        for t in ("vista", "aqua", "clam"):
            if t in st.theme_names():
                st.theme_use(t)
                break
        self._bold = tkfont.nametofont("TkDefaultFont").copy()
        self._bold.configure(weight="bold")
        st.configure("Accent.TButton", font=self._bold)

        menu = tk.Menu(r)
        r.config(menu=menu)
        fm = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="File", menu=fm)
        fm.add_command(label="Open file…", accelerator="Ctrl+O", command=self.open_file)
        fm.add_command(label="Save image…", accelerator="Ctrl+S", command=self.save_image)
        fm.add_separator()
        fm.add_command(label="Quit", command=self.close)
        em = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Edit", menu=em)
        em.add_command(label="Paste from clipboard", command=self.paste_clipboard)
        em.add_command(label="Load example", command=self.load_example)
        em.add_command(label="Clear", command=self.clear)
        vm = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="View", menu=vm)
        vm.add_checkbutton(label="Live preview", variable=self.live)
        vm.add_checkbutton(label="Show labels", variable=self.labels)
        vm.add_checkbutton(label="Show grid", variable=self.grid_on)
        vm.add_checkbutton(label="Show origin (0,0)", variable=self.origin)
        vm.add_separator()
        vm.add_command(label="Reset view", command=self.reset_view)
        hm = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Help", menu=hm)
        hm.add_command(label="Tips and shortcuts", command=lambda: messagebox.showinfo("Tips and shortcuts", HELP_TEXT))

        self.status_lbl = ttk.Label(r, textvariable=self.status, anchor="w", padding=(10, 4))
        self.status_lbl.pack(side="bottom", fill="x")
        ttk.Separator(r).pack(side="bottom", fill="x")
        paned = ttk.PanedWindow(r, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, padding=(8, 8, 4, 4))
        right = ttk.Frame(paned, padding=(4, 8, 8, 4))
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        self.nb = ttk.Notebook(left)
        self.nb.pack(fill="both", expand=True)
        tab_in = ttk.Frame(self.nb)
        self.nb.add(tab_in, text="Input")
        self.txt = tk.Text(tab_in, width=46, wrap="none", undo=True, font="TkFixedFont",
                           borderwidth=0, highlightthickness=0, padx=6, pady=4)
        ys = ttk.Scrollbar(tab_in, orient="vertical", command=self.txt.yview)
        xs = ttk.Scrollbar(tab_in, orient="horizontal", command=self.txt.xview)
        self.txt.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.txt.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        tab_in.rowconfigure(0, weight=1)
        tab_in.columnconfigure(0, weight=1)
        self.txt.tag_configure("warn", background="#fde2e2")
        self.txt.bind("<<Modified>>", self._on_modified)

        tab_tb = ttk.Frame(self.nb)
        self.nb.add(tab_tb, text="Waypoints (0)")
        frow = ttk.Frame(tab_tb)
        frow.pack(fill="x", padx=6, pady=6)
        ttk.Label(frow, text="Filter").pack(side="left")
        ttk.Entry(frow, textvariable=self.filter).pack(side="left", fill="x", expand=True, padx=(6, 0))
        cols = ("n", "name", "x", "y", "theta")
        self.tree = ttk.Treeview(tab_tb, columns=cols, show="headings", selectmode="extended")
        for col, text, w, anc in (("n", "#", 32, "e"), ("name", "name", 150, "w"), ("x", "x", 80, "e"),
                                  ("y", "y", 80, "e"), ("theta", "theta °", 64, "e")):
            self.tree.heading(col, text=text, command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=w, anchor=anc, stretch=(col == "name"))
        tsb = ttk.Scrollbar(tab_tb, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tsb.set)
        tsb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True, padx=(6, 0), pady=(0, 6))
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-1>", self._on_tree_dbl)

        self.prob_frame = ttk.LabelFrame(left, text="Problems", padding=4)
        self.prob = tk.Listbox(self.prob_frame, height=4, activestyle="none", borderwidth=0,
                               highlightthickness=0, foreground="#b3261e")
        self.prob.pack(fill="x")
        self.prob.bind("<Double-1>", self._on_problem_dbl)

        self.opts = ttk.LabelFrame(left, text="Options", padding=8)
        self.opts.pack(fill="x", pady=(8, 0))
        o = self.opts
        ttk.Label(o, text="Theta unit").grid(row=0, column=0, sticky="w")
        ttk.Combobox(o, textvariable=self.unit, values=("deg", "rad"), width=6, state="readonly").grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(o, text="Font size").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Spinbox(o, from_=2, to=14, increment=0.2, format="%.1f", textvariable=self.font, width=6).grid(row=0, column=3, padx=6)
        ttk.Label(o, text="Arrow length").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(o, textvariable=self.arrow, width=9).grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(o, text="map units, blank = auto", foreground="#888").grid(row=1, column=2, columnspan=2, sticky="w", padx=(12, 0), pady=(6, 0))
        ttk.Checkbutton(o, text="Live preview", variable=self.live).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(o, text="Labels", variable=self.labels).grid(row=2, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Checkbutton(o, text="Grid", variable=self.grid_on).grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))
        ttk.Checkbutton(o, text="Origin", variable=self.origin).grid(row=3, column=2, sticky="w", padx=(12, 0), pady=(2, 0))

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="Generate", style="Accent.TButton", command=self.generate).pack(side="left")
        ttk.Button(btns, text="Save image…", command=self.save_image).pack(side="left", padx=6)
        ttk.Button(btns, text="Reset view", command=self.reset_view).pack(side="right")

        self.preview = ttk.Frame(right, width=700, height=600)
        self.preview.pack(fill="both", expand=True)
        self.preview.pack_propagate(False)
        self.plot_host = ttk.Frame(self.preview)
        self.plot_host.pack(fill="both", expand=True)
        self.placeholder = ttk.Label(self.plot_host, justify="center", foreground="#888", font=self._bold,
                                     text="Paste waypoints on the left\nor choose Edit > Load example")
        self.placeholder.pack(expand=True)
        self.preview.bind("<Configure>", self._on_resize)

        r.bind("<Control-Return>", lambda _e: self.generate())
        r.bind("<Control-o>", lambda _e: self.open_file())
        r.bind("<Control-s>", lambda _e: self.save_image())

    # ------------------------------------------------------------- helpers
    def _status(self, msg: str, error: bool = False) -> None:
        self.status.set(msg)
        self.status_lbl.configure(foreground="#b3261e" if error else "")

    def _settings(self) -> Dict[str, object]:
        return dict(font_size=float(self.font.get()),
                    arrow_len=float(self.arrow.get()) if self.arrow.get().strip() else None,
                    show_origin=self.origin.get(), show_labels=self.labels.get(), show_grid=self.grid_on.get())

    def _fit_inches(self) -> Tuple[float, float]:
        w = max(self.preview.winfo_width(), 300)
        h = max(self.preview.winfo_height() - self.TOOLBAR_PX, 300)
        return w / self.DPI, h / self.DPI

    def _goto_line(self, n: int) -> None:
        if n <= 0:
            return
        self.nb.select(0)
        self.txt.see(f"{n}.0")
        self.txt.tag_remove("sel", "1.0", "end")
        self.txt.tag_add("sel", f"{n}.0", f"{n}.end")
        self.txt.mark_set("insert", f"{n}.0")
        self.txt.focus_set()

    # ---------------------------------------------------- change scheduling
    def _on_modified(self, _e=None) -> None:
        if self.txt.edit_modified():
            self.txt.edit_modified(False)
            self._schedule_live()

    def _schedule_live(self, *_a) -> None:
        if not self.live.get():
            return
        if self._live_job:
            self.root.after_cancel(self._live_job)
        self._live_job = self.root.after(450, self.generate)

    def _on_display_toggle(self, *_a) -> None:
        if self.wps:
            self.root.after_idle(lambda: self.generate(reparse=False))

    def _on_resize(self, _e=None) -> None:
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(300, self._refit)

    def _refit(self) -> None:
        self._resize_job = None
        size = (self.preview.winfo_width(), self.preview.winfo_height())
        if self.wps and (self._last_fit is None or abs(size[0] - self._last_fit[0]) > 12
                         or abs(size[1] - self._last_fit[1]) > 12):
            self.generate(reparse=False)

    # ------------------------------------------------------------ generate
    def generate(self, *_a, reparse: bool = True) -> None:
        if self._live_job:
            self.root.after_cancel(self._live_job)
            self._live_job = None
        try:
            kw = self._settings()
        except ValueError:
            self._status("Font size and arrow length must be numbers.", error=True)
            return
        if reparse:
            text = self.txt.get("1.0", "end")
            self.wps, self.warns = parse_waypoints(text, self.unit.get())
            self.sel = []
            self._refresh_problems()
            self._fill_table()
            if not text.strip():
                self._clear_plot()
                self._status("Paste waypoints on the left to get started.")
                return
        if not self.wps:
            self._clear_plot()
            self._status("No valid waypoints found - check the Problems list.", error=True)
            return
        mw, mh = self._fit_inches()
        try:
            fig = render(self.wps, dpi=self.DPI, max_size=(mw, mh), **kw)  # type: ignore[arg-type]
        except Exception as e:  # keep the GUI alive on any render error
            self._status(f"Render failed: {e}", error=True)
            return
        self._show(fig)
        self._last_fit = (self.preview.winfo_width(), self.preview.winfo_height())
        n = len(self.wps)
        self._status(f"{n} waypoint{'s' if n != 1 else ''} plotted"
                     + (f"  |  {len(self.warns)} problem{'s' if len(self.warns) != 1 else ''}" if self.warns else "")
                     + "  |  hover for details, click to select, wheel to zoom")

    def _drop_canvas(self) -> None:
        for w in (self.toolbar, self.canvas):
            if w is not None:
                (w.get_tk_widget() if w is self.canvas else w).destroy()
        self.toolbar = self.canvas = self.fig = self.ax = self.hl = self.tip = None

    def _clear_plot(self) -> None:
        self._drop_canvas()
        self.placeholder.pack(expand=True)

    def _show(self, fig: Figure) -> None:
        keep = None
        if self.ax is not None and hasattr(self.ax, "_wp_init_lims"):
            cur = (self.ax.get_xlim(), self.ax.get_ylim())
            if cur != self.ax._wp_init_lims:
                keep = (cur, self.ax._wp_init_lims)  # user had zoomed / panned
        self._drop_canvas()
        self.placeholder.pack_forget()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_host)
        toolbar = NavigationToolbar2Tk(canvas, self.plot_host, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(side="top", anchor="center")
        ax = fig.axes[0]
        toolbar.push_current()
        if keep and keep[1] == ax._wp_init_lims:  # same data extent: keep the zoom while editing
            ax.set_xlim(*keep[0][0])
            ax.set_ylim(*keep[0][1])
        self.hl, = ax.plot([], [], "o", ms=10, mfc="none", mec="#e24b4a", mew=1.3, zorder=10)
        self.tip = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points", fontsize=6,
                               zorder=20, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#888", lw=0.5),
                               annotation_clip=False)
        self.tip.set_multialignment("left")
        self.tip.set_visible(False)
        self.fig, self.ax, self.canvas, self.toolbar = fig, ax, canvas, toolbar
        self.hover_key = None
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_press_event", self._on_click)
        canvas.mpl_connect("scroll_event", self._on_scroll)
        canvas.mpl_connect("axes_leave_event", lambda _e: self._hide_tip())
        self._apply_selection()
        canvas.draw()

    def reset_view(self) -> None:
        if self.ax is not None:
            self.ax.set_xlim(*self.ax._wp_init_lims[0])
            self.ax.set_ylim(*self.ax._wp_init_lims[1])
            self.canvas.draw_idle()

    # ---------------------------------------------------- map interactions
    def _near(self, event) -> List[int]:
        import numpy as np
        if event.inaxes is not self.ax or not self.wps:
            return []
        pts = self.ax.transData.transform(np.array([[w.x, w.y] for w in self.wps]))
        d = np.hypot(pts[:, 0] - event.x, pts[:, 1] - event.y)
        return [int(i) for i in np.argsort(d) if d[i] <= self.HIT_PX][:5]

    def _hide_tip(self) -> None:
        if self.tip is not None and self.tip.get_visible():
            self.tip.set_visible(False)
            self.hover_key = None
            self.canvas.draw_idle()

    def _on_motion(self, event) -> None:
        if self.ax is None:
            return
        if event.inaxes is not self.ax:
            self._hide_tip()
            return
        idx = self._near(event)
        self._status(f"x {event.xdata:,.1f}    y {event.ydata:,.1f}"
                     + (f"    |    {', '.join(self.wps[i].name for i in idx)}" if idx else ""))
        key = tuple(idx)
        if key == self.hover_key:
            return
        self.hover_key = key
        if not idx:
            self.tip.set_visible(False)
        else:
            self.tip.set_text("\n".join(
                f"{self.wps[i].name}\nx {self.wps[i].x:,.2f}   y {self.wps[i].y:,.2f}   theta {self.wps[i].theta_deg:.2f}"
                for i in idx))
            self.tip.xy = (self.wps[idx[0]].x, self.wps[idx[0]].y)
            fw, fh = self.canvas.get_width_height()
            right_half, top_half = event.x > fw * 0.6, event.y > fh * 0.6
            self.tip.xyann = (-10 if right_half else 10, -10 if top_half else 10)
            self.tip.set_ha("right" if right_half else "left")
            self.tip.set_va("top" if top_half else "bottom")
            self.tip.set_visible(True)
        self.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if event.button != 1 or event.inaxes is not self.ax or str(self.toolbar.mode):
            return
        self._select(self._near(event))

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        f = 0.8 if event.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(event.xdata - (event.xdata - x0) * f, event.xdata + (x1 - event.xdata) * f)
        self.ax.set_ylim(event.ydata - (event.ydata - y0) * f, event.ydata + (y1 - event.ydata) * f)
        self.canvas.draw_idle()

    # ------------------------------------------------- selection and table
    def _apply_selection(self) -> None:
        if self.hl is not None:
            self.hl.set_data([self.wps[i].x for i in self.sel], [self.wps[i].y for i in self.sel])

    def _select(self, idx: List[int], from_tree: bool = False) -> None:
        self.sel = idx
        self._apply_selection()
        if self.canvas is not None:
            self.canvas.draw_idle()
        if not from_tree:
            self._syncing = True
            ids = tuple(str(i) for i in idx if self.tree.exists(str(i)))
            self.tree.selection_set(ids)
            if ids:
                self.tree.see(ids[0])
            self._syncing = False
        if idx:
            w = self.wps[idx[0]]
            self._status(f"{', '.join(self.wps[i].name for i in idx)}    x {w.x:,.2f}   y {w.y:,.2f}   theta {w.theta_deg:.2f}")

    def _on_tree_select(self, _e=None) -> None:
        if not self._syncing:
            self._select([int(i) for i in self.tree.selection()], from_tree=True)

    def _on_tree_dbl(self, e) -> None:
        row = self.tree.identify_row(e.y)
        if row:
            self._goto_line(self.wps[int(row)].line)

    def _sort_by(self, col: str) -> None:
        self.sort = (col, not self.sort[1] if self.sort[0] == col else False)
        self._fill_table()

    def _fill_table(self) -> None:
        ch = self.tree.get_children()
        if ch:
            self.tree.delete(*ch)
        q = self.filter.get().strip().lower()
        order = list(range(len(self.wps)))
        col, rev = self.sort
        keys = {"n": lambda i: i, "name": lambda i: self.wps[i].name.lower(), "x": lambda i: self.wps[i].x,
                "y": lambda i: self.wps[i].y, "theta": lambda i: self.wps[i].theta_deg}
        if col:
            order.sort(key=keys[col], reverse=rev)
        for i in order:
            w = self.wps[i]
            if q and q not in w.name.lower():
                continue
            self.tree.insert("", "end", iid=str(i),
                             values=(i + 1, w.name, f"{w.x:.2f}", f"{w.y:.2f}", f"{w.theta_deg:.2f}"))
        self.nb.tab(1, text=f"Waypoints ({len(self.wps)})")

    # ------------------------------------------------------------- problems
    def _refresh_problems(self) -> None:
        self.txt.tag_remove("warn", "1.0", "end")
        self.prob.delete(0, "end")
        for w in self.warns:
            self.prob.insert("end", w)
            m = re.search(r"line (\d+)", w)
            if m:
                self.txt.tag_add("warn", f"{m.group(1)}.0", f"{m.group(1)}.end+1c")
        if self.warns:
            self.prob_frame.pack(fill="x", pady=(6, 0), before=self.opts)
        else:
            self.prob_frame.pack_forget()

    def _on_problem_dbl(self, _e=None) -> None:
        sel = self.prob.curselection()
        if sel:
            m = re.search(r"line (\d+)", self.prob.get(sel[0]))
            if m:
                self._goto_line(int(m.group(1)))

    # ------------------------------------------------------------- commands
    def _set_text(self, text: str) -> None:
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", text)
        self.txt.edit_modified(False)
        self.nb.select(0)
        self.generate()

    def open_file(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, initialdir=self.cfg.get("last_dir") or None,
                                          filetypes=[("Text files", "*.txt *.yaml *.yml *.log"), ("All files", "*.*")])
        if not path:
            return
        self.cfg["last_dir"] = os.path.dirname(path)
        with open(path, encoding="utf-8", errors="replace") as f:
            self._set_text(f.read())
        self._status(f"Opened {path}")

    def paste_clipboard(self) -> None:
        try:
            self._set_text(self.root.clipboard_get())
        except tk.TclError:
            self._status("Clipboard is empty or not text.", error=True)

    def load_example(self) -> None:
        self._set_text(EXAMPLE)

    def clear(self) -> None:
        self._set_text("")

    def save_image(self) -> None:
        if not self.wps:
            self._status("Nothing to save yet - add some waypoints first.", error=True)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".png", initialfile="waypoints_map.png",
            initialdir=self.cfg.get("last_dir") or None,
            filetypes=[("PNG image", "*.png"), ("SVG vector", "*.svg"), ("PDF", "*.pdf")])
        if not path:
            return
        try:
            save_png(path, self.wps, **self._settings())  # type: ignore[arg-type]
        except Exception as e:
            self._status(f"Save failed: {e}", error=True)
            return
        self.cfg["last_dir"] = os.path.dirname(path)
        self._status(f"Saved {path}")

    def close(self) -> None:
        self.cfg.update(geometry=self.root.geometry(), unit=self.unit.get(), font=self.font.get(),
                        arrow=self.arrow.get(), origin=self.origin.get(), labels=self.labels.get(),
                        grid=self.grid_on.get(), live=self.live.get(), text=self.txt.get("1.0", "end").rstrip())
        _save_cfg(self.cfg)
        self.root.destroy()


def run_gui(initial_text: str = "") -> None:
    if _TK_ERR is not None:
        raise RuntimeError(f"tkinter/TkAgg unavailable: {_TK_ERR}")
    try:
        root = tk.Tk()
    except tk.TclError as e:  # no display
        raise RuntimeError(f"cannot open a display: {e}") from e
    App(root, initial_text)
    root.mainloop()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Plot named (x, y, theta) waypoints as heading arrows.")
    ap.add_argument("-i", "--input", help="waypoint text file ('-' = stdin). Omit to open the GUI.")
    ap.add_argument("-o", "--output", default="waypoints_map.png", help="output PNG/SVG/PDF (default: %(default)s)")
    ap.add_argument("--theta-unit", choices=("deg", "rad"), default="deg")
    ap.add_argument("--font-size", type=float, default=4.2)
    ap.add_argument("--arrow-len", type=float, default=None, help="arrow length in map units (default: 3.5%% of extent)")
    ap.add_argument("--no-origin", action="store_true", help="do not force (0,0) into the view")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--gui", action="store_true", help="open the GUI (optionally preloaded with -i FILE)")
    ap.add_argument("--cli", action="store_true", help="paste input in the terminal instead of opening the GUI")
    args = ap.parse_args(argv)

    file_text = None
    if args.input not in (None, "-"):
        with open(args.input, encoding="utf-8", errors="replace") as f:
            file_text = f.read()

    if args.gui or (args.input is None and not args.cli and sys.stdin.isatty()):
        try:
            run_gui(file_text or "")
            return 0
        except RuntimeError as e:
            print(f"GUI unavailable ({e}).", file=sys.stderr)
            if args.gui:
                return 1
            print("Falling back to terminal input.", file=sys.stderr)

    if file_text is None:
        if sys.stdin.isatty():
            print("Paste waypoints (name / x: / y: / theta:), then Ctrl-D (Ctrl-Z + Enter on Windows):", file=sys.stderr)
        file_text = sys.stdin.read()

    wps, warns = parse_waypoints(file_text, args.theta_unit)
    for w in warns:
        print(f"warning: {w}", file=sys.stderr)
    if not wps:
        print("error: no valid waypoints found", file=sys.stderr)
        return 1
    save_png(args.output, wps, font_size=args.font_size, arrow_len=args.arrow_len, show_origin=not args.no_origin,
             show_labels=not args.no_labels, show_grid=not args.no_grid, dpi=args.dpi)
    print(f"{len(wps)} waypoints -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
