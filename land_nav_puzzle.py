"""
Land-navigation puzzle generator and daily archive manager.

Variants
--------
resection          Bearings TO landmarks are given; find where YOU are (A/B/C map markers).
reverse_resection  Your position is known; a bearing to an unidentified feature is given;
                   which lettered feature (A/B/C) did you sight?
intersection       Fixed observation stations report bearings TO a target (e.g. smoke);
                   find the target (A/B/C map markers).
running_fix        One landmark, two bearings taken before/after a leg of known course,
                   speed and time; where are you at the end of the leg (A/B/C)?
error_budget       Each sighting has an angular uncertainty; which feature contributes the
                   largest lateral position error (distance x tan(uncertainty))?

Map is 10 km x 10 km (UNITS_PER_KM = 100). Each grid square = 1 km.
Declination convention: magnetic = true - declination (E positive).

Usage
-----
    python land_nav_puzzle.py --variant running_fix --difficulty medium --seed 7
    python land_nav_puzzle.py --daily
    python land_nav_puzzle.py --backfill 2026-11-19 2026-11-25
    python land_nav_puzzle.py --organize
    python land_nav_puzzle.py --rebuild-index
    python land_nav_puzzle.py --status
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import textwrap
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
ARCHIVE_DIR = Path(os.environ.get("ARCHIVE_DIR", REPO_ROOT / "archive"))
EPOCH = date.fromisoformat(os.environ.get("PUZZLE_EPOCH", "2026-01-01"))
DAILY_SEED_BASE = int(os.environ.get("DAILY_SEED_BASE", "20260000"))
DAILY_DIFFICULTIES = ["easy", "medium", "hard", "very_hard"]

MAP_SIZE = 1000
UNITS_PER_KM = 100  # map is 10 km x 10 km

FEATURE_NAMES = [
    "Overlook", "Saddle", "Water Tower", "Fire Tower", "Radio Mast",
    "Church Spire", "Quarry Rim", "Bridge", "Trail Junction", "Cairn",
    "Old Barn", "Lookout Rock", "Windmill", "Dam", "Ranger Station",
]

DIFFICULTY_PRESETS = {
    "easy":      dict(n_landmarks=3, spread_deg=100, error_deg=0.0, declination=None, round_to=10),
    "medium":    dict(n_landmarks=3, spread_deg=60,  error_deg=1.5, declination=None, round_to=5),
    "hard":      dict(n_landmarks=2, spread_deg=40,  error_deg=2.5, declination=8.0,  round_to=1),
    "very_hard": dict(n_landmarks=3, spread_deg=25,  error_deg=4.0, declination=-6.0, round_to=1),
}

ERROR_BUDGET_SIGMAS = {
    "easy":      [1.0, 2.0, 3.0],
    "medium":    [1.0, 2.0, 3.0, 4.0],
    "hard":      [1.0, 1.5, 2.5, 3.5, 5.0],
    "very_hard": [0.5, 1.5, 2.5, 3.5, 5.0, 6.0],
}

VARIANTS = ["resection", "reverse_resection", "intersection", "running_fix", "error_budget"]

COLORS = {"A": "#d95f02", "B": "#7570b3", "C": "#e7298a"}
LETTERS = ["A", "B", "C"]


# --------------------------------------------------------------------------- geometry
def bearing(from_pt, to_pt) -> float:
    dx = to_pt[0] - from_pt[0]
    dy = to_pt[1] - from_pt[1]
    return math.degrees(math.atan2(dx, dy)) % 360.0


def back_bearing(b: float) -> float:
    return (b + 180.0) % 360.0


def project(from_pt, brg: float, dist_: float):
    rad = math.radians(brg)
    return (from_pt[0] + dist_ * math.sin(rad), from_pt[1] + dist_ * math.cos(rad))


def round_bearing(b: float, step: int) -> float:
    return round(b / step) * step % 360


def dist(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def in_bounds(pt, margin: float) -> bool:
    return margin <= pt[0] <= MAP_SIZE - margin and margin <= pt[1] <= MAP_SIZE - margin


def clamp_pt(pt, margin: float = 50):
    return (
        min(max(pt[0], margin), MAP_SIZE - margin),
        min(max(pt[1], margin), MAP_SIZE - margin),
    )


def observe(rng: random.Random, true_b: float, preset: dict) -> float:
    """Simulate a compass reading: instrument error, declination (magnetic), then rounding."""
    b = true_b + rng.uniform(-preset["error_deg"], preset["error_deg"])
    if preset["declination"] is not None:
        b -= preset["declination"]
    return round_bearing(b % 360.0, preset["round_to"])


def true_from_shown(shown: float, preset: dict) -> float:
    """What a trainee gets after correcting a shown (magnetic) bearing to true."""
    return (shown + (preset["declination"] or 0.0)) % 360.0


def declination_lines(preset: dict) -> list[str]:
    d = preset["declination"]
    if d is None:
        return ["Bearings are true; map north is up."]
    return [
        f"Bearings are MAGNETIC. Declination: {d:+.1f}° (E positive).",
        "Map north (up) is TRUE north.",
    ]


# --------------------------------------------------------------------------- data
@dataclass
class Landmark:
    name: str
    pos: tuple[float, float]


@dataclass
class Puzzle:
    variant: str
    difficulty: str
    landmarks: list[Landmark]
    unknown_pos: Optional[tuple[float, float]]
    given: dict = field(default_factory=dict)
    answer: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    map_seed: int = 0


# --------------------------------------------------------------------------- generation helpers
def generate_landmarks(rng: random.Random, n: int, margin: float = 80,
                       min_sep: float = 150) -> list[Landmark]:
    names = rng.sample(FEATURE_NAMES, n)
    pts: list[Landmark] = []
    for name in names:
        p = None
        for _ in range(200):
            p = (rng.uniform(margin, MAP_SIZE - margin), rng.uniform(margin, MAP_SIZE - margin))
            if all(dist(p, q.pos) >= min_sep for q in pts):
                break
        pts.append(Landmark(name, p))
    return pts


def pick_unknown_with_spread(rng: random.Random, landmarks: list[Landmark],
                             min_spread_deg: float, margin: float = 120,
                             tries: int = 500, min_dist: float = 100):
    """Find a point from which the landmarks are well spread in bearing."""
    best_pt, best_score = None, -1.0
    for _ in range(tries):
        pt = (rng.uniform(margin, MAP_SIZE - margin), rng.uniform(margin, MAP_SIZE - margin))
        if any(dist(pt, lm.pos) < min_dist for lm in landmarks):
            continue
        bearings = sorted(bearing(pt, lm.pos) for lm in landmarks)
        gaps = [
            (bearings[(i + 1) % len(bearings)] - bearings[i]) % 360
            for i in range(len(bearings))
        ]
        score = min(gaps)
        if score > best_score:
            best_score, best_pt = score, pt
        if score >= min_spread_deg:
            return pt
    return best_pt or (MAP_SIZE / 2, MAP_SIZE / 2)


def generate_options(rng: random.Random, true_pos, preset: dict, extra=()):
    """A/B/C position markers. `extra` = deliberate 'common mistake' distractors."""
    lo = 140 + 20 * preset["error_deg"]
    hi = lo + 120
    others = [tuple(p) for p in extra][:2]
    pool = [true_pos] + others
    while len(others) < 2:
        cand = None
        for _ in range(300):
            pt = project(true_pos, rng.uniform(0, 360), rng.uniform(lo, hi))
            if in_bounds(pt, 50) and all(dist(pt, q) >= 100 for q in pool):
                cand = pt
                break
        if cand is None:
            cand = clamp_pt(project(true_pos, rng.uniform(0, 360), rng.uniform(lo, hi)))
        others.append(cand)
        pool.append(cand)

    correct = rng.choice(LETTERS)
    it = iter(others)
    options = {lab: (true_pos if lab == correct else next(it)) for lab in LETTERS}
    return options, correct


def _new_puzzle(rng, variant, difficulty, landmarks, unknown) -> Puzzle:
    return Puzzle(variant, difficulty, landmarks, unknown, map_seed=rng.randrange(2 ** 31))


def _fmt(b: float) -> str:
    return f"{b:03.0f}"


# --------------------------------------------------------------------------- variant builders
def build_resection(rng, preset, difficulty) -> Puzzle:
    """Bearings TO landmarks are known; where are you?"""
    landmarks = generate_landmarks(rng, preset["n_landmarks"])
    unknown = pick_unknown_with_spread(rng, landmarks, preset["spread_deg"])
    options, correct = generate_options(rng, unknown, preset)

    p = _new_puzzle(rng, "resection", difficulty, landmarks, unknown)
    readings, lines, plot_lines, notes = [], ["Bearings from your position TO each landmark:"], [], []
    if preset["declination"] is not None:
        notes.append("Convert magnetic to true first: true = magnetic + declination.")
    notes.append("Back-bearing = bearing +/- 180°. Plot a line from each landmark along its back-bearing.")
    for lm in landmarks:
        shown = observe(rng, bearing(unknown, lm.pos), preset)
        est_true = true_from_shown(shown, preset)
        back = back_bearing(est_true)
        readings.append({"landmark": lm.name, "bearing_shown": shown})
        lines.append(f"  {lm.name:<14} {_fmt(shown)}°")
        plot_lines.append((lm.pos, back))
        notes.append(f"  {lm.name}: {_fmt(shown)}° -> true {_fmt(est_true)}° -> back-bearing {_fmt(back)}°")
    notes.append(f"The lines cross at your position -> marker {correct}.")

    p.given = {
        "readings": readings,
        "options": options,
        "option_type": "position",
        "declination": preset["declination"],
        "instructions": (
            "Determine your position using the bearings below. "
            "Which location marker (A, B, or C) represents where you are standing?"
        ),
        "lines": lines + declination_lines(preset),
    }
    p.answer = {
        "position": unknown,
        "correct_option": correct,
        "lines": plot_lines,
        "star_label": "TRUE POSITION",
    }
    p.notes = notes
    return p


def build_reverse_resection(rng, preset, difficulty) -> Puzzle:
    """Your position is known; identify the feature you sighted."""
    landmarks = generate_landmarks(rng, 3)
    observer = pick_unknown_with_spread(rng, landmarks, preset["spread_deg"])
    options = {LETTERS[i]: i for i in range(3)}
    target_idx = rng.randrange(3)
    correct = LETTERS[target_idx]

    shown = observe(rng, bearing(observer, landmarks[target_idx].pos), preset)
    est_true = true_from_shown(shown, preset)
    decl = preset["declination"]

    notes = []
    if decl is not None:
        notes.append(f"Sighted {_fmt(shown)}° magnetic -> true {_fmt(est_true)}° (true = magnetic + declination).")
    else:
        notes.append(f"Sighted bearing {_fmt(shown)}°.")
    notes.append("Draw that bearing from your position; the feature it points at is the answer.")
    notes.append("Bearing from you to each candidate:")
    for i, lm in enumerate(landmarks):
        tb = bearing(observer, lm.pos)
        extra = f" / {_fmt((tb - decl) % 360)}° mag" if decl is not None else ""
        notes.append(f"  {LETTERS[i]} {lm.name}: {_fmt(tb)}° true{extra}")

    p = _new_puzzle(rng, "reverse_resection", difficulty, landmarks, observer)
    p.given = {
        "readings": [{"bearing_shown": shown}],
        "options": options,
        "option_type": "landmark",
        "declination": decl,
        "known_marker": {"pos": observer, "label": "YOU"},
        "instructions": (
            "You are standing at the marked position and sight an unidentified "
            "feature. Which lettered feature (A, B, or C) did you sight?"
        ),
        "lines": [f"Bearing to the unidentified feature: {_fmt(shown)}°"] + declination_lines(preset),
    }
    p.answer = {
        "position": observer,
        "correct_option": correct,
        "lines": [(observer, est_true)],
        "star_label": "YOU",
    }
    p.notes = notes
    return p


def build_intersection(rng, preset, difficulty) -> Puzzle:
    """Known stations report bearings TO an unknown target (e.g. smoke)."""
    stations = generate_landmarks(rng, preset["n_landmarks"])
    target = pick_unknown_with_spread(rng, stations, preset["spread_deg"])
    options, correct = generate_options(rng, target, preset)

    p = _new_puzzle(rng, "intersection", difficulty, stations, target)
    readings, lines, plot_lines, notes = [], ["Bearing from each station TO the target (smoke column):"], [], []
    if preset["declination"] is not None:
        notes.append("Convert magnetic to true first: true = magnetic + declination.")
    notes.append("Plot each bearing FORWARD from its station (no back-bearing needed).")
    for st in stations:
        shown = observe(rng, bearing(st.pos, target), preset)
        est_true = true_from_shown(shown, preset)
        readings.append({"landmark": st.name, "bearing_shown": shown})
        lines.append(f"  {st.name:<14} {_fmt(shown)}°")
        plot_lines.append((st.pos, est_true))
        notes.append(f"  {st.name}: {_fmt(shown)}° -> true {_fmt(est_true)}°")
    notes.append(f"The lines cross at the target -> marker {correct}.")

    p.given = {
        "readings": readings,
        "options": options,
        "option_type": "position",
        "declination": preset["declination"],
        "instructions": (
            "The marked stations each report the bearing to a column of smoke. "
            "Which marker (A, B, or C) is the location of the fire?"
        ),
        "lines": lines + declination_lines(preset),
    }
    p.answer = {
        "position": target,
        "correct_option": correct,
        "lines": plot_lines,
        "star_label": "TARGET",
    }
    p.notes = notes
    return p


def build_running_fix(rng, preset, difficulty) -> Puzzle:
    """Two bearings to one landmark either side of a leg of known course/speed/time."""
    step = preset["round_to"]
    decl = preset["declination"] or 0.0
    d_min = 140 + 20 * preset["error_deg"]
    cross_min = max(20.0, 0.6 * preset["spread_deg"])

    found = None
    for _ in range(5000):
        lm_pos = (rng.uniform(80, MAP_SIZE - 80), rng.uniform(80, MAP_SIZE - 80))
        p1 = (rng.uniform(100, MAP_SIZE - 100), rng.uniform(100, MAP_SIZE - 100))
        mag_heading = rng.randrange(0, 360, step)
        true_heading = (mag_heading + decl) % 360
        speed = rng.choice([3, 4, 5])
        minutes = rng.choice([20, 30, 40, 45, 60])
        d = speed * minutes / 60 * UNITS_PER_KM
        if not d_min <= d <= 450:
            continue
        p2 = project(p1, true_heading, d)
        if not in_bounds(p2, 100):
            continue
        d1, d2 = dist(p1, lm_pos), dist(p2, lm_pos)
        if min(d1, d2) < 100 or max(d1, d2) > 650:
            continue
        cross = angle_diff(bearing(p1, lm_pos), bearing(p2, lm_pos))
        if not cross_min <= cross <= 130:
            continue
        found = (lm_pos, p1, p2, mag_heading, true_heading, speed, minutes, d)
        break
    if found is None:
        raise RuntimeError("Could not construct a running-fix geometry; try another seed.")
    lm_pos, p1, p2, mag_heading, true_heading, speed, minutes, d = found

    lm = Landmark(rng.choice(FEATURE_NAMES), lm_pos)
    b1 = observe(rng, bearing(p1, lm_pos), preset)
    b2 = observe(rng, bearing(p2, lm_pos), preset)
    t1, t2 = true_from_shown(b1, preset), true_from_shown(b2, preset)
    options, correct = generate_options(rng, p2, preset, extra=[p1])

    heading_txt = f"{_fmt(mag_heading)}°" + (" (magnetic)" if preset["declination"] is not None else "")
    lines = [
        f"Course: {heading_txt}    Speed: {speed} km/h    Time: {minutes} min",
        f"Bearing to {lm.name} at START of leg: {_fmt(b1)}°",
        f"Bearing to {lm.name} at END of leg:   {_fmt(b2)}°",
    ] + declination_lines(preset)

    notes = [
        f"Distance = {speed} km/h x {minutes} min = {d / UNITS_PER_KM:.2f} km "
        f"({d / UNITS_PER_KM:.2f} grid squares).",
        f"True course = {_fmt(true_heading)}°.",
        f"Back-bearings from {lm.name}: start {_fmt(back_bearing(t1))}°, end {_fmt(back_bearing(t2))}°.",
        "Plot both back-bearing lines, then slide a course line of the travelled distance between them "
        "until it fits; the far end is your position at the END of the leg.",
        f"Marker {correct} is the end-of-leg position (the start point is a common wrong answer).",
    ]

    p = _new_puzzle(rng, "running_fix", difficulty, [lm], p2)
    p.given = {
        "readings": [
            {"landmark": lm.name, "bearing_shown": b1, "when": "start"},
            {"landmark": lm.name, "bearing_shown": b2, "when": "end"},
        ],
        "options": options,
        "option_type": "position",
        "declination": preset["declination"],
        "course_magnetic": mag_heading,
        "speed_kmh": speed,
        "minutes": minutes,
        "instructions": (
            "You walk a straight leg at a steady speed. Which marker (A, B, or C) "
            "is your position at the END of the leg?"
        ),
        "lines": lines,
    }
    p.answer = {
        "position": p2,
        "correct_option": correct,
        "star_label": "END POSITION",
        "lines": [(lm_pos, back_bearing(t1)), (lm_pos, back_bearing(t2))],
        "path": (p1, p2),
    }
    p.notes = notes
    return p


def build_error_budget(rng, preset, difficulty) -> Puzzle:
    """Which sighting contributes the largest lateral error to the fix?"""
    sigmas = ERROR_BUDGET_SIGMAS[difficulty]
    decl = preset["declination"]
    decl_unc = 0.0 if decl is None else round(abs(decl) * 0.25, 1)

    for _ in range(300):
        landmarks = generate_landmarks(rng, 3)
        observer = pick_unknown_with_spread(rng, landmarks, min(preset["spread_deg"], 60))
        ds = [dist(observer, lm.pos) for lm in landmarks]
        if min(ds) < 150:
            continue
        sig = rng.sample(sigmas, 3)
        totals = [math.hypot(s, decl_unc) for s in sig]
        lateral = [(d / UNITS_PER_KM) * 1000 * math.tan(math.radians(t)) for d, t in zip(ds, totals)]
        top = sorted(lateral, reverse=True)
        if top[0] >= 1.3 * top[1]:
            break

    win_idx = max(range(3), key=lambda i: lateral[i])
    correct = LETTERS[win_idx]
    options = {LETTERS[i]: i for i in range(3)}

    budget, lines, notes = [], ["Angular uncertainty of each sighting:"], []
    for i, lm in enumerate(landmarks):
        brg = bearing(observer, lm.pos)
        budget.append({
            "letter": LETTERS[i],
            "landmark": lm.name,
            "dist_km": ds[i] / UNITS_PER_KM,
            "sigma": sig[i],
            "sigma_total": totals[i],
            "lateral_m": lateral[i],
            "left": project(observer, brg - totals[i], ds[i]),
            "right": project(observer, brg + totals[i], ds[i]),
        })
        lines.append(f"  {LETTERS[i]}: {lm.name:<14} +/- {sig[i]:.1f}°")
        notes.append(
            f"  {LETTERS[i]} {lm.name}: {ds[i] / UNITS_PER_KM:.2f} km x tan({totals[i]:.2f}°) "
            f"= {lateral[i]:.0f} m"
        )
    if decl_unc:
        lines.append(f"  Declination uncertainty: +/- {decl_unc:.1f}° (add in quadrature to each)")
    notes = [
        "Lateral error = distance x tan(total angular uncertainty)."
        + (f" Total = sqrt(sighting² + declination²), declination term ±{decl_unc:.1f}°." if decl_unc else "")
    ] + notes
    rss = math.sqrt(sum(l ** 2 for l in lateral))
    notes.append(f"Largest contributor: {correct}. RSS of the three lateral errors ≈ {rss:.0f} m.")

    p = _new_puzzle(rng, "error_budget", difficulty, landmarks, observer)
    p.given = {
        "readings": [{"landmark": lm.name, "sigma_deg": sig[i]} for i, lm in enumerate(landmarks)],
        "options": options,
        "option_type": "landmark",
        "declination": decl,
        "declination_uncertainty": decl_unc,
        "known_marker": {"pos": observer, "label": "YOU"},
        "instructions": (
            "Your position (star) is fixed. Lateral error at a feature = distance x "
            "tan(uncertainty). Measure distances with the grid (1 square = 1 km). "
            "Which feature (A, B, or C) adds the LARGEST lateral position error?"
        ),
        "lines": lines,
    }
    p.answer = {
        "position": observer,
        "correct_option": correct,
        "star_label": "YOU",
        "budget": budget,
    }
    p.notes = notes
    return p


BUILDERS = {
    "resection": build_resection,
    "reverse_resection": build_reverse_resection,
    "intersection": build_intersection,
    "running_fix": build_running_fix,
    "error_budget": build_error_budget,
}


def build_puzzle(variant: str, difficulty: str, seed: int) -> Puzzle:
    rng = random.Random(seed)
    return BUILDERS[variant](rng, DIFFICULTY_PRESETS[difficulty], difficulty)


# --------------------------------------------------------------------------- rendering
def _draw_base_map(ax, map_seed: int):
    rng = random.Random(map_seed)
    x = np.linspace(0, MAP_SIZE, 200)
    y = np.linspace(0, MAP_SIZE, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for _ in range(5):
        cx, cy = rng.uniform(0, MAP_SIZE), rng.uniform(0, MAP_SIZE)
        amp = rng.uniform(0.5, 1.0)
        spread = rng.uniform(150, 350)
        Z += amp * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * spread ** 2)))
    ax.contour(X, Y, Z, levels=12, colors="saddlebrown", linewidths=0.5, alpha=0.5)
    ax.set_facecolor("#f5f1e6")

    ticks = range(0, MAP_SIZE + 1, UNITS_PER_KM)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.grid(True, color="#8c7b5a", alpha=0.25, lw=0.6)
    ax.plot([40, 40 + UNITS_PER_KM], [40, 40], color="black", lw=3, solid_capstyle="butt")
    ax.text(40 + UNITS_PER_KM / 2, 55, "1 km", ha="center", fontsize=8, fontweight="bold")


def _draw_north_arrow(ax):
    ax.annotate(
        "N", xy=(0.95, 0.90), xytext=(0.95, 0.78),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=2, color="black"),
        ha="center", fontsize=12, fontweight="bold",
    )


def _draw_landmarks(ax, puzzle: Puzzle):
    letters = {}
    if puzzle.given.get("option_type") == "landmark":
        letters = {idx: lab for lab, idx in puzzle.given["options"].items()}
    for i, lm in enumerate(puzzle.landmarks):
        ax.plot(*lm.pos, marker="^", color="black", markersize=10, zorder=4)
        if i in letters:
            lab = letters[i]
            ax.annotate(
                f"{lab}: {lm.name}", lm.pos, textcoords="offset points", xytext=(8, 8),
                fontsize=10, fontweight="bold", color=COLORS[lab], zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85,
                          edgecolor=COLORS[lab]),
            )
        else:
            ax.annotate(lm.name, lm.pos, textcoords="offset points", xytext=(8, 8), fontsize=10)


def _draw_options(ax, puzzle: Puzzle, reveal: bool):
    opts = puzzle.given["options"]
    correct = puzzle.answer["correct_option"]
    if puzzle.given["option_type"] == "position":
        for label, pos in opts.items():
            ax.plot(*pos, marker="o", color=COLORS[label], markersize=12,
                    markeredgecolor="black", zorder=5)
            ax.annotate(
                label, pos, textcoords="offset points", xytext=(8, -4),
                fontsize=11, fontweight="bold", color=COLORS[label], zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85,
                          edgecolor=COLORS[label]),
            )
        ring_pos = opts[correct]
    else:
        ring_pos = puzzle.landmarks[opts[correct]].pos
    if reveal:
        ax.plot(*ring_pos, marker="o", markersize=28, markerfacecolor="none",
                markeredgecolor="limegreen", markeredgewidth=3, zorder=4)


def _draw_star(ax, pos, label, color):
    ax.plot(*pos, marker="*", color=color, markersize=18, markeredgecolor="black", zorder=7)
    ax.annotate(label, pos, textcoords="offset points", xytext=(10, -16),
                fontsize=9, color=color, fontweight="bold", zorder=7)


def _finish_map(ax, title: str):
    _draw_north_arrow(ax)
    ax.set_xlim(0, MAP_SIZE)
    ax.set_ylim(0, MAP_SIZE)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, fontweight="bold")


def _new_figure():
    fig, (ax, tax) = plt.subplots(2, 1, figsize=(8, 10.8), gridspec_kw={"height_ratios": [8, 2.8]})
    tax.axis("off")
    return fig, ax, tax


def _text_panel(tax, lines: list[str], width: int = 95):
    wrapped = []
    for ln in lines:
        wrapped.extend(textwrap.wrap(ln, width, subsequent_indent="    ",
                                     replace_whitespace=False) or [""])
    tax.text(
        0.0, 1.0, "\n".join(wrapped), transform=tax.transAxes, va="top", ha="left",
        family="monospace", fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def _title(puzzle: Puzzle) -> str:
    return f"{puzzle.variant.replace('_', ' ').title()} -- {puzzle.difficulty.replace('_', ' ').title()}"


def render_puzzle(puzzle: Puzzle, out_path: str):
    fig, ax, tax = _new_figure()
    _draw_base_map(ax, puzzle.map_seed)
    _draw_landmarks(ax, puzzle)
    _draw_options(ax, puzzle, reveal=False)
    km = puzzle.given.get("known_marker")
    if km:
        _draw_star(ax, km["pos"], km["label"], "navy")
    _finish_map(ax, _title(puzzle))

    _text_panel(tax, [puzzle.given["instructions"], ""] + puzzle.given["lines"])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_solution(puzzle: Puzzle, out_path: str):
    fig, ax, tax = _new_figure()
    _draw_base_map(ax, puzzle.map_seed)
    _draw_landmarks(ax, puzzle)
    _draw_options(ax, puzzle, reveal=True)

    for origin, direction in puzzle.answer.get("lines", []):
        end = project(origin, direction, MAP_SIZE * 1.5)
        ax.plot([origin[0], end[0]], [origin[1], end[1]], "--", color="royalblue", lw=1, zorder=3)

    path = puzzle.answer.get("path")
    if path:
        ax.annotate(
            "", xy=path[1], xytext=path[0],
            arrowprops=dict(arrowstyle="-|>", color="darkgreen", lw=2.5), zorder=6,
        )
        ax.plot(*path[0], marker="s", color="darkgreen", markersize=9, markeredgecolor="black", zorder=6)
        ax.annotate("START", path[0], textcoords="offset points", xytext=(10, 6),
                    fontsize=9, color="darkgreen", fontweight="bold")

    for b in puzzle.answer.get("budget", []):
        o = puzzle.unknown_pos
        for side in ("left", "right"):
            ax.plot([o[0], b[side][0]], [o[1], b[side][1]], color="darkorange", lw=1, zorder=3)
        ax.plot([b["left"][0], b["right"][0]], [b["left"][1], b["right"][1]],
                color="darkorange", lw=3, zorder=3)
        mid = ((b["left"][0] + b["right"][0]) / 2, (b["left"][1] + b["right"][1]) / 2)
        ax.annotate(f"±{b['lateral_m']:.0f} m", mid, textcoords="offset points", xytext=(8, -16),
                    fontsize=9, color="darkorange", fontweight="bold")

    _draw_star(ax, puzzle.unknown_pos, puzzle.answer.get("star_label", "TRUE POSITION"), "crimson")
    _finish_map(ax, f"SOLUTION -- {puzzle.variant.replace('_', ' ').title()}")

    correct = puzzle.answer["correct_option"]
    head = f"ANSWER: {correct}"
    if puzzle.given["option_type"] == "landmark":
        head += f" ({puzzle.landmarks[puzzle.given['options'][correct]].name})"
    _text_panel(tax, [head, ""] + puzzle.notes)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- driver
def _show_pair(puzzle_path, solution_path, label):
    img_p, img_s = Image.open(puzzle_path), Image.open(solution_path)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    axes[0].imshow(img_p)
    axes[0].axis("off")
    axes[0].set_title(f"Puzzle Image (Trainee View) -- {label}", fontsize=14, fontweight="bold")
    axes[1].imshow(img_s)
    axes[1].axis("off")
    axes[1].set_title("Solution Image (Answer Key)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def generate(variant="resection", difficulty="hard", seed=42, outdir=None, show=False):
    """Render puzzle + solution PNGs. variant/difficulty may be 'all'."""
    variants = VARIANTS if variant == "all" else [variant]
    difficulties = list(DIFFICULTY_PRESETS) if difficulty == "all" else [difficulty]
    outdir = Path(outdir or datetime.now().strftime("%m-%d-%Y"))
    outdir.mkdir(parents=True, exist_ok=True)
    multi = len(variants) * len(difficulties) > 1

    results = []
    for vi, v in enumerate(variants):
        for di, d in enumerate(difficulties):
            s = seed + 1000 * vi + 10 * di if multi else seed
            puzzle = build_puzzle(v, d, s)
            pp = outdir / f"{v}_{d}_puzzle.png"
            sp = outdir / f"{v}_{d}_solution.png"
            render_puzzle(puzzle, str(pp))
            render_solution(puzzle, str(sp))
            print(f"{v:<18} {d:<10} seed={s:<6} answer={puzzle.answer['correct_option']}  -> {pp.name}")
            results.append((puzzle, pp, sp))

    if show:
        for puzzle, pp, sp in results:
            _show_pair(pp, sp, _title(puzzle))
    return results


# --------------------------------------------------------------------------- daily archive & catalog
# Layout:
#   archive/YYYY/YYYY-MM-DD/{puzzle.png, solution.png, meta.json, caption.txt}
#   archive/index.json     (machine-readable index of all daily puzzles)
#   archive/ARCHIVE.md     (human-readable catalog table)
#   archive/legacy/        (archived ad-hoc/manual runs)

INDEX_KEYS = (
    "date", "number", "variant", "difficulty", "seed",
    "answer", "answer_text", "puzzle_image", "solution_image", "posted",
)


def _to_repo_rel(path: Path) -> str:
    """Convert a path under REPO_ROOT to a repo-relative POSIX path for raw URLs."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def daily_plan(day: date) -> tuple[str, str, int]:
    """Deterministic (variant, difficulty, seed) for a calendar day.

    Variants cycle daily; difficulty steps up every full variant cycle,
    so every variant is seen at every level before it repeats.
    """
    n = (day - EPOCH).days
    variant = VARIANTS[n % len(VARIANTS)]
    difficulty = DAILY_DIFFICULTIES[(n // len(VARIANTS)) % len(DAILY_DIFFICULTIES)]
    return variant, difficulty, DAILY_SEED_BASE + n


def day_dir(day: date, archive: Path = ARCHIVE_DIR) -> Path:
    return Path(archive) / f"{day:%Y}" / day.isoformat()


def build_caption(puzzle: Puzzle, day: date, number: int) -> str:
    title = puzzle.variant.replace("_", " ").title()
    lines = [
        f"Land Nav Puzzle #{number} - {title} ({puzzle.difficulty.replace('_', ' ')})",
        "",
        puzzle.given["instructions"],
        "",
    ]
    lines += [ln.rstrip() for ln in puzzle.given["lines"]]
    lines += [
        "",
        "Drop your answer (A, B, or C) in the comments! Full plotted solution posts tomorrow.",
        "",
        "#landnavigation #mapandcompass #orienteering #navigation #hiking #survival #puzzle #backcountry",
    ]
    return "\n".join(lines)[:2200]


def _load_index(archive: Path) -> list[dict]:
    f = Path(archive) / "index.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def _save_index(archive: Path, entries: list[dict]):
    archive = Path(archive)
    archive.mkdir(parents=True, exist_ok=True)
    entries.sort(key=lambda e: e["date"])
    (archive / "index.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    write_catalog_md(archive, entries)


def write_catalog_md(archive: Path = ARCHIVE_DIR, entries: Optional[list[dict]] = None):
    """Generate a human-readable Markdown catalog of all archived daily puzzles."""
    archive = Path(archive)
    entries = entries if entries is not None else _load_index(archive)
    lines = [
        "# Land Navigation Puzzle Archive",
        "",
        f"Total daily puzzles archived: **{len(entries)}**",
        "",
        "| Date | # | Variant | Difficulty | Answer | Puzzle | Solution | Instagram |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for e in reversed(entries):
        d = e["date"]
        yr = d[:4]
        rel_p = f"{yr}/{d}/puzzle.png"
        rel_s = f"{yr}/{d}/solution.png"
        posted = e.get("posted")
        ig_status = f"Posted (`{posted['id']}`)" if posted else "Pending"
        ans = e.get("answer_text") or e.get("answer", "-")
        lines.append(
            f"| {d} | {e['number']} | {e['variant']} | {e['difficulty']} | "
            f"**{ans}** | [Puzzle]({rel_p}) | [Solution]({rel_s}) | {ig_status} |"
        )
    (archive / "ARCHIVE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _index_entry_from_meta(meta: dict) -> dict:
    return {k: meta.get(k) for k in INDEX_KEYS}


def generate_daily(day: Optional[date] = None, archive: Path = ARCHIVE_DIR,
                   force: bool = False) -> dict:
    """Create (or reuse) the archived puzzle for `day`. Returns its meta dict.

    Idempotent: if the day's folder already exists the files are left untouched
    (so a re-run never changes an already-published puzzle) unless force=True.
    """
    day = day or date.today()
    archive = Path(archive)
    d = day_dir(day, archive)
    meta_f = d / "meta.json"
    if meta_f.exists() and not force:
        return json.loads(meta_f.read_text(encoding="utf-8"))

    variant, difficulty, seed = daily_plan(day)
    puzzle = build_puzzle(variant, difficulty, seed)
    d.mkdir(parents=True, exist_ok=True)
    puzzle_png = d / "puzzle.png"
    solution_png = d / "solution.png"
    render_puzzle(puzzle, str(puzzle_png))
    render_solution(puzzle, str(solution_png))

    number = (day - EPOCH).days + 1
    caption = build_caption(puzzle, day, number)
    (d / "caption.txt").write_text(caption, encoding="utf-8")

    correct = puzzle.answer["correct_option"]
    answer_text = correct
    if puzzle.given["option_type"] == "landmark":
        answer_text += f" ({puzzle.landmarks[puzzle.given['options'][correct]].name})"

    meta = {
        "date": day.isoformat(),
        "number": number,
        "variant": variant,
        "difficulty": difficulty,
        "seed": seed,
        "answer": correct,
        "answer_text": answer_text,
        "puzzle_image": _to_repo_rel(puzzle_png),
        "solution_image": _to_repo_rel(solution_png),
        "posted": None,
    }
    meta_f.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    entries = [e for e in _load_index(archive) if e["date"] != meta["date"]]
    entries.append(_index_entry_from_meta(meta))
    _save_index(archive, entries)
    print(f"[archive] {day} #{number} {variant}/{difficulty} seed={seed} answer={answer_text}")
    return meta


def load_meta(day: date, archive: Path = ARCHIVE_DIR) -> Optional[dict]:
    f = day_dir(day, Path(archive)) / "meta.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def find_previous_puzzle(day: date, archive: Path = ARCHIVE_DIR) -> Optional[dict]:
    """Return yesterday's puzzle metadata, falling back to the most recent prior entry."""
    archive = Path(archive)
    yday_meta = load_meta(day - timedelta(days=1), archive)
    if yday_meta:
        return yday_meta
    prior = [e for e in _load_index(archive) if e["date"] < day.isoformat()]
    if not prior:
        return None
    prior.sort(key=lambda e: e["date"])
    return load_meta(date.fromisoformat(prior[-1]["date"]), archive) or prior[-1]


def mark_posted(day: date, post_id: str, archive: Path = ARCHIVE_DIR):
    archive = Path(archive)
    f = day_dir(day, archive) / "meta.json"
    meta = json.loads(f.read_text(encoding="utf-8"))
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    meta["posted"] = {"id": post_id, "at": ts}
    f.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    entries = [e for e in _load_index(archive) if e["date"] != meta["date"]]
    entries.append(_index_entry_from_meta(meta))
    _save_index(archive, entries)


def rebuild_index(archive: Path = ARCHIVE_DIR) -> list[dict]:
    """Scan all archive/YYYY/YYYY-MM-DD/meta.json files and rebuild index.json + ARCHIVE.md."""
    archive = Path(archive)
    entries = []
    if archive.exists():
        for meta_f in sorted(archive.glob("????/????-??-??/meta.json")):
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
            entries.append(_index_entry_from_meta(meta))
    _save_index(archive, entries)
    print(f"[archive] rebuilt index with {len(entries)} puzzle(s)")
    return entries


def archive_legacy(src: str, archive: Path = ARCHIVE_DIR):
    """Move an old flat output folder (e.g. MM-DD-YYYY/) into archive/legacy/."""
    src_p = Path(src)
    dest = Path(archive) / "legacy" / src_p.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        stamp = datetime.now().strftime("%H%M%S")
        dest = dest.parent / f"{src_p.name}_{stamp}"
    shutil.move(str(src_p), str(dest))
    print(f"[archive] moved {src_p} -> {dest}")


def organize_legacy(root: Path = REPO_ROOT, archive: Path = ARCHIVE_DIR) -> int:
    """Find legacy MM-DD-YYYY/ folders and loose *_puzzle.png/*_solution.png files and archive them."""
    root = Path(root)
    archive = Path(archive)
    moved = 0

    for item in sorted(root.iterdir()):
        if item.resolve() == archive.resolve() or item.name.startswith("."):
            continue
        if item.is_dir() and re.fullmatch(r"\d{2}-\d{2}-\d{4}", item.name):
            archive_legacy(str(item), archive)
            moved += 1

    loose = sorted(root.glob("*_puzzle.png")) + sorted(root.glob("*_solution.png"))
    if loose:
        bucket = archive / "legacy" / f"loose_{date.today().isoformat()}"
        bucket.mkdir(parents=True, exist_ok=True)
        for f in loose:
            shutil.move(str(f), str(bucket / f.name))
            moved += 1
        print(f"[archive] moved {len(loose)} loose PNG(s) -> {bucket}")

    if moved == 0:
        print("[archive] no legacy folders or loose puzzle files found")
    return moved


def backfill(start_day: date, end_day: date, archive: Path = ARCHIVE_DIR, force: bool = False):
    """Generate daily puzzles for every date from start_day through end_day inclusive."""
    if end_day < start_day:
        raise ValueError("end_day must be on or after start_day")
    cur = start_day
    while cur <= end_day:
        generate_daily(cur, archive, force=force)
        cur += timedelta(days=1)


def archive_status(archive: Path = ARCHIVE_DIR) -> str:
    archive = Path(archive)
    entries = _load_index(archive)
    posted = sum(1 for e in entries if e.get("posted"))
    today = date.today()
    v, diff, seed = daily_plan(today)
    lines = [
        f"Archive directory : {archive.resolve()}",
        f"Total puzzles     : {len(entries)} ({posted} posted, {len(entries) - posted} pending)",
    ]
    if entries:
        lines.append(f"Date range        : {entries[0]['date']} -> {entries[-1]['date']}")
    lines.append(f"Today ({today}) : {v} / {diff} (seed={seed})")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate and archive land-navigation puzzles.")
    ap.add_argument("--variant", default="resection", choices=VARIANTS + ["all"])
    ap.add_argument("--difficulty", default="hard", choices=list(DIFFICULTY_PRESETS) + ["all"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=None, help="default: today's date (MM-DD-YYYY)")
    ap.add_argument("--show", action="store_true", help="display puzzle/solution side by side")
    ap.add_argument("--daily", action="store_true",
                    help="generate today's puzzle into the archive (rotates variant/difficulty)")
    ap.add_argument("--date", default=None, help="with --daily: YYYY-MM-DD (default today)")
    ap.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                    help="generate daily puzzles for YYYY-MM-DD range inclusive")
    ap.add_argument("--organize", action="store_true",
                    help="move legacy MM-DD-YYYY folders and loose PNGs into archive/legacy/")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="rebuild archive/index.json and archive/ARCHIVE.md from meta.json files")
    ap.add_argument("--status", action="store_true", help="print archive status summary")
    ap.add_argument("--archive", default=str(ARCHIVE_DIR))
    ap.add_argument("--force", action="store_true", help="regenerate existing archived days")
    args, _ = ap.parse_known_args()

    arch = Path(args.archive)
    if args.organize:
        organize_legacy(REPO_ROOT, arch)
    elif args.rebuild_index:
        rebuild_index(arch)
    elif args.status:
        print(archive_status(arch))
    elif args.backfill:
        backfill(date.fromisoformat(args.backfill[0]), date.fromisoformat(args.backfill[1]),
                 arch, args.force)
    elif args.daily:
        generate_daily(date.fromisoformat(args.date) if args.date else None, arch, args.force)
    else:
        generate(args.variant, args.difficulty, args.seed, args.outdir, args.show)