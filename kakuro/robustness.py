"""
robustness.py — Barrido de robustez: genera imágenes aleatorias con degradaciones
de intensidad creciente y mide la precisión de la visión en cada nivel.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List

import numpy as np

from .evaluate import compare
from .generator import make_puzzle
from .render import photo_effects, render_puzzle
from .vision import read_puzzle

FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts", "render")

LEVELS = {
    "blur":        [0.5, 1.0, 1.5, 2.0, 2.5],
    "perspective": [0.02, 0.05, 0.08, 0.11, 0.14],
    "rotation":    [0, 8, 16, 24, 32],
    "noise":       [2, 6, 10, 14, 18],
    "dark":        [0.9, 0.7, 0.5, 0.35, 0.25],
}


def sweep(classifier, factor: str, n_per_level: int = 4, seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    fonts = sorted(os.listdir(FONT_DIR))
    rows = []
    for lvl in LEVELS[factor]:
        for k in range(n_per_level):
            size = rng.choice([7, 8, 9, 10])
            p = make_puzzle(size, size, seed=rng.randint(0, 10**6), unique=False)
            style = rng.choice(["classic", "classic", "gray", "navy"])
            board = render_puzzle(p, style, os.path.join(FONT_DIR, rng.choice(fonts)))
            kw = {"perspective": 0.03, "seed": rng.randint(0, 10**6)}
            if factor == "blur":
                kw["blur"] = lvl
            elif factor == "perspective":
                kw["perspective"] = lvl
            elif factor == "rotation":
                kw["rotation"] = lvl * rng.choice([-1, 1])
            elif factor == "noise":
                kw["noise"] = lvl
            elif factor == "dark":
                kw["light"] = (lvl * 0.8, lvl)
            img = photo_effects(board, **kw)
            try:
                vr = read_puzzle(img, classifier)
                m = compare(vr.puzzle, p)
            except Exception:
                m = {"grid_ok": False, "cells_acc": 0.0, "clue_acc": 0.0, "exact": False}
            rows.append({"factor": factor, "level": lvl, **{k2: m[k2] for k2 in
                         ("grid_ok", "cells_acc", "clue_acc", "exact")}})
    return rows


def summarize(rows: List[Dict]) -> Dict:
    out = {}
    for r in rows:
        out.setdefault(r["level"], []).append(r)
    return {lvl: {"grid": np.mean([x["grid_ok"] for x in v]),
                  "cells": np.mean([x["cells_acc"] for x in v]),
                  "clues": np.mean([x["clue_acc"] for x in v]),
                  "exact": np.mean([x["exact"] for x in v])} for lvl, v in out.items()}
