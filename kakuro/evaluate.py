"""
evaluate.py — Métricas de la Fase 1 contra el ground truth y pipeline extremo a extremo.
"""
from __future__ import annotations

import glob
import os
import time
from typing import Dict, List

import cv2
import numpy as np

from .puzzle import Puzzle
from .solver import solve
from .vision import read_puzzle


def compare(pred: Puzzle, gt: Puzzle) -> Dict:
    """Compara la estructura extraída con el ground truth."""
    res = {"grid_ok": pred.rows == gt.rows and pred.cols == gt.cols}
    if not res["grid_ok"]:
        res.update(cells_acc=0.0, clue_acc=0.0, clues_total=sum(
            (a > 0) + (d > 0) for ra, rd in zip(gt.across, gt.down) for a, d in zip(ra, rd)),
            clues_ok=0, exact=False)
        return res
    n = gt.rows * gt.cols
    cell_ok = sum(pred.white[r][c] == gt.white[r][c] for r in range(gt.rows) for c in range(gt.cols))
    tot = ok = 0
    errors = []
    for r in range(gt.rows):
        for c in range(gt.cols):
            for name in ("across", "down"):
                g = getattr(gt, name)[r][c]
                if g:
                    tot += 1
                    p = getattr(pred, name)[r][c]
                    ok += p == g
                    if p != g:
                        errors.append((r, c, name, g, p))
    res.update(cells_acc=cell_ok / n, cells_ok=cell_ok, cells_total=n, clue_acc=ok / max(tot, 1),
               clues_ok=ok, clues_total=tot, clue_errors=errors,
               exact=(cell_ok == n and ok == tot))
    return res


def evaluate_dataset(data_dir: str, classifier, ocr: str = "mlp", correction: bool = True,
                     solve_backend: str = "auto", do_solve: bool = True) -> List[Dict]:
    rows = []
    for img_path in sorted(glob.glob(os.path.join(data_dir, "images", "*.*"))):
        name = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(data_dir, "ground_truth", name + ".json")
        gt = Puzzle.load(gt_path) if os.path.exists(gt_path) else None
        img = cv2.imread(img_path)
        row = {"image": name}
        try:
            vr = read_puzzle(img, classifier, ocr=ocr, use_constraint_correction=correction)
            row["vision_s"] = vr.timings["total"]
            row["corrections"] = vr.debug["corrections"]
            if gt is not None:
                row.update(compare(vr.puzzle, gt))
            if do_solve:
                t0 = time.perf_counter()
                sr = solve(vr.puzzle, backend=solve_backend, model="global_dom")
                row["solve_s"] = time.perf_counter() - t0
                row["solved"] = sr.solved
                row["unique"] = sr.unique
                row["backend"] = sr.backend
                row["correct_solution"] = bool(sr.solved and gt is not None and gt.solution == sr.solution)
        except Exception as e:  # el pipeline no debe caerse con una imagen mala
            row.update(error=str(e), grid_ok=False, cells_acc=0.0, cells_ok=0, cells_total=0,
                       clue_acc=0.0, clues_ok=0, clues_total=0, exact=False,
                       vision_s=row.get("vision_s", float("nan")), corrections=row.get("corrections", 0),
                       solve_s=float("nan"), solved=False, unique=False, backend=None,
                       correct_solution=False)
        rows.append(row)
    return rows
