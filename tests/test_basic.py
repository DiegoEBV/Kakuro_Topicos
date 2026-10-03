"""Pruebas básicas: `pytest -q` desde la raíz del repositorio."""
import glob
import os
import sys

import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kakuro.generator import make_puzzle          # noqa: E402
from kakuro.ocr import DigitClassifier            # noqa: E402
from kakuro.puzzle import Puzzle, combos, sum_bounds  # noqa: E402
from kakuro.solver import HAS_ORTOOLS, MODELS, solve  # noqa: E402
from kakuro.vision import read_puzzle             # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_combinatorics():
    assert sum_bounds(2) == (3, 17)
    assert combos(2, 3) == ((1, 2),)
    assert combos(9, 45) == ((1, 2, 3, 4, 5, 6, 7, 8, 9),)


@pytest.mark.parametrize("model", MODELS)
def test_native_solver_unique(model):
    p = make_puzzle(8, 8, seed=3)
    r = solve(p, backend="native", model=model)
    assert r.solved and p.check_solution(r.solution) and r.unique


@pytest.mark.skipif(not HAS_ORTOOLS, reason="OR-Tools no instalado")
@pytest.mark.parametrize("model", MODELS)
def test_ortools_matches_native(model):
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "ground_truth", "*.json"))):
        p = Puzzle.load(f)
        r = solve(p, backend="ortools", model=model)
        assert r.solved and r.unique and r.solution == p.solution, f


def test_infeasible_detected():
    p = make_puzzle(6, 6, seed=1)
    run = p.runs()[0]
    r0, c0 = run.clue_cell
    (p.across if run.direction == "across" else p.down)[r0][c0] = 1  # pista imposible
    assert not solve(p, backend="native").solved


def test_end_to_end_one_image():
    clf = DigitClassifier.load(os.path.join(ROOT, "models", "digit_mlp.joblib"))
    img = cv2.imread(os.path.join(ROOT, "data", "images", "05_photo_angle.jpg"))
    vr = read_puzzle(img, clf)
    gt = Puzzle.load(os.path.join(ROOT, "data", "ground_truth", "05_photo_angle.json"))
    assert (vr.puzzle.rows, vr.puzzle.cols) == (gt.rows, gt.cols)
    assert solve(vr.puzzle).solution == gt.solution
