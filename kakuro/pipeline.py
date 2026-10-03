"""
pipeline.py — Integración extremo a extremo (sin intervención manual):

    imagen  ->  [Fase 1: visión + OCR]  ->  Puzzle (JSON)  ->  [Fase 2: CP]  ->  solución
            ->  [Fase 3: visualización sobre la imagen original]

Uso por consola:
    python -m kakuro.pipeline data/images/05_photo_angle.jpg --out resultado.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2

from .ocr import DigitClassifier
from .solver import SolveResult, solve
from .vision import VisionResult, read_puzzle
from .visualize import overlay_solution

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "digit_mlp.joblib")


@dataclass
class PipelineResult:
    vision: VisionResult
    solve: SolveResult
    overlay: Optional[object]
    total_s: float


def solve_image(img_or_path, classifier: Optional[DigitClassifier] = None,
                backend: str = "auto", model: str = "global_dom") -> PipelineResult:
    t0 = time.perf_counter()
    img = cv2.imread(img_or_path) if isinstance(img_or_path, str) else img_or_path
    if img is None:
        raise FileNotFoundError(img_or_path)
    classifier = classifier or DigitClassifier.load(DEFAULT_MODEL)
    vr = read_puzzle(img, classifier)                       # Fase 1
    sr = solve(vr.puzzle, backend=backend, model=model)     # Fase 2 (entrada = salida de Fase 1)
    ov = overlay_solution(img, vr, sr.solution) if sr.solved else None   # Fase 3
    return PipelineResult(vr, sr, ov, time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Resuelve un Kakuro a partir de una imagen")
    ap.add_argument("image")
    ap.add_argument("--out", default="solucion.jpg")
    ap.add_argument("--json", default=None, help="guardar el puzzle extraído en JSON")
    ap.add_argument("--backend", default="auto", choices=["auto", "ortools", "native"])
    ap.add_argument("--model", default="global_dom", choices=["global", "global_dom", "reified"])
    a = ap.parse_args()
    res = solve_image(a.image, backend=a.backend, model=a.model)
    p = res.vision.puzzle
    print(f"Grilla detectada: {p.rows}x{p.cols}, celdas blancas: {len(p.white_cells())}")
    for w in res.vision.warnings:
        print("  aviso:", w)
    print(f"Solver: {res.solve.backend}/{res.solve.model} -> {res.solve.status}, "
          f"única={res.solve.unique}, {res.solve.time_s * 1000:.1f} ms")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(p.to_dict(), f, indent=1)
    if res.overlay is not None:
        cv2.imwrite(a.out, res.overlay)
        print("Solución guardada en", a.out)
    print(f"Tiempo total: {res.total_s:.2f} s")


if __name__ == "__main__":
    main()
