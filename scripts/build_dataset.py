"""
build_dataset.py — Genera el dataset de prueba (imágenes + ground truth JSON).

Uso:  python scripts/build_dataset.py --out data
Cada imagen se acompaña de un JSON con el puzzle, la solución y la condición de captura.
Las imágenes reales tomadas con celular pueden añadirse a data/images/ con su JSON
(misma estructura) y el evaluador las incluirá automáticamente.
"""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kakuro.generator import make_puzzle          # noqa: E402
from kakuro.render import render_puzzle, photo_effects, digital_effects  # noqa: E402

FONTS = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts", "render")

# (id, filas, columnas, estilo, fuente, tipo, parámetros)
CONDITIONS = [
    ("01_digital_clean",   7, 7,  "classic", "DejaVuSans-Bold.ttf",     "digital", {}),
    ("02_digital_gray",    8, 8,  "gray",    "LiberationSans-Bold.ttf", "digital", {}),
    ("03_digital_lowres",  9, 9,  "classic", "FreeSansBold.ttf",        "digital", {"scale": 0.45, "jpeg": 35}),
    ("04_photo_mild",      8, 8,  "classic", "Carlito-Bold.ttf",        "photo", {"perspective": 0.03}),
    ("05_photo_angle",     9, 9,  "classic", "DejaVuSans-Bold.ttf",     "photo", {"perspective": 0.10}),
    ("06_photo_dim",       8, 8,  "classic", "LiberationSans-Bold.ttf", "photo", {"light": (0.35, 0.55), "noise": 7}),
    ("07_photo_shadow",    10, 10, "classic", "FreeSansBold.ttf",       "photo", {"shadow": 0.55}),
    ("08_photo_warm",      9, 9,  "gray",    "Carlito-Bold.ttf",        "photo", {"tint": (1.0, 0.88, 0.65), "light": (0.7, 1.0)}),
    ("09_photo_blur",      8, 8,  "classic", "DejaVuSans-Bold.ttf",     "photo", {"blur": 1.8, "jpeg": 60}),
    ("10_photo_rotated",   9, 9,  "classic", "LiberationSans-Bold.ttf", "photo", {"rotation": 14}),
    ("11_photo_overexp",   8, 8,  "gray",    "FreeSansBold.ttf",        "photo", {"light": (1.05, 1.35)}),
    ("12_scan_navy",       10, 10, "navy",   "Carlito-Bold.ttf",        "photo", {"perspective": 0.01, "rotation": -3, "bg_kind": "gray"}),
    ("13_rect_photo",      8, 11, "classic", "DejaVuSans-Bold.ttf",     "photo", {"perspective": 0.05, "rotation": 6, "bg_kind": "gray"}),
    ("14_large_digital",   12, 12, "classic", "LiberationSans-Bold.ttf", "digital", {"scale": 0.8}),
]


def main(out: str) -> None:
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "ground_truth"), exist_ok=True)
    for k, (name, R, C, style, font, kind, params) in enumerate(CONDITIONS):
        p = make_puzzle(R, C, seed=100 + k)
        board = render_puzzle(p, style=style, font_path=os.path.join(FONTS, font))
        img = digital_effects(board, **params) if kind == "digital" else \
            photo_effects(board, seed=k, **params)
        cv2.imwrite(os.path.join(out, "images", name + ".jpg"), img)
        p.meta.update({"image": name + ".jpg", "style": style, "font": font, "capture": kind,
                       "params": {a: list(b) if isinstance(b, tuple) else b for a, b in params.items()}})
        p.save(os.path.join(out, "ground_truth", name + ".json"))
        print(f"{name}: {R}x{C}, {p.stats()['white_cells']} celdas blancas, única={p.meta['unique']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    main(ap.parse_args().out)
