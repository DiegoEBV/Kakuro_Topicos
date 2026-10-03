"""
visualize.py — Fase 3: proyección de la solución.

* overlay_solution: dibuja los dígitos en el tablero rectificado y los re-proyecta sobre
  la foto original con la homografía inversa (realidad aumentada sencilla).
* draw_board: grilla limpia con pistas y solución (matplotlib).
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .puzzle import Puzzle
from .vision import CELL_PX, VisionResult


def overlay_solution(img_bgr: np.ndarray, vr: VisionResult, solution: List[List[int]],
                     color=(40, 160, 40)) -> np.ndarray:
    p = vr.puzzle
    cp = CELL_PX
    W, H = p.cols * cp, p.rows * cp
    layer = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), np.uint8)
    for r in range(p.rows):
        for c in range(p.cols):
            if p.white[r][c] and solution[r][c]:
                txt = str(solution[r][c])
                fs = cp / 40.0
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs, 3)
                org = (int(c * cp + (cp - tw) / 2), int(r * cp + (cp + th) / 2))
                cv2.putText(layer, txt, org, cv2.FONT_HERSHEY_DUPLEX, fs, color, 3, cv2.LINE_AA)
                cv2.putText(mask, txt, org, cv2.FONT_HERSHEY_DUPLEX, fs, 255, 3, cv2.LINE_AA)
    Hinv = np.linalg.inv(vr.H)
    h0, w0 = img_bgr.shape[:2]
    lw = cv2.warpPerspective(layer, Hinv, (w0, h0))
    mw = cv2.warpPerspective(mask, Hinv, (w0, h0)).astype(np.float32)[..., None] / 255.0
    out = (img_bgr.astype(np.float32) * (1 - mw) + lw.astype(np.float32) * mw).astype(np.uint8)
    cv2.polylines(out, [vr.corners.astype(np.int32)], True, (0, 0, 255), max(2, w0 // 400))
    return out


def draw_board(p: Puzzle, solution: Optional[List[List[int]]] = None, ax=None,
               title: Optional[str] = None, highlight_errors: Optional[set] = None):
    """Grilla limpia con matplotlib (pistas en blanco sobre negro, solución en azul)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(p.cols * 0.55, p.rows * 0.55))
    for r in range(p.rows):
        for c in range(p.cols):
            x, y = c, p.rows - 1 - r
            if p.white[r][c]:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, fc="white", ec="black", lw=1))
                if solution and solution[r][c]:
                    ax.text(x + .5, y + .5, str(solution[r][c]), ha="center", va="center",
                            fontsize=13, color="#1f4e9c", fontweight="bold")
            else:
                bad = highlight_errors and (r, c) in highlight_errors
                ax.add_patch(plt.Rectangle((x, y), 1, 1, fc="#b22" if bad else "#222", ec="black", lw=1))
                d, a = p.down[r][c], p.across[r][c]
                if d or a:
                    ax.plot([x, x + 1], [y + 1, y], color="white", lw=0.8)
                if a:
                    ax.text(x + .72, y + .7, str(a), ha="center", va="center", fontsize=8, color="white")
                if d:
                    ax.text(x + .3, y + .28, str(d), ha="center", va="center", fontsize=8, color="white")
    ax.set_xlim(0, p.cols); ax.set_ylim(0, p.rows)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    return ax
