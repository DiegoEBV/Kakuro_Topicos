"""
render.py — Dibuja un Kakuro (para crear el dataset) y aplica degradaciones que
simulan fotografías de un puzzle impreso (perspectiva, iluminación, sombras, ruido...).
"""
from __future__ import annotations

import math
import random
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .puzzle import Puzzle

STYLES = {
    #            bloque           diagonal / texto pista   línea grilla
    "classic": {"block": (20, 20, 20), "ink": (255, 255, 255), "line": (0, 0, 0)},
    "gray":    {"block": (185, 185, 185), "ink": (0, 0, 0), "line": (0, 0, 0)},
    "navy":    {"block": (38, 52, 92), "ink": (255, 255, 255), "line": (30, 30, 30)},
}


def render_puzzle(p: Puzzle, style: str = "classic", font_path: Optional[str] = None,
                  cell: int = 72, margin: int = 40, show_solution: bool = False) -> np.ndarray:
    """Renderiza el tablero como imagen RGB (uint8)."""
    st = STYLES[style]
    W, H = p.cols * cell + 2 * margin, p.rows * cell + 2 * margin
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, int(cell * 0.30)) if font_path else ImageFont.load_default()
    big = ImageFont.truetype(font_path, int(cell * 0.6)) if font_path else font
    lw = max(1, cell // 36)
    for r in range(p.rows):
        for c in range(p.cols):
            x0, y0 = margin + c * cell, margin + r * cell
            x1, y1 = x0 + cell, y0 + cell
            if p.white[r][c]:
                d.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=st["line"], width=lw)
                if show_solution and p.solution:
                    d.text(((x0 + x1) / 2, (y0 + y1) / 2), str(p.solution[r][c]),
                           fill=(0, 0, 0), font=big, anchor="mm")
            else:
                d.rectangle([x0, y0, x1, y1], fill=st["block"], outline=st["line"], width=lw)
                dn, ac = p.down[r][c], p.across[r][c]
                if dn or ac:
                    d.line([x0 + lw, y0 + lw, x1 - lw, y1 - lw], fill=st["ink"], width=lw + 1)
                if ac:   # pista horizontal: triángulo superior derecho
                    d.text((x0 + cell * 0.70, y0 + cell * 0.30), str(ac), fill=st["ink"],
                           font=font, anchor="mm")
                if dn:   # pista vertical: triángulo inferior izquierdo
                    d.text((x0 + cell * 0.30, y0 + cell * 0.72), str(dn), fill=st["ink"],
                           font=font, anchor="mm")
    bw = max(3, cell // 18)  # borde exterior grueso
    d.rectangle([margin - bw // 2, margin - bw // 2, margin + p.cols * cell + bw // 2,
                 margin + p.rows * cell + bw // 2], outline=(0, 0, 0), width=bw)
    return np.array(img)


# --------------------------------------------------------------- degradaciones
def _background(h: int, w: int, rng: np.random.Generator, kind: str = "wood") -> np.ndarray:
    if kind == "wood":
        base = np.array([95, 140, 175], np.float32)      # BGR madera
        y = np.arange(h)[:, None]
        x = np.arange(w)[None, :]
        grain = 18 * np.sin(x / 9.0 + 6 * np.sin(y / 120.0)) + rng.normal(0, 6, (h, w))
        bg = base[None, None, :] + grain[..., None]
    else:  # mesa gris / tela
        bg = np.full((h, w, 3), 120, np.float32) + rng.normal(0, 12, (h, w, 1))
        bg = cv2.GaussianBlur(bg, (0, 0), 2)
    return np.clip(bg, 0, 255).astype(np.uint8)


def photo_effects(board_rgb: np.ndarray, seed: int = 0, perspective: float = 0.06,
                  rotation: float = 0.0, light: Tuple[float, float] = (0.85, 1.05),
                  shadow: float = 0.0, noise: float = 4.0, blur: float = 0.6,
                  tint: Tuple[float, float, float] = (1.0, 1.0, 1.0), jpeg: int = 85,
                  out_long_side: int = 1400, bg_kind: str = "wood",
                  paper_texture: float = 3.0) -> np.ndarray:
    """Simula la foto de una hoja impresa con el puzzle. Devuelve BGR (convención OpenCV)."""
    rng = np.random.default_rng(seed)
    board = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    # textura de papel
    board += rng.normal(0, paper_texture, board.shape[:2])[..., None]
    h, w = board.shape[:2]
    pad = int(0.35 * max(h, w))
    H, W = h + 2 * pad, w + 2 * pad
    canvas = _background(H, W, rng, bg_kind).astype(np.float32)
    # hoja de papel un poco más grande que el tablero
    sheet = np.full((h + 80, w + 80, 3), 245, np.float32)
    sheet[40:40 + h, 40:40 + w] = board
    src = np.float32([[0, 0], [w + 80, 0], [w + 80, h + 80], [0, h + 80]])
    cx, cy = W / 2, H / 2
    # rotación + perspectiva aleatoria de las esquinas
    ang = math.radians(rotation)
    dst = []
    for (x, y) in src:
        x, y = x - (w + 80) / 2, y - (h + 80) / 2
        xr = x * math.cos(ang) - y * math.sin(ang)
        yr = x * math.sin(ang) + y * math.cos(ang)
        j = perspective * max(h, w)
        dst.append([cx + xr + rng.uniform(-j, j), cy + yr + rng.uniform(-j, j)])
    M = cv2.getPerspectiveTransform(src, np.float32(dst))
    warped = cv2.warpPerspective(sheet, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpPerspective(np.ones(sheet.shape[:2], np.float32), M, (W, H))
    canvas = canvas * (1 - mask[..., None]) + warped * mask[..., None]
    # iluminación: gradiente lineal en dirección aleatoria
    th = rng.uniform(0, 2 * math.pi)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = (np.cos(th) * (xx - cx) + np.sin(th) * (yy - cy)) / max(H, W)
    gain = light[0] + (light[1] - light[0]) * (g - g.min()) / (np.ptp(g) + 1e-6)
    canvas *= gain[..., None]
    # sombra suave (polígono oscuro desenfocado)
    if shadow > 0:
        poly = np.int32([[rng.uniform(0, W), 0], [rng.uniform(0, W), 0],
                         [rng.uniform(0, W), H], [rng.uniform(0, W * 0.5), H]])
        sm = np.zeros((H, W), np.float32)
        cv2.fillPoly(sm, [poly], 1.0)
        sm = cv2.GaussianBlur(sm, (0, 0), max(H, W) / 40)
        canvas *= (1 - shadow * sm)[..., None]
    canvas *= np.float32(tint)[None, None, ::-1]
    if blur > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
    canvas += rng.normal(0, noise, canvas.shape)
    img = np.clip(canvas, 0, 255).astype(np.uint8)
    s = out_long_side / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def digital_effects(board_rgb: np.ndarray, scale: float = 1.0, jpeg: int = 95) -> np.ndarray:
    """Versión digital (captura de pantalla / web), opcionalmente de baja resolución."""
    img = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2BGR)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)
