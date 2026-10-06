"""
vision.py — Fase 1: de la imagen a la estructura de datos del Kakuro.

Pipeline
--------
1. Preprocesamiento: escala de grises, desenfoque gaussiano, umbral adaptativo.
2. Localización del tablero: contornos externos -> cuadrilátero más grande y "rectangular".
3. Corrección de perspectiva: homografía de las 4 esquinas a un rectángulo.
4. Estimación del tamaño de la grilla: perfiles de gradiente (Sobel) proyectados por
   filas/columnas y búsqueda del número de celdas n que maximiza el contraste
   "líneas - puntos medios" (robusto a armónicos n/2 y 2n).
5. Clasificación de celdas blanca / negra: mediana del interior normalizada por un
   modelo de iluminación cuadrático ajustado de forma robusta + umbral de Otsu.
6. Lectura de pistas: polaridad automática, binarización de Otsu, eliminación de la
   diagonal, componentes conexas, asignación al triángulo (horizontal / vertical)
   y clasificación de cada dígito (MLP propio o Tesseract).
7. Corrección guiada por restricciones: cada pista debe estar en el rango alcanzable
   por su suma; se elige la lectura más probable que lo cumpla.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .puzzle import Puzzle, sum_bounds

CELL_PX = 96  # resolución de cada celda en el tablero rectificado


@dataclass
class VisionResult:
    puzzle: Puzzle
    corners: np.ndarray            # 4x2 esquinas en la imagen original (TL, TR, BR, BL)
    H: np.ndarray                  # homografía original -> rectificado
    warped: np.ndarray             # tablero rectificado (BGR)
    timings: Dict[str, float] = field(default_factory=dict)
    debug: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ------------------------------------------------------------------ utilidades
def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(-1, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def preprocess(img_bgr: np.ndarray, max_side: int = 1600) -> Tuple[np.ndarray, np.ndarray, float]:
    s = min(1.0, max_side / max(img_bgr.shape[:2]))
    img = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray, s


def find_board(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve las esquinas del tablero y la imagen binaria usada."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    block = int(max(gray.shape) / 25) | 1
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, 7)
    # Otsu sobre todo el tablero capta los bloques oscuros; se combina con el adaptativo
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cands = []
    img_area = gray.shape[0] * gray.shape[1]
    for b in (binary, cv2.bitwise_or(binary, otsu)):
        bb = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(bb, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 0.04 * img_area:
                continue
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            approx = None
            for eps in (0.01, 0.02, 0.03, 0.05):
                a = cv2.approxPolyDP(hull, eps * peri, True)
                if len(a) == 4:
                    approx = a
                    break
            if approx is None:
                continue
            quad_area = cv2.contourArea(approx)
            fill = area / max(quad_area, 1)        # ≈1 si el contorno es realmente un cuadrilátero
            if fill < 0.75 or quad_area > 0.992 * img_area:
                continue
            corners = order_corners(approx)
            band = _border_darkness(gray, corners)
            cands.append((quad_area, band, corners))
    if not cands:
        h, w = gray.shape
        full_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        return full_corners, binary
    # El tablero tiene un borde exterior grueso y oscuro; la hoja de papel no.
    best_band = max(c[1] for c in cands)
    pool = [c for c in cands if c[1] >= 0.80 * best_band]
    # Además, la región debe mostrar una grilla periódica clara
    qual = []
    for area, band, corners in pool:
        w, _ = warp(gray, corners, 600, 600)
        _, _, d = estimate_grid(w)
        qual.append(min(max(d["score_x"]), max(d["score_y"])))
    best_q = max(qual)
    pool = [c for c, q in zip(pool, qual) if q >= 0.6 * best_q]
    pool.sort(key=lambda t: -t[0])
    return pool[0][2], binary


def _border_darkness(gray: np.ndarray, corners: np.ndarray, size: int = 400) -> float:
    """Oscuridad de las bandas superior e izquierda justo dentro del cuadrilátero.

    En un Kakuro la primera fila y la primera columna son siempre bloques (oscuros),
    mientras que el margen de una hoja de papel es claro.
    """
    w, _ = warp(gray, corners, size, size)
    paper = np.percentile(w, 95)                      # nivel del papel / celdas blancas
    b = (w < 0.9 * paper).astype(np.float32)          # bloques negros, grises o de color
    k0, k1 = size // 100, size // 33
    band = np.concatenate([b[k0:k1, k1:].ravel(), b[k1:, k0:k1].ravel()])
    return float(band.mean())


def warp(img: np.ndarray, corners: np.ndarray, out_w: int, out_h: int) -> Tuple[np.ndarray, np.ndarray]:
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    H = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_CUBIC), H


def _period_score(profile: np.ndarray, n: int) -> float:
    L = len(profile)
    step = L / n
    xs = np.arange(L)
    lines = np.interp(np.arange(1, n) * step, xs, profile)
    mids = np.interp((np.arange(n) + 0.5) * step, xs, profile)
    return float(lines.mean() - mids.mean()) / (profile.std() + 1e-6)


def estimate_grid(gray_warped: np.ndarray, n_min: int = 3, n_max: int = 30) -> Tuple[int, int, Dict]:
    """Número de filas y columnas a partir de los perfiles de gradiente."""
    g = cv2.GaussianBlur(gray_warped, (3, 3), 0).astype(np.float32)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    # recorta un pequeño margen para excluir el borde exterior grueso
    m = 4
    px = cv2.GaussianBlur(gx[m:-m, :].sum(0)[None, :], (0, 0), 1.5).ravel()
    py = cv2.GaussianBlur(gy[:, m:-m].sum(1)[None, :], (0, 0), 1.5).ravel()
    ns = list(range(n_min, n_max + 1))
    sx = [_period_score(px, n) for n in ns]
    sy = [_period_score(py, n) for n in ns]
    # Los subarmónicos impares (n/3, n/5...) puntúan casi igual que el n real, mientras que
    # los múltiplos (2n) puntúan bajo: se elige el mayor n con puntaje cercano al máximo.
    def pick(scores):
        sc = dict(zip(ns, scores))
        n0 = max(sc, key=sc.get)
        best = n0
        for k in range(2, n_max // n0 + 1):          # ¿algún múltiplo también es consistente?
            if sc.get(k * n0, -1) >= 0.5 * sc[n0]:
                best = k * n0
        return best
    cols, rows = pick(sx), pick(sy)
    return rows, cols, {"profile_x": px, "profile_y": py, "ns": ns, "score_x": sx, "score_y": sy}


def classify_cells(gray: np.ndarray, rows: int, cols: int,
                   ring: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clasifica cada celda como blanca (1) o bloque (0).

    gray: tablero rectificado (CELL_PX px por celda).
    ring: muestras (fila, col, intensidad) del papel justo fuera del tablero, en unidades
          de celda. Con ellas se ajusta un modelo de iluminación cuadrático L(fila, col)
          (en log), que luego se refina con las celdas clasificadas como blancas.
    Devuelve (white, ratio = mediana / iluminación, iluminación).
    """
    cp = CELL_PX
    med = np.zeros((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            roi = gray[r * cp + cp // 5:(r + 1) * cp - cp // 5, c * cp + cp // 5:(c + 1) * cp - cp // 5]
            med[r, c] = np.median(roi)

    def design(rr, cc):
        rr, cc = rr / rows - 0.5, cc / cols - 0.5
        return np.stack([np.ones_like(rr), rr, cc, rr * rr, cc * cc, rr * cc], -1)

    rr, cc = np.mgrid[0:rows, 0:cols].astype(np.float32) + 0.5
    A_cells = design(rr.ravel(), cc.ravel())
    y_cells = np.log(med.ravel() + 1)
    if ring is not None and len(ring) >= 6:
        A_ring, y_ring = design(ring[:, 0], ring[:, 1]), np.log(ring[:, 2] + 1)
    else:  # sin margen visible: semilla con las celdas más claras
        sel = med.ravel() >= np.percentile(med, 70)
        A_ring, y_ring = A_cells[sel], y_cells[sel]
    cand = np.zeros(rows * cols, bool)
    for it in range(4):
        A = np.vstack([A_ring, A_cells[cand]])
        y = np.concatenate([y_ring, y_cells[cand]])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        illum = np.exp(A_cells @ coef) - 1
        ratio = med.ravel() / np.maximum(illum, 1)
        thr = min(max(otsu_1d(ratio), 0.55), 0.96)
        cand = ratio > thr
    ratio = ratio.reshape(rows, cols)
    white = (ratio > thr).astype(int)
    white[0, :] = 0   # la primera fila y columna de un Kakuro siempre son bloques
    white[:, 0] = 0
    return white, ratio, illum.reshape(rows, cols)


def paper_ring(img_gray: np.ndarray, corners: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Muestras de intensidad del papel en un anillo exterior al tablero (0.25-0.5 celdas)."""
    cp = CELL_PX
    m = cp // 2
    W, Hh = cols * cp, rows * cp
    dst = np.float32([[m, m], [m + W - 1, m], [m + W - 1, m + Hh - 1], [m, m + Hh - 1]])
    Hm = cv2.getPerspectiveTransform(corners, dst)
    ext = cv2.warpPerspective(img_gray, Hm, (W + 2 * m, Hh + 2 * m), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a, b = int(0.08 * cp), int(0.35 * cp)   # distancia (px) hacia fuera del borde
    pts = []
    for k in range(cols):   # bordes superior e inferior
        x0, x1 = m + k * cp, m + (k + 1) * cp
        top = ext[m - b:m - a, x0:x1]; bot = ext[m + Hh + a:m + Hh + b, x0:x1]
        pts.append((-0.25, k + 0.5, np.median(top))); pts.append((rows + 0.25, k + 0.5, np.median(bot)))
    for k in range(rows):   # bordes izquierdo y derecho
        y0, y1 = m + k * cp, m + (k + 1) * cp
        lef = ext[y0:y1, m - b:m - a]; rig = ext[y0:y1, m + W + a:m + W + b]
        pts.append((k + 0.5, -0.25, np.median(lef))); pts.append((k + 0.5, cols + 0.25, np.median(rig)))
    pts = np.array(pts, np.float32)
    # descarta muestras fuera de la imagen o del papel (oscuras respecto al resto)
    keep = pts[:, 2] > 0.6 * np.percentile(pts[:, 2], 90)
    return pts[keep]


def otsu_1d(v: np.ndarray) -> float:
    v = np.sort(v.ravel())
    best, thr = -1, v.mean()
    for i in range(1, len(v)):
        w0, w1 = i / len(v), 1 - i / len(v)
        m0, m1 = v[:i].mean(), v[i:].mean()
        b = w0 * w1 * (m0 - m1) ** 2
        if b > best:
            best, thr = b, (v[i - 1] + v[i]) / 2
    return float(thr)


# ---------------------------------------------------------- lectura de pistas
def extract_clue_glyphs(cell_gray: np.ndarray) -> Dict[str, List[Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
    """Separa los dígitos de una celda negra en {'across': [...], 'down': [...]}.

    Cada glifo es una imagen binaria (blanco sobre negro) del tamaño de la celda y su bbox.
    """
    cp = cell_gray.shape[0]
    g = cell_gray.astype(np.float32)
    bg = np.median(g)
    hi, lo = np.percentile(g, 98), np.percentile(g, 2)
    # polaridad: si el fondo está más cerca del extremo oscuro, la tinta es clara
    contrast = (g - bg) if (bg - lo) < (hi - bg) else (bg - g)
    contrast = np.clip(contrast, 0, None)
    if contrast.max() < 25:
        return {"across": [], "down": []}
    c8 = (255 * contrast / contrast.max()).astype(np.uint8)
    _, b = cv2.threshold(c8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yy, xx = np.mgrid[0:cp, 0:cp]
    dd = (yy - xx)
    max_off = max(4, int(round(0.05 * cp)))
    near = (b > 0) & (np.abs(dd) <= max_off)
    off = 0
    if near.sum() > 0.25 * cp:
        hist = np.bincount((dd[near] + max_off).ravel(), minlength=2 * max_off + 1)
        hist = np.convolve(hist, np.ones(3), "same")
        peak = int(np.argmax(hist)) - max_off
        if abs(peak) <= max_off:
            off = peak
    b[np.abs(dd - off) < 0.08 * cp] = 0
    n, lab, st, cen = cv2.connectedComponentsWithStats(b)
    out = {"across": [], "down": []}
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if w > 0.65 * cp or h > 0.65 * cp:   # restos de bordes de la celda
            continue
        if h < 0.11 * cp or h > 0.55 * cp or area < 0.0035 * cp * cp or w > 0.6 * cp:
            continue
        if w <= 0.035 * cp or area / float(w * h) < 0.16:   # restos de líneas (diagonal/bordes)
            continue
        mask = (lab == i).astype(np.uint8) * 255
        cx, cy = cen[i]
        side = "across" if cx > cy else "down"
        # dígitos pegados (ej. "11" en negrita): separar si el ancho es excesivo
        if w > 1.15 * h:
            half = x + w // 2
            m1, m2 = mask.copy(), mask.copy()
            m1[:, half:] = 0
            m2[:, :half] = 0
            out[side] += [(m1, (x, y, w // 2, h)), (m2, (half, y, w - w // 2, h))]
        else:
            out[side].append((mask, (x, y, w, h)))
    for side in out:
        out[side].sort(key=lambda t: t[1][0])
        # fusiona fragmentos superpuestos verticalmente (p.ej. partes de un mismo dígito)
        merged = []
        for m, bb in out[side]:
            if merged and bb[0] < merged[-1][1][0] + merged[-1][1][2] * 0.5:
                pm, pb = merged[-1]
                x0, y0 = min(pb[0], bb[0]), min(pb[1], bb[1])
                x1 = max(pb[0] + pb[2], bb[0] + bb[2]); y1 = max(pb[1] + pb[3], bb[1] + bb[3])
                merged[-1] = (cv2.bitwise_or(pm, m), (x0, y0, x1 - x0, y1 - y0))
            else:
                merged.append((m, bb))
        out[side] = merged
    return out


def number_candidates(probs: np.ndarray, length: int, top_k: int = 4) -> List[Tuple[int, float]]:
    """Lecturas posibles de una pista ordenadas por log-probabilidad, restringidas al
    rango alcanzable [suma mínima, suma máxima] de una suma de `length` celdas."""
    lo, hi = sum_bounds(length)
    logp = np.log(probs + 1e-9)
    top = [np.argsort(-p)[:top_k] for p in probs]
    out = {}
    for combo in product(*top):
        if combo[0] == 0:
            continue
        val = int("".join(map(str, combo)))
        if lo <= val <= hi:
            out[val] = max(out.get(val, -np.inf), float(sum(logp[i, d] for i, d in enumerate(combo))))
    return sorted(out.items(), key=lambda t: -t[1])


def decode_number(probs: np.ndarray, length: int) -> Tuple[int, float, bool]:
    """Elige el número más probable compatible con la longitud de su suma.
    Devuelve (valor, log-probabilidad, ¿difiere de la lectura cruda?)."""
    if probs.shape[0] == 0:
        return 0, -np.inf, False
    raw = int("".join(str(int(np.argmax(p))) for p in probs))
    cands = number_candidates(probs, length)
    if not cands:
        return raw, float(np.log(probs.max(1) + 1e-9).sum()), False
    return cands[0][0], cands[0][1], cands[0][0] != raw


def global_sum_correction(entries: List[Dict], across, down) -> int:
    """Restricción global: Σ pistas horizontales = Σ pistas verticales (= Σ de todas las
    celdas). Si no se cumple, aplica el cambio de UNA pista (entre sus lecturas
    alternativas) que la restablece con la menor pérdida de log-probabilidad."""
    ta = sum(map(sum, across)); td = sum(map(sum, down))
    if ta == td:
        return 0
    best = None
    for e in entries:
        need = (td - ta) if e["side"] == "across" else (ta - td)
        base = e["cands"][0][1] if e["cands"] else 0.0
        for val, sc in e["cands"]:
            if val - e["value"] == need and (best is None or base - sc < best[0]):
                best = (base - sc, e, val)
    if best is None:
        return 0
    _, e, val = best
    r, c = e["cell"]
    (across if e["side"] == "across" else down)[r][c] = val
    e["value"], e["corrected"] = val, True
    return 1


# ------------------------------------------------------------------ pipeline
def read_puzzle(img_bgr: np.ndarray, classifier=None, ocr: str = "mlp",
                use_constraint_correction: bool = True) -> VisionResult:
    """Imagen BGR -> VisionResult con el Puzzle extraído.

    ocr: "mlp" (clasificador propio) o "tesseract".
    """
    from .ocr import tesseract_number

    t = {}
    t0 = time.perf_counter()
    img, gray, scale = preprocess(img_bgr)
    t["preproc"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    corners, binary = find_board(gray)
    t["board"] = time.perf_counter() - t0

    # 1ª rectificación (resolución fija) para estimar la grilla
    t0 = time.perf_counter()
    wtop = np.linalg.norm(corners[1] - corners[0]); wbot = np.linalg.norm(corners[2] - corners[3])
    hl = np.linalg.norm(corners[3] - corners[0]); hr = np.linalg.norm(corners[2] - corners[1])
    aspect = (wtop + wbot) / (hl + hr)
    W0 = 1000
    H0 = int(round(W0 / aspect))
    w0, _ = warp(gray, corners, W0, H0)
    rows, cols, grid_dbg = estimate_grid(w0)
    t["grid"] = time.perf_counter() - t0

    # 2ª rectificación: exactamente CELL_PX píxeles por celda
    t0 = time.perf_counter()
    warped, H = warp(img, corners, cols * CELL_PX, rows * CELL_PX)
    wg = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    ring = paper_ring(gray, corners, rows, cols)
    white, ratio, illum = classify_cells(wg, rows, cols, ring)
    t["cells"] = time.perf_counter() - t0

    # lectura de pistas
    t0 = time.perf_counter()
    down = [[0] * cols for _ in range(rows)]
    across = [[0] * cols for _ in range(rows)]
    pad = int(0.04 * CELL_PX)
    wg_padded = cv2.copyMakeBorder(wg, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    glyph_log = []
    warnings = []
    corrections = 0
    for r in range(rows):
        for c in range(cols):
            if white[r][c]:
                continue
            need = {"across": c + 1 < cols and white[r][c + 1],
                    "down": r + 1 < rows and white[r + 1][c]}
            if not (need["across"] or need["down"]):
                continue
            # longitud de la suma que nace en esta celda
            la = 0
            while c + 1 + la < cols and white[r][c + 1 + la]:
                la += 1
            ld = 0
            while r + 1 + ld < rows and white[r + 1 + ld][c]:
                ld += 1
            crop = wg_padded[r * CELL_PX : (r + 1) * CELL_PX + 2 * pad,
                             c * CELL_PX : (c + 1) * CELL_PX + 2 * pad]
            glyphs = extract_clue_glyphs(crop)
            for side, length in (("across", la), ("down", ld)):
                if not need[side]:
                    continue
                gs = glyphs[side]
                if not gs:
                    warnings.append(f"no se leyó la pista {side} en ({r},{c})")
                    continue
                if ocr == "tesseract":
                    union = np.zeros_like(gs[0][0])
                    for m, _ in gs:
                        union = cv2.bitwise_or(union, m)
                    val = tesseract_number(union) or 0
                    corrected = False
                    cands = []
                else:
                    probs = classifier.predict_proba([m for m, _ in gs])
                    cands = number_candidates(probs, length)
                    if use_constraint_correction:
                        val, _, corrected = decode_number(probs, length)
                    else:
                        val = int("".join(str(int(np.argmax(p))) for p in probs))
                        corrected = False
                corrections += corrected
                (across if side == "across" else down)[r][c] = val
                glyph_log.append({"cell": (r, c), "side": side, "value": val, "n_glyphs": len(gs),
                                  "corrected": corrected, "cands": cands,
                                  "glyphs": [m for m, _ in gs]})
    if use_constraint_correction and ocr != "tesseract":
        corrections += global_sum_correction(glyph_log, across, down)
    t["ocr"] = time.perf_counter() - t0
    t["total"] = sum(t.values())

    # esquinas y homografía en coordenadas de la imagen original
    corners_orig = corners / scale
    S = np.diag([scale, scale, 1.0])
    H_orig = H @ S
    p = Puzzle(rows, cols, white.tolist(), down, across)
    warnings += p.validate()
    return VisionResult(p, corners_orig, H_orig, warped, t,
                        {"grid": grid_dbg, "ratio": ratio, "binary": binary, "glyphs": glyph_log,
                         "corrections": corrections}, warnings)
