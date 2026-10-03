"""
ocr.py — Reconocimiento de los dígitos de las pistas.

Modelo propio: red neuronal MLP (scikit-learn) sobre descriptores HOG + píxeles,
entrenada con dígitos sintéticos renderizados en 12 tipografías (distintas a las
del dataset de prueba) con aumentos de datos (rotación, grosor, desenfoque, ruido).

Comparación: Tesseract OCR 5 (LSTM) en modo línea con lista blanca de dígitos.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import hog

SIZE = 28


# ------------------------------------------------------------ normalización
def normalize_glyph(binary: np.ndarray) -> np.ndarray:
    """Recorta el glifo (blanco sobre negro), lo centra en un cuadrado y lo lleva a 28x28."""
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return np.zeros((SIZE, SIZE), np.float32)
    g = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = g.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = g
    inner = SIZE - 6
    sq = cv2.resize(sq, (inner, inner), interpolation=cv2.INTER_AREA)
    out = np.zeros((SIZE, SIZE), np.float32)
    out[3:3 + inner, 3:3 + inner] = sq.astype(np.float32) / 255.0
    return out


def features(glyph28: np.ndarray, aspect: float) -> np.ndarray:
    """HOG (orientación de gradientes) + píxeles 14x14 + relación de aspecto del glifo."""
    h = hog(glyph28, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2),
            feature_vector=True)
    px = cv2.resize(glyph28, (14, 14), interpolation=cv2.INTER_AREA).ravel()
    return np.concatenate([h, px, [aspect]]).astype(np.float32)


def glyph_aspect(binary: np.ndarray) -> float:
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return 0.0
    return (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)


# ------------------------------------------------------- datos sintéticos
def synth_digits(font_dir: str, per_font_digit: int = 120, seed: int = 0):
    """Genera (X, y) con dígitos 0-9 renderizados y degradados aleatoriamente."""
    rng = np.random.default_rng(seed)
    fonts = sorted(glob.glob(os.path.join(font_dir, "*.[ot]tf")))
    X, y = [], []
    for fp in fonts:
        for d in range(10):
            for _ in range(per_font_digit):
                px = int(rng.integers(12, 40))
                font = ImageFont.truetype(fp, px)
                im = Image.new("L", (px * 2, px * 2), 0)
                ImageDraw.Draw(im).text((px, px), str(d), fill=255, font=font, anchor="mm")
                a = np.array(im)
                M = cv2.getRotationMatrix2D((px, px), rng.uniform(-7, 7), rng.uniform(0.9, 1.1))
                M[:, 2] += rng.uniform(-1, 1, 2)
                a = cv2.warpAffine(a, M, a.shape[::-1])
                k = int(rng.integers(0, 3))
                if k == 1:
                    a = cv2.erode(a, np.ones((2, 2), np.uint8))
                elif k == 2:
                    a = cv2.dilate(a, np.ones((2, 2), np.uint8))
                if rng.random() < 0.5:   # baja resolución: reducir y volver a ampliar
                    f = rng.uniform(0.3, 0.8)
                    small = cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                    a = cv2.resize(small, a.shape[::-1], interpolation=cv2.INTER_LINEAR)
                if rng.random() < 0.6:
                    a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.3, 2.0) * px / 25)
                a = np.clip(a + rng.normal(0, rng.uniform(0, 25), a.shape), 0, 255).astype(np.uint8)
                _, b = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                n, lab, st, _ = cv2.connectedComponentsWithStats(b)
                if n > 1:  # quedarse con los componentes significativos
                    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.05 * st[1:, cv2.CC_STAT_AREA].max()]
                    b = np.isin(lab, keep).astype(np.uint8) * 255
                X.append(features(normalize_glyph(b), glyph_aspect(b)))
                y.append(d)
    return np.array(X), np.array(y)


# ------------------------------------------------------------- clasificador
class DigitClassifier:
    def __init__(self, model=None):
        self.model = model

    @staticmethod
    def train(font_dir: str, per_font_digit: int = 120, seed: int = 0, verbose: bool = True):
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = synth_digits(font_dir, per_font_digit, seed)
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        model = make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3,
                                            max_iter=200, early_stopping=True, random_state=seed))
        model.fit(Xtr, ytr)
        acc = model.score(Xva, yva)
        if verbose:
            print(f"muestras={len(X)}  exactitud de validación={acc:.4f}")
        clf = DigitClassifier(model)
        clf.val_accuracy = acc
        clf.val_data = (Xva, yva)
        return clf

    def predict_proba(self, binaries: List[np.ndarray]) -> np.ndarray:
        F = np.stack([features(normalize_glyph(b), glyph_aspect(b)) for b in binaries])
        return self.model.predict_proba(F)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str) -> "DigitClassifier":
        import joblib
        return DigitClassifier(joblib.load(path))


# ----------------------------------------------------------------- Tesseract
def tesseract_number(binary_white_on_black: np.ndarray) -> Optional[int]:
    """Lee un número con Tesseract (tinta negra sobre blanco, con margen y escalado)."""
    try:
        import pytesseract
    except ImportError:
        return None
    img = 255 - binary_white_on_black
    ys, xs = np.nonzero(binary_white_on_black)
    if len(xs) == 0:
        return None
    img = img[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3]
    s = 48.0 / max(1, img.shape[0])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    txt = pytesseract.image_to_string(
        img, config="--psm 7 -c tessedit_char_whitelist=0123456789").strip()
    digits = "".join(ch for ch in txt if ch.isdigit())
    return int(digits) if digits else None
