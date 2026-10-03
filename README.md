# Kakuro Solver — Visión Computacional + Constraint Programming

Sistema **extremo a extremo** que recibe la foto (o captura) de un Kakuro, extrae su
estructura con visión computacional y una red neuronal propia, lo modela como un
problema de **Constraint Programming** y proyecta la solución sobre la imagen original.

CC58 – Tópicos en Ciencia de la Computación · Trabajo 1

```
imagen ─► preprocesamiento ─► tablero + homografía ─► tamaño de grilla ─► celdas blanca/negra
       ─► OCR de pistas (MLP) + corrección por restricciones ─► Puzzle (JSON)
       ─► modelo CP (AllDifferent, Sum, Table) ─► solución + prueba de unicidad
       ─► superposición sobre la foto (H⁻¹)
```

## Resultados (dataset de 14 imágenes)

| Métrica | MLP + corrección | MLP sin corrección | Tesseract 5 |
|---|---|---|---|
| Grilla detectada | 100 % | 100 % | 100 % |
| Celdas blanca/negra | 100 % | 100 % | 100 % |
| Pistas leídas | **100 %** (314/314) | 99.7 % | 92.4 % |
| Puzzles resueltos correctamente | **14/14** | 13/14 | 4/14 |

Tiempo medio por imagen ≈ 0.15 s (visión, CPU) + < 2 ms (solver). Detalles en el notebook y el informe.

## Estructura

```
kakuro-cv-cp/
├── Kakuro_CV_CP.ipynb        # notebook principal (ejecutado, con todas las salidas)
├── kakuro/                   # paquete Python
│   ├── puzzle.py             # estructura de datos (JSON) y utilidades combinatorias
│   ├── vision.py             # Fase 1: preprocesamiento, tablero, grilla, celdas, pistas
│   ├── ocr.py                # clasificador de dígitos (MLP sobre HOG) + Tesseract
│   ├── solver.py             # Fase 2: modelos CP en OR-Tools CP-SAT (+ respaldo)
│   ├── cp_engine.py          # motor CP propio: propagación GAC + búsqueda MRV
│   ├── pipeline.py           # Fase 3: integración imagen → solución (CLI)
│   ├── visualize.py          # superposición sobre la foto y grilla limpia
│   ├── generator.py          # generador de Kakuros con solución única
│   ├── render.py             # render del tablero y simulación de fotos
│   ├── evaluate.py           # métricas contra ground truth
│   ├── robustness.py         # barrido de degradaciones
│   └── bench.py              # benchmark del solver
├── data/
│   ├── images/               # 14 imágenes de prueba
│   ├── ground_truth/         # JSON con estructura, pistas y solución de cada imagen
│   └── bench/                # instancias únicas 6×6 … 16×16 para medir tiempos
├── models/digit_mlp.joblib   # red neuronal entrenada
├── assets/fonts/             # tipografías libres (entrenamiento y render)
├── scripts/                  # build_dataset.py, build_bench.py
├── results/                  # CSV y tablas .tex generadas por el notebook
├── report/                   # informe IEEE en LaTeX (main.tex, figuras, main.pdf)
└── tests/test_basic.py
```

## Instalación

Requiere Python ≥ 3.9.

```bash
git clone https://github.com/USUARIO/kakuro-cv-cp.git
cd kakuro-cv-cp
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Opcional (sólo para la comparación con Tesseract):
#   Ubuntu: sudo apt install tesseract-ocr   ·   macOS: brew install tesseract
```

## Ejecución

**Resolver una imagen (consola):**
```bash
python -m kakuro.pipeline data/images/05_photo_angle.jpg --out solucion.jpg --json puzzle.json
```
Opciones: `--backend {auto,ortools,native}` y `--model {global,global_dom,reified}`.

**Desde Python:**
```python
from kakuro.pipeline import solve_image
res = solve_image("mi_foto.jpg")
print(res.vision.puzzle)          # estructura extraída
print(res.solve.solution)         # solución
import cv2; cv2.imwrite("out.jpg", res.overlay)
```

**Notebook:** abrir `Kakuro_CV_CP.ipynb` (Jupyter o Colab) y ejecutar todas las celdas.
Reproduce todas las métricas, figuras y tablas del informe (`results/`, `report/figures/`).

**Regenerar datos / modelo (opcional):**
```bash
python scripts/build_dataset.py --out data         # dataset de imágenes + ground truth
python scripts/build_bench.py                       # instancias para el benchmark
python -c "from kakuro.ocr import DigitClassifier as D; D.train('assets/fonts/train', 200).save('models/digit_mlp.joblib')"
pytest -q                                           # pruebas
```

**Informe:** `cd report && pdflatex main && pdflatex main` (o subir la carpeta a Overleaf).

## Fotos reales
Para añadir fotos propias al dataset: copiar la imagen a `data/images/NOMBRE.jpg` y, si se
quiere medir el error, su ground truth en `data/ground_truth/NOMBRE.json`
(mismo formato que los existentes; `solution` es opcional). El tablero debe verse completo,
con algo de margen alrededor, y la primera fila/columna son bloques (como en todo Kakuro).

## Referencias
- OR-Tools CP-SAT — https://developers.google.com/optimization
- Kakuro — https://en.wikipedia.org/wiki/Kakuro
- Tesseract OCR — https://github.com/tesseract-ocr/tesseract
- Dalal & Triggs (2005), *Histograms of Oriented Gradients for Human Detection*, CVPR.
