# Kakuro Solver — Visión Computacional + Constraint Programming

Sistema **extremo a extremo (End-to-End)** que recibe la foto o captura digital de un acertijo **Kakuro**, extrae su topología y pistas con técnicas de **Visión Computacional** y una **red neuronal propia (MLP sobre HOG)**, lo modela como un problema formal de **Constraint Programming (CP)** en **Google OR-Tools CP-SAT** (y motor CP nativo alternativo), y proyecta la solución directamente sobre la imagen original mediante realidad aumentada (homografía inversa).

> **CC58 – Tópicos en Ciencia de la Computación · Trabajo 1**

```
imagen ─► preprocesamiento ─► tablero + homografía (H) ─► tamaño de grilla (Sobel) ─► celdas blanca/negra
       ─► OCR de pistas (MLP HOG) + corrección guiada por restricciones ─► Puzzle (JSON)
       ─► modelo CP (AllDifferent, Sum, Table, Reified) ─► solución + prueba de unicidad
       ─► superposición sobre la foto (H⁻¹) y grilla limpia
```

---

## 🏆 Resultados y Métricas (Dataset de 14 imágenes)

| Métrica | MLP Propio + Corrección | MLP sin corrección | Tesseract OCR 5 |
|---|---|---|---|
| **Detección de Grilla** | **100 %** (14/14) | 100 % | 100 % |
| **Clasificación Celdas** | **100 %** | 100 % | 100 % |
| **Pistas Leídas** | **100 %** (314/314) | 99.7 % | 92.4 % |
| **Puzzles Resueltos Correctamente** | **14 / 14 (100 %)** | 13 / 14 | 4 / 14 |

* **Tiempo medio por imagen**: ≈ 0.15 s (visión por CPU) + < 25 ms (solver OR-Tools CP-SAT).
* **Demostración de unicidad**: < 3 ms para tableros de hasta $16\times16$.

---

## 📋 Cumplimiento con la Rúbrica de Evaluación

| Criterio de la Rúbrica | Puntos | Implementación en este Proyecto |
|---|:---:|---|
| **Visión Computacional e IA** | 3 pts | Pipeline OpenCV con umbral adaptativo + Otsu, detección de 4 esquinas con validación de borde oscuro, homografía $H$, estimación de grilla por perfiles de gradiente de Sobel, modelo cuadrático de iluminación de papel, y clasificador MLP propio sobre descriptores HOG con eliminación de diagonales. |
| **Bajo diferentes condiciones de imagen** | 1 pt | Dataset de 14 imágenes que cubre: capturas digitales limpias, compresión JPEG agresiva, fotos con perspectiva pronunciada, rotación de 14°, baja iluminación con ruido de sensor, sombras diagonales, luz cálida, desenfoque y tableros asimétricos. |
| **Modelado CP: Formulación matemática** | 3 pts | Variables $X_{r,c} \in \{1,\dots,9\}$ para cada celda jugable. Formulación matemática formal detallada en el informe IEEE y en el notebook. Totalmente parametrizado para cualquier tamaño e instancia. |
| **Uso eficiente de restricciones globales** | 3 pts | Implementación con `AllDifferent` y `Sum` en OR-Tools CP-SAT, y extensión `global_dom` con filtrado de dominios y restricción global `Table` (`AddAllowedAssignments`). Motor alternativo con consistencia GAC por emparejamientos bipartitos. |
| **Uso eficiente de restricciones reificadas** | 1 pt | 1) Verificación formal de solución única mediante restricciones semi-reificadas ($\text{diff}_c \implies X_c \ne s_c$ y $\bigvee \text{diff}_c$). 2) Modelo alternativo one-hot booleano con reificación bidireccional ($b_{c,d} \iff X_c = d$) con `OnlyEnforceIf`. |
| **Puente IA $\to$ CP sin intervención manual** | 1 pt | Pipeline 100% automático: una sola función (`solve_image`) o comando CLI/Web recibe el archivo de imagen y entrega la solución sin ajustes humanos. |
| **Visualización clara** | 1 pt | Dos modalidades: 1) Proyección de realidad aumentada sobre la perspectiva original usando $H^{-1}$. 2) Render vectorial limpio del tablero resuelto en alta resolución. Interfaz Web interactiva en Streamlit. |
| **Código limpio y buenas prácticas** | 2 pts | Paquete modularizado `kakuro/`, tipado estático (`typing`), docstrings, scripts CLI limpios, y suite de pruebas unitarias con `pytest` pasando al 100%. |
| **Informe técnico en LaTeX (formato IEEE)** | 5 pts | Artículo completo de 7 páginas en formato IEEE (`report/main.tex` y `report/main.pdf`), con algoritmos, tablas empíricas de complejidad y figuras vectoriales de cada etapa. |
| **TOTAL** | **20 / 20** | |

---

## 🗂️ Estructura del Repositorio

```
TP1_TCC/
├── app.py                    # Interfaz Web interactiva en Streamlit (Dashboard completo)
├── main.py                   # Punto de entrada unificado por consola (CLI, UI, Tests)
├── Kakuro_CV_CP.ipynb        # Notebook interactivo principal (Jupyter / Google Colab)
├── VIDEO_GUION.md            # Guion detallado para el video demostrativo de 5 minutos
├── requirements.txt          # Dependencias del proyecto
├── kakuro/                   # Paquete Python principal
│   ├── puzzle.py             # Estructuras de datos (Kakuro, Runs, Clues) y combinatoria
│   ├── vision.py             # Fase 1: Preprocesamiento, homografía, celdas y pistas
│   ├── ocr.py                # Red neuronal propia (MLP sobre HOG) y wrapper Tesseract
│   ├── solver.py             # Fase 2: Modelos CP (OR-Tools CP-SAT y respaldo nativo)
│   ├── cp_engine.py          # Motor CP nativo: propagación GAC + heurística MRV
│   ├── pipeline.py           # Fase 3: Integración imagen → solución
│   ├── visualize.py          # Proyección sobre foto original y grilla limpia
│   ├── generator.py          # Generador de instancias con solución única
│   ├── render.py             # Renderizado y simulación física de fotografías
│   ├── evaluate.py           # Cálculo de métricas contra Ground Truth
│   ├── bench.py              # Medición de tiempos y complejidad del solver
│   └── robustness.py         # Barrido de degradaciones y pruebas de robustez
├── data/
│   ├── images/               # 14 imágenes de prueba bajo diversas condiciones
│   ├── ground_truth/         # JSON con la estructura y solución exacta de cada imagen
│   └── bench/                # Instancias de 6x6 a 16x16 para benchmarking
├── models/
│   └── digit_mlp.joblib      # Pesos entrenados de la red neuronal propia
├── assets/fonts/             # Tipografías libres utilizadas para el entrenamiento
├── report/                   # Informe técnico en LaTeX (formato IEEE)
│   ├── main.tex              # Código fuente LaTeX
│   ├── main.pdf              # Documento compilado listo para entrega
│   ├── figures/              # Figuras de alta resolución
│   └── tables/               # Tablas empíricas generadas
├── scripts/                  # Generadores de dataset y benchmark
└── tests/
    └── test_basic.py         # Suite de pruebas automatizadas con pytest
```

---

## 🚀 Instalación

Requiere **Python ≥ 3.9** (probado y verificado en Python 3.12).

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/USUARIO/TP1_TCC.git
   cd TP1_TCC
   ```

2. Crear y activar un entorno virtual:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Modos de Ejecución

### 1. Interfaz Web Interactiva (Streamlit)
La forma más visual y recomendada para probar el sistema o realizar la demostración:
```bash
python main.py --ui
# o directamente:
streamlit run app.py
```
Permite:
* Seleccionar cualquiera de las 14 imágenes del dataset o cargar una foto propia.
* Elegir el backend (`ortools` / `native`) y el modelo CP (`global`, `global_dom`, `reified`).
* Ver la solución superpuesta con realidad aumentada y la grilla limpia.
* Explorar visualmente cada fase intermedia del pipeline de Visión Computacional.
* Descargar el resultado y el archivo JSON estructurado.

### 2. Ejecución por Consola (CLI)
Resolver cualquier imagen en un solo comando:
```bash
python main.py data/images/05_photo_angle.jpg --out solucion.jpg
```
Opciones adicionales:
```bash
python main.py data/images/05_photo_angle.jpg --backend ortools --model global_dom --json puzzle.json
```

### 3. Suite de Pruebas Automatizadas
Para verificar la integridad matemática y del pipeline:
```bash
python main.py --test
# o bien:
pytest tests/ -v
```

### 4. Uso desde Código Python
```python
from kakuro.pipeline import solve_image
import cv2

# Resolver de extremo a extremo
res = solve_image("data/images/05_photo_angle.jpg", backend="ortools", model="global_dom")

print("Filas y Columnas:", res.vision.puzzle.rows, res.vision.puzzle.cols)
print("Estado del Solver:", res.solve.status, "| ¿Única?:", res.solve.unique)
print("Tiempo CP-SAT:", res.solve.time_s * 1000, "ms")

# Guardar imagen con solución proyectada
cv2.imwrite("resultado.jpg", res.overlay)
```

### 5. Jupyter Notebook
Abrir `Kakuro_CV_CP.ipynb` en JupyterLab, VS Code o Google Colab para reproducir todos los experimentos, gráficos de gradientes, barridos de robustez y tablas de benchmarks del informe.

---

## 📄 Informe Técnico y Compilación LaTeX

El informe técnico se encuentra en la carpeta `report/`:
* Para compilar a PDF con LaTeX local:
  ```bash
  cd report
  pdflatex main
  pdflatex main
  ```
* También puede subirse la carpeta `report/` directamente a **Overleaf** para compilar con un solo clic.

---

## 👥 Integrantes
* Integrante 1: Diego Ballon
* Integrante 2: Jeffrey Diaz

Curso: **CC58 - Tópicos en Ciencia de la Computación** · Universidad Peruana de Ciencias Aplicadas (UPC)
