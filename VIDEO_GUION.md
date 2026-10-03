# Guion sugerido para el video (≤ 5 min, 3 integrantes)

| Tiempo | Quién | Contenido | Qué mostrar |
|---|---|---|---|
| 0:00–0:30 | Integrante 1 | Problema: reglas del Kakuro y objetivo del sistema (imagen → solución) | Una foto del dataset y la misma con la solución superpuesta |
| 0:30–1:45 | Integrante 1 | Fase 1: umbral adaptativo, contornos + validación de borde oscuro, homografía, perfil de gradiente S(n), modelo de iluminación | Notebook §2.1–2.3 (figuras `preproc`, `grid`, `cells`) |
| 1:45–2:30 | Integrante 2 | OCR: segmentación de pistas, eliminación de la diagonal, MLP sobre HOG, corrección por rango e invariante global; comparación con Tesseract | §2.4–2.6, tabla resumen y gráfico por imagen |
| 2:30–3:45 | Integrante 3 | Fase 2: variables, dominios, AllDifferent + Sum + Table, por qué no hace falta reificación y dónde se usa (unicidad / modelo one-hot), propagación GAC | `kakuro/solver.py` (`_build_ortools`), §3.2 tabla de propagación |
| 3:45–4:20 | Integrante 3 | Complejidad y tiempos (NP-completo, nodos vs. tamaño) | §3.4 gráfico `bench` |
| 4:20–5:00 | Integrante 2 | Demo en vivo: `python -m kakuro.pipeline foto.jpg --out solucion.jpg` con una foto tomada con el celular; abrir el resultado | Terminal + imagen de salida |
