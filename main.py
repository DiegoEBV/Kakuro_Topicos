"""
main.py — Punto de entrada unificado para Kakuro Solver (CC58 - Tópicos en CC).

Uso:
    python main.py --ui                                       # Inicia la interfaz web en Streamlit
    python main.py data/images/05_photo_angle.jpg            # Resuelve una imagen por consola
    python main.py data/images/05_photo_angle.jpg --out sol.jpg --backend ortools
    python main.py --test                                    # Ejecuta la suite de pruebas unitarias
"""
import argparse
import os
import subprocess
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_streamlit():
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    print(f"🚀 Iniciando servidor Streamlit desde {app_path} ...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)


def run_tests():
    print("🧪 Ejecutando suite de pruebas automatizadas...")
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Kakuro Solver — Visión Computacional + Constraint Programming (CC58)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py --ui
  python main.py data/images/05_photo_angle.jpg --out solucion.jpg
  python main.py --test
        """,
    )
    parser.add_argument("image", nargs="?", default=None, help="Ruta a la imagen del Kakuro a resolver")
    parser.add_argument("--ui", action="store_true", help="Lanza la interfaz web interactiva en Streamlit")
    parser.add_argument("--test", action="store_true", help="Ejecuta las pruebas unitarias con pytest")
    parser.add_argument("--out", default="solucion.jpg", help="Ruta de guardado para la imagen con solución superpuesta")
    parser.add_argument("--json", default=None, help="Ruta de guardado para la definición del puzzle en JSON")
    parser.add_argument("--backend", default="auto", choices=["auto", "ortools", "native"], help="Motor CP a utilizar")
    parser.add_argument("--model", default="global_dom", choices=["global", "global_dom", "reified"], help="Modelo CP")

    args = parser.parse_args()

    if args.ui:
        run_streamlit()
        return

    if args.test:
        run_tests()
        return

    if args.image is None:
        print("💡 No se especificó ninguna imagen ni comando. Opciones rápidas:")
        print("   1. Para abrir la interfaz web interactiva: python main.py --ui")
        print("   2. Para resolver una imagen del dataset:   python main.py data/images/05_photo_angle.jpg")
        print("   3. Para ejecutar las pruebas unitarias:   python main.py --test")
        print("\nMostrando ayuda general:\n")
        parser.print_help()
        return

    # Delegate to pipeline
    from kakuro.pipeline import solve_image
    import cv2
    import json

    print(f"📷 Cargando y analizando imagen: {args.image}")
    res = solve_image(args.image, backend=args.backend, model=args.model)
    p = res.vision.puzzle
    print(f"✅ Grilla detectada: {p.rows}x{p.cols}, celdas jugables: {len(p.white_cells())}")
    for w in res.vision.warnings:
        print(f"⚠️  Aviso: {w}")
    print(
        f"🧩 Solver CP: {res.solve.backend}/{res.solve.model} -> {res.solve.status}, "
        f"Única={res.solve.unique}, {res.solve.time_s * 1000:.1f} ms"
    )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(p.to_dict(), f, indent=2)
        print(f"💾 Puzzle guardado en JSON: {args.json}")

    if res.overlay is not None:
        cv2.imwrite(args.out, res.overlay)
        print(f"🖼️  Solución superpuesta guardada en: {args.out}")

    print(f"⏱️  Tiempo total de ejecución: {res.total_s:.2f} s")


if __name__ == "__main__":
    main()
