"""
build_bench.py — Genera (una sola vez) las instancias únicas usadas en el análisis de
tiempos del solver y las guarda en data/bench/. La generación de puzzles grandes con
solución única es costosa, por eso se cachean.

Uso:  python scripts/build_bench.py --sizes 6 8 10 12 14 16 --per-size 3
"""
import argparse
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kakuro.generator import make_puzzle  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "bench")


def job(args):
    n, k = args
    path = os.path.join(OUT, f"bench_{n:02d}_{k}.json")
    if os.path.exists(path):
        return path
    p = make_puzzle(n, n, seed=5000 + 100 * n + k)
    p.save(path)
    print(path, p.meta, p.stats(), flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 8, 10, 12, 14, 16])
    ap.add_argument("--per-size", type=int, default=3)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(n, k) for n in a.sizes for k in range(a.per_size)]
    jobs.sort()  # los pequeños primero
    with Pool(a.procs) as pool:
        pool.map(job, jobs, chunksize=1)
