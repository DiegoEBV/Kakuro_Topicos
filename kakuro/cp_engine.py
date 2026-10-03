"""
cp_engine.py — Motor de Constraint Programming en Python puro (sin dependencias).

Se usa como respaldo cuando OR-Tools no está instalado y como punto de comparación
académico: implementa explícitamente propagación + búsqueda.

  * Variables: una por celda blanca, dominio representado como máscara de bits (1..9).
  * Restricción global por suma: SumAllDifferent(cells, total) = alldifferent ∧ sum.
      - nivel "gac":    consistencia de arco generalizada exacta sobre la conjunción,
                        mediante programación dinámica sobre subconjuntos de dígitos
                        (el conjunto de dígitos usados determina la suma).
      - nivel "bounds": alldifferent por eliminación de valores asignados +
                        razonamiento de cotas sobre la suma (más débil).
  * Búsqueda: DFS con heurística MRV (dominio mínimo) y propagación hasta punto fijo
    (cola AC-3 sobre restricciones).
"""
from __future__ import annotations

import random
import time
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

FULL = 0b1111111110  # bits 1..9
POP = [bin(i).count("1") for i in range(1 << 10)]
MASK_SUM = [sum(d for d in range(1, 10) if (m >> d) & 1) for m in range(1 << 10)]
VALUES = [[d for d in range(1, 10) if (m >> d) & 1] for m in range(1 << 10)]


# ----------------------------------------------------------------- propagadores
@lru_cache(maxsize=200_000)
def gac_sum_alldiff(doms: Tuple[int, ...], total: int) -> Optional[Tuple[int, ...]]:
    """GAC exacto para alldifferent(x) ∧ sum(x) = total.

    forward[i]  = máscaras de dígitos usados alcanzables asignando x_0..x_{i-1}
    backward[i] = máscaras desde las que se puede completar x_i..x_{L-1} y llegar
                  a una máscara final de suma `total`.
    Un valor d es soportado para x_i si existe m en forward[i] ∩ (backward[i+1] - d).
    Devuelve los dominios podados o None si la restricción es insatisfacible.
    """
    L = len(doms)
    fwd: List[set] = [set() for _ in range(L + 1)]
    fwd[0].add(0)
    for i, dm in enumerate(doms):
        nxt = fwd[i + 1]
        for m in fwd[i]:
            free = dm & ~m
            while free:
                b = free & -free
                nxt.add(m | b)
                free ^= b
        if not nxt:
            return None
    goal = {m for m in fwd[L] if MASK_SUM[m] == total}
    if not goal:
        return None
    bwd: List[set] = [set() for _ in range(L + 1)]
    bwd[L] = goal
    new = [0] * L
    for i in range(L - 1, -1, -1):
        dm = doms[i]
        ok_next = bwd[i + 1]
        cur = bwd[i]
        sup = 0
        for m in fwd[i]:
            free = dm & ~m
            while free:
                b = free & -free
                if (m | b) in ok_next:
                    cur.add(m)
                    sup |= b
                free ^= b
        if not sup:
            return None
        new[i] = sup
    return tuple(new)


def bounds_sum_alldiff(doms: Tuple[int, ...], total: int) -> Optional[Tuple[int, ...]]:
    """Propagación débil: alldifferent por valores fijos + cotas de la suma."""
    doms = list(doms)
    changed = True
    while changed:
        changed = False
        fixed = 0
        for dm in doms:
            if POP[dm] == 1:
                if fixed & dm:
                    return None
                fixed |= dm
        for i, dm in enumerate(doms):
            if POP[dm] > 1 and dm & fixed:
                nd = dm & ~fixed
                if not nd:
                    return None
                doms[i] = nd; changed = True
        mins = [VALUES[d][0] for d in doms]
        maxs = [VALUES[d][-1] for d in doms]
        smin, smax = sum(mins), sum(maxs)
        if smin > total or smax < total:
            return None
        for i, dm in enumerate(doms):
            lo = total - (smax - maxs[i])
            hi = total - (smin - mins[i])
            nd = dm & sum(1 << v for v in range(max(1, lo), min(9, hi) + 1))
            if nd != dm:
                if not nd:
                    return None
                doms[i] = nd; changed = True
                mins[i], maxs[i] = VALUES[nd][0], VALUES[nd][-1]
                smin, smax = sum(mins), sum(maxs)
    return tuple(doms)


# ------------------------------------------------------------------- el solver
class CPSolver:
    def __init__(self, n_vars: int, constraints: Sequence[Tuple[Sequence[int], int]],
                 level: str = "gac", rng: Optional[random.Random] = None):
        self.n = n_vars
        self.cons = [(tuple(v), t) for v, t in constraints]
        self.watch: List[List[int]] = [[] for _ in range(n_vars)]
        for k, (vs, _) in enumerate(self.cons):
            for v in vs:
                self.watch[v].append(k)
        self.prop = gac_sum_alldiff if level == "gac" else bounds_sum_alldiff
        self.rng = rng
        self.stats = {"nodes": 0, "failures": 0, "propagations": 0}

    def propagate(self, doms: List[int], queue: List[int]) -> bool:
        inq = set(queue)
        while queue:
            k = queue.pop()
            inq.discard(k)
            vs, total = self.cons[k]
            self.stats["propagations"] += 1
            if total == 0:  # restricción solo alldifferent (usada por el generador)
                res = self._alldiff_only(tuple(doms[v] for v in vs))
            else:
                res = self.prop(tuple(doms[v] for v in vs), total)
            if res is None:
                return False
            for v, nd in zip(vs, res):
                if nd != doms[v]:
                    doms[v] = nd
                    for k2 in self.watch[v]:
                        if k2 != k and k2 not in inq:
                            queue.append(k2); inq.add(k2)
        return True

    @staticmethod
    def _alldiff_only(doms):
        doms = list(doms)
        changed = True
        while changed:
            changed = False
            for i, dm in enumerate(doms):
                if POP[dm] == 1:
                    for j in range(len(doms)):
                        if j != i and doms[j] & dm:
                            doms[j] &= ~dm
                            if not doms[j]:
                                return None
                            changed = True
        return tuple(doms)

    def solve(self, init_domains: Optional[List[int]] = None, limit: int = 1,
              time_limit: float = 60.0) -> List[List[int]]:
        """Devuelve hasta `limit` soluciones (listas de valores por variable)."""
        doms = list(init_domains) if init_domains else [FULL] * self.n
        sols: List[List[int]] = []
        t_end = time.perf_counter() + time_limit
        if not self.propagate(doms, list(range(len(self.cons)))):
            return sols

        def dfs(doms: List[int]) -> bool:
            if time.perf_counter() > t_end:
                raise TimeoutError
            self.stats["nodes"] += 1
            # MRV: variable no asignada con el menor dominio
            best, bsize = -1, 99
            for v, dm in enumerate(doms):
                p = POP[dm]
                if 1 < p < bsize:
                    best, bsize = v, p
                    if p == 2:
                        break
            if best < 0:
                sols.append([VALUES[d][0] for d in doms])
                return len(sols) >= limit
            vals = list(VALUES[doms[best]])
            if self.rng:
                self.rng.shuffle(vals)
            for val in vals:
                nd = doms.copy()
                nd[best] = 1 << val
                if self.propagate(nd, list(self.watch[best])):
                    if dfs(nd):
                        return True
                else:
                    self.stats["failures"] += 1
            return False

        try:
            dfs(doms)
        except TimeoutError:
            self.stats["timeout"] = True
        return sols


def domain_mask(values) -> int:
    m = 0
    for v in values:
        m |= 1 << v
    return m
