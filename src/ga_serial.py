"""
ga_serial.py — GA con evaluación de fitness SERIAL (bucle en Python puro).

Es la línea base para medir el speedup. Los operadores genéticos usan la
versión Numba *serial* (sin paralelismo) para que la diferencia de tiempos
entre backends provenga de la evaluación de fitness.

Uso:
    python src/ga_serial.py instances/A-n32-k5.vrp --pop 2000 --gens 300
"""
from __future__ import annotations

import argparse

import numpy as np

from common import fitness_py, make_offspring_serial, read_vrp, warmup
from ga_engine import GAParams, run_ga


class SerialEvaluator:
    def __init__(self, inst, penalty):
        self.D = inst.D.tolist()           # listas: acceso más rápido en Python
        self.dem = inst.demand.tolist()
        self.cap = inst.capacity
        self.nc = inst.n_customers
        self.penalty = penalty
        self.fit = None

    def evaluate(self, pop):
        P = pop.shape[0]
        fit = np.empty(P); dist = np.empty(P); exc = np.empty(P, dtype=np.int64)
        for i, chrom in enumerate(pop.tolist()):          # bucle Python
            f, d, e = fitness_py(chrom, self.D, self.dem, self.cap, self.nc, self.penalty)
            fit[i] = f; dist[i] = d; exc[i] = e
        self.fit = fit
        return fit, dist, exc

    def topk(self, k):
        # selección de los k mejores: ordenamiento serial
        return np.argsort(self.fit, kind="stable")[:k]


def solve(inst, params: GAParams, verbose=False):
    warmup()
    ev = SerialEvaluator(inst, params.penalty)
    return run_ga(inst, ev, make_offspring_serial, params, verbose=verbose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--pop", type=int, default=2000)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    inst = read_vrp(a.instance)
    r = solve(inst, GAParams(pop_size=a.pop, generations=a.gens, seed=a.seed), verbose=True)
    gap = 100 * (r.best_cost - inst.optimum) / inst.optimum
    print(f"{inst.name}: costo={r.best_cost:.0f} gap={gap:.2f}% "
          f"tiempo={r.total_time:.2f}s (fitness {r.eval_time:.2f}s)")
    for k, rt in enumerate(r.routes, 1):
        print(f"  Ruta #{k}: {' '.join(map(str, rt))}")


if __name__ == "__main__":
    main()
