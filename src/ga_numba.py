"""
ga_numba.py — GA con evaluación de fitness paralela en CPU usando
Numba @njit(parallel=True) + prange (un individuo por iteración).

La generación de descendencia (torneo + OX + mutación + 2-opt opcional)
también se paraleliza con prange, ya que cada hijo es independiente.

Uso:
    python src/ga_numba.py instances/A-n60-k9.vrp --pop 2000 --gens 300 --threads 2
"""
from __future__ import annotations

import argparse

import numba
import numpy as np

from common import (eval_population_parallel, eval_population_serial,
                    make_offspring_parallel, make_offspring_serial, read_vrp, warmup)
from ga_engine import GAParams, run_ga


class NumbaEvaluator:
    """parallel=True usa el kernel con prange; parallel=False el mismo código
    compilado sin hilos (útil para separar 'compilado' de 'paralelo')."""

    def __init__(self, inst, penalty, parallel=True):
        self.inst = inst
        self.penalty = penalty
        self.kernel = eval_population_parallel if parallel else eval_population_serial
        self.fit = None

    def evaluate(self, pop):
        P = pop.shape[0]
        fit = np.empty(P); dist = np.empty(P); exc = np.empty(P, dtype=np.int64)
        i = self.inst
        self.kernel(pop, i.D, i.demand, i.capacity, i.n_customers, self.penalty, fit, dist, exc)
        self.fit = fit
        return fit, dist, exc

    def topk(self, k):
        # reducción parcial O(P) en lugar de ordenar todo
        idx = np.argpartition(self.fit, k)[:k]
        return idx[np.argsort(self.fit[idx])]


def solve(inst, params: GAParams, parallel=True, init_pop=None, verbose=False):
    warmup()
    ev = NumbaEvaluator(inst, params.penalty, parallel)
    off = make_offspring_parallel if parallel else make_offspring_serial
    return run_ga(inst, ev, off, params, init_pop=init_pop, verbose=verbose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--pop", type=int, default=2000)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=numba.config.NUMBA_NUM_THREADS)
    a = ap.parse_args()
    numba.set_num_threads(a.threads)
    inst = read_vrp(a.instance)
    r = solve(inst, GAParams(pop_size=a.pop, generations=a.gens, seed=a.seed), verbose=True)
    gap = 100 * (r.best_cost - inst.optimum) / inst.optimum
    print(f"{inst.name}: costo={r.best_cost:.0f} gap={gap:.2f}% hilos={a.threads} "
          f"tiempo={r.total_time:.2f}s (fitness {r.eval_time:.3f}s)")
    for k, rt in enumerate(r.routes, 1):
        print(f"  Ruta #{k}: {' '.join(map(str, rt))}")


if __name__ == "__main__":
    main()
