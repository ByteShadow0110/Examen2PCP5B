"""
ga_engine.py — Bucle generacional del GA, independiente del backend.

Cada backend (serial / Numba / CUDA) aporta un "evaluador" con:
    evaluate(pop) -> (fit, dist, exc)  arrays en host
    topk(k)       -> índices de los k mejores de la última evaluación
y una función para generar descendencia.

El motor mide por separado el tiempo de evaluación de fitness, que es la
métrica principal del proyecto (speedup serial vs Numba vs CUDA).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from common import Instance, decode, random_population, route_cost


@dataclass
class GAParams:
    pop_size: int = 2000
    generations: int = 300
    elite: int = 20
    tournament: int = 6
    pc: float = 0.9
    pm: float = 0.4
    penalty: float = 50.0
    ls_prob: float = 0.3       # prob. de aplicar 2-opt intra-ruta a un hijo
    seed: int = 0


@dataclass
class GAResult:
    best_chrom: np.ndarray
    best_cost: float           # distancia de la mejor solución factible
    feasible: bool
    routes: list
    total_time: float
    eval_time: float
    history: list = field(default_factory=list)
    final_pop: np.ndarray | None = None
    final_fit: np.ndarray | None = None


def run_ga(inst: Instance, evaluator, offspring_fn, params: GAParams,
           init_pop: np.ndarray | None = None, verbose: bool = False) -> GAResult:
    rng = np.random.default_rng(params.seed)
    np.random.seed(params.seed)
    P, E = params.pop_size, params.elite
    maxval = inst.chrom_len
    pop = init_pop.copy() if init_pop is not None else random_population(P, inst, rng)
    children = np.empty((P - E, pop.shape[1]), dtype=np.int32)

    best_cost, best_chrom = np.inf, None
    history = []
    eval_time = 0.0
    t0 = time.perf_counter()

    for gen in range(params.generations + 1):
        te = time.perf_counter()
        fit, dist, exc = evaluator.evaluate(pop)
        eval_time += time.perf_counter() - te

        # mejor solución factible de la generación
        feas = exc == 0
        if feas.any():
            idx = np.flatnonzero(feas)
            j = idx[np.argmin(dist[idx])]
            if dist[j] < best_cost:
                best_cost = float(dist[j])
                best_chrom = pop[j].copy()
        history.append(best_cost)
        if verbose and gen % 50 == 0:
            print(f"  gen {gen:4d}  best={best_cost:.0f}  minfit={fit.min():.0f}")
        if gen == params.generations:
            break

        elite_idx = evaluator.topk(E)
        offspring_fn(pop, fit, children, params.tournament, params.pc,
                     params.pm, maxval, inst.D, inst.n_customers, params.ls_prob)
        new_pop = np.empty_like(pop)
        new_pop[:E] = pop[elite_idx]
        new_pop[E:] = children
        pop = new_pop

    total = time.perf_counter() - t0
    feasible = best_chrom is not None
    if not feasible:   # devolvemos el de menor fitness aunque viole capacidad
        best_chrom = pop[int(np.argmin(fit))].copy()
        best_cost = float(dist[int(np.argmin(fit))])
    routes = decode(best_chrom, inst.n_customers)
    assert not feasible or abs(route_cost(routes, inst.D) - best_cost) < 1e-6
    return GAResult(best_chrom, best_cost, feasible, routes, total, eval_time,
                    history, pop, fit)
