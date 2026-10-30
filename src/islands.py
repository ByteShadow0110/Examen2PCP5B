"""
islands.py — Modelo de islas del GA orquestado con Dask.

* Se crea un LocalCluster(n_workers=4) — un proceso por isla.
* La población total se divide en N islas (4–8). Cada isla evoluciona de
  forma independiente con el GA de Numba durante `epoch_gens` generaciones
  (tarea dask.delayed) y devuelve su población final y su mejor solución.
* Migración (topología en anillo): los `n_migrants` mejores de la isla i
  reemplazan a los peores de la isla i+1. Se repite `epochs` veces; tras
  la última época se hace la migración final y se elige el mejor global.

Con epochs=1 y epoch_gens=100 se obtiene exactamente el esquema del
enunciado (4 islas x 100 generaciones + migración final).

Uso:
    python src/islands.py instances/A-n60-k9.vrp --islands 4 --pop 8000 --epochs 5 --epoch-gens 100
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def _island_task(inst_path, pop, params_dict, island_id, epoch, numba_threads):
    """Se ejecuta dentro de un worker de Dask."""
    import numba
    numba.set_num_threads(numba_threads)
    from common import read_vrp
    from ga_engine import GAParams
    from ga_numba import solve

    inst = read_vrp(inst_path)
    p = GAParams(**params_dict)
    p.seed = 1000 * epoch + island_id + p.seed
    r = solve(inst, p, parallel=numba_threads > 1, init_pop=pop)
    order = np.argsort(r.final_fit)
    return {
        "island": island_id,
        "pop": r.final_pop[order],          # ordenada de mejor a peor
        "best_cost": r.best_cost if r.feasible else np.inf,
        "best_chrom": r.best_chrom,
        "eval_time": r.eval_time,
        "time": r.total_time,
    }


def migrate(results, n_migrants):
    """Anillo: los mejores de i sustituyen a los peores de (i+1) mod N."""
    N = len(results)
    pops = [r["pop"].copy() for r in results]
    for i in range(N):
        dst = (i + 1) % N
        pops[dst][-n_migrants:] = results[i]["pop"][:n_migrants]
    return pops


def run_islands(inst_path, n_islands=4, total_pop=8000, epochs=5, epoch_gens=100,
                n_migrants=10, n_workers=4, numba_threads=1, seed=0,
                client=None, verbose=True, **ga_kwargs):
    import dask
    from dask.distributed import Client, LocalCluster

    from common import random_population, read_vrp
    from ga_engine import GAParams

    inst = read_vrp(inst_path)
    own = client is None
    if own:
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1,
                               processes=True, dashboard_address=None)
        client = Client(cluster)
    sub = total_pop // n_islands
    params = GAParams(pop_size=sub, generations=epoch_gens, seed=seed, **ga_kwargs)
    pd = params.__dict__.copy()
    rng = np.random.default_rng(seed)
    pops = [random_population(sub, inst, rng) for _ in range(n_islands)]

    best_cost, best_chrom = np.inf, None
    history = []
    t0 = time.perf_counter()
    eval_time = 0.0
    for ep in range(epochs):
        tasks = [dask.delayed(_island_task)(inst_path, pops[i], pd, i, ep, numba_threads)
                 for i in range(n_islands)]
        results = dask.compute(*tasks)                  # islas en paralelo
        for r in results:
            eval_time += r["eval_time"]
            if r["best_cost"] < best_cost or best_chrom is None:
                best_cost, best_chrom = r["best_cost"], r["best_chrom"]
        history.append(best_cost)
        if verbose:
            costs = ", ".join(f"{r['best_cost']:.0f}" for r in results)
            print(f"  época {ep + 1}/{epochs}: islas=[{costs}]  mejor global={best_cost:.0f}")
        pops = migrate(results, n_migrants)             # migración
    total = time.perf_counter() - t0
    if own:
        client.close()
        cluster.close()
    from common import decode
    return {
        "best_cost": best_cost, "best_chrom": best_chrom,
        "routes": decode(best_chrom, inst.n_customers),
        "time": total, "eval_time_sum": eval_time, "history": history,
        "optimum": inst.optimum, "name": inst.name,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--islands", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pop", type=int, default=8000, help="población TOTAL")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--epoch-gens", type=int, default=100)
    ap.add_argument("--migrants", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = run_islands(a.instance, a.islands, a.pop, a.epochs, a.epoch_gens,
                    a.migrants, a.workers, seed=a.seed)
    gap = 100 * (r["best_cost"] - r["optimum"]) / r["optimum"]
    print(f"{r['name']}: costo={r['best_cost']:.0f} gap={gap:.2f}% tiempo={r['time']:.2f}s")
    for k, rt in enumerate(r["routes"], 1):
        print(f"  Ruta #{k}: {' '.join(map(str, rt))}")


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
