"""
bench_scaling.py — Eficiencia y escalabilidad.

  A) Escalabilidad fuerte (fitness): población fija, 1..N hilos Numba.
       speedup S(p) = T(1)/T(p)      eficiencia E(p) = S(p)/p
  B) Escalabilidad fuerte (GA completo Numba): P=10,000, 100 generaciones, 1..N hilos.
  C) Escalabilidad débil (fitness): 100,000 individuos POR hilo.
       eficiencia débil E(p) = T(1)/T(p)   (ideal = 1)
  D) Islas Dask: población total fija, 4 islas, 1..W workers (escalabilidad fuerte
     del modelo de islas) y 1/2/4/8 islas (efecto en calidad).

Escribe benchmarks/results/scaling_*.csv. Con más núcleos, simplemente se
extiende el rango de hilos/workers (se detectan automáticamente).

Uso:
    python benchmarks/bench_scaling.py
"""
import csv
import os
import statistics
import sys
import time

import numba
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from common import random_population, read_vrp, warmup     # noqa: E402
from ga_engine import GAParams                             # noqa: E402
from ga_numba import NumbaEvaluator, solve                 # noqa: E402
from islands import run_islands                            # noqa: E402

RES = os.path.join(ROOT, "benchmarks", "results")
INST = os.path.join(ROOT, "instances", "A-n60-k9.vrp")
MAXT = numba.config.NUMBA_NUM_THREADS


def med(fn, reps=15):
    """Mínimo de `reps` corridas: el estimador menos sensible al ruido de una VM."""
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)


def write(name, rows):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def strong_fitness(inst):
    print("A) escalabilidad fuerte — fitness")
    rows = []
    for P in [50_000, 200_000]:
        pop = random_population(P, inst, np.random.default_rng(0))
        ev = NumbaEvaluator(inst, 50.0, parallel=True)
        t1 = None
        for p in range(1, MAXT + 1):
            numba.set_num_threads(p)
            t = med(lambda: ev.evaluate(pop))
            t1 = t1 or t
            rows.append({"poblacion": P, "hilos": p, "tiempo_s": t,
                         "speedup": t1 / t, "eficiencia": t1 / t / p})
    write("scaling_strong_fitness.csv", rows)


def strong_ga(inst):
    print("B) escalabilidad fuerte — GA completo (Numba)")
    rows = []
    t1 = None
    for p in range(1, MAXT + 1):
        numba.set_num_threads(p)
        ts, cs = [], []
        for s in range(2):
            r = solve(inst, GAParams(pop_size=10_000, generations=100, seed=s), parallel=True)
            ts.append(r.total_time); cs.append(r.best_cost)
        t = statistics.mean(ts)
        t1 = t1 or t
        rows.append({"hilos": p, "tiempo_s": t, "speedup": t1 / t,
                     "eficiencia": t1 / t / p, "costo_prom": statistics.mean(cs)})
    write("scaling_strong_ga.csv", rows)


def weak_fitness(inst):
    print("C) escalabilidad débil — fitness (100,000 individuos por hilo)")
    rows = []
    t1 = None
    for p in range(1, MAXT + 1):
        numba.set_num_threads(p)
        pop = random_population(100_000 * p, inst, np.random.default_rng(0))
        ev = NumbaEvaluator(inst, 50.0, parallel=True)
        t = med(lambda: ev.evaluate(pop))
        t1 = t1 or t
        rows.append({"hilos": p, "poblacion": 100_000 * p, "tiempo_s": t,
                     "eficiencia_debil": t1 / t})
    write("scaling_weak_fitness.csv", rows)


def islands():
    from dask.distributed import Client, LocalCluster
    print("D) islas Dask")
    rows = []
    maxw = max(2, MAXT)
    t1 = None
    for w in sorted({1, 2, 4, maxw}):
        cluster = LocalCluster(n_workers=w, threads_per_worker=1, dashboard_address=None)
        client = Client(cluster)
        run_islands(INST, 4, 160, 1, 2, n_migrants=2, client=client, verbose=False)  # warm-up
        ts, cs = [], []
        for s in range(2):
            r = run_islands(INST, 4, 8000, 3, 100, n_migrants=5, seed=s, client=client,
                            verbose=False)
            ts.append(r["time"]); cs.append(r["best_cost"])
        client.close(); cluster.close()
        t = statistics.mean(ts)
        t1 = t1 or t
        rows.append({"experimento": "workers", "islas": 4, "workers": w, "tiempo_s": t,
                     "speedup": t1 / t, "eficiencia": t1 / t / min(w, MAXT),
                     "costo_prom": statistics.mean(cs)})
    # número de islas con población total fija y mismo presupuesto de generaciones
    cluster = LocalCluster(n_workers=min(4, maxw), threads_per_worker=1, dashboard_address=None)
    client = Client(cluster)
    run_islands(INST, 4, 160, 1, 2, n_migrants=2, client=client, verbose=False)
    for n in [1, 2, 4, 8]:
        ts, cs = [], []
        for s in range(3):
            r = run_islands(INST, n, 8000, 3, 100, n_migrants=5, seed=s, client=client,
                            verbose=False)
            ts.append(r["time"]); cs.append(r["best_cost"])
        rows.append({"experimento": "num_islas", "islas": n, "workers": min(4, maxw),
                     "tiempo_s": statistics.mean(ts), "speedup": float("nan"),
                     "eficiencia": float("nan"), "costo_prom": statistics.mean(cs)})
    client.close(); cluster.close()
    write("scaling_islands.csv", rows)


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    warmup()
    inst = read_vrp(INST)
    print(f"Hilos disponibles: {MAXT}")
    strong_fitness(inst)
    strong_ga(inst)
    weak_fitness(inst)
    islands()
