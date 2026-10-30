"""
bench_ga.py — Calidad de solución y tiempo del GA completo por enfoque:
    serial | Numba (paralelo) | CUDA (si hay GPU) | Islas Dask

Mismo presupuesto para todos: población total y número de generaciones.
Se ejecutan varias semillas y se reporta mejor costo, promedio, gap y tiempo.

Uso:
    python benchmarks/bench_ga.py --pop 2000 --gens 500 --seeds 3
"""
import argparse
import csv
import json
import os
import statistics
import sys

import numba
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from common import read_vrp                                     # noqa: E402
from ga_engine import GAParams                                  # noqa: E402
import ga_numba, ga_serial                                       # noqa: E402
from islands import run_islands                                 # noqa: E402

try:
    from numba import cuda
    HAS_GPU = cuda.is_available() and os.environ.get("NUMBA_ENABLE_CUDASIM") != "1"
except Exception:
    HAS_GPU = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=2000)
    ap.add_argument("--gens", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--islands", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tag = f"_{a.tag}" if a.tag else ""

    out = os.path.join(ROOT, "benchmarks", "results")
    os.makedirs(out, exist_ok=True)
    rows, best_routes = [], {}
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=a.islands, threads_per_worker=1, dashboard_address=None)
    client = Client(cluster)
    # calentamiento: compila los kernels Numba en cada worker antes de medir
    run_islands(os.path.join(ROOT, "instances", "A-n32-k5.vrp"), a.islands, 4 * 40, 1, 2,
                n_migrants=2, client=client, verbose=False)

    for name in ["A-n32-k5", "A-n60-k9"]:
        path = os.path.join(ROOT, "instances", f"{name}.vrp")
        inst = read_vrp(path)
        approaches = ["serial", "numba"] + (["cuda"] if HAS_GPU else []) + ["islas_dask"]
        for ap_name in approaches:
            costs, times, etimes, sols = [], [], [], []
            for s in range(a.seeds):
                p = GAParams(pop_size=a.pop, generations=a.gens, seed=s)
                if ap_name == "serial":
                    numba.set_num_threads(1)
                    r = ga_serial.solve(inst, p)
                    c, t, et, ch = r.best_cost, r.total_time, r.eval_time, r.routes
                elif ap_name == "numba":
                    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
                    r = ga_numba.solve(inst, p, parallel=True)
                    c, t, et, ch = r.best_cost, r.total_time, r.eval_time, r.routes
                elif ap_name == "cuda":
                    import ga_cuda
                    r = ga_cuda.solve(inst, p)
                    c, t, et, ch = r.best_cost, r.total_time, r.eval_time, r.routes
                else:
                    epochs = max(1, a.gens // 100)
                    r = run_islands(path, a.islands, a.pop, epochs, 100, n_migrants=5,
                                    seed=s, client=client, verbose=False)
                    c, t, et, ch = r["best_cost"], r["time"], float("nan"), r["routes"]
                costs.append(c); times.append(t); etimes.append(et); sols.append((c, ch))
                print(f"{name:9s} {ap_name:10s} seed={s} costo={c:.0f} t={t:.2f}s fit={et:.3f}s",
                      flush=True)
            best = min(costs)
            rows.append({
                "instancia": name, "enfoque": ap_name, "optimo": inst.optimum,
                "mejor_costo": best, "costo_promedio": statistics.mean(costs),
                "gap_mejor_%": 100 * (best - inst.optimum) / inst.optimum,
                "gap_prom_%": 100 * (statistics.mean(costs) - inst.optimum) / inst.optimum,
                "tiempo_total_s": statistics.mean(times),
                "tiempo_fitness_s": statistics.mean(etimes),
            })
            best_routes[f"{name}|{ap_name}"] = min(sols)[1]
    client.close(); cluster.close()
    with open(os.path.join(out, f"ga{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out, f"best_routes{tag}.json"), "w") as f:
        json.dump(best_routes, f, indent=1)


if __name__ == "__main__":
    main()
