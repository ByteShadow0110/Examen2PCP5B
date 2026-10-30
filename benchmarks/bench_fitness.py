"""
bench_fitness.py — Speedup de la EVALUACIÓN DE FITNESS de toda la población:
    serial (Python) vs Numba 1 hilo vs Numba paralelo (N hilos) vs CUDA.

Uso:
    python benchmarks/bench_fitness.py            # escribe benchmarks/results/fitness.csv
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

from common import random_population, read_vrp, warmup          # noqa: E402
from ga_numba import NumbaEvaluator                             # noqa: E402
from ga_serial import SerialEvaluator                           # noqa: E402

try:
    from numba import cuda
    HAS_GPU = cuda.is_available() and os.environ.get("NUMBA_ENABLE_CUDASIM") != "1"
except Exception:
    HAS_GPU = False

POP_SIZES = [1000, 10000, 50000]
REPS = 7


def timeit(fn, reps):
    fn()                                   # calentamiento
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    warmup()
    max_threads = numba.config.NUMBA_NUM_THREADS
    out = os.path.join(ROOT, "benchmarks", "results")
    os.makedirs(out, exist_ok=True)
    rows = []
    print(f"CPU hilos disponibles: {max_threads} | GPU: {HAS_GPU}")
    for name in ["A-n32-k5", "A-n60-k9"]:
        inst = read_vrp(os.path.join(ROOT, "instances", f"{name}.vrp"))
        for P in POP_SIZES:
            pop = random_population(P, inst, np.random.default_rng(0))
            ser = SerialEvaluator(inst, 50.0)
            t_ser = timeit(lambda: ser.evaluate(pop), 3 if P >= 10000 else REPS)
            numba.set_num_threads(1)
            nb1 = NumbaEvaluator(inst, 50.0, parallel=False)
            t_nb1 = timeit(lambda: nb1.evaluate(pop), REPS)
            res = {"instancia": name, "poblacion": P, "serial_s": t_ser, "numba_1h_s": t_nb1}
            for th in range(2, max_threads + 1):
                numba.set_num_threads(th)
                nbp = NumbaEvaluator(inst, 50.0, parallel=True)
                res[f"numba_{th}h_s"] = timeit(lambda: nbp.evaluate(pop), REPS)
            numba.set_num_threads(max_threads)
            if HAS_GPU:
                from ga_cuda import CudaEvaluator
                ce = CudaEvaluator(inst, 50.0)
                res["cuda_s"] = timeit(lambda: ce.evaluate(pop), REPS)
            rows.append(res)
            msg = "  ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in res.items())
            print(msg, flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k)
                  if k in rows[-1] else 99)
    with open(os.path.join(out, "fitness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
