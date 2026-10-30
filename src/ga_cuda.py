"""
ga_cuda.py — GA con evaluación de fitness en GPU (Numba @cuda.jit,
un hilo por individuo) y selección de los mejores con reducciones en GPU.

Flujo por generación
--------------------
1. La población (P x L) se copia a la GPU en layout *columna-mayor* (L x P):
   el hilo i lee pop[k, i], de modo que hilos consecutivos de un warp leen
   direcciones consecutivas (acceso coalescido).
2. Kernel `fitness_kernel`: cada hilo decodifica y evalúa su individuo
   (distancia + penalización por exceso de capacidad).
3. Reducción / selección en GPU:
     - mejor individuo  -> thrust::min_element (libthrust_sel.so) o cupy.argmin (CUB)
     - top-k (élite)    -> thrust::sort_by_key  (libthrust_sel.so) o cupy.argsort
                           (CuPy implementa sort/argsort con Thrust)
4. Se copian a host fit/dist/exc para los operadores genéticos (Numba CPU).

Sin GPU se puede validar el kernel con el simulador de Numba:
    NUMBA_ENABLE_CUDASIM=1 python src/ga_cuda.py instances/A-n32-k5.vrp --pop 64 --gens 5

Uso real (GPU NVIDIA + CUDA 13.x):
    python src/ga_cuda.py instances/A-n60-k9.vrp --pop 10000 --gens 500
"""
from __future__ import annotations

import argparse
import ctypes
import os

import numpy as np
from numba import cuda

from common import make_offspring_parallel, read_vrp, warmup
from ga_engine import GAParams, run_ga

SIM = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"

try:                                    # CuPy: arrays en GPU + reducciones CUB/Thrust
    if SIM:
        raise ImportError
    import cupy as cp
except Exception:                       # pragma: no cover
    cp = None

THREADS_PER_BLOCK = 128


@cuda.jit
def fitness_kernel(popT, D, demand, cap, n_customers, penalty, fit, dist, exc):
    """popT: (L, P) int32 — columna i = individuo i. Un hilo por individuo."""
    i = cuda.grid(1)
    P = popT.shape[1]
    if i >= P:
        return
    L = popT.shape[0]
    d = 0.0
    load = 0
    excess = 0
    prev = 0
    for k in range(L):
        g = popT[k, i]
        if g > n_customers:             # separador de ruta
            d += D[prev, 0]
            if load > cap:
                excess += load - cap
            load = 0
            prev = 0
        else:
            d += D[prev, g]
            load += demand[g]
            prev = g
    d += D[prev, 0]
    if load > cap:
        excess += load - cap
    fit[i] = d + penalty * excess
    dist[i] = d
    exc[i] = excess


# ---------------------------------------------------------------------------
# Thrust (C++) opcional: libthrust_sel.so compilada desde src/cuda/thrust_sel.cu
# ---------------------------------------------------------------------------
class ThrustSelector:
    def __init__(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "cuda", "libthrust_sel.so")
        self.lib = None
        if cp is not None and os.path.exists(path):
            lib = ctypes.CDLL(path)
            lib.thrust_argmin.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.thrust_argmin.restype = ctypes.c_int
            lib.thrust_topk.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_void_p, ctypes.c_void_p]
            self.lib = lib

    @property
    def available(self):
        return self.lib is not None


class CudaEvaluator:
    def __init__(self, inst, penalty):
        self.inst = inst
        self.penalty = penalty
        self.d_D = cuda.to_device(inst.D.astype(np.float64))
        self.d_dem = cuda.to_device(inst.demand.astype(np.int64))
        self.P = None
        self.thrust = ThrustSelector()
        self.backend = ("thrust" if self.thrust.available
                        else "cupy" if cp is not None else "numpy(sim)")

    def _alloc(self, P):
        self.P = P
        self.d_fit = cuda.device_array(P, dtype=np.float64)
        self.d_dist = cuda.device_array(P, dtype=np.float64)
        self.d_exc = cuda.device_array(P, dtype=np.int64)

    def evaluate(self, pop):
        P = pop.shape[0]
        if self.P != P:
            self._alloc(P)
        d_pop = cuda.to_device(np.ascontiguousarray(pop.T))      # H2D (L x P)
        blocks = (P + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        fitness_kernel[blocks, THREADS_PER_BLOCK](
            d_pop, self.d_D, self.d_dem, self.inst.capacity,
            self.inst.n_customers, self.penalty, self.d_fit, self.d_dist, self.d_exc)
        cuda.synchronize()
        fit = self.d_fit.copy_to_host()
        dist = self.d_dist.copy_to_host()
        exc = self.d_exc.copy_to_host()
        self._fit_host = fit
        return fit, dist, exc

    def best(self):
        """Índice del mejor individuo (reducción en GPU)."""
        if self.thrust.available:
            return self.thrust.lib.thrust_argmin(self.d_fit.device_ctypes_pointer.value, self.P)
        if cp is not None:
            return int(cp.argmin(cp.asarray(self.d_fit)))
        return int(np.argmin(self._fit_host))

    def topk(self, k):
        """Índices de los k mejores (élite) calculados en GPU."""
        if self.thrust.available:
            keys = cp.array(cp.asarray(self.d_fit))            # copia: sort_by_key la reordena
            out = cp.empty(k, dtype=cp.int32)
            self.thrust.lib.thrust_topk(keys.data.ptr, self.P, k,
                                        cp.empty(self.P, dtype=cp.int32).data.ptr, out.data.ptr)
            return cp.asnumpy(out)
        if cp is not None:
            return cp.asnumpy(cp.argsort(cp.asarray(self.d_fit))[:k])   # sort de Thrust en CuPy
        return np.argsort(self._fit_host, kind="stable")[:k]


def solve(inst, params: GAParams, verbose=False):
    warmup()
    ev = CudaEvaluator(inst, params.penalty)
    if verbose:
        print(f"  [CUDA] simulador={SIM}  selección={ev.backend}")
    return run_ga(inst, ev, make_offspring_parallel, params, verbose=verbose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--pop", type=int, default=2000)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    inst = read_vrp(a.instance)
    r = solve(inst, GAParams(pop_size=a.pop, generations=a.gens, seed=a.seed,
                             elite=min(20, a.pop // 4)), verbose=True)
    gap = 100 * (r.best_cost - inst.optimum) / inst.optimum
    print(f"{inst.name}: costo={r.best_cost:.0f} gap={gap:.2f}% "
          f"tiempo={r.total_time:.2f}s (fitness {r.eval_time:.3f}s)")
    for k, rt in enumerate(r.routes, 1):
        print(f"  Ruta #{k}: {' '.join(map(str, rt))}")


if __name__ == "__main__":
    main()
