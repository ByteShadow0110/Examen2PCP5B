"""
common.py — Utilidades compartidas para el GA del CVRP.

Contiene:
  * Lectura de instancias en formato TSPLIB/CVRPLIB (Augerat set A).
  * Matriz de distancias EUC_2D redondeada (convención TSPLIB: nint).
  * Representación: permutación de clientes con separadores de ruta.
  * Decodificación y función de fitness (versión Python pura y núcleo Numba).
  * Operadores genéticos compilados con Numba (selección por torneo,
    cruce OX, mutaciones swap / inversión / inserción, 2-opt intra-ruta).

Representación
--------------
Para una instancia con n nodos (nodo 0 = depósito, clientes 1..n-1) y K
vehículos, un cromosoma es una permutación de longitud L = (n-1) + (K-1)
de los enteros 1..n-1+K-1. Los valores 1..n-1 son clientes; los valores
>= n son "separadores" que cierran la ruta actual (el vehículo vuelve al
depósito). Así cada cromosoma codifica como máximo K rutas.

Fitness = distancia_total + penalizacion * exceso_de_capacidad
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

# Óptimos conocidos (CVRPLIB)
KNOWN_OPT = {"A-n32-k5": 784, "A-n60-k9": 1354}


# ---------------------------------------------------------------------------
# Instancias
# ---------------------------------------------------------------------------
@dataclass
class Instance:
    name: str
    n: int                 # nodos incluyendo depósito
    K: int                 # vehículos
    capacity: int
    coords: np.ndarray     # (n, 2)
    demand: np.ndarray     # (n,) int64, demand[0] = 0
    D: np.ndarray          # (n, n) float64
    optimum: float | None

    @property
    def n_customers(self) -> int:
        return self.n - 1

    @property
    def chrom_len(self) -> int:
        return (self.n - 1) + (self.K - 1)


def read_vrp(path: str) -> Instance:
    with open(path) as f:
        lines = [ln.strip() for ln in f]
    name = os.path.splitext(os.path.basename(path))[0]
    cap = None
    dim = None
    opt = None
    coords, demand = {}, {}
    section = None
    for ln in lines:
        if not ln:
            continue
        up = ln.upper()
        if up.startswith("NAME"):
            name = ln.split(":", 1)[1].strip()
        elif up.startswith("DIMENSION"):
            dim = int(ln.split(":", 1)[1])
        elif up.startswith("CAPACITY"):
            cap = int(ln.split(":", 1)[1])
        elif up.startswith("COMMENT"):
            m = re.search(r"Optimal value:\s*(\d+)", ln)
            if m:
                opt = float(m.group(1))
        elif up.startswith("NODE_COORD_SECTION"):
            section = "coord"
        elif up.startswith("DEMAND_SECTION"):
            section = "demand"
        elif up.startswith("DEPOT_SECTION") or up.startswith("EOF"):
            section = None
        elif section == "coord":
            i, x, y = ln.split()[:3]
            coords[int(i)] = (float(x), float(y))
        elif section == "demand":
            i, d = ln.split()[:2]
            demand[int(i)] = int(d)
    n = dim or len(coords)
    xy = np.array([coords[i + 1] for i in range(n)], dtype=np.float64)
    dem = np.array([demand[i + 1] for i in range(n)], dtype=np.int64)
    m = re.search(r"-k(\d+)", name)
    K = int(m.group(1)) if m else int(math.ceil(dem.sum() / cap))
    if opt is None:
        opt = KNOWN_OPT.get(name)
    D = distance_matrix(xy)
    return Instance(name, n, K, cap, xy, dem, D, opt)


def distance_matrix(xy: np.ndarray) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    return np.floor(d + 0.5)   # nint() de TSPLIB (EUC_2D)


# ---------------------------------------------------------------------------
# Decodificación y fitness — Python puro (referencia serial)
# ---------------------------------------------------------------------------
def decode(chrom, n_customers: int):
    """Convierte un cromosoma en una lista de rutas (sin depósito)."""
    routes, cur = [], []
    for g in chrom:
        g = int(g)
        if g > n_customers:
            if cur:
                routes.append(cur)
            cur = []
        else:
            cur.append(g)
    if cur:
        routes.append(cur)
    return routes


def fitness_py(chrom, D, demand, cap, n_customers, penalty):
    """Fitness de UN individuo con un bucle Python puro.

    D y demand pueden ser listas (más rápido en Python puro) o arrays.
    Devuelve (fitness, distancia, exceso_de_capacidad).
    """
    dist = 0.0
    load = 0
    excess = 0
    prev = 0
    for g in chrom:
        if g > n_customers:            # separador → regresar al depósito
            dist += D[prev][0]
            if load > cap:
                excess += load - cap
            load = 0
            prev = 0
        else:
            dist += D[prev][g]
            load += demand[g]
            prev = g
    dist += D[prev][0]
    if load > cap:
        excess += load - cap
    return dist + penalty * excess, dist, excess


def route_cost(routes, D):
    c = 0.0
    for r in routes:
        prev = 0
        for v in r:
            c += D[prev, v]
            prev = v
        c += D[prev, 0]
    return c


# ---------------------------------------------------------------------------
# Núcleo Numba: evaluación de un individuo (reutilizado por CPU paralelo)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True, inline="always")
def eval_one(chrom, D, demand, cap, n_customers, penalty):
    dist = 0.0
    load = 0
    excess = 0
    prev = 0
    for k in range(chrom.shape[0]):
        g = chrom[k]
        if g > n_customers:
            dist += D[prev, 0]
            if load > cap:
                excess += load - cap
            load = 0
            prev = 0
        else:
            dist += D[prev, g]
            load += demand[g]
            prev = g
    dist += D[prev, 0]
    if load > cap:
        excess += load - cap
    return dist + penalty * excess, dist, excess


def _eval_population_impl(pop, D, demand, cap, n_customers, penalty, fit, dist, exc):
    for i in prange(pop.shape[0]):
        f, d, e = eval_one(pop[i], D, demand, cap, n_customers, penalty)
        fit[i] = f
        dist[i] = d
        exc[i] = e


# Dos compilaciones del MISMO código: serial (prange == range) y paralela.
eval_population_serial = njit(fastmath=True)(_eval_population_impl)
eval_population_parallel = njit(fastmath=True, parallel=True)(_eval_population_impl)


# ---------------------------------------------------------------------------
# Operadores genéticos (Numba)
# ---------------------------------------------------------------------------
@njit(cache=True)
def tournament(fit, k):
    P = fit.shape[0]
    best = np.random.randint(P)
    for _ in range(k - 1):
        c = np.random.randint(P)
        if fit[c] < fit[best]:
            best = c
    return best


@njit(cache=True)
def ox_crossover(p1, p2, child, maxval):
    """Order Crossover (OX) para permutaciones."""
    L = p1.shape[0]
    a = np.random.randint(L)
    b = np.random.randint(L)
    if a > b:
        a, b = b, a
    used = np.zeros(maxval + 1, dtype=np.bool_)
    for i in range(a, b + 1):
        child[i] = p1[i]
        used[p1[i]] = True
    pos = (b + 1) % L
    for k in range(L):
        g = p2[(b + 1 + k) % L]
        if not used[g]:
            child[pos] = g
            used[g] = True
            pos = (pos + 1) % L


@njit(cache=True)
def mutate(c, pm):
    L = c.shape[0]
    if np.random.random() < pm:
        r = np.random.random()
        i = np.random.randint(L)
        j = np.random.randint(L)
        if i > j:
            i, j = j, i
        if r < 0.34:                       # swap
            t = c[i]; c[i] = c[j]; c[j] = t
        elif r < 0.67:                     # inversión
            while i < j:
                t = c[i]; c[i] = c[j]; c[j] = t
                i += 1; j -= 1
        else:                              # inserción (mueve c[j] a i)
            t = c[j]
            for k in range(j, i, -1):
                c[k] = c[k - 1]
            c[i] = t


@njit(cache=True, fastmath=True)
def two_opt_routes(c, D, n_customers, max_pass):
    """2-opt dentro de cada ruta (entre separadores). Mejora local barata;
    no cambia la asignación de clientes a vehículos, sólo el orden."""
    L = c.shape[0]
    start = 0
    while start < L:
        end = start
        while end < L and c[end] <= n_customers:
            end += 1
        # ruta en c[start:end]
        m = end - start
        if m >= 3:
            for _ in range(max_pass):
                improved = False
                for i in range(start, end - 1):
                    a = 0 if i == start else c[i - 1]
                    b = c[i]
                    for j in range(i + 1, end):
                        cc = c[j]
                        d = 0 if j == end - 1 else c[j + 1]
                        delta = D[a, cc] + D[b, d] - D[a, b] - D[cc, d]
                        if delta < -1e-9:
                            lo, hi = i, j
                            while lo < hi:
                                t = c[lo]; c[lo] = c[hi]; c[hi] = t
                                lo += 1; hi -= 1
                            b = c[i]
                            improved = True
                if not improved:
                    break
        start = end + 1


def _make_offspring_impl(pop, fit, out, tk, pc, pm, maxval, D, n_customers, ls_prob):
    n_off = out.shape[0]
    L = pop.shape[1]
    for i in prange(n_off):
        a = tournament(fit, tk)
        b = tournament(fit, tk)
        if np.random.random() < pc:
            ox_crossover(pop[a], pop[b], out[i], maxval)
        else:
            for k in range(L):
                out[i, k] = pop[a, k]
        mutate(out[i], pm)
        if ls_prob > 0.0 and np.random.random() < ls_prob:
            two_opt_routes(out[i], D, n_customers, 2)


make_offspring_serial = njit(_make_offspring_impl)
make_offspring_parallel = njit(parallel=True)(_make_offspring_impl)


def random_population(P, inst: Instance, rng: np.random.Generator) -> np.ndarray:
    L = inst.chrom_len
    base = np.arange(1, L + 1, dtype=np.int32)
    pop = np.empty((P, L), dtype=np.int32)
    for i in range(P):
        pop[i] = rng.permutation(base)
    return pop


def warmup():
    """Compila los kernels Numba (para que la compilación no cuente en tiempos)."""
    D = np.ones((4, 4))
    dem = np.ones(4, dtype=np.int64)
    pop = np.array([[1, 2, 4, 3], [3, 4, 1, 2]], dtype=np.int32)
    f = np.empty(2); d = np.empty(2); e = np.empty(2, dtype=np.int64)
    eval_population_serial(pop, D, dem, 10, 3, 1.0, f, d, e)
    eval_population_parallel(pop, D, dem, 10, 3, 1.0, f, d, e)
    out = np.empty_like(pop)
    make_offspring_serial(pop, f, out, 2, 0.9, 0.3, 4, D, 3, 0.5)
    make_offspring_parallel(pop, f, out, 2, 0.9, 0.3, 4, D, 3, 0.5)
