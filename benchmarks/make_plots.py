"""make_plots.py — Gráficas a partir de benchmarks/results/*.csv"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import read_vrp  # noqa: E402

RES = os.path.join(ROOT, "benchmarks", "results")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#52514e"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.labelcolor": "#222"})


def speedup_plot():
    rows = list(csv.DictReader(open(os.path.join(RES, "fitness.csv"))))
    series = [k for k in rows[0] if k.endswith("_s") and k != "serial_s"]
    labels = [f"{r['instancia']}\nP={int(r['poblacion']):,}" for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(series)
    for j, s in enumerate(series):
        sp = [float(r["serial_s"]) / float(r[s]) for r in rows]
        xs = [i + (j - (len(series) - 1) / 2) * w for i in range(len(rows))]
        bars = ax.bar(xs, sp, w * 0.92, color=C[j], label=s.replace("_s", "").replace("_", " "))
        for x, v in zip(xs, sp):
            ax.text(x, v + 1, f"{v:.0f}×", ha="center", fontsize=8, color="#333")
    ax.set_xticks(range(len(rows)), labels, fontsize=8)
    ax.set_ylabel("Speedup vs. serial (Python)")
    ax.set_title("Speedup en la evaluación de fitness de toda la población", loc="left")
    ax.grid(axis="y", color="#e5e5e5"); ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "speedup_fitness.png"), dpi=160)


def routes_plot(tag="p10000", name="A-n60-k9", approach="islas_dask"):
    routes = json.load(open(os.path.join(RES, f"best_routes_{tag}.json")))[f"{name}|{approach}"]
    inst = read_vrp(os.path.join(ROOT, "instances", f"{name}.vrp"))
    xy = inst.coords
    cost = sum(inst.D[a, b] for r in routes for a, b in zip([0] + r, r + [0]))
    fig, ax = plt.subplots(figsize=(6, 6))
    for k, r in enumerate(routes):
        p = [0] + r + [0]
        ax.plot(xy[p, 0], xy[p, 1], "-o", color=C[k % len(C)], lw=2, ms=4,
                label=f"Ruta {k + 1} (carga {inst.demand[r].sum()})")
    ax.plot(*xy[0], "s", color="black", ms=10, label="Depósito")
    ax.set_title(f"{name} — mejor solución ({approach}), costo {cost:.0f} "
                 f"(óptimo {inst.optimum:.0f})", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(RES, f"rutas_{name}.png"), dpi=160)


def scaling_plot():
    rd = lambda f: list(csv.DictReader(open(os.path.join(RES, f))))
    sf = [r for r in rd("scaling_strong_fitness.csv") if r["poblacion"] == "200000"]
    sg = rd("scaling_strong_ga.csv")
    si = [r for r in rd("scaling_islands.csv") if r["experimento"] == "workers"]
    ni = [r for r in rd("scaling_islands.csv") if r["experimento"] == "num_islas"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    maxp = max(int(r["workers"]) for r in si)
    a.plot([1, maxp], [1, maxp], "--", color="#9a9a96", lw=1.5, label="Ideal (lineal)")
    for k, (rows, key, lab) in enumerate([(sf, "hilos", "Fitness Numba (P=200k)"),
                                          (sg, "hilos", "GA completo Numba (P=10k)"),
                                          (si, "workers", "Islas Dask (4 islas)")]):
        a.plot([int(r[key]) for r in rows], [float(r["speedup"]) for r in rows], "-o",
               color=C[k], lw=2, ms=8, label=lab)
    a.set_xlabel("Hilos / workers"); a.set_ylabel("Speedup  T(1)/T(p)")
    a.set_title("Escalabilidad fuerte (CPU de 2 núcleos)", loc="left")
    a.grid(color="#e5e5e5"); a.legend(frameon=False, fontsize=8)
    xs = [int(r["islas"]) for r in ni]
    b.plot([str(x) for x in xs], [float(r["costo_prom"]) for r in ni], "-o", color=C[0], lw=2, ms=9)
    b.axhline(1354, ls="--", color="#9a9a96", lw=1.5)
    b.text(-0.4, 1356, "óptimo 1354", fontsize=8, color="#52514e")
    for i, r in enumerate(ni):
        b.text(i, float(r["costo_prom"]) + 3, f"{float(r['costo_prom']):.0f}", ha="center", va="bottom", fontsize=8)
    b.set_ylim(1330, max(float(r["costo_prom"]) for r in ni) + 25)
    b.set_xlabel("Número de islas (población total 8,000)"); b.set_ylabel("Costo promedio (3 semillas)")
    b.set_title("A-n60-k9: calidad vs número de islas", loc="left")
    b.grid(axis="y", color="#e5e5e5"); b.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "escalabilidad.png"), dpi=160)


if __name__ == "__main__":
    scaling_plot()
    speedup_plot()
    routes_plot()
    routes_plot(name="A-n32-k5")
