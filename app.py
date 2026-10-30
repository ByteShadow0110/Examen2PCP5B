"""
app.py — Interfaz web del optimizador CVRP (herramienta evaluada con SUS).

Ejecutar:
    streamlit run app.py
Se abre en http://localhost:8501

Permite: elegir una instancia incluida o subir un .vrp (formato CVRPLIB),
elegir el enfoque (serial / Numba / CUDA / islas Dask), ajustar parámetros,
ejecutar, y ver costo, gap, tiempos, mapa de rutas y curva de convergencia.
"""
import os
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from common import read_vrp, route_cost        # noqa: E402
from ga_engine import GAParams                  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#6250d6", "#e34948", "#52514e", "#0d366b"]


def gpu_available():
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False


@st.cache_resource
def get_dask_client(n_workers):
    from dask.distributed import Client, LocalCluster
    # Se envuelve en un dict: dask.Client es "awaitable" y st.cache_resource
    # lo rechazaría si se devolviera directamente.
    return {"client": Client(LocalCluster(n_workers=n_workers, threads_per_worker=1,
                                          dashboard_address=None))}


def plot_routes(inst, routes):
    xy = inst.coords
    fig, ax = plt.subplots(figsize=(6, 6))
    for k, r in enumerate(routes):
        p = [0] + list(r) + [0]
        ax.plot(xy[p, 0], xy[p, 1], "-o", color=COLORS[k % len(COLORS)], lw=2, ms=4,
                label=f"Ruta {k + 1}")
    ax.plot(*xy[0], "s", color="black", ms=10, label="Depósito")
    for i in range(1, inst.n):
        ax.annotate(str(i), xy[i], fontsize=6, color="#52514e",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_history(hist, xlabel):
    h = [v for v in hist if np.isfinite(v)]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(len(hist) - len(h), len(hist)), h, color=COLORS[0], lw=2)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(xlabel); ax.set_ylabel("Mejor costo factible")
    ax.grid(color="#e5e5e5")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="CVRP — GA paralelo", layout="wide")
st.title("Optimización de rutas de reparto (CVRP)")
st.caption("Algoritmo genético paralelo: Numba (CPU), CUDA (GPU) e islas con Dask")

with st.sidebar:
    st.header("1. Instancia")
    builtin = sorted(f for f in os.listdir(os.path.join(ROOT, "instances")) if f.endswith(".vrp"))
    source = st.radio("Origen", ["Incluida", "Subir archivo .vrp"], horizontal=True)
    if source == "Incluida":
        inst_path = os.path.join(ROOT, "instances", st.selectbox("Instancia", builtin))
    else:
        up = st.file_uploader("Archivo CVRPLIB (.vrp)", type=["vrp", "txt"])
        inst_path = None
        if up is not None:
            tmp = os.path.join(tempfile.gettempdir(), up.name)
            with open(tmp, "wb") as f:
                f.write(up.getbuffer())
            inst_path = tmp

    st.header("2. Enfoque")
    options = ["Numba (CPU paralelo)", "Islas Dask", "Serial (Python)"]
    if gpu_available():
        options.insert(1, "CUDA (GPU)")
    method = st.selectbox("Backend", options,
                          help="Serial es la línea base lenta; Numba usa todos los núcleos; "
                               "Islas divide la población en subpoblaciones que migran.")

    st.header("3. Parámetros")
    pop = st.number_input("Población (total)", 100, 100_000, 2000, step=500)
    gens = st.number_input("Generaciones", 10, 5000, 300, step=50)
    seed = st.number_input("Semilla", 0, 10_000, 0)
    n_islands = 4
    if method == "Islas Dask":
        n_islands = st.slider("Número de islas", 2, 8, 4)
        st.caption("Cada isla corre 100 generaciones por época y luego migran los mejores.")
    with st.expander("Avanzado"):
        tk = st.slider("Tamaño de torneo", 2, 10, 6)
        pm = st.slider("Prob. de mutación", 0.0, 1.0, 0.4)
        ls = st.slider("Prob. de 2-opt por hijo", 0.0, 1.0, 0.3)
    run = st.button("Optimizar rutas", type="primary", use_container_width=True)

if inst_path is None:
    st.info("Sube un archivo .vrp o elige una instancia incluida para comenzar.")
    st.stop()

try:
    inst = read_vrp(inst_path)
except Exception as e:                                   # archivo inválido
    st.error(f"No se pudo leer la instancia: {e}. Verifica que tenga formato CVRPLIB "
             "(NODE_COORD_SECTION y DEMAND_SECTION).")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Instancia", inst.name)
c2.metric("Clientes", inst.n_customers)
c3.metric("Vehículos (K)", inst.K)
c4.metric("Capacidad", inst.capacity)

if run:
    params = GAParams(pop_size=int(pop), generations=int(gens), seed=int(seed),
                      tournament=tk, pm=pm, ls_prob=ls, elite=min(20, int(pop) // 4))
    with st.spinner(f"Optimizando con {method}…"):
        t0 = time.perf_counter()
        if method.startswith("Islas"):
            from islands import run_islands
            client = get_dask_client(min(4, os.cpu_count() or 2))["client"]
            r = run_islands(inst_path, n_islands, int(pop), max(1, int(gens) // 100), 100,
                            n_migrants=5, seed=int(seed), client=client, verbose=False,
                            tournament=tk, pm=pm, ls_prob=ls)
            res = dict(cost=r["best_cost"], routes=r["routes"], time=r["time"],
                       eval=None, hist=r["history"], xlabel="Época (100 generaciones)")
        else:
            if method.startswith("Numba"):
                import ga_numba as mod
            elif method.startswith("CUDA"):
                import ga_cuda as mod
            else:
                import ga_serial as mod
            r = mod.solve(inst, params)
            res = dict(cost=r.best_cost, routes=r.routes, time=r.total_time,
                       eval=r.eval_time, hist=r.history, xlabel="Generación",
                       feasible=r.feasible)
    st.session_state["res"] = res
    st.session_state["res_inst"] = inst_path
    st.session_state["res_method"] = method

if "res" in st.session_state and st.session_state.get("res_inst") == inst_path:
    res = st.session_state["res"]
    st.subheader(f"Resultado — {st.session_state['res_method']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Distancia total", f"{res['cost']:.0f}")
    if inst.optimum:
        m2.metric("Gap vs óptimo", f"{100 * (res['cost'] - inst.optimum) / inst.optimum:.2f} %",
                  help=f"Óptimo conocido: {inst.optimum:.0f}")
    else:
        m2.metric("Gap vs óptimo", "N/D", help="La instancia no trae óptimo conocido")
    m3.metric("Tiempo total", f"{res['time']:.2f} s")
    m4.metric("Tiempo de fitness", f"{res['eval']:.3f} s" if res["eval"] is not None else "—")
    if res.get("feasible") is False:
        st.warning("No se encontró solución factible: aumenta generaciones o población.")

    g1, g2 = st.columns([1.1, 1])
    with g1:
        st.pyplot(plot_routes(inst, res["routes"]))
    with g2:
        st.pyplot(plot_history(res["hist"], res["xlabel"]))
        rows = [{"Ruta": k + 1, "Clientes": " → ".join(map(str, r)),
                 "Carga": int(inst.demand[r].sum()),
                 "Uso de capacidad": f"{100 * inst.demand[r].sum() / inst.capacity:.0f} %"}
                for k, r in enumerate(res["routes"])]
        st.dataframe(rows, hide_index=True, use_container_width=True)
        sol = "\n".join(f"Route #{k + 1}: {' '.join(map(str, r))}"
                        for k, r in enumerate(res["routes"]))
        sol += f"\nCost {route_cost(res['routes'], inst.D):.0f}\n"
        st.download_button("Descargar solución (.sol)", sol, file_name=f"{inst.name}.sol")
else:
    st.info("Configura los parámetros en la barra lateral y presiona **Optimizar rutas**.")
