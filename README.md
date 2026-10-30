# Proyecto 4 — CVRP con algoritmo genético paralelo

Optimización de rutas de vehículos con capacidad (CVRP) mediante un algoritmo genético (GA)
paralelizado en tres niveles:

| Nivel | Tecnología | Archivo |
|---|---|---|
| Población en CPU | Numba `@njit(parallel=True)` + `prange` | `src/ga_numba.py` |
| Población en GPU | Numba `@cuda.jit` (un hilo por individuo) + Thrust | `src/ga_cuda.py`, `src/cuda/thrust_sel.cu` |
| Islas | Dask `LocalCluster(n_workers=4)` + `dask.delayed` | `src/islands.py` |
| Línea base | Bucle en Python puro | `src/ga_serial.py` |

## Entregables

| Entregable (rúbrica) | Dónde está |
|---|---|
| Código funcional y documentación | `src/`, `app.py` (cada módulo documenta su papel en la arquitectura paralela) |
| README | este archivo |
| Informe de pruebas — métricas | [`INFORME_PRUEBAS.md`](INFORME_PRUEBAS.md) §1–4, §6 (speedup, eficiencia, escalabilidad, gap) |
| Informe de pruebas — SUS y comentarios | [`INFORME_PRUEBAS.md`](INFORME_PRUEBAS.md) §5 + `sus/` (protocolo, plantilla y análisis) |
| Conclusión | [`CONCLUSION.md`](CONCLUSION.md) |

## Arquitectura paralela

```text
                 ┌────────────── Dask LocalCluster (islas) ──────────────┐
                 │  isla 1        isla 2        isla 3        isla 4      │  dask.delayed por época
                 │  GA Numba      GA Numba      GA Numba      GA Numba    │  (100 generaciones)
                 └──────┬─────────────┬────────────┬─────────────┬───────┘
                        └── migración en anillo: 5 mejores → isla siguiente ──┘
 Dentro de cada GA (por generación):
   evaluar población ─┬─ serial: bucle Python
                      ├─ Numba:  prange sobre individuos (1 hilo CPU por bloque de individuos)
                      └─ CUDA:   1 hilo GPU por individuo  → élite con Thrust (sort_by_key / min_element)
   generar hijos ──── Numba prange (torneo + OX + mutación + 2-opt), cada hijo independiente
```

## Estructura

```text
cvrp_par/
├── instances/          # A-n32-k5 (óptimo 784), A-n60-k9 (óptimo 1354) — Augerat set A
├── src/
│   ├── common.py       # lectura de instancias, representación, fitness, operadores (Numba)
│   ├── ga_engine.py    # bucle generacional común a todos los backends
│   ├── ga_serial.py
│   ├── ga_numba.py
│   ├── ga_cuda.py
│   ├── islands.py      # orquestación Dask
│   └── cuda/thrust_sel.cu   # thrust::min_element / reduce / sort_by_key
├── app.py              # interfaz web (Streamlit) — herramienta evaluada con SUS
├── benchmarks/
│   ├── bench_fitness.py     # speedup de la evaluación de fitness
│   ├── bench_ga.py          # mejor costo, gap y tiempo por enfoque
│   ├── bench_scaling.py     # eficiencia, escalabilidad fuerte/débil, islas
│   ├── make_plots.py
│   └── results/             # CSV + gráficas
├── sus/                # protocolo SUS, plantilla de respuestas y script de análisis
├── docs/               # capturas de la interfaz
├── INFORME_PRUEBAS.md
└── CONCLUSION.md
```

## Instalación

```bash
pip install -r requirements.txt
# Sólo con GPU NVIDIA (CUDA 13.x):
pip install cupy-cuda13x "numba-cuda[cu13]"
cd src/cuda && nvcc -O3 -shared -Xcompiler -fPIC -o libthrust_sel.so thrust_sel.cu
```

## Ejecución

### Interfaz web (forma más sencilla)

```bash
streamlit run app.py        # abre http://localhost:8501
```

Elige la instancia, el enfoque y los parámetros en la barra lateral y presiona **Optimizar rutas**.

### Línea de comandos

```bash
python src/ga_serial.py  instances/A-n32-k5.vrp --pop 2000 --gens 500
python src/ga_numba.py   instances/A-n60-k9.vrp --pop 10000 --gens 300 --threads 8
python src/ga_cuda.py    instances/A-n60-k9.vrp --pop 10000 --gens 300      # requiere GPU
NUMBA_ENABLE_CUDASIM=1 python src/ga_cuda.py instances/A-n32-k5.vrp --pop 128 --gens 5   # sin GPU (Linux/macOS)
# Windows (PowerShell):  $env:NUMBA_ENABLE_CUDASIM="1"; python src/ga_cuda.py instances/A-n32-k5.vrp --pop 128 --gens 5
python src/islands.py instances/A-n60-k9.vrp --islands 4 --pop 10000 --epochs 3 --epoch-gens 100

python benchmarks/bench_fitness.py
python benchmarks/bench_ga.py --pop 10000 --gens 300 --seeds 2 --tag p10000
python benchmarks/bench_scaling.py
python benchmarks/make_plots.py
python sus/analizar_sus.py sus/respuestas_sus.csv   # después de aplicar la encuesta
```

Todos los comandos se ejecutan desde la carpeta raíz `cvrp_par/`. Tiempos aproximados en 2 núcleos:
`bench_fitness` ~1 min, `bench_ga` (P=10,000) ~3 min, `bench_scaling` ~3 min.

## Diseño

**Representación.** Permutación de los clientes `1..n-1` más `K-1` separadores (valores `≥ n`).
Al leer un separador el vehículo regresa al depósito, así que cada cromosoma codifica a lo más K rutas.
Fitness = distancia total + 50 × (unidades de carga que exceden la capacidad).
Distancias EUC_2D redondeadas (convención TSPLIB); la solución óptima publicada de A-n32-k5
se decodifica exactamente a 784.

**GA.** Torneo (k=6), cruce OX (pc=0.9), mutación swap/inversión/inserción (pm=0.4),
élite de 20, y con probabilidad 0.3 un 2-opt dentro de cada ruta del hijo.

**Paralelismo de población.** Evaluar cada individuo es independiente: en Numba el bucle
sobre la población es un `prange`; en CUDA cada hilo evalúa una columna de la población,
que se transpone a layout (L × P) para que las lecturas de un warp sean coalescidas.
La élite se obtiene en GPU con `thrust::sort_by_key` y el mejor con `thrust::min_element`
(si no está la librería compilada se usa `cupy.argsort`/`cupy.argmin`).

**Paralelismo de islas.** La población total se divide en 4 islas (configurable 4–8). Cada época,
cada isla corre 100 generaciones como tarea `dask.delayed`; después los 5 mejores de cada isla
migran a la siguiente (anillo) reemplazando a sus peores. Con `--epochs 1` es exactamente
el esquema del enunciado (100 generaciones y una migración final).

## Resultados (CPU de 2 núcleos, Intel Xeon 2.1 GHz, sin GPU)

Resumen; el análisis completo (incluida eficiencia y escalabilidad) está en `INFORME_PRUEBAS.md`.

### Speedup en la evaluación de fitness (mediana de 7 repeticiones)

| Instancia | Población | Serial (Python) | Numba 1 hilo | Numba 2 hilos | Speedup Numba 2h |
|---|---:|---:|---:|---:|---:|
| A-n32-k5 | 10,000 | 26.1 ms | 0.99 ms | 0.50 ms | 52× |
| A-n32-k5 | 50,000 | 141.8 ms | 5.09 ms | 3.57 ms | 40× |
| A-n60-k9 | 10,000 | 64.7 ms | 1.87 ms | 1.45 ms | 45× |
| A-n60-k9 | 50,000 | 271.3 ms | 9.21 ms | 4.53 ms | 60× |

Unas ~30× vienen de compilar (Numba 1 hilo vs Python) y el resto del paralelismo (hasta 2× con 2 núcleos).
Los 6–8× de la referencia comparan contra código ya compilado en CPUs con más núcleos; aquí
sólo hay 2. **CUDA:** el kernel se validó con el simulador de Numba (resultados idénticos a la
versión serial) y la librería Thrust compila con CUDA 13.4, pero no hubo GPU para medir tiempos;
`bench_fitness.py` agrega la columna `cuda_s` automáticamente si detecta una.

### Calidad y tiempo del GA completo (P = 10,000, 300 generaciones, 2 semillas)

| Instancia | Enfoque | Mejor costo | Gap mejor | Gap promedio | Tiempo total | Tiempo fitness |
|---|---|---:|---:|---:|---:|---:|
| A-n32-k5 | Serial | 784 | 0.00 % | 0.77 % | 13.34 s | 9.31 s |
| A-n32-k5 | Numba | 784 | 0.00 % | 0.00 % | 2.25 s | 0.15 s |
| A-n32-k5 | Islas Dask (4×2,500) | 784 | 0.00 % | 0.00 % | 2.09 s | — |
| A-n32-k5 | CUDA | N/D | — | — | — | — |
| A-n60-k9 | Serial | 1415 | 4.51 % | 4.65 % | 22.39 s | 16.81 s |
| A-n60-k9 | Numba | 1428 | 5.47 % | 6.94 % | 3.15 s | 0.28 s |
| A-n60-k9 | Islas Dask (4×2,500) | **1358** | **0.30 %** | 2.10 % | 2.90 s | — |
| A-n60-k9 | CUDA | N/D | — | — | — | — |

Con P = 2,000 y 500 generaciones (5 semillas) ver `benchmarks/results/ga_p2000.csv`
(A-n60-k9: serial 1378, Numba 1390, islas 1380; gap promedio de las islas 2.9 % vs ~5 % de una sola población).

Serial y Numba ejecutan el mismo algoritmo; sus costos difieren sólo por las secuencias aleatorias.
El modelo de islas mejora la calidad porque mantiene más diversidad con el mismo presupuesto.
