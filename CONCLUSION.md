# Conclusión — Proyecto 4: CVRP con algoritmo genético paralelo

> **Nota para el equipo:** las preguntas de abajo son las preguntas guía típicas del proyecto.
> Si el profesor entregó una lista de "preguntas adaptadas", sustituyan los encabezados por las
> suyas y conserven la evidencia (cada respuesta cita una tabla del `INFORME_PRUEBAS.md`).
> Completen la §5 con los resultados SUS reales.

## 1. ¿La paralelización acelera el algoritmo genético? ¿Cuánto?

Sí. La evaluación de fitness, que en la versión serial ocupa el 75 % del tiempo del GA, se aceleró
de **40× a 60×** con Numba en 2 núcleos (A-n60-k9, P = 50,000: 271 ms → 4.5 ms; §2 del informe).
De ese factor, unas 30× vienen de compilar a código máquina y 2× de usar los dos núcleos. El GA completo
con P = 10,000 bajó de 22.4 s a 3.2 s (**7.1×**). No se llega al speedup del fitness porque Amdahl
limita lo que se gana si quedan partes seriales; por eso también se paralelizó la generación de hijos.

## 2. ¿La solución escala al agregar recursos?

En esta máquina, sí, de forma casi ideal. La eficiencia fuerte fue 1.00 en el fitness, 0.94 en el GA
completo y 1.02 en el modelo de islas; la eficiencia débil del fitness fue 0.98 (§3). El fitness escala bien
porque cada individuo se evalúa sin comunicarse con los demás. La fracción paralela estimada del GA
(f ≈ 0.93) predice ~5.4× con 8 núcleos. La evidencia también muestra el límite: poner 4 workers de Dask
en 2 núcleos no dio más velocidad.

## 3. ¿Se mantiene la calidad de la solución?

Sí. Serial y Numba obtienen gaps equivalentes (4.7 % vs 5.4 % promedio en A-n60-k9, 5 semillas),
porque es el mismo algoritmo. El modelo de islas **mejora** la calidad con el mismo presupuesto:
gap promedio de 2.1–2.9 % contra ~5 % de una sola población. Encontró la mejor solución del proyecto,
**1358 en A-n60-k9 (0.30 % del óptimo)**. A-n32-k5 se resolvió al óptimo (784) de forma consistente (§4).

## 4. ¿Qué enfoque conviene y cuándo?

* **Numba CPU:** la mejor opción por defecto, sin hardware especial y con 30–60× sobre Python.
* **Islas Dask:** cuando importa la calidad. Paraleliza a nivel de proceso y, además, mantiene diversidad.
  Es el único enfoque que escala a varias máquinas, porque Dask puede usar un clúster real.
* **CUDA:** conviene con poblaciones muy grandes (≥ 10,000), donde miles de hilos compensan copiar
  la población a la GPU en cada generación. No pudimos medirlo por falta de GPU.
* **Serial:** sólo sirve como referencia y para verificar que las otras versiones son correctas.

## 5. Usabilidad (SUS)

*Completar con `sus/resultados_sus.md`:* puntaje promedio [__], interpretación [__], tema con más
problemas [__], y diferencia entre perfiles técnico y no técnico [__].

## 6. Errores presentados y cómo se resolvieron

1. **El kernel "paralelo" no escalaba (speedup 1.0×).** Con `cache=True`, Numba guardó en caché la
   versión serial y la paralela, compiladas desde la misma función, bajo la misma clave. La versión
   paralela cargaba el código serial. Lo detectamos porque la prueba de escalabilidad no mostraba
   ganancia y `parallel_diagnostics` falló. Se quitó el caché de esas funciones y el speedup pasó a 2.0×.
2. **Convergencia prematura.** Con la configuración inicial (torneo 4, sin búsqueda local) el gap
   de A-n60-k9 era de 10–14 %, y aumentar la población lo empeoraba. Se ajustó a torneo 6 con 2-opt
   intra-ruta en el 30 % de los hijos y el gap bajó a ~3 %.
3. **Fallo en las islas sin solución factible.** Una corrida corta de calentamiento no encontró
   soluciones que respetaran la capacidad y `decode(None)` lanzó `TypeError`. Se agregó un respaldo
   que conserva el individuo de menor fitness.
4. **Error en la interfaz:** `st.cache_resource` rechazó el `Client` de Dask por ser un objeto
   *awaitable*. Se resolvió envolviéndolo en un diccionario.
5. **Mediciones ruidosas** (eficiencias > 1 con tiempos de 5 ms). Se usaron poblaciones grandes y el mínimo de 15 repeticiones.
6. **Instancias y hardware.** El sitio de CVRPLIB no era accesible; A-n60-k9 se obtuvo de un espejo en
   GitHub y se validó contra el óptimo publicado. No hubo GPU: CUDA se validó en el simulador.

## 7. Propuesta de mejoramiento

1. **Medir CUDA en una GPU real** y optimizar el kernel. Cargar la matriz de distancias en memoria
   compartida y mantener la población en la GPU (operadores genéticos también en CUDA) para eliminar
   las copias por generación, que serán el cuello de botella.
2. **Mejor representación:** usar el *giant tour* con el procedimiento *Split* de Prins, que reparte
   rutas óptimamente y elimina las violaciones de capacidad, en lugar de los separadores con penalización.
3. **Búsqueda local entre rutas** (relocate, swap, 2-opt*) para cerrar el gap restante de A-n60-k9.
4. **Islas más robustas.** Tras varias migraciones todas las islas convergen a la misma solución. Proponemos
   migración aleatoria o menos frecuente, parámetros distintos por isla y ejecución asíncrona con
   `client.submit` para que una isla lenta no detenga a las demás.
5. **Penalización adaptativa** de la capacidad según la fracción de individuos factibles.
6. **Escalabilidad real:** repetir `bench_scaling.py` en una máquina de 8–16 núcleos y en un clúster Dask de varias máquinas.
7. **Interfaz:** aplicar las mejoras que señale el análisis SUS (por ejemplo, explicar términos como
   "gap" o "islas" y exportar resultados a Excel/PDF).
