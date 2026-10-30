# Protocolo de evaluación de usabilidad (SUS)

**Herramienta evaluada:** la interfaz web `app.py` (Streamlit) del optimizador CVRP.
**Instrumento:** System Usability Scale (SUS), Brooke (1996), 10 ítems en escala Likert de 1 a 5.
**Participantes recomendados:** de 5 a 10 personas (compañeros de otro equipo, familiares,
compañeros de trabajo). Anotar su perfil: *técnico* (programa o estudia sistemas) o *no técnico*.

## 1. Preparación (la hace el equipo)

```bash
pip install -r requirements.txt
streamlit run app.py          # abre http://localhost:8501
```

Explicar en una frase qué es el problema: *"La herramienta calcula las rutas más cortas
para que una flotilla de camiones entregue a todos los clientes sin pasarse de su capacidad."*
No explicar cómo se usa la interfaz: eso es lo que se evalúa.

## 2. Tareas que realiza el participante (sin ayuda)

| # | Tarea | Criterio de éxito |
|---|---|---|
| T1 | Optimiza las rutas de la instancia **A-n32-k5** con la configuración por defecto. | Aparece el mapa de rutas. |
| T2 | Di cuál es la distancia total obtenida y qué tan lejos está del óptimo. | Lee "Distancia total" y "Gap". |
| T3 | Repite con **A-n60-k9** usando el enfoque **Islas Dask** con **6 islas**. | Resultado de islas en pantalla. |
| T4 | Indica qué ruta va más cargada (porcentaje de capacidad). | Lee la tabla de rutas. |
| T5 | Descarga el archivo de la solución. | Se descarga el `.sol`. |

El observador anota por participante: tareas completadas (0–5), tiempo total en minutos
y dónde se atoró (eso alimenta el análisis de comentarios).

## 3. Cuestionario SUS (lo contesta el participante al terminar)

Escala: **1 = Totalmente en desacuerdo … 5 = Totalmente de acuerdo**.

| Ítem | Afirmación |
|---|---|
| P1 | Creo que me gustaría usar esta herramienta con frecuencia. |
| P2 | Encontré la herramienta innecesariamente compleja. |
| P3 | Pensé que la herramienta era fácil de usar. |
| P4 | Creo que necesitaría el apoyo de un experto para poder usar esta herramienta. |
| P5 | Encontré que las distintas funciones de la herramienta estaban bien integradas. |
| P6 | Pensé que había demasiada inconsistencia en la herramienta. |
| P7 | Imagino que la mayoría de las personas aprendería a usar esta herramienta muy rápidamente. |
| P8 | Encontré la herramienta muy difícil de usar. |
| P9 | Me sentí muy seguro(a) usando la herramienta. |
| P10 | Necesité aprender muchas cosas antes de poder usar la herramienta. |

Preguntas abiertas:
1. ¿Qué fue lo que más te gustó?
2. ¿Qué fue lo más confuso o lo que menos te gustó?
3. ¿Qué cambiarías o agregarías?

Puede hacerse en papel o en un Google Forms con las mismas preguntas; al final se
vacían las respuestas en `respuestas_sus.csv` (una fila por participante).

## 4. Cálculo del puntaje

Para cada participante:

* ítems impares (1, 3, 5, 7, 9): aporte = respuesta − 1
* ítems pares (2, 4, 6, 8, 10): aporte = 5 − respuesta
* **SUS = (suma de aportes) × 2.5** → rango 0–100

Interpretación (Bangor, Kortum y Miller, 2009; Sauro, 2011):

| SUS | Adjetivo | Aceptabilidad |
|---|---|---|
| ≥ 85 | Excelente | Aceptable |
| 73 – 84 | Bueno | Aceptable |
| 68 | Promedio de la industria | — |
| 51 – 67 | Regular (OK) | Marginal |
| ≤ 50 | Pobre | No aceptable |

## 5. Análisis

```bash
python sus/analizar_sus.py sus/respuestas_sus.csv
```

Genera `sus/resultados_sus.md` (puntajes, media, desviación estándar, intervalo de confianza
del 95 %, promedio por ítem, comparación técnico/no técnico, comentarios agrupados por tema)
y `sus/sus_puntajes.png`. Ese archivo se copia/resume en la sección SUS del informe.
