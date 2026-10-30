"""
analizar_sus.py — Calcula el puntaje SUS y analiza los comentarios.

Uso:
    python sus/analizar_sus.py sus/respuestas_sus.csv

Entrada (CSV, una fila por participante):
    participante, perfil (tecnico/no_tecnico), p1..p10 (1-5),
    tareas_completadas (0-5), tiempo_min, gusto, confuso, sugerencia

Salida:
    sus/resultados_sus.md   — tablas y análisis listos para el informe
    sus/sus_puntajes.png    — puntaje por participante vs referencia 68
"""
import csv
import math
import os
import statistics
import sys
import unicodedata
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

ITEMS = [
    "Me gustaría usarla con frecuencia",
    "Innecesariamente compleja (−)",
    "Fácil de usar",
    "Necesitaría apoyo de un experto (−)",
    "Funciones bien integradas",
    "Demasiada inconsistencia (−)",
    "Se aprende rápido",
    "Muy difícil de usar (−)",
    "Me sentí seguro(a)",
    "Necesité aprender mucho antes (−)",
]

# Temas para clasificar comentarios (palabras clave sin acentos, en minúsculas)
THEMES = {
    "Rendimiento / velocidad": ["rapid", "lent", "tard", "velocidad", "tiempo", "espera", "veloz"],
    "Visualización / mapa": ["mapa", "grafic", "visual", "color", "ruta", "dibujo", "linea"],
    "Claridad de términos": ["gap", "fitness", "termino", "tecnic", "entend", "significa",
                             "confus", "explic", "jerga", "ayuda", "isla", "semilla"],
    "Diseño / navegación": ["boton", "menu", "barra", "diseno", "orden", "encontr", "lateral",
                            "pantalla", "interfaz", "navega"],
    "Parámetros / configuración": ["parametr", "poblacion", "generacion", "configur", "avanzad",
                                   "opcion", "slider"],
    "Resultados / exportación": ["descarg", "export", "archivo", "tabla", "resultado", "excel",
                                 "pdf", "sol"],
    "Errores / fallas": ["error", "falla", "trab", "crash", "no funciona", "se cerro", "bug"],
}


def norm(s):
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def sus_score(r):
    total = 0
    for i in range(1, 11):
        v = int(r[f"p{i}"])
        if not 1 <= v <= 5:
            raise ValueError(f"{r['participante']}: p{i}={v} fuera de rango 1-5")
        total += (v - 1) if i % 2 == 1 else (5 - v)
    return total * 2.5


def adjective(s):
    if s >= 85: return "Excelente"
    if s >= 73: return "Bueno"
    if s >= 68: return "Arriba del promedio (OK)"
    if s >= 51: return "Regular / marginal"
    return "Pobre / no aceptable"


def ci95(xs):
    if len(xs) < 2:
        return float("nan")
    # t de Student para n-1 g.l. (tabla corta; n>=30 → 1.96)
    t = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31,
         9: 2.26, 10: 2.23, 15: 2.13, 20: 2.09, 29: 2.05}
    df = len(xs) - 1
    tv = t.get(df) or next((t[k] for k in sorted(t) if k >= df), 1.96)
    return tv * statistics.stdev(xs) / math.sqrt(len(xs))


def main(path):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r.get("participante", "").strip()]
    if not rows:
        sys.exit("El CSV no tiene respuestas todavía. Llena una fila por participante.")
    for r in rows:
        r["sus"] = sus_score(r)
    scores = [r["sus"] for r in rows]
    mean = statistics.mean(scores)
    sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
    ci = ci95(scores)

    L = []
    L.append("# Resultados SUS\n")
    L.append(f"Participantes: **{len(rows)}**  \n"
             f"SUS promedio: **{mean:.1f}** (DE {sd:.1f}; IC 95 % "
             f"{max(0, mean - ci):.1f} – {min(100, mean + ci):.1f})  \n"
             f"Mediana: {statistics.median(scores):.1f} · Mín {min(scores):.1f} · Máx {max(scores):.1f}  \n"
             f"Interpretación: **{adjective(mean)}** "
             f"({'por encima' if mean >= 68 else 'por debajo'} del promedio de referencia 68)\n")

    L.append("## Puntaje por participante\n")
    L.append("| Participante | Perfil | SUS | Adjetivo | Tareas (0-5) | Tiempo (min) |")
    L.append("|---|---|---:|---|---:|---:|")
    for r in rows:
        L.append(f"| {r['participante']} | {r.get('perfil', '')} | {r['sus']:.1f} | "
                 f"{adjective(r['sus'])} | {r.get('tareas_completadas', '')} | {r.get('tiempo_min', '')} |")

    L.append("\n## Promedio por ítem\n")
    L.append("Ítems impares: más alto es mejor. Ítems pares (−): más bajo es mejor. "
             "La columna *aporte* (0–4) normaliza ambos: más alto siempre es mejor.\n")
    L.append("| Ítem | Afirmación | Promedio (1-5) | Aporte (0-4) |")
    L.append("|---|---|---:|---:|")
    contrib = []
    for i in range(1, 11):
        m = statistics.mean(int(r[f"p{i}"]) for r in rows)
        c = (m - 1) if i % 2 == 1 else (5 - m)
        contrib.append((c, i))
        L.append(f"| P{i} | {ITEMS[i - 1]} | {m:.2f} | {c:.2f} |")
    worst = sorted(contrib)[:2]
    best = sorted(contrib, reverse=True)[:2]
    L.append(f"\nFortalezas (aporte más alto): " +
             ", ".join(f"P{i} ({ITEMS[i - 1]})" for _, i in best) + ".  ")
    L.append("Áreas de mejora (aporte más bajo): " +
             ", ".join(f"P{i} ({ITEMS[i - 1]})" for _, i in worst) + ".\n")

    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("perfil") or "sin_perfil").strip()].append(r["sus"])
    if len(groups) > 1:
        L.append("## Por perfil\n")
        L.append("| Perfil | n | SUS promedio |")
        L.append("|---|---:|---:|")
        for g, xs in groups.items():
            L.append(f"| {g} | {len(xs)} | {statistics.mean(xs):.1f} |")
        L.append("")

    tasks = [int(r["tareas_completadas"]) for r in rows if r.get("tareas_completadas", "").strip()]
    if tasks:
        L.append(f"**Tasa de éxito en tareas:** {100 * sum(tasks) / (5 * len(tasks)):.0f} % "
                 f"({sum(tasks)} de {5 * len(tasks)} tareas).\n")

    # ---- comentarios ----
    L.append("## Análisis de comentarios\n")
    L.append("Cada comentario se clasificó por tema con palabras clave (un comentario puede "
             "caer en varios temas). Revisar y ajustar manualmente si alguno quedó mal clasificado.\n")
    tally = defaultdict(lambda: {"gusto": [], "confuso": [], "sugerencia": []})
    for r in rows:
        for field in ("gusto", "confuso", "sugerencia"):
            txt = (r.get(field) or "").strip()
            if not txt:
                continue
            n = norm(txt)
            hit = [t for t, kws in THEMES.items() if any(k in n for k in kws)] or ["Otros"]
            for t in hit:
                tally[t][field].append(f"{txt} *({r['participante']})*")
    L.append("| Tema | Positivos | Problemas | Sugerencias |")
    L.append("|---|---:|---:|---:|")
    order = sorted(tally, key=lambda t: -(len(tally[t]["confuso"]) + len(tally[t]["sugerencia"])))
    for t in order:
        d = tally[t]
        L.append(f"| {t} | {len(d['gusto'])} | {len(d['confuso'])} | {len(d['sugerencia'])} |")
    for t in order:
        L.append(f"\n### {t}\n")
        for field, lab in (("gusto", "Lo que gustó"), ("confuso", "Problemas"),
                           ("sugerencia", "Sugerencias")):
            if tally[t][field]:
                L.append(f"**{lab}:**")
                L.extend(f"- {c}" for c in tally[t][field])
    if order:
        top = order[0]
        L.append(f"\n**Tema con más problemas/sugerencias:** {top}. "
                 "Es la prioridad para la propuesta de mejora.")

    out_md = os.path.join(HERE, "resultados_sus.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5))
        names = [r["participante"] for r in rows]
        ax.bar(names, scores, color="#2a78d6", width=0.6)
        ax.axhline(68, ls="--", color="#9a9a96", lw=1.5)
        ax.text(len(names) - 0.5, 69.5, "referencia 68", ha="right", fontsize=8, color="#52514e")
        ax.axhline(mean, color="#eb6834", lw=2)
        ax.text(-0.4, mean + 1.5, f"promedio {mean:.1f}", fontsize=8, color="#52514e")
        ax.set_ylim(0, 100); ax.set_ylabel("Puntaje SUS")
        ax.set_title("SUS por participante", loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "sus_puntajes.png"), dpi=160)
    except Exception as e:
        print("No se generó la gráfica:", e)
    print(f"SUS promedio = {mean:.1f} ({adjective(mean)}); reporte en {out_md}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "respuestas_sus.csv"))
