# -*- coding: utf-8 -*-
"""
ANÁLISIS MATRICIAL DE ESTRUCTURAS PLANAS  (armaduras, vigas y marcos 2D)
==========================================================================
Método de rigidez directa, 3 grados de libertad por nodo (u, v, theta).

Soporta:
  - Estructuras determinadas e indeterminadas (calcula el grado GI).
  - Apoyos: empotre (1,1,1), articulación (1,1,0), rodillo (0,1,0) o (1,0,0),
    guiados (1,0,1)/(0,1,1) y cualquier combinación.
  - Resortes elásticos en nodos (kx, ky, kz).
  - Asentamientos / giros impuestos en apoyos.
  - Articulaciones internas (liberación de momento) en cualquier extremo de barra.
    Una barra con ambos extremos articulados trabaja como barra de armadura.
  - Cargas nodales (Fx, Fy, M).
  - Cargas en barras: distribuidas (uniformes o trapezoidales, totales o parciales),
    puntuales y momentos concentrados, en dirección local o global (o global proyectada).
  - Cambios de temperatura en barras: uniforme (dT) y gradiente entre caras (dTg).
  - Peso propio automático (peso volumétrico x área).
  - Casos de carga (CM, CV, ...) y combinaciones con factores, con envolvente.
  - Resultados: desplazamientos, reacciones, fuerzas en extremos, diagramas N, V, M,
    deformada, flechas por barra y valores por estaciones; PDF con el desarrollo
    matricial paso a paso y exportación a Excel.
  - Calculadora de propiedades de secciones (rectangular, circular, tubo, I, cajón).

Unidades (usa un sistema consistente):  kN, m, kPa (=kN/m2), m2, m4.
    Acero: E = 200e6 kPa      Concreto f'c 250: E aprox. 22e6 kPa

Convenciones
  - Eje local x: del nodo i al nodo j.  Eje local y: x girado 90 grados antihorario.
  - Fuerzas en extremos: positivas según ejes locales; momentos + antihorario.
  - Diagramas: N + tensión; V + si en el extremo izquierdo la fuerza sube;
    M + si tensiona la fibra del lado -y local (en vigas i->j de izquierda a derecha:
    fibra inferior). El diagrama de momento se dibuja del lado en tensión.

Uso en Colab: pega el archivo en una celda y ejecuta, o  %run analisis_matricial_marcos.py
Uso desde otro programa (p. ej. Streamlit):
    from analisis_matricial_marcos import analizar, generar_pdf
    res = analizar(modelo)          # modelo = dict con el formato de MODELO
    generar_pdf(res, buffer_o_ruta)
    # con combinaciones:
    todos = analizar_combinaciones(modelo)     # {nombre: res}
    env = envolvente(todos)
    exportar_excel(todos, "resultados.xlsx", env)
"""

import sys
import io
import math
import subprocess


def _asegurar(paquete, modulo=None):
    try:
        __import__(modulo or paquete)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", paquete])


_asegurar("numpy")
_asegurar("matplotlib")
_asegurar("reportlab")
_asegurar("pillow", "PIL")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether)
from PIL import Image as PILImage


# =====================================================================
# 1) MODELO DE EJEMPLO  (edita esta sección)
# =====================================================================
MODELO = {
    "titulo": "Marco con voladizo - Análisis matricial",

    # Propiedades por defecto (se pueden sobrescribir en cada barra)
    "E": 200e6,        # kPa
    "A": 0.006,        # m2
    "I": 1.2e-4,       # m4

    # Nodos: {id: (x, y)} en m
    "nodos": {
        1: (0.0, 0.0),
        2: (0.0, 4.0),
        3: (6.0, 4.0),
        4: (6.0, 0.0),
        5: (8.0, 4.0),
    },

    # Barras: {id: {"i": nodo, "j": nodo, opcionales: "E", "A", "I", "art_i", "art_j"}}
    #   art_i / art_j = True  -> articulación interna (momento cero) en ese extremo
    "barras": {
        1: {"i": 1, "j": 2},
        2: {"i": 2, "j": 3, "I": 2.0e-4},
        3: {"i": 4, "j": 3},
        4: {"i": 3, "j": 5, "I": 2.0e-4},
    },

    # Apoyos: {nodo: (rx, ry, rz)}  1 = restringido, 0 = libre
    "apoyos": {
        1: (1, 1, 1),      # empotre
        4: (1, 1, 0),      # articulación
    },

    # Resortes (opcional): {nodo: (kx kN/m, ky kN/m, kz kN.m/rad)}
    "resortes": {},

    # Cargas nodales: {nodo: (Fx kN, Fy kN, M kN.m)}  (+x derecha, +y arriba, M + antihorario)
    "cargas_nodales": {
        2: (15.0, 0.0, 0.0),
        5: (0.0, -10.0, 0.0),
    },

    # Cargas en barras (lista). Campos:
    #   "barra": id
    #   "tipo": "distribuida" | "puntual" | "momento" | "termica"
    #   "dir":  "global_y" | "global_x" | "local_y" | "local_x"
    #           | "global_y_proy" | "global_x_proy"  (por unidad de proyección)
    #   distribuida: "w1", "w2" (kN/m, opcional w2=w1), "a", "b" (m desde el nodo i, opcionales)
    #   puntual:     "P" (kN), "a" (m desde el nodo i)
    #   momento:     "M" (kN.m, + antihorario), "a" (m desde el nodo i)
    #   termica:     "dT" (cambio uniforme, grados C), "dTg" (T cara +y local menos T cara -y local),
    #                "h" (peralte en m, necesario si hay dTg), "alpha" (opcional)
    #   "caso": nombre del caso de carga (opcional, por defecto "CM")
    #   Signo: + en el sentido positivo del eje indicado (global_y negativo = hacia abajo)
    "cargas_barras": [
        {"barra": 2, "tipo": "distribuida", "w1": -20.0, "dir": "global_y"},
        {"barra": 1, "tipo": "distribuida", "w1": 5.0, "dir": "global_x"},
        {"barra": 4, "tipo": "distribuida", "w1": -8.0, "dir": "global_y"},
    ],

    # Asentamientos / giros impuestos (opcional): {nodo: (dx m, dy m, giro rad)}
    "asentamientos": {},

    # ---- Opcionales ----
    # "alpha": 1.2e-5,             # coeficiente de dilatación térmica (1/grado C)
    # "peso_propio": True,         # agrega w = gamma*A hacia abajo en cada barra
    # "gamma": 77.0,               # peso volumétrico kN/m3 (acero 77, concreto 24); también por barra
    # "caso_pp": "CM",             # caso al que pertenece el peso propio
    # Las cargas nodales también pueden darse como lista con caso:
    #   "cargas_nodales": [{"nodo": 2, "Fx": 15, "Fy": 0, "M": 0, "caso": "VIENTO"}]
    # Combinaciones {nombre: {caso: factor}}; los asentamientos entran con factor 1 en todas.
    # "combinaciones": {"Servicio": {"CM": 1.0, "CV": 1.0}, "Resistencia": {"CM": 1.3, "CV": 1.5}},
}

ARCHIVO_PDF = "marco_resultados.pdf"


# =====================================================================
# 2) MOTOR DE CÁLCULO
# =====================================================================
class ErrorEstructura(Exception):
    pass


_GP, _GW = np.polynomial.legendre.leggauss(6)


def k_local(E, A, I, L):
    a = E * A / L
    b = 12 * E * I / L ** 3
    c = 6 * E * I / L ** 2
    d = 4 * E * I / L
    e = 2 * E * I / L
    return np.array([[a, 0, 0, -a, 0, 0],
                     [0, b, c, 0, -b, c],
                     [0, c, d, 0, -c, e],
                     [-a, 0, 0, a, 0, 0],
                     [0, -b, -c, 0, b, -c],
                     [0, c, e, 0, -c, d]], dtype=float)


def matriz_T(c, s):
    T = np.zeros((6, 6))
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    T[:3, :3] = R
    T[3:, 3:] = R
    return T


def _condensar(k, f, r):
    """Condensación estática del GDL r (liberación de momento)."""
    k = k.copy()
    f = f.copy()
    if abs(k[r, r]) > 1e-12 * max(1.0, np.abs(k).max()):
        kc = k - np.outer(k[:, r], k[r, :]) / k[r, r]
        fc = f - k[:, r] * f[r] / k[r, r]
    else:
        kc, fc = k, f
    kc[r, :] = 0.0
    kc[:, r] = 0.0
    fc[r] = 0.0
    return kc, fc


def _dir_vectores(dir_, c, s):
    """Devuelve (componente local x, componente local y, factor proyección, vector global de dibujo)."""
    dir_ = dir_.lower()
    if dir_ == "local_x":
        return 1.0, 0.0, 1.0, (c, s)
    if dir_ == "local_y":
        return 0.0, 1.0, 1.0, (-s, c)
    if dir_ in ("global_x", "global_x_proy"):
        gx, gy = 1.0, 0.0
        fac = abs(s) if dir_.endswith("proy") else 1.0
    elif dir_ in ("global_y", "global_y_proy"):
        gx, gy = 0.0, 1.0
        fac = abs(c) if dir_.endswith("proy") else 1.0
    else:
        raise ErrorEstructura(f"Dirección de carga no reconocida: '{dir_}'")
    return gx * c + gy * s, -gx * s + gy * c, fac, (gx, gy)


def _num(v, defecto=0.0):
    """float(v) tolerando None y NaN."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return defecto
    return defecto if math.isnan(v) else v


def _piezas_de_carga(carga, L, c, s, alpha0=1.2e-5):
    tipo = str(carga.get("tipo") or "distribuida").lower()
    dir_ = str(carga.get("dir") or "global_y")
    tol = 1e-9 * L
    if tipo.startswith("ter") or tipo.startswith("temp"):
        dT, dTg = _num(carga.get("dT")), _num(carga.get("dTg"))
        al = _num(carga.get("alpha"), alpha0) or alpha0
        h = _num(carga.get("h"))
        if dTg != 0 and h <= 0:
            raise ErrorEstructura(f"Carga térmica en barra {carga['barra']}: el gradiente dTg requiere el peralte h > 0.")
        kap = -al * dTg / h if dTg != 0 else 0.0     # curvatura térmica (v'' libre)
        return dict(k="t", eps=al * dT, kap=kap), dict(k="t", dT=dT, dTg=dTg, h=h, alpha=al)
    if tipo.startswith("mom"):
        M0 = float(carga["M"])
        a = _num(carga.get("a"), L / 2)
        if a < -tol or a > L + tol:
            raise ErrorEstructura(f"Momento en barra {carga['barra']}: se requiere 0 <= a <= L = {L:.4g} m.")
        a = min(max(a, 0.0), L)
        return dict(k="m", a=a, M=M0), dict(k="m", a=a, M=M0)
    ux, uy, fac, gvec = _dir_vectores(dir_, c, s)
    if tipo.startswith("dist"):
        w1 = float(carga["w1"])
        w2 = float(carga.get("w2", w1) if carga.get("w2") is not None else w1)
        a = _num(carga.get("a"), 0.0)
        b = _num(carga.get("b"), L)
        if a < -tol or b > L + tol or b - a <= tol:
            raise ErrorEstructura(f"Carga distribuida en barra {carga['barra']}: se requiere 0 <= a < b <= L = {L:.4g} m.")
        a, b = max(a, 0.0), min(b, L)
        pieza = dict(k="d", a=a, b=b,
                     qx1=w1 * fac * ux, qx2=w2 * fac * ux,
                     qy1=w1 * fac * uy, qy2=w2 * fac * uy)
        dib = dict(k="d", a=a, b=b, w1=w1, w2=w2, gvec=gvec, dir=dir_, pp=bool(carga.get("pp")))
    elif tipo.startswith("pun"):
        P = float(carga["P"])
        a = _num(carga.get("a"), L / 2)
        if a < -tol or a > L + tol:
            raise ErrorEstructura(f"Carga puntual en barra {carga['barra']}: se requiere 0 <= a <= L = {L:.4g} m.")
        a = min(max(a, 0.0), L)
        pieza = dict(k="p", a=a, Px=P * ux, Py=P * uy)
        dib = dict(k="p", a=a, P=P, gvec=gvec, dir=dir_)
    else:
        raise ErrorEstructura(f"Tipo de carga no reconocido: '{tipo}'")
    return pieza, dib


def _equivalente(piezas, L):
    """Vector de cargas nodales equivalentes (local) con funciones de forma de Hermite."""
    q = np.zeros(6)

    def formas(r):
        return (1 - r, 1 - 3 * r ** 2 + 2 * r ** 3, L * (r - 2 * r ** 2 + r ** 3),
                r, 3 * r ** 2 - 2 * r ** 3, L * (-r ** 2 + r ** 3))

    for p in piezas:
        if p["k"] == "t":
            continue                                  # la temperatura se trata aparte (depende de EA, EI)
        if p["k"] == "m":                             # trabajo = M0 * dv/dx  ->  derivadas de Hermite
            r = p["a"] / L
            q += [0.0, p["M"] * (-6 * r + 6 * r ** 2) / L, p["M"] * (1 - 4 * r + 3 * r ** 2),
                  0.0, p["M"] * (6 * r - 6 * r ** 2) / L, p["M"] * (-2 * r + 3 * r ** 2)]
            continue
        if p["k"] == "d":
            a, b = p["a"], p["b"]
            xs = (b - a) / 2 * _GP + (a + b) / 2
            ws = (b - a) / 2 * _GW
            t = (xs - a) / (b - a)
            qx = p["qx1"] + (p["qx2"] - p["qx1"]) * t
            qy = p["qy1"] + (p["qy2"] - p["qy1"]) * t
            Na1, N1, N2, Na2, N3, N4 = formas(xs / L)
            q += [np.sum(ws * qx * Na1), np.sum(ws * qy * N1), np.sum(ws * qy * N2),
                  np.sum(ws * qx * Na2), np.sum(ws * qy * N3), np.sum(ws * qy * N4)]
        else:
            Na1, N1, N2, Na2, N3, N4 = formas(p["a"] / L)
            q += [p["Px"] * Na1, p["Py"] * N1, p["Py"] * N2,
                  p["Px"] * Na2, p["Py"] * N3, p["Py"] * N4]
    return q


def _acumulados(piezas, x):
    """Fx(x)=int qx, Fy(x)=int qy, Gy(x)=int qy(x-xi) de 0 a x (forma cerrada).
    Mc(x) = -(suma de momentos concentrados aplicados antes de x)."""
    Fx = np.zeros_like(x)
    Fy = np.zeros_like(x)
    Gy = np.zeros_like(x)
    Mc = np.zeros_like(x)
    for p in piezas:
        if p["k"] == "t":
            continue
        if p["k"] == "m":
            Mc -= p["M"] * (x > p["a"]).astype(float)
            continue
        if p["k"] == "d":
            a, b = p["a"], p["b"]
            t = np.clip(x, a, b)
            for comp, q1, q2 in (("x", p["qx1"], p["qx2"]), ("y", p["qy1"], p["qy2"])):
                if q1 == 0 and q2 == 0:
                    continue
                beta = (q2 - q1) / (b - a)
                alpha = q1 - beta * a
                I0 = alpha * (t - a) + beta * (t ** 2 - a ** 2) / 2
                if comp == "x":
                    Fx += I0
                else:
                    I1 = alpha * (t ** 2 - a ** 2) / 2 + beta * (t ** 3 - a ** 3) / 3
                    Fy += I0
                    Gy += x * I0 - I1
        else:
            H = (x > p["a"]).astype(float)
            Fx += p["Px"] * H
            Fy += p["Py"] * H
            Gy += p["Py"] * (x - p["a"]) * H
    return Fx, Fy, Gy, Mc


def _resultantes(piezas):
    """Resultantes locales (Fx, Fy) y momento respecto al nodo i."""
    FX = FY = MI = 0.0
    for p in piezas:
        if p["k"] == "t":
            continue
        if p["k"] == "m":
            MI += p["M"]
            continue
        if p["k"] == "d":
            x = np.array([p["b"]])
            fx, fy, gy, _ = _acumulados([p], x)
            FX += fx[0]
            FY += fy[0]
            MI += p["b"] * fy[0] - gy[0]          # = int qy * xi
        else:
            FX += p["Px"]
            FY += p["Py"]
            MI += p["Py"] * p["a"]
    return FX, FY, MI


def _malla(L, piezas, n=241):
    x = list(np.linspace(0.0, L, n))
    eps = L * 1e-7
    for p in piezas:
        if p["k"] == "t":
            continue
        pts = [p["a"], p["b"]] if p["k"] == "d" else [p["a"]]
        for a in pts:
            if eps < a < L - eps:
                x += [a - eps, a, a + eps]
    return np.unique(np.array(x))


def _cumtrapz(y, x):
    out = np.zeros_like(y)
    out[1:] = np.cumsum((y[1:] + y[:-1]) / 2 * np.diff(x))
    return out


CASO_DEF = "CM"
GAMMA_ACERO = 77.0          # kN/m3
GAMMA_CONCRETO = 24.0       # kN/m3
SIN_COMBINACION = "Todas las cargas"


def _caso(v):
    v = "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip().upper()
    return v or CASO_DEF


def _factores(modelo, combinacion):
    """Devuelve (nombre, {caso: factor}) o (None, None) si se analizan todas las cargas tal cual."""
    if combinacion is None:
        return None, None
    if isinstance(combinacion, dict):
        return "Personalizada", {_caso(k): float(v) for k, v in combinacion.items()}
    combos = modelo.get("combinaciones") or {}
    if combinacion not in combos:
        raise ErrorEstructura(f"La combinación '{combinacion}' no existe en el modelo.")
    return str(combinacion), {_caso(k): float(v) for k, v in combos[combinacion].items()}


def _escalar(c, f):
    c = dict(c)
    for k in ("w1", "w2", "P", "M", "dT", "dTg"):
        if c.get(k) is not None:
            c[k] = _num(c[k]) * f
    return c


def _normalizar(modelo, factores=None):
    nodos = {int(k): (float(v[0]), float(v[1])) for k, v in modelo["nodos"].items()}
    E0, A0, I0 = modelo.get("E"), modelo.get("A"), modelo.get("I")
    pp = bool(modelo.get("peso_propio"))
    gamma0 = _num(modelo.get("gamma"), GAMMA_ACERO)
    barras = {}
    for b, d in modelo["barras"].items():
        if isinstance(d, (tuple, list)):
            d = {"i": d[0], "j": d[1]}
        E = d.get("E", E0)
        A = d.get("A", A0)
        I = d.get("I", I0)
        if E is None or A is None or I is None:
            raise ErrorEstructura(f"La barra {b} no tiene E, A o I (ni valores por defecto).")
        if float(E) <= 0 or float(A) <= 0 or float(I) < 0:
            raise ErrorEstructura(f"La barra {b} tiene E, A o I no válidos.")
        barras[int(b)] = dict(i=int(d["i"]), j=int(d["j"]), E=float(E), A=float(A), I=float(I),
                              art_i=bool(d.get("art_i", False)), art_j=bool(d.get("art_j", False)),
                              gamma=_num(d.get("gamma"), gamma0))

    def trip(v):
        v = list(v) + [0] * (3 - len(v))
        return tuple(v[:3])

    apoyos = {int(k): tuple(int(bool(x)) for x in trip(v)) for k, v in modelo.get("apoyos", {}).items()}
    apoyos = {k: v for k, v in apoyos.items() if any(v)}
    resortes = {int(k): tuple(float(x) for x in trip(v)) for k, v in (modelo.get("resortes") or {}).items()}
    resortes = {k: v for k, v in resortes.items() if any(abs(x) > 0 for x in v)}
    asent = {int(k): tuple(float(x) for x in trip(v)) for k, v in (modelo.get("asentamientos") or {}).items()}

    # cargas nodales: dict {nodo: (Fx, Fy, M)} (caso CM) o lista de dicts con "caso"
    cn_raw = modelo.get("cargas_nodales") or {}
    if isinstance(cn_raw, dict):
        cn_lista = [dict(nodo=k, Fx=trip(v)[0], Fy=trip(v)[1], M=trip(v)[2], caso=CASO_DEF) for k, v in cn_raw.items()]
    else:
        cn_lista = [dict(c) for c in cn_raw]
    cb = [dict(c) for c in (modelo.get("cargas_barras") or [])]
    if pp:
        caso_pp = _caso(modelo.get("caso_pp"))
        for b in sorted(barras):
            w = barras[b]["gamma"] * barras[b]["A"]
            if w > 0:
                cb.append({"barra": b, "tipo": "distribuida", "dir": "global_y", "w1": -w, "caso": caso_pp, "pp": True})

    casos = sorted({_caso(c.get("caso")) for c in cn_lista} | {_caso(c.get("caso")) for c in cb})

    def fac(c):
        return 1.0 if factores is None else factores.get(_caso(c.get("caso")), 0.0)

    cn = {}
    for c in cn_lista:
        f = fac(c)
        n = int(c["nodo"])
        v = (_num(c.get("Fx")) * f, _num(c.get("Fy")) * f, _num(c.get("M")) * f)
        if any(v):
            cn[n] = tuple(a + b_ for a, b_ in zip(cn.get(n, (0.0, 0.0, 0.0)), v))
        elif n not in nodos:
            cn[n] = (0.0, 0.0, 0.0)                    # para que la validación lo detecte
    cb = [_escalar(c, fac(c)) for c in cb if fac(c) != 0]
    return nodos, barras, apoyos, resortes, cn, cb, asent, casos


def analizar(modelo, combinacion=None):
    """Resuelve la estructura.

    combinacion: None (todas las cargas con factor 1), el nombre de una combinación de
    modelo["combinaciones"], o un dict {caso: factor}.
    """
    nombre_combo, factores = _factores(modelo, combinacion)
    nodos, barras, apoyos, resortes, cn, cb, asent, casos = _normalizar(modelo, factores)
    alpha0 = _num(modelo.get("alpha"), 1.2e-5)
    if len(nodos) < 2 or len(barras) < 1:
        raise ErrorEstructura("Se requieren al menos 2 nodos y 1 barra.")

    ids = sorted(nodos)
    idx = {n: k for k, n in enumerate(ids)}
    nn = len(ids)
    ndof = 3 * nn

    # --- validaciones ---
    for b, d in barras.items():
        for n in (d["i"], d["j"]):
            if n not in nodos:
                raise ErrorEstructura(f"La barra {b} usa el nodo {n}, que no existe.")
        if d["i"] == d["j"]:
            raise ErrorEstructura(f"La barra {b} conecta el nodo {d['i']} consigo mismo.")
    for nombre, dic in (("apoyo", apoyos), ("resorte", resortes), ("carga nodal", cn), ("asentamiento", asent)):
        for n in dic:
            if n not in nodos:
                raise ErrorEstructura(f"El {nombre} está en el nodo {n}, que no existe.")
    for c in cb:
        if int(c.get("barra", -1)) not in barras:
            raise ErrorEstructura(f"Carga en barra {c.get('barra')}: esa barra no existe.")
    usados = {n for d in barras.values() for n in (d["i"], d["j"])}
    sueltos = [n for n in ids if n not in usados]
    if sueltos:
        raise ErrorEstructura(f"Nodos sin ninguna barra conectada: {sueltos}")

    # --- barras ---
    miembros = []
    for b in sorted(barras):
        d = barras[b]
        i, j = d["i"], d["j"]
        xi, yi = nodos[i]
        xj, yj = nodos[j]
        L = math.hypot(xj - xi, yj - yi)
        if L < 1e-9:
            raise ErrorEstructura(f"La barra {b} tiene longitud cero.")
        c, s = (xj - xi) / L, (yj - yi) / L
        piezas, dibujos = [], []
        for carga in cb:
            if int(carga["barra"]) == b:
                p, dib = _piezas_de_carga(carga, L, c, s, alpha0)
                piezas.append(p)
                dibujos.append(dib)
        kl0 = k_local(d["E"], d["A"], d["I"], L)
        qeq0 = _equivalente(piezas, L)
        # temperatura: deformación axial libre eps_T y curvatura libre kap_T (constantes en la barra)
        eps_T = sum(p["eps"] for p in piezas if p["k"] == "t")
        kap_T = sum(p["kap"] for p in piezas if p["k"] == "t")
        EA_, EI_ = d["E"] * d["A"], d["E"] * d["I"]
        fef_T = np.array([EA_ * eps_T, 0.0, EI_ * kap_T, -EA_ * eps_T, 0.0, -EI_ * kap_T])
        fef0 = -qeq0 + fef_T
        kl, fef = kl0.copy(), fef0.copy()
        if d["art_i"]:
            kl, fef = _condensar(kl, fef, 2)
        if d["art_j"]:
            kl, fef = _condensar(kl, fef, 5)
        T = matriz_T(c, s)
        kg = T.T @ kl @ T
        dofs = [3 * idx[i], 3 * idx[i] + 1, 3 * idx[i] + 2, 3 * idx[j], 3 * idx[j] + 1, 3 * idx[j] + 2]
        miembros.append(dict(id=b, i=i, j=j, E=d["E"], A=d["A"], I=d["I"], L=L, c=c, s=s,
                             art_i=d["art_i"], art_j=d["art_j"], kl0=kl0, kl=kl, T=T, kg=kg,
                             fef0=fef0, fef=fef, peq_g=T.T @ (-fef), dofs=dofs,
                             piezas=piezas, dibujos=dibujos, eps_T=eps_T, kap_T=kap_T, gamma=d["gamma"]))

    # --- ensamble ---
    K = np.zeros((ndof, ndof))
    Peq = np.zeros(ndof)
    for m in miembros:
        dd = m["dofs"]
        K[np.ix_(dd, dd)] += m["kg"]
        Peq[dd] += m["peq_g"]
    Kest = K.copy()                           # sin resortes (para el reporte)

    Pn = np.zeros(ndof)
    for n, v in cn.items():
        Pn[3 * idx[n]:3 * idx[n] + 3] += v
    P = Pn + Peq

    restr = np.zeros(ndof, dtype=bool)
    for n, v in apoyos.items():
        restr[3 * idx[n]:3 * idx[n] + 3] = np.array(v, dtype=bool)

    kres = np.zeros(ndof)
    for n, v in resortes.items():
        for q in range(3):
            if v[q] != 0:
                g = 3 * idx[n] + q
                if restr[g]:
                    raise ErrorEstructura(f"El nodo {n} tiene resorte y apoyo en la misma dirección.")
                kres[g] = v[q]
                K[g, g] += v[q]

    ur = np.zeros(ndof)
    for n, v in asent.items():
        for q in range(3):
            g = 3 * idx[n] + q
            if v[q] != 0 and not restr[g]:
                raise ErrorEstructura(f"Desplazamiento impuesto en el nodo {n} (componente {q+1}) "
                                      f"pero ese GDL no está restringido.")
            ur[g] = v[q]

    # --- giros de nodos donde todas las barras están articuladas ---
    diag_max = max(np.abs(np.diag(K)).max(), 1e-30)
    eliminados = []
    for n in ids:
        g = 3 * idx[n] + 2
        if not restr[g] and abs(K[g, g]) < 1e-10 * diag_max:
            if abs(Pn[g]) > 0:
                raise ErrorEstructura(f"Hay un momento aplicado en el nodo {n}, pero todas las barras "
                                      f"llegan articuladas y el nodo no tiene restricción al giro (inestable).")
            eliminados.append(g)

    fijos_mask = restr.copy()
    fijos_mask[eliminados] = True
    libres = np.where(~fijos_mask)[0]
    fijos = np.where(restr)[0]
    fijos_todos = np.where(fijos_mask)[0]

    # --- grado de indeterminación ---
    inc = sum(3 - int(m["art_i"]) - int(m["art_j"]) for m in miembros)
    r = int(restr.sum())
    nres = int((kres != 0).sum())
    ecs = 3 * nn - len(eliminados)
    GI = inc + r + nres - ecs

    Kff = K[np.ix_(libres, libres)]
    Kfr = K[np.ix_(libres, fijos_todos)]
    u = np.zeros(ndof)
    u[fijos_todos] = ur[fijos_todos]
    Pf = P[libres]
    if len(libres) > 0:
        w = np.linalg.eigvalsh(Kff)
        if w.max() <= 0 or np.sum(w < 1e-10 * w.max()) > 0:
            n_mec = int(np.sum(w < 1e-10 * max(w.max(), 1e-30)))
            raise ErrorEstructura(
                f"La estructura es INESTABLE: {n_mec} modo(s) de mecanismo o cuerpo rígido. "
                f"Revisa apoyos, articulaciones y conectividad (GI calculado = {GI}).")
        rhs = Pf - Kfr @ ur[fijos_todos]
        uf = np.linalg.solve(Kff, rhs)
        u[libres] = uf
        cond = float(w.max() / w.min())
    else:                                   # todo restringido (p. ej. viga empotrada de una sola barra)
        rhs = np.zeros(0)
        uf = np.zeros(0)
        cond = 1.0

    Rtot = K @ u - P
    reac = np.zeros(ndof)
    reac[fijos] = Rtot[fijos]
    fres = -kres * u                          # fuerza que el resorte ejerce sobre la estructura

    # --- fuerzas en barras y diagramas ---
    for m in miembros:
        ul = m["T"] @ u[m["dofs"]]
        f = m["kl"] @ ul + m["fef"]
        m["u_loc"] = ul
        m["k_u"] = m["kl"] @ ul
        m["f"] = f
        x = _malla(m["L"], m["piezas"])
        Fx, Fy, Gy, Mc = _acumulados(m["piezas"], x)
        N = -(f[0] + Fx)
        V = f[1] + Fy
        M = -f[2] + f[1] * x + Gy + Mc
        esc0 = 1e-9 * max(np.abs(P).max(), np.abs(f).max(), 1.0)
        for arr in (N, V, M, f):
            arr[np.abs(arr) < esc0] = 0.0
        m["f"] = f
        m["x"], m["N"], m["V"], m["M"] = x, N, V, M
        # deformada: v'' = M/EI + kap_T ;  u' = N/EA + eps_T
        EI = m["E"] * m["I"]
        if EI > 0:
            v0 = _cumtrapz(_cumtrapz(M / EI + m["kap_T"], x), x)
        else:
            v0 = np.zeros_like(x)
        vi, vj = ul[1], ul[4]
        v = v0 + vi + (vj - vi - v0[-1]) * x / m["L"]
        ua = ul[0] + _cumtrapz(N / (m["E"] * m["A"]) + m["eps_T"], x)
        m["def_u"], m["def_v"] = ua, v
        # flecha relativa a la cuerda (línea recta entre los extremos desplazados)
        rel = v - (vi + (vj - vi) * x / m["L"])
        kf = int(np.argmax(np.abs(rel)))
        fl = float(rel[kf])
        m["flecha"] = dict(valor=fl, x=float(x[kf]),
                           relacion=(m["L"] / abs(fl) if abs(fl) > 1e-12 * m["L"] else float("inf")))
        tot = np.hypot(ua, v)
        kt = int(np.argmax(tot))
        m["desp_max"] = (float(tot[kt]), float(x[kt]))
        m["cierre"] = max(abs(N[-1] - f[3]), abs(V[-1] + f[4]), abs(M[-1] - f[5]))
        for key in ("N", "V", "M"):
            arr = m[key]
            kmax, kmin = int(np.argmax(arr)), int(np.argmin(arr))
            m[key + "_max"] = (float(arr[kmax]), float(x[kmax]))
            m[key + "_min"] = (float(arr[kmin]), float(x[kmin]))
        m["FX"], m["FY"], m["MI"] = _resultantes(m["piezas"])

    # --- equilibrio global ---
    sfx = sfy = sm = 0.0
    for n in ids:
        g = 3 * idx[n]
        x0, y0 = nodos[n]
        fx = Pn[g] + reac[g] + fres[g]
        fy = Pn[g + 1] + reac[g + 1] + fres[g + 1]
        mz = Pn[g + 2] + reac[g + 2] + fres[g + 2]
        sfx += fx
        sfy += fy
        sm += x0 * fy - y0 * fx + mz
    for m in miembros:
        xi, yi = nodos[m["i"]]
        gx = m["FX"] * m["c"] - m["FY"] * m["s"]
        gy = m["FX"] * m["s"] + m["FY"] * m["c"]
        sfx += gx
        sfy += gy
        sm += xi * gy - yi * gx + m["MI"]
    escala_f = max(1.0, np.abs(Pn).max(), max([abs(m["FX"]) + abs(m["FY"]) for m in miembros] + [0]))

    return dict(titulo=modelo.get("titulo", "Análisis matricial de estructura plana"),
                ids=ids, idx=idx, nodos=nodos, miembros=miembros, K=K, Kest=Kest, P=P, Pn=Pn,
                Peq=Peq, restr=restr, libres=libres, fijos=fijos, eliminados=eliminados,
                Kff=Kff, Pf=Pf, rhs=rhs, uf=uf, u=u, reac=reac, fres=fres, kres=kres,
                GI=GI, inc=inc, r=r, nres=nres, ecs=ecs, j=nn, m=len(miembros),
                cond=cond, equilibrio=(sfx, sfy, sm), escala_f=escala_f,
                apoyos=apoyos, resortes=resortes, cargas_nodales=cn, asentamientos=asent,
                combinacion=nombre_combo, factores=factores, casos=casos, alpha=alpha0,
                peso_propio=bool(modelo.get("peso_propio")))


def equilibrio_ok(res):
    sfx, sfy, sm = res["equilibrio"]
    return max(abs(sfx), abs(sfy), abs(sm)) < 1e-6 * res["escala_f"] * max(1.0, _dims(res))


def analizar_combinaciones(modelo):
    """Analiza todas las combinaciones del modelo. Devuelve {nombre: res}.
    Si el modelo no define combinaciones, devuelve una sola entrada con todas las cargas."""
    combos = modelo.get("combinaciones") or {}
    if not combos:
        return {SIN_COMBINACION: analizar(modelo)}
    return {str(nombre): analizar(modelo, nombre) for nombre in combos}


def estaciones(m, n=11):
    """Valores de N, V, M y desplazamientos locales en n puntos equidistantes de la barra."""
    xs = np.linspace(0.0, m["L"], n)
    out = {"x": xs}
    for clave, arr in (("N", m["N"]), ("V", m["V"]), ("M", m["M"]), ("u", m["def_u"]), ("v", m["def_v"])):
        out[clave] = np.interp(xs, m["x"], arr)
    return out


def envolvente(resultados, n=41):
    """Envolvente de varias combinaciones. resultados = {nombre: res}.

    Devuelve dict con:
      barras[id]   -> x, y para N, V, M: máx/mín por estación y extremos globales con su combinación
      reacciones[(nodo, q)] -> (máx, combinación, mín, combinación)   q = 0 Rx, 1 Ry, 2 Mz
      flechas[id]  -> (flecha relativa de mayor valor absoluto, x, combinación)
    """
    nombres = list(resultados)
    r0 = resultados[nombres[0]]
    env = {"combinaciones": nombres, "barras": {}, "reacciones": {}, "flechas": {}}
    for k, m0 in enumerate(r0["miembros"]):
        xs = np.linspace(0.0, m0["L"], n)
        d = {"x": xs}
        for clave in ("N", "V", "M"):
            curvas = np.array([np.interp(xs, resultados[c]["miembros"][k]["x"], resultados[c]["miembros"][k][clave])
                               for c in nombres])
            d[clave + "_max"], d[clave + "_min"] = curvas.max(axis=0), curvas.min(axis=0)
            mx = [(resultados[c]["miembros"][k][clave + "_max"], c) for c in nombres]
            mn = [(resultados[c]["miembros"][k][clave + "_min"], c) for c in nombres]
            (vmax, xmax), cmax = max(mx, key=lambda t: t[0][0])
            (vmin, xmin), cmin = min(mn, key=lambda t: t[0][0])
            d[clave] = dict(max=vmax, x_max=xmax, combo_max=cmax, min=vmin, x_min=xmin, combo_min=cmin)
        env["barras"][m0["id"]] = d
        fl = [(resultados[c]["miembros"][k]["flecha"], c) for c in nombres]
        f, c = max(fl, key=lambda t: abs(t[0]["valor"]))
        env["flechas"][m0["id"]] = (f["valor"], f["x"], c)
    for n_ in r0["ids"]:
        g = 3 * r0["idx"][n_]
        for q in range(3):
            if r0["restr"][g + q]:
                vals = [(float(resultados[c]["reac"][g + q]), c) for c in nombres]
                vmax, cmax = max(vals, key=lambda t: t[0])
                vmin, cmin = min(vals, key=lambda t: t[0])
                env["reacciones"][(n_, q)] = (vmax, cmax, vmin, cmin)
    return env


# =====================================================================
# 2b) SECCIONES Y MATERIALES  (dimensiones en m -> A en m2, I en m4)
# =====================================================================
def seccion_rectangular(b, h):
    return dict(A=b * h, I=b * h ** 3 / 12)


def seccion_circular(d):
    return dict(A=math.pi * d ** 2 / 4, I=math.pi * d ** 4 / 64)


def seccion_tubo_circular(d, t):
    di = d - 2 * t
    if di < 0:
        raise ValueError("El espesor no puede ser mayor que el radio.")
    return dict(A=math.pi * (d ** 2 - di ** 2) / 4, I=math.pi * (d ** 4 - di ** 4) / 64)


def seccion_cajon(b, h, t):
    """Tubo rectangular (PTR/HSS) de ancho b, peralte h y espesor t."""
    bi, hi = b - 2 * t, h - 2 * t
    if bi < 0 or hi < 0:
        raise ValueError("El espesor es demasiado grande para la sección.")
    return dict(A=b * h - bi * hi, I=(b * h ** 3 - bi * hi ** 3) / 12)


def seccion_I(d, bf, tf, tw):
    """Perfil I/H de peralte total d, patín bf x tf y alma tw (flexión respecto al eje fuerte)."""
    hw = d - 2 * tf
    if hw < 0 or tw > bf:
        raise ValueError("Dimensiones del perfil I no válidas.")
    return dict(A=2 * bf * tf + hw * tw, I=(bf * d ** 3 - (bf - tw) * hw ** 3) / 12)


def E_concreto(fc_MPa, criterio="ACI"):
    """Módulo de elasticidad del concreto en kPa a partir de f'c en MPa.
    "ACI": 4700*raiz(f'c).   "NTC": 4400*raiz(f'c) (concreto clase 1, agregado calizo)."""
    k = 4400.0 if str(criterio).upper().startswith("NTC") else 4700.0
    return k * math.sqrt(fc_MPa) * 1000.0


# =====================================================================
# 3) FIGURAS
# =====================================================================
C_BARRA = "#1f3b57"
C_CARGA = "#c0392b"


def _dims(res):
    xs = [p[0] for p in res["nodos"].values()]
    ys = [p[1] for p in res["nodos"].values()]
    return max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)


def _encuadre(ax, res, frac=0.45):
    """Evita figuras aplastadas en vigas horizontales: alto mínimo proporcional al tamaño."""
    D = _dims(res)
    for get, set_ in ((ax.get_ylim, ax.set_ylim), (ax.get_xlim, ax.set_xlim)):
        a, b = get()
        if b - a < frac * D:
            c = (a + b) / 2
            set_(c - frac * D / 2, c + frac * D / 2)


def _png(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _apoyos(ax, res, t):
    for n, (rx, ry, rz) in res["apoyos"].items():
        x, y = res["nodos"][n]
        if rx and ry and rz:
            ax.plot([x - t, x + t], [y, y], "k-", lw=2.2, zorder=4)
            ax.add_patch(Rectangle((x - t, y - 0.6 * t), 2 * t, 0.6 * t, fill=False,
                                   hatch="////", ec="k", lw=0, zorder=3))
        elif rx and ry:
            ax.add_patch(Polygon([[x, y], [x - 0.7 * t, y - 1.2 * t], [x + 0.7 * t, y - 1.2 * t]],
                                 closed=True, fc="#dddddd", ec="k", lw=1, zorder=3))
        elif ry and not rx:
            ax.add_patch(Polygon([[x, y], [x - 0.7 * t, y - t], [x + 0.7 * t, y - t]],
                                 closed=True, fc="#dddddd", ec="k", lw=1, zorder=3))
            for dx in (-0.35, 0.35):
                ax.add_patch(Circle((x + dx * t, y - 1.22 * t), 0.2 * t, fc="w", ec="k", zorder=3))
            if rz:
                ax.plot([x - t, x + t], [y, y], "k-", lw=2.2, zorder=4)
        elif rx and not ry:
            ax.add_patch(Polygon([[x, y], [x - t, y - 0.7 * t], [x - t, y + 0.7 * t]],
                                 closed=True, fc="#dddddd", ec="k", lw=1, zorder=3))
            for dy in (-0.35, 0.35):
                ax.add_patch(Circle((x - 1.22 * t, y + dy * t), 0.2 * t, fc="w", ec="k", zorder=3))
            if rz:
                ax.plot([x, x], [y - t, y + t], "k-", lw=2.2, zorder=4)
        else:   # solo giro
            ax.add_patch(Rectangle((x - 0.5 * t, y - 0.5 * t), t, t, fc="#dddddd", ec="k", zorder=3))
    for n, v in res["resortes"].items():
        x, y = res["nodos"][n]
        ax.text(x + 0.8 * t, y - 0.8 * t, "resorte", fontsize=6.5, color="#6c3483", zorder=6)


def _articulaciones(ax, res, t):
    for m in res["miembros"]:
        xi, yi = res["nodos"][m["i"]]
        d = min(0.9 * t, 0.2 * m["L"])
        if m["art_i"]:
            ax.add_patch(Circle((xi + m["c"] * d, yi + m["s"] * d), 0.28 * t, fc="w", ec="k", lw=1.2, zorder=6))
        if m["art_j"]:
            xj, yj = res["nodos"][m["j"]]
            ax.add_patch(Circle((xj - m["c"] * d, yj - m["s"] * d), 0.28 * t, fc="w", ec="k", lw=1.2, zorder=6))


def _base(ax, res, color=C_BARRA, lw=2.2, etiquetas=True, t=None):
    D = _dims(res)
    t = t or D * 0.03
    for m in res["miembros"]:
        xi, yi = res["nodos"][m["i"]]
        xj, yj = res["nodos"][m["j"]]
        ax.plot([xi, xj], [yi, yj], "-", color=color, lw=lw, zorder=2, solid_capstyle="round")
    _apoyos(ax, res, t)
    _articulaciones(ax, res, t)
    for n, (x, y) in res["nodos"].items():
        ax.plot(x, y, "o", color="k", ms=3.5, zorder=7)
        if etiquetas:
            ax.text(x + 0.5 * t, y + 0.5 * t, str(n), fontsize=8.5, fontweight="bold", zorder=8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return D, t


def texto_combinacion(factores):
    """{'CM': 1.3, 'CV': 1.5} -> '1.3 CM + 1.5 CV'."""
    if not factores:
        return "todas las cargas con factor 1"
    out = ""
    for k, (caso, f) in enumerate(factores.items()):
        signo = (" - " if k else "-") if f < 0 else (" + " if k else "")
        out += f"{signo}{abs(f):g} {caso}"
    return out


def fig_modelo(res):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    D, t = _base(ax, res)
    for m in res["miembros"]:
        xi, yi = res["nodos"][m["i"]]
        xj, yj = res["nodos"][m["j"]]
        ax.text((xi + xj) / 2 - m["s"] * 0.6 * t, (yi + yj) / 2 + m["c"] * 0.6 * t, f"({m['id']})",
                color="#555", fontsize=7.5, ha="center", va="center", zorder=5)
    # cargas en barras
    wmax = max([max(abs(d["w1"]), abs(d["w2"])) for m in res["miembros"] for d in m["dibujos"]
                if d["k"] == "d" and not d.get("pp")] + [1e-12])
    Pmax = max([abs(d["P"]) for m in res["miembros"] for d in m["dibujos"] if d["k"] == "p"]
               + [math.hypot(v[0], v[1]) for v in res["cargas_nodales"].values()] + [1e-12])
    for m in res["miembros"]:
        xi, yi = res["nodos"][m["i"]]
        for d in m["dibujos"]:
            if d.get("pp"):
                continue                               # el peso propio no se dibuja (se indica en el título)
            if d["k"] == "m":
                px, py = xi + m["c"] * d["a"], yi + m["s"] * d["a"]
                ax.plot(px, py, "o", color=C_CARGA, ms=5, zorder=7)
                ax.text(px, py + 1.3 * t, f"{'↺' if d['M'] > 0 else '↻'} {abs(d['M']):g} kN·m", color=C_CARGA,
                        fontsize=7.5, ha="center", va="bottom", zorder=6,
                        bbox=dict(fc="w", ec="none", alpha=0.8, pad=0.5))
                continue
            if d["k"] == "t":
                px, py = xi + m["c"] * m["L"] * 0.5, yi + m["s"] * m["L"] * 0.5
                partes = ([f"ΔT = {d['dT']:g} °C"] if d["dT"] else []) + ([f"ΔTg = {d['dTg']:g} °C"] if d["dTg"] else [])
                ax.text(px + m["s"] * 1.6 * t, py - m["c"] * 1.6 * t, ", ".join(partes), color="#8e44ad",
                        fontsize=7, ha="center", va="center", zorder=6,
                        bbox=dict(fc="w", ec="#8e44ad", lw=0.5, alpha=0.9, pad=1.0))
                continue
            gx, gy = d["gvec"]
            if d["k"] == "d":
                colas = []
                for xx in np.linspace(d["a"], d["b"], 7):
                    w = d["w1"] + (d["w2"] - d["w1"]) * (xx - d["a"]) / (d["b"] - d["a"])
                    lng = 0.12 * D * abs(w) / wmax
                    sg = 1 if w >= 0 else -1
                    px, py = xi + m["c"] * xx, yi + m["s"] * xx
                    tx, ty = px - sg * gx * lng, py - sg * gy * lng
                    colas.append((tx, ty))
                    if lng > 1e-9:
                        ax.annotate("", xy=(px, py), xytext=(tx, ty), zorder=5,
                                    arrowprops=dict(arrowstyle="-|>", color=C_CARGA, lw=0.9, mutation_scale=7))
                cx, cy = zip(*colas)
                ax.plot(cx, cy, "-", color=C_CARGA, lw=0.9, zorder=5)
                txt = f"{d['w1']:g}" if d["w1"] == d["w2"] else f"{d['w1']:g} a {d['w2']:g}"
                ax.text(cx[3], cy[3], f"{txt} kN/m", color=C_CARGA, fontsize=7, ha="center",
                        va="bottom" if gy <= 0 else "top", zorder=6,
                        bbox=dict(fc="w", ec="none", alpha=0.8, pad=0.5))
            else:
                lng = 0.16 * D * (0.5 + 0.5 * abs(d["P"]) / Pmax)
                sg = 1 if d["P"] >= 0 else -1
                px, py = xi + m["c"] * d["a"], yi + m["s"] * d["a"]
                tx, ty = px - sg * gx * lng, py - sg * gy * lng
                ax.annotate("", xy=(px, py), xytext=(tx, ty), zorder=5,
                            arrowprops=dict(arrowstyle="-|>", color=C_CARGA, lw=1.5))
                ax.text(tx, ty, f"{d['P']:g} kN", color=C_CARGA, fontsize=7, ha="center", va="center",
                        bbox=dict(fc="w", ec="none", alpha=0.8, pad=0.5), zorder=6)
    # cargas nodales
    for n, (fx, fy, mz) in res["cargas_nodales"].items():
        x, y = res["nodos"][n]
        F = math.hypot(fx, fy)
        if F > 0:
            lng = 0.16 * D * (0.5 + 0.5 * F / Pmax)
            tx, ty = x - fx / F * lng, y - fy / F * lng
            ax.annotate("", xy=(x, y), xytext=(tx, ty), zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color=C_CARGA, lw=1.6))
            etq = f"{math.hypot(fx, fy):g} kN" if (fx == 0 or fy == 0) else f"({fx:g}, {fy:g}) kN"
            ax.text(tx, ty, etq, color=C_CARGA, fontsize=7, ha="center", va="center",
                    bbox=dict(fc="w", ec="none", alpha=0.8, pad=0.5), zorder=6)
        if mz != 0:
            ax.text(x - 1.2 * t, y + 1.0 * t, f"{'↺' if mz > 0 else '↻'} {abs(mz):g} kN·m",
                    color=C_CARGA, fontsize=7.5, zorder=6)
    ax.margins(0.15)
    tit = "Modelo: geometría, apoyos, articulaciones y cargas"
    if res.get("combinacion"):
        tit += f"\n{res['combinacion']}: {texto_combinacion(res['factores'])}"
    if res.get("peso_propio"):
        tit += "  (+ peso propio, no dibujado)"
    _encuadre(ax, res)
    ax.set_title(tit, fontsize=10 if "\n" not in tit else 9)
    return _png(fig)


def fig_deformada(res):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    D = _dims(res)
    t = D * 0.03
    umax = 1e-30
    for m in res["miembros"]:
        umax = max(umax, np.abs(m["def_u"]).max(), np.abs(m["def_v"]).max())
    fac = 0.08 * D / umax
    for m in res["miembros"]:
        xi, yi = res["nodos"][m["i"]]
        xj, yj = res["nodos"][m["j"]]
        ax.plot([xi, xj], [yi, yj], "--", color="#aaaaaa", lw=1.2, zorder=1)
        x, ua, v = m["x"], m["def_u"], m["def_v"]
        X = xi + m["c"] * x + fac * (ua * m["c"] - v * m["s"])
        Y = yi + m["s"] * x + fac * (ua * m["s"] + v * m["c"])
        ax.plot(X, Y, "-", color=C_CARGA, lw=2, zorder=3)
    _apoyos(ax, res, t)
    u = res["u"]
    for n, (x, y) in res["nodos"].items():
        g = 3 * res["idx"][n]
        ax.plot(x + fac * u[g], y + fac * u[g + 1], "o", color=C_CARGA, ms=3.5, zorder=4)
        ax.text(x + fac * u[g] + 0.4 * t, y + fac * u[g + 1] + 0.4 * t, str(n), fontsize=8, zorder=5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.margins(0.15)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    _encuadre(ax, res)
    ax.set_title(f"Deformada (roja) vs. original (punteada) - amplificación x{fac:,.0f}", fontsize=10)
    return _png(fig)


def fig_diagrama(res, clave):
    info = {"N": ("Fuerza axial N (kN)  [+ tensión]", "#27ae60", 1),
            "V": ("Fuerza cortante V (kN)  [+ dibujado hacia el lado +y local]", "#2471a3", 1),
            "M": ("Momento flexionante M (kN·m)  [dibujado del lado en tensión]", "#c0392b", -1)}
    titulo, col, sgn = info[clave]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    D, t = _base(ax, res, color="#555555", lw=1.4, etiquetas=True)
    vmax = max([np.abs(m[clave]).max() for m in res["miembros"]] + [1e-30])
    esc = 0.13 * D / vmax
    tol = 1e-6 * vmax
    for m in res["miembros"]:
        vals = m[clave]
        if np.abs(vals).max() < tol:
            continue
        xi, yi = res["nodos"][m["i"]]
        x = m["x"]
        nx, ny = -m["s"], m["c"]
        bx, by = xi + m["c"] * x, yi + m["s"] * x
        ox, oy = bx + sgn * vals * esc * nx, by + sgn * vals * esc * ny
        polx = np.concatenate([bx, ox[::-1]])
        poly = np.concatenate([by, oy[::-1]])
        ax.fill(polx, poly, color=col, alpha=0.22, lw=0, zorder=1)
        ax.plot(ox, oy, "-", color=col, lw=1.3, zorder=2)
        # etiquetas: extremos y extremo interior
        marcas = {0, len(x) - 1}
        for k in (int(np.argmax(vals)), int(np.argmin(vals))):
            if 2 < k < len(x) - 3 and min(abs(vals[k] - vals[0]), abs(vals[k] - vals[-1])) > 0.03 * vmax:
                marcas.add(k)
        for k in sorted(marcas):
            if abs(vals[k]) < tol:
                continue
            ax.text(ox[k], oy[k], f"{vals[k]:.2f}", fontsize=6.8, color="k", ha="center", va="center",
                    zorder=6, bbox=dict(fc="w", ec=col, lw=0.6, pad=0.8, alpha=0.9))
    ax.margins(0.18)
    _encuadre(ax, res)
    ax.set_title(titulo, fontsize=10)
    return _png(fig)


def formula_img(latex, fontsize=12, ancho_max=6.6 * inch):
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, f"${latex}$", fontsize=fontsize)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)
    buf.seek(0)
    w, h = PILImage.open(buf).size
    buf.seek(0)
    W, H = w * 72 / 300, h * 72 / 300
    if W > ancho_max:
        H *= ancho_max / W
        W = ancho_max
    img = Image(buf, width=W, height=H)
    img.hAlign = "LEFT"
    return img


# =====================================================================
# 4) PDF
# =====================================================================
def _f(v, dec=4):
    if abs(v) < 1e-10:
        return "0"
    return f"{v:.{dec}g}"


def _c(v, dec=3):
    """Formato fijo sin '-0.000'."""
    t = f"{v:.{dec}f}"
    return t[1:] if t.startswith("-") and float(t) == 0 else t


def _tabla_matriz(M, filas, cols, dec=4, fs=7, ancho_total=7.0 * inch):
    mx = np.abs(M).max()
    p = int(math.floor(math.log10(mx))) if mx > 0 else 0
    Ms = M / 10 ** p
    data = [[""] + cols]
    for a, fl in enumerate(filas):
        data.append([fl] + [("0" if abs(Ms[a, b]) < 0.5 * 10 ** (-dec) else f"{Ms[a, b]:.{dec}f}")
                            for b in range(len(cols))])
    w0 = 0.42 * inch
    cw = min(0.85 * inch, (ancho_total - w0) / max(len(cols), 1))
    t = Table(data, colWidths=[w0] + [cw] * len(cols), hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#e8eef5")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t, p


def _tabla(data, col_w=None, fs=7.8):
    t = Table(data, colWidths=col_w, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f9")]),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t


def _img(buf, ancho=6.6 * inch, alto_max=4.3 * inch):
    w, h = PILImage.open(buf).size
    buf.seek(0)
    H = ancho * h / w
    if H > alto_max:
        ancho *= alto_max / H
        H = alto_max
    return Image(buf, width=ancho, height=H)


def generar_pdf(res, archivo=ARCHIVO_PDF, max_gdl_matriz=18, env=None, limite_flecha=240):
    """archivo puede ser una ruta o un objeto tipo BytesIO.
    env: resultado de envolvente() para agregar la sección de combinaciones (opcional).
    limite_flecha: n del límite L/n con el que se revisan las flechas."""
    titulo = res["titulo"]
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=colors.HexColor(C_BARRA))
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, textColor=colors.HexColor(C_BARRA),
                        spaceBefore=10, spaceAfter=4)
    H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=9.3, spaceBefore=6, spaceAfter=2)
    N = ParagraphStyle("N", parent=ss["Normal"], fontSize=8.8, leading=11.5)
    S = ParagraphStyle("S", parent=ss["Normal"], fontSize=7.2, leading=9, textColor=colors.HexColor("#555555"))
    SIM = ParagraphStyle("SIM", parent=ss["Normal"], fontSize=8.5, alignment=1)

    def pie(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(0.6 * inch, 0.42 * inch, titulo)
        canvas.drawRightString(letter[0] - 0.6 * inch, 0.42 * inch, f"Página {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(archivo, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.65 * inch, title=titulo)
    st = []
    ids, idx = res["ids"], res["idx"]
    etq = []
    for n in ids:
        etq += [f"u{n}", f"v{n}", f"θ{n}"]
    etq_pdf = [e.replace("θ", "t") for e in etq]    # Helvetica no trae theta

    # ------------------------------ 1. DATOS ------------------------------
    st.append(Paragraph(titulo, H1))
    st.append(Paragraph("Método de rigidez directa - estructura plana con 3 grados de libertad por nodo "
                        "(u: desplazamiento horizontal, v: vertical, t: giro). Unidades: kN, m, kPa, rad.", N))
    if res.get("combinacion"):
        st.append(Paragraph(f"<b>Combinación de carga: {res['combinacion']}</b> = "
                            f"{texto_combinacion(res['factores'])}. Las cargas mostradas ya incluyen sus factores. "
                            f"Casos definidos en el modelo: {', '.join(res['casos']) or '-'}.", N))
    st.append(Paragraph("1. Datos de entrada", H2))

    data = [["Nodo", "x (m)", "y (m)", "Apoyo (rx,ry,rz)", "Resorte (kx,ky,kz)", "Fx (kN)", "Fy (kN)", "M (kN·m)"]]
    for n in ids:
        x, y = res["nodos"][n]
        ap = res["apoyos"].get(n)
        rs = res["resortes"].get(n)
        fx, fy, mz = res["cargas_nodales"].get(n, (0, 0, 0))
        data.append([str(n), _f(x), _f(y), f"({ap[0]}, {ap[1]}, {ap[2]})" if ap else "-",
                     ", ".join(_f(v) for v in rs) if rs else "-", _f(fx), _f(fy), _f(mz)])
    st.append(_tabla(data, col_w=[0.45*inch, 0.65*inch, 0.65*inch, 1.1*inch, 1.35*inch, 0.75*inch, 0.75*inch, 0.8*inch]))
    st.append(Spacer(1, 5))

    data = [["Barra", "i", "j", "L (m)", "cos a", "sen a", "E (kPa)", "A (m2)", "I (m4)", "Articul."]]
    for m in res["miembros"]:
        art = ("i" if m["art_i"] else "") + ("," if m["art_i"] and m["art_j"] else "") + ("j" if m["art_j"] else "")
        data.append([str(m["id"]), str(m["i"]), str(m["j"]), _f(m["L"], 5), _f(m["c"], 5), _f(m["s"], 5),
                     f"{m['E']:.4g}", f"{m['A']:.4g}", f"{m['I']:.4g}", art or "-"])
    st.append(_tabla(data, col_w=[0.45*inch, 0.35*inch, 0.35*inch, 0.65*inch, 0.7*inch, 0.7*inch,
                                  0.85*inch, 0.75*inch, 0.8*inch, 0.65*inch]))

    cargas_b = [(m["id"], d) for m in res["miembros"] for d in m["dibujos"]]
    if cargas_b:
        st.append(Spacer(1, 5))
        data = [["Barra", "Tipo", "Dirección", "Valor", "Posición (m desde i)"]]
        for b, d in cargas_b:
            if d["k"] == "d":
                val = f"{d['w1']:g} kN/m" if d["w1"] == d["w2"] else f"{d['w1']:g} a {d['w2']:g} kN/m"
                data.append([str(b), "Peso propio" if d.get("pp") else "Distribuida", d["dir"], val,
                             f"{_f(d['a'])} a {_f(d['b'])}"])
            elif d["k"] == "m":
                data.append([str(b), "Momento", "+ antihorario", f"{d['M']:g} kN·m", _f(d["a"])])
            elif d["k"] == "t":
                val = f"dT = {d['dT']:g} °C" + (f"; dTg = {d['dTg']:g} °C, h = {d['h']:g} m" if d["dTg"] else "")
                data.append([str(b), "Térmica", f"alfa = {d['alpha']:g}", val, "toda la barra"])
            else:
                data.append([str(b), "Puntual", d["dir"], f"{d['P']:g} kN", _f(d["a"])])
        st.append(_tabla(data, col_w=[0.5*inch, 0.9*inch, 1.2*inch, 2.3*inch, 1.5*inch]))
    if res["asentamientos"]:
        st.append(Paragraph(f"<b>Desplazamientos impuestos</b> (dx, dy, giro): {res['asentamientos']}", N))

    GI = res["GI"]
    if GI == 0:
        clas = "ESTÁTICAMENTE DETERMINADA"
    elif GI > 0:
        clas = f"ESTÁTICAMENTE INDETERMINADA de grado {GI}"
    else:
        clas = "con GI negativo (revisar)"
    st.append(Spacer(1, 4))
    st.append(Paragraph(
        f"<b>Clasificación:</b> incógnitas internas = suma de (3 - articulaciones) por barra = {res['inc']}; "
        f"reacciones = {res['r']}" + (f"; resortes = {res['nres']}" if res["nres"] else "") +
        f"; ecuaciones de equilibrio = 3j - giros nulos = 3({res['j']}) - {len(res['eliminados'])} = {res['ecs']}. "
        f"GI = {res['inc']} + {res['r'] + res['nres']} - {res['ecs']} = <b>{GI}</b>. "
        f"La estructura es <b>{clas}</b> y estable (matriz reducida definida positiva).", N))
    if res["eliminados"]:
        st.append(Paragraph("Nota: en nodos donde todas las barras llegan articuladas, el giro del nodo no tiene "
                            "rigidez y se elimina del sistema (" +
                            ", ".join(etq_pdf[g] for g in res["eliminados"]) + ").", S))
    st.append(Spacer(1, 4))
    st.append(_img(fig_modelo(res), alto_max=3.7 * inch))

    # ------------------------------ 2. FORMULACIÓN ------------------------------
    st.append(PageBreak())
    st.append(Paragraph("2. Formulación", H2))
    st.append(Paragraph("<b>Matriz de rigidez local</b> de un elemento de marco (orden: u<sub>i</sub>, v<sub>i</sub>, "
                        "t<sub>i</sub>, u<sub>j</sub>, v<sub>j</sub>, t<sub>j</sub>):", N))
    simb = [["EA/L", "0", "0", "-EA/L", "0", "0"],
            ["0", "12EI/L<super>3</super>", "6EI/L<super>2</super>", "0", "-12EI/L<super>3</super>", "6EI/L<super>2</super>"],
            ["0", "6EI/L<super>2</super>", "4EI/L", "0", "-6EI/L<super>2</super>", "2EI/L"],
            ["-EA/L", "0", "0", "EA/L", "0", "0"],
            ["0", "-12EI/L<super>3</super>", "-6EI/L<super>2</super>", "0", "12EI/L<super>3</super>", "-6EI/L<super>2</super>"],
            ["0", "6EI/L<super>2</super>", "2EI/L", "0", "-6EI/L<super>2</super>", "4EI/L"]]
    t = Table([[Paragraph(c_, SIM) for c_ in fila] for fila in simb], colWidths=[0.95 * inch] * 6, hAlign="LEFT")
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.black),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    st.append(t)
    st.append(Spacer(1, 5))
    st.append(Paragraph("<b>Matriz de transformación</b> T (c = cos a, s = sen a), y rigidez en coordenadas globales:", N))
    simbT = [["c", "s", "0", "0", "0", "0"], ["-s", "c", "0", "0", "0", "0"], ["0", "0", "1", "0", "0", "0"],
             ["0", "0", "0", "c", "s", "0"], ["0", "0", "0", "-s", "c", "0"], ["0", "0", "0", "0", "0", "1"]]
    t = Table([[Paragraph(c_, SIM) for c_ in fila] for fila in simbT], colWidths=[0.4 * inch] * 6, hAlign="LEFT")
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.black),
                           ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    st.append(t)
    st.append(formula_img(r"[K]_e = [T]^T\,[k]_e\,[T] \qquad\qquad \{f\}_e = [k]_e\,[T]\,\{U\}_e + \{f_{emp}\}_e"))
    st.append(Paragraph("<b>Cargas en barras.</b> Las fuerzas de empotramiento se obtienen con las funciones de forma "
                        "de Hermite (exactas para viga de Euler-Bernoulli):", N))
    st.append(formula_img(r"\{f_{emp}\} = -\int_0^L [N(x)]^T\,q(x)\,dx \qquad\qquad \{P_{eq}\} = -[T]^T\{f_{emp}\}", fontsize=11))
    st.append(formula_img(r"N_1 = 1-3r^2+2r^3,\;\; N_2 = L(r-2r^2+r^3),\;\; N_3 = 3r^2-2r^3,\;\; N_4 = L(r^3-r^2),\;\; r = x/L", fontsize=10.5))
    hay_m = any(d["k"] == "m" for m in res["miembros"] for d in m["dibujos"])
    hay_t = any(d["k"] == "t" for m in res["miembros"] for d in m["dibujos"])
    if hay_m:
        st.append(Paragraph("<b>Momento concentrado</b> M<sub>0</sub> a la distancia a: trabaja con el giro "
                            "(derivadas de las funciones de forma) y produce un salto en el diagrama de momento:", N))
        st.append(formula_img(r"\{P_{eq}\}_{local} = M_0\,\left[\frac{dN}{dx}\right]^T_{x=a} \qquad\qquad M(a^+) = M(a^-) - M_0", fontsize=11))
    if hay_t:
        st.append(Paragraph("<b>Temperatura.</b> Un cambio uniforme dT produce la deformación libre "
                            "e<sub>T</sub> = alfa·dT; un gradiente dTg (cara +y menos cara -y) en el peralte h produce "
                            "la curvatura libre k<sub>T</sub> = -alfa·dTg/h. Fuerzas de empotramiento:", N))
        st.append(formula_img(r"\{f_{emp}\}_T = \left[\,EA\,\epsilon_T,\;0,\;EI\,\kappa_T,\;-EA\,\epsilon_T,\;0,\;-EI\,\kappa_T\,\right]^T \qquad v'' = \frac{M}{EI} + \kappa_T \qquad u' = \frac{N}{EA} + \epsilon_T", fontsize=10.5))
    st.append(Paragraph("<b>Articulaciones internas.</b> Se condensa el giro liberado r del extremo:", N))
    st.append(formula_img(r"[k]^* = [k] - \frac{\{k_r\}\{k_r\}^T}{k_{rr}} \qquad\qquad \{f_{emp}\}^* = \{f_{emp}\} - \{k_r\}\,\frac{f_{emp,r}}{k_{rr}}", fontsize=11))
    st.append(Paragraph("<b>Sistema global</b>, particionado en GDL libres (f) y restringidos (r):", N))
    st.append(formula_img(r"[K_{ff}]\{u_f\} = \{P_f\} - [K_{fr}]\{u_r\} \qquad\qquad \{R\} = [K_{rf}]\{u_f\} + [K_{rr}]\{u_r\} - \{P_r\}", fontsize=11))
    st.append(Paragraph("<b>Diagramas</b> (x desde el nodo i):", N))
    st.append(formula_img(r"N(x) = -f_1 - \int_0^x q_x\,d\xi \qquad V(x) = f_2 + \int_0^x q_y\,d\xi \qquad M(x) = -f_3 + f_2\,x + \int_0^x q_y\,(x-\xi)\,d\xi", fontsize=10.5))

    # ------------------------------ 3. MATRICES POR BARRA ------------------------------
    st.append(PageBreak())
    st.append(Paragraph("3. Matrices de cada barra", H2))
    for m in res["miembros"]:
        ci, cj = [f"u{m['i']}", f"v{m['i']}", f"t{m['i']}"], [f"u{m['j']}", f"v{m['j']}", f"t{m['j']}"]
        cols = ci + cj
        loc = ["x'i", "y'i", "t'i", "x'j", "y'j", "t'j"]
        bloque = [Paragraph(f"Barra {m['id']}  (nodo {m['i']} a {m['j']}):  L = {m['L']:.5g} m,  c = {m['c']:.5f},  "
                            f"s = {m['s']:.5f},  EA = {m['E']*m['A']:.5g} kN,  EI = {m['E']*m['I']:.5g} kN·m2", H3)]
        t1, p1 = _tabla_matriz(m["kl0"], loc, loc, fs=6.8, ancho_total=6.9 * inch)
        bloque += [Paragraph("[k] local", S), t1, Paragraph(f"x 10^{p1}", S)]
        if m["art_i"] or m["art_j"]:
            t2, p2 = _tabla_matriz(m["kl"], loc, loc, fs=6.8, ancho_total=6.9 * inch)
            lado = " y ".join([s_ for s_, f_ in (("i", m["art_i"]), ("j", m["art_j"])) if f_])
            bloque += [Paragraph(f"[k]* local condensada (articulación en extremo {lado})", S), t2, Paragraph(f"x 10^{p2}", S)]
        t3, p3 = _tabla_matriz(m["kg"], cols, cols, fs=6.8, ancho_total=6.9 * inch)
        bloque += [Paragraph("[K] global = [T]<super>T</super>[k][T]", S), t3, Paragraph(f"x 10^{p3}", S)]
        st.append(KeepTogether(bloque))
        if m["piezas"]:
            data = [["Componente", "f emp (local)", "f emp* (condensado)", "GDL global", "P eq (global)"]]
            nomb = ["Fx' i", "Fy' i", "M i", "Fx' j", "Fy' j", "M j"]
            for q in range(6):
                data.append([nomb[q], f"{m['fef0'][q]:.4f}", f"{m['fef'][q]:.4f}", cols[q], f"{m['peq_g'][q]:.4f}"])
            st.append(Paragraph("Fuerzas de empotramiento y cargas nodales equivalentes (kN, kN·m)", S))
            st.append(_tabla(data, col_w=[0.9*inch, 1.2*inch, 1.4*inch, 0.9*inch, 1.2*inch], fs=7.2))
        st.append(Spacer(1, 6))

    # ------------------------------ 4. SISTEMA GLOBAL ------------------------------
    st.append(PageBreak())
    ndof = 3 * len(ids)
    st.append(Paragraph("4. Ensamble y solución del sistema global", H2))
    st.append(Paragraph(f"Tamaño de [K]: {ndof} x {ndof}. GDL libres: {', '.join(etq_pdf[g] for g in res['libres'])}. "
                        f"GDL restringidos: {', '.join(etq_pdf[g] for g in res['fijos']) or '-'}.", N))
    if ndof <= max_gdl_matriz:
        t, p = _tabla_matriz(res["Kest"], etq_pdf, etq_pdf, dec=3, fs=5.4 if ndof > 12 else 6.4, ancho_total=7.2 * inch)
        st.append(Paragraph("[K] global ensamblada (sin resortes)", H3))
        st.append(t)
        st.append(Paragraph(f"x 10^{p}", S))
    else:
        st.append(Paragraph(f"(Matriz global de {ndof} x {ndof} omitida por tamaño.)", S))
    if res["nres"]:
        st.append(Paragraph("Los resortes se suman a la diagonal: " +
                            ", ".join(f"{etq_pdf[g]} + {res['kres'][g]:g}" for g in np.where(res['kres'] != 0)[0]), S))

    st.append(Paragraph("Vector de cargas {P} = cargas nodales + cargas equivalentes de barras", H3))
    data = [["GDL", "P nodal", "P equivalente", "P total", "Estado"]]
    for g in range(ndof):
        estado = "libre" if g in set(res["libres"]) else ("restringido" if res["restr"][g] else "giro nulo")
        data.append([etq_pdf[g], _f(res["Pn"][g], 6), _f(res["Peq"][g], 6), _f(res["P"][g], 6), estado])
    st.append(_tabla(data, col_w=[0.7*inch, 1.1*inch, 1.2*inch, 1.1*inch, 1.0*inch], fs=7))

    nf = len(res["libres"])
    lib = [etq_pdf[g] for g in res["libres"]]
    if nf <= max_gdl_matriz:
        t, p = _tabla_matriz(res["Kff"], lib, lib, dec=3, fs=5.4 if nf > 12 else 6.4, ancho_total=7.2 * inch)
        st.append(KeepTogether([Paragraph("[Kff] (incluye resortes)", H3), t, Paragraph(f"x 10^{p}", S)]))
    st.append(Paragraph("Solución [Kff]{uf} = {Pf} - [Kfr]{ur}", H3))
    data = [["GDL", "Pf", "Pf - Kfr·ur", "uf", "uf (mm o mrad)"]]
    for k_, g in enumerate(res["libres"]):
        data.append([etq_pdf[g], _f(res["Pf"][k_], 6), _f(res["rhs"][k_], 6),
                     f"{res['uf'][k_]:.6e}", f"{res['uf'][k_]*1000:.5f}"])
    st.append(_tabla(data, col_w=[0.7*inch, 1.1*inch, 1.2*inch, 1.3*inch, 1.2*inch], fs=7))
    st.append(Paragraph(f"Número de condición de [Kff] = {res['cond']:.3e}", S))

    # ------------------------------ 5. RESULTADOS ------------------------------
    st.append(PageBreak())
    st.append(Paragraph("5. Resultados", H2))
    st.append(Paragraph("Desplazamientos nodales", H3))
    data = [["Nodo", "u (mm)", "v (mm)", "giro (mrad)"]]
    for n in ids:
        g = 3 * idx[n]
        data.append([str(n)] + [f"{res['u'][g+q]*1000:.5f}" for q in range(3)])
    st.append(_tabla(data, col_w=[0.7*inch, 1.3*inch, 1.3*inch, 1.3*inch]))

    st.append(Paragraph("Reacciones", H3))
    data = [["Nodo", "Rx (kN)", "Ry (kN)", "Mz (kN·m)"]]
    for n in ids:
        g = 3 * idx[n]
        if n in res["apoyos"]:
            fila = [str(n)]
            for q in range(3):
                fila.append(_c(res['reac'][g+q], 4) if res["restr"][g + q] else "-")
            data.append(fila)
        if n in res["resortes"]:
            data.append([f"{n} (resorte)"] + [f"{res['fres'][g+q]:.4f}" if res["kres"][g + q] else "-" for q in range(3)])
    st.append(_tabla(data, col_w=[1.0*inch, 1.3*inch, 1.3*inch, 1.3*inch]))

    sfx, sfy, sm = res["equilibrio"]
    ok = equilibrio_ok(res)
    st.append(Paragraph(f"<b>Equilibrio global</b> (cargas nodales + cargas en barras + reacciones): "
                        f"suma Fx = {sfx:.2e} kN, suma Fy = {sfy:.2e} kN, suma M (origen) = {sm:.2e} kN·m "
                        f"&rarr; <b>{'CUMPLE' if ok else 'NO CUMPLE, revisar'}</b>", N))

    st.append(Paragraph("Fuerzas en los extremos de cada barra (ejes locales): {f} = [k]{u'} + {f emp}", H3))
    for m in res["miembros"]:
        data = [[f"Barra {m['id']}", "u' local", "[k]{u'}", "f emp", "f total"]]
        nomb = ["Fx' i (kN)", "Fy' i (kN)", "M i (kN·m)", "Fx' j (kN)", "Fy' j (kN)", "M j (kN·m)"]
        for q in range(6):
            data.append([nomb[q], f"{m['u_loc'][q]:.4e}", _c(m['k_u'][q], 4), _c(m['fef'][q], 4), _c(m['f'][q], 4)])
        st.append(KeepTogether([_tabla(data, col_w=[1.0*inch, 1.2*inch, 1.1*inch, 1.0*inch, 1.1*inch], fs=7), Spacer(1, 4)]))

    st.append(Paragraph("Resumen de elementos mecánicos (x medida desde el nodo i)", H3))
    data = [["Barra", "N i", "N j", "V i", "V j", "M i", "M j", "M máx (x)", "M mín (x)"]]
    for m in res["miembros"]:
        data.append([str(m["id"]), _c(m['N'][0]), _c(m['N'][-1]), _c(m['V'][0]), _c(m['V'][-1]),
                     _c(m['M'][0]), _c(m['M'][-1]),
                     f"{_c(m['M_max'][0])} ({m['M_max'][1]:.2f})", f"{_c(m['M_min'][0])} ({m['M_min'][1]:.2f})"])
    st.append(_tabla(data, col_w=[0.45*inch] + [0.68*inch] * 6 + [1.05*inch, 1.05*inch], fs=7))
    st.append(Paragraph("N en kN (+ tensión), V en kN, M en kN·m (+ tensiona el lado -y local). "
                        "Los valores en los extremos coinciden con las fuerzas de extremo: N(L) = f4, V(L) = -f5, M(L) = f6.", S))

    st.append(Paragraph("Flechas por barra", H3))
    data = [["Barra", "L (m)", "Flecha relativa máx. (mm)", "x (m)", "L / flecha", f"Límite L/{limite_flecha:g}",
             "Desp. total máx. (mm)"]]
    for m in res["miembros"]:
        fl = m["flecha"]
        rel_ = fl["relacion"]
        data.append([str(m["id"]), _f(m["L"], 5), _c(fl["valor"] * 1000, 3), f"{fl['x']:.2f}",
                     "-" if math.isinf(rel_) or rel_ > 99999 else f"{rel_:.0f}",
                     "Cumple" if rel_ >= limite_flecha else "EXCEDE", _c(m["desp_max"][0] * 1000, 3)])
    st.append(_tabla(data, col_w=[0.5*inch, 0.65*inch, 1.5*inch, 0.6*inch, 0.8*inch, 0.95*inch, 1.35*inch], fs=7))
    st.append(Paragraph("La flecha relativa se mide perpendicular a la barra respecto a la cuerda que une sus extremos "
                        "ya desplazados (+ hacia +y local). En voladizos revisa el desplazamiento total del extremo libre. "
                        "El límite L/n es indicativo: usa el que marque tu reglamento.", S))

    if env and len(env.get("combinaciones", [])) > 1:
        st.append(PageBreak())
        st.append(Paragraph("Envolvente de combinaciones", H2))
        st.append(Paragraph("Combinaciones analizadas: " + ", ".join(env["combinaciones"]) +
                            ". Para cada barra se da el valor máximo y mínimo de cada efecto y la combinación que lo produce.", N))
        data = [["Barra", "Efecto", "Máximo", "x (m)", "Combinación", "Mínimo", "x (m)", "Combinación"]]
        unid = {"N": "N (kN)", "V": "V (kN)", "M": "M (kN·m)"}
        for bid, d in env["barras"].items():
            for clave in ("N", "V", "M"):
                e = d[clave]
                data.append([str(bid) if clave == "N" else "", unid[clave], _c(e["max"]), f"{e['x_max']:.2f}",
                             e["combo_max"], _c(e["min"]), f"{e['x_min']:.2f}", e["combo_min"]])
        st.append(_tabla(data, col_w=[0.45*inch, 0.75*inch, 0.8*inch, 0.5*inch, 1.5*inch, 0.8*inch, 0.5*inch, 1.5*inch], fs=7))
        if env["reacciones"]:
            st.append(Paragraph("Reacciones extremas", H3))
            data = [["Nodo", "Componente", "Máxima", "Combinación", "Mínima", "Combinación"]]
            comp = ["Rx (kN)", "Ry (kN)", "Mz (kN·m)"]
            for (n_, q), (vmax, cmax, vmin, cmin) in env["reacciones"].items():
                data.append([str(n_), comp[q], _c(vmax), cmax, _c(vmin), cmin])
            st.append(_tabla(data, col_w=[0.5*inch, 0.9*inch, 0.9*inch, 1.6*inch, 0.9*inch, 1.6*inch], fs=7))
        st.append(Paragraph("Flecha relativa crítica por barra", H3))
        data = [["Barra", "Flecha (mm)", "x (m)", "Combinación"]]
        for bid, (v_, x_, c_) in env["flechas"].items():
            data.append([str(bid), _c(v_ * 1000, 3), f"{x_:.2f}", c_])
        st.append(_tabla(data, col_w=[0.5*inch, 1.0*inch, 0.7*inch, 1.8*inch], fs=7))

    # ------------------------------ 6. DIAGRAMAS ------------------------------
    st.append(PageBreak())
    st.append(Paragraph("6. Diagramas", H2))
    st.append(_img(fig_deformada(res)))
    st.append(Spacer(1, 6))
    st.append(_img(fig_diagrama(res, "N")))
    st.append(PageBreak())
    st.append(_img(fig_diagrama(res, "V")))
    st.append(Spacer(1, 6))
    st.append(_img(fig_diagrama(res, "M")))

    doc.build(st, onFirstPage=pie, onLaterPages=pie)
    return archivo


# =====================================================================
# 4b) EXCEL
# =====================================================================
def exportar_excel(resultados, archivo, env=None, n_est=11):
    """Exporta resultados a un .xlsx. resultados = res de analizar() o {nombre: res}.
    archivo puede ser una ruta o un BytesIO."""
    _asegurar("openpyxl")
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    if "miembros" in resultados:
        resultados = {resultados.get("combinacion") or SIN_COMBINACION: resultados}
    wb = Workbook()
    wb.remove(wb.active)

    def r6(v):
        v = float(v)
        return 0.0 if abs(v) < 1e-12 else round(v, 6)

    def hoja(nombre, enc, filas):
        ws = wb.create_sheet(nombre)
        ws.append(enc)
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="1F3B57")
            c.alignment = Alignment(horizontal="center", wrap_text=True)
        for f in filas:
            ws.append(f)
        for k, e in enumerate(enc, 1):
            ws.column_dimensions[get_column_letter(k)].width = max(12, min(28, len(str(e)) + 4))
        ws.freeze_panes = "A2"
        return ws

    r0 = next(iter(resultados.values()))
    hoja("Barras", ["Barra", "Nodo i", "Nodo j", "L (m)", "E (kPa)", "A (m2)", "I (m4)", "Articulada i", "Articulada j"],
         [[m["id"], m["i"], m["j"], r6(m["L"]), m["E"], m["A"], m["I"], "sí" if m["art_i"] else "no",
           "sí" if m["art_j"] else "no"] for m in r0["miembros"]])
    hoja("Combinaciones", ["Combinación", "Factores"],
         [[c, texto_combinacion(r["factores"])] for c, r in resultados.items()])

    f_desp, f_reac, f_ext, f_mec, f_est = [], [], [], [], []
    for combo, res in resultados.items():
        idx = res["idx"]
        for n in res["ids"]:
            g = 3 * idx[n]
            f_desp.append([combo, n] + [r6(res["u"][g + q] * 1000) for q in range(3)])
            if n in res["apoyos"]:
                f_reac.append([combo, n, "apoyo"] + [r6(res["reac"][g + q]) if res["restr"][g + q] else None for q in range(3)])
            if n in res["resortes"]:
                f_reac.append([combo, n, "resorte"] + [r6(res["fres"][g + q]) if res["kres"][g + q] else None for q in range(3)])
        for m in res["miembros"]:
            f_ext.append([combo, m["id"]] + [r6(v) for v in m["f"]])
            fl = m["flecha"]
            f_mec.append([combo, m["id"], r6(m["L"]), r6(m["N"][0]), r6(m["N"][-1]), r6(m["V"][0]), r6(m["V"][-1]),
                          r6(m["M"][0]), r6(m["M"][-1]), r6(m["M_max"][0]), r6(m["M_max"][1]), r6(m["M_min"][0]),
                          r6(m["M_min"][1]), r6(fl["valor"] * 1000), r6(fl["x"]),
                          None if math.isinf(fl["relacion"]) else round(fl["relacion"], 1)])
            e = estaciones(m, n_est)
            for k in range(n_est):
                f_est.append([combo, m["id"], r6(e["x"][k]), r6(e["N"][k]), r6(e["V"][k]), r6(e["M"][k]),
                              r6(e["u"][k] * 1000), r6(e["v"][k] * 1000)])
    hoja("Desplazamientos", ["Combinación", "Nodo", "u (mm)", "v (mm)", "giro (mrad)"], f_desp)
    hoja("Reacciones", ["Combinación", "Nodo", "Tipo", "Rx (kN)", "Ry (kN)", "Mz (kN·m)"], f_reac)
    hoja("Fuerzas en extremos", ["Combinación", "Barra", "Fx' i (kN)", "Fy' i (kN)", "M i (kN·m)",
                                 "Fx' j (kN)", "Fy' j (kN)", "M j (kN·m)"], f_ext)
    hoja("Elementos mecánicos", ["Combinación", "Barra", "L (m)", "N i (kN)", "N j (kN)", "V i (kN)", "V j (kN)",
                                 "M i (kN·m)", "M j (kN·m)", "M máx (kN·m)", "x M máx (m)", "M mín (kN·m)",
                                 "x M mín (m)", "Flecha rel. (mm)", "x flecha (m)", "L / flecha"], f_mec)
    hoja("Estaciones", ["Combinación", "Barra", "x (m)", "N (kN)", "V (kN)", "M (kN·m)", "u local (mm)",
                        "v local (mm)"], f_est)
    if env:
        filas = []
        for bid, d in env["barras"].items():
            for clave, un in (("N", "N (kN)"), ("V", "V (kN)"), ("M", "M (kN·m)")):
                e = d[clave]
                filas.append([bid, un, r6(e["max"]), r6(e["x_max"]), e["combo_max"], r6(e["min"]), r6(e["x_min"]),
                              e["combo_min"]])
        comp = ["Rx (kN)", "Ry (kN)", "Mz (kN·m)"]
        for (n_, q), (vmax, cmax, vmin, cmin) in env["reacciones"].items():
            filas.append([f"Nodo {n_}", comp[q], r6(vmax), None, cmax, r6(vmin), None, cmin])
        hoja("Envolvente", ["Barra / nodo", "Efecto", "Máximo", "x (m)", "Combinación", "Mínimo", "x (m)",
                            "Combinación"], filas)
    wb.save(archivo)
    return archivo


# =====================================================================
# 5) EJECUCIÓN
# =====================================================================
def imprimir_resumen(res):
    print(f"GI = {res['GI']}   (incógnitas internas {res['inc']}, reacciones {res['r']}, "
          f"resortes {res['nres']}, ecuaciones {res['ecs']})")
    print("\nDesplazamientos:  u (mm)   v (mm)   giro (mrad)")
    for n in res["ids"]:
        g = 3 * res["idx"][n]
        print(f"  Nodo {n}: " + "  ".join(f"{res['u'][g+q]*1000:+.5f}" for q in range(3)))
    print("\nReacciones:  Rx (kN)   Ry (kN)   Mz (kN.m)")
    for n in res["ids"]:
        if n in res["apoyos"]:
            g = 3 * res["idx"][n]
            print(f"  Nodo {n}: " + "  ".join(f"{res['reac'][g+q]:+.4f}" if res['restr'][g+q] else "    -    "
                                             for q in range(3)))
    print("\nBarra:   N_i      V_i      M_i   |   N_j      V_j      M_j   |  M max (x)")
    for m in res["miembros"]:
        print(f"  {m['id']:>3}: {m['N'][0]+0:+8.3f} {m['V'][0]:+8.3f} {m['M'][0]:+8.3f} | "
              f"{m['N'][-1]:+8.3f} {m['V'][-1]:+8.3f} {m['M'][-1]:+8.3f} | "
              f"{m['M_max'][0]:+.3f} ({m['M_max'][1]:.2f} m)")
    print("\nFlechas relativas:  barra: flecha (mm) en x (m)   L/flecha")
    for m in res["miembros"]:
        fl = m["flecha"]
        rel = "-" if math.isinf(fl["relacion"]) else f"L/{fl['relacion']:.0f}"
        print(f"  {m['id']:>3}: {fl['valor']*1000:+.3f} en {fl['x']:.2f}   {rel}")
    sfx, sfy, sm = res["equilibrio"]
    print(f"\nEquilibrio global: SFx={sfx:.2e}  SFy={sfy:.2e}  SM={sm:.2e}")


def main(modelo=MODELO, archivo=ARCHIVO_PDF):
    """Analiza el modelo. Si define combinaciones, imprime cada una y el PDF desarrolla la primera
    (con la envolvente de todas al final de la sección de resultados)."""
    try:
        todos = analizar_combinaciones(modelo)
    except ErrorEstructura as e:
        print("ERROR EN LA ESTRUCTURA:", e)
        return None
    for nombre, r in todos.items():
        if len(todos) > 1:
            print(f"\n=========== {nombre}: {texto_combinacion(r['factores'])} ===========")
        imprimir_resumen(r)
    res = next(iter(todos.values()))
    generar_pdf(res, archivo, env=envolvente(todos) if len(todos) > 1 else None)
    print(f"\nPDF generado: {archivo}")
    try:
        from google.colab import files
        files.download(archivo)
    except Exception:
        pass
    return res


if __name__ == "__main__":
    main()
