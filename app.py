# -*- coding: utf-8 -*-
"""
App de Streamlit - Análisis matricial de estructuras planas
Ejecutar local:   streamlit run app.py
Requiere en la misma carpeta: analisis_matricial_marcos.py
"""
import copy
import json
import math
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analisis_matricial_marcos as motor

st.set_page_config(page_title="Análisis matricial 2D", page_icon="📐", layout="wide")

AZUL = "#1f3b57"
ROJO = "#c0392b"
NARANJA = "#d68910"

# ---------------------------------------------------------------------
# Tablas del modelo
# ---------------------------------------------------------------------
COLUMNAS = {
    "nodos": ["id", "x", "y"],
    "barras": ["id", "i", "j", "E", "A", "I", "art_i", "art_j"],
    "apoyos": ["nodo", "rx", "ry", "rz"],
    "resortes": ["nodo", "kx", "ky", "kz"],
    "cargas_nodales": ["nodo", "Fx", "Fy", "M"],
    "cargas_barras": ["barra", "tipo", "dir", "w1", "w2", "a", "b", "P"],
    "asentamientos": ["nodo", "dx", "dy", "giro"],
}
BOOL_COLS = {"barras": ["art_i", "art_j"], "apoyos": ["rx", "ry", "rz"]}
TEXTO_COLS = {"cargas_barras": ["tipo", "dir"]}
TIPOS_CARGA = ["distribuida", "puntual"]
DIRS = ["global_y", "global_x", "local_y", "local_x", "global_y_proy", "global_x_proy"]
APOYOS_PRESET = {
    "Empotre": (1, 1, 1),
    "Articulación": (1, 1, 0),
    "Rodillo (se desliza en x)": (0, 1, 0),
    "Rodillo (se desliza en y)": (1, 0, 0),
    "Quitar apoyo": None,
}


def _vacio(nombre):
    df = pd.DataFrame({c: pd.Series(dtype="float") for c in COLUMNAS[nombre]})
    for c in BOOL_COLS.get(nombre, []):
        df[c] = df[c].astype(bool)
    for c in TEXTO_COLS.get(nombre, []):
        df[c] = df[c].astype(object)
    return df


def _tipar(nombre, df):
    df = df.copy()
    for c in COLUMNAS[nombre]:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNAS[nombre]]
    for c in COLUMNAS[nombre]:
        if c in BOOL_COLS.get(nombre, []):
            df[c] = df[c].fillna(False).astype(bool)
        elif c in TEXTO_COLS.get(nombre, []):
            df[c] = df[c].astype(object)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df.reset_index(drop=True)


def modelo_a_tablas(mod):
    t = {}
    t["nodos"] = pd.DataFrame([{"id": k, "x": v[0], "y": v[1]} for k, v in mod.get("nodos", {}).items()])
    filas = []
    for k, d in mod.get("barras", {}).items():
        if isinstance(d, (list, tuple)):
            d = {"i": d[0], "j": d[1]}
        filas.append({"id": k, "i": d["i"], "j": d["j"], "E": d.get("E"), "A": d.get("A"), "I": d.get("I"),
                      "art_i": bool(d.get("art_i", False)), "art_j": bool(d.get("art_j", False))})
    t["barras"] = pd.DataFrame(filas)

    def trip(v):
        v = list(v) + [0] * (3 - len(v))
        return v[:3]

    t["apoyos"] = pd.DataFrame([dict(zip(["nodo", "rx", "ry", "rz"], [k] + [bool(x) for x in trip(v)]))
                                for k, v in mod.get("apoyos", {}).items()])
    t["resortes"] = pd.DataFrame([dict(zip(["nodo", "kx", "ky", "kz"], [k] + trip(v)))
                                  for k, v in (mod.get("resortes") or {}).items()])
    t["cargas_nodales"] = pd.DataFrame([dict(zip(["nodo", "Fx", "Fy", "M"], [k] + trip(v)))
                                        for k, v in (mod.get("cargas_nodales") or {}).items()])
    t["cargas_barras"] = pd.DataFrame([{c: cb.get(c) for c in COLUMNAS["cargas_barras"]}
                                       for cb in (mod.get("cargas_barras") or [])])
    t["asentamientos"] = pd.DataFrame([dict(zip(["nodo", "dx", "dy", "giro"], [k] + trip(v)))
                                       for k, v in (mod.get("asentamientos") or {}).items()])
    return {k: (_tipar(k, v) if len(v) else _vacio(k)) for k, v in t.items()}


def _ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and not pd.isna(v)


def tablas_a_modelo(t, E, A, I, titulo):
    """Convierte las tablas al formato del motor. Devuelve (modelo, lista_de_avisos)."""
    avisos = []
    mod = {"titulo": titulo, "E": E, "A": A, "I": I, "nodos": {}, "barras": {}, "apoyos": {}, "resortes": {},
           "cargas_nodales": {}, "cargas_barras": [], "asentamientos": {}}
    for _, r in t["nodos"].iterrows():
        if _ok(r["id"]) and _ok(r["x"]) and _ok(r["y"]):
            nid = int(r["id"])
            if nid in mod["nodos"]:
                avisos.append(f"Nodo {nid} repetido: se usa el primero.")
                continue
            mod["nodos"][nid] = (float(r["x"]), float(r["y"]))
        elif any(_ok(r[c]) for c in ("id", "x", "y")):
            avisos.append("Una fila de nodos está incompleta y se ignoró.")
    for _, r in t["barras"].iterrows():
        if _ok(r["id"]) and _ok(r["i"]) and _ok(r["j"]):
            d = {"i": int(r["i"]), "j": int(r["j"]), "art_i": bool(r["art_i"]), "art_j": bool(r["art_j"])}
            for c in ("E", "A", "I"):
                if _ok(r[c]):
                    d[c] = float(r[c])
            mod["barras"][int(r["id"])] = d
        elif any(_ok(r[c]) for c in ("id", "i", "j")):
            avisos.append("Una fila de barras está incompleta y se ignoró.")
    for _, r in t["apoyos"].iterrows():
        if _ok(r["nodo"]):
            mod["apoyos"][int(r["nodo"])] = (int(bool(r["rx"])), int(bool(r["ry"])), int(bool(r["rz"])))
    for nombre, cols in (("resortes", ["kx", "ky", "kz"]), ("cargas_nodales", ["Fx", "Fy", "M"]),
                         ("asentamientos", ["dx", "dy", "giro"])):
        for _, r in t[nombre].iterrows():
            if _ok(r["nodo"]):
                v = tuple(float(r[c]) if _ok(r[c]) else 0.0 for c in cols)
                n = int(r["nodo"])
                if nombre == "cargas_nodales" and n in mod[nombre]:
                    v = tuple(a + b for a, b in zip(mod[nombre][n], v))
                mod[nombre][n] = v
    for _, r in t["cargas_barras"].iterrows():
        if not _ok(r["barra"]):
            continue
        tipo = r["tipo"] if isinstance(r["tipo"], str) and r["tipo"] else "distribuida"
        c = {"barra": int(r["barra"]), "tipo": tipo, "dir": r["dir"] if isinstance(r["dir"], str) and r["dir"] else "global_y"}
        if tipo.startswith("dist"):
            if not _ok(r["w1"]):
                avisos.append(f"Carga distribuida en barra {c['barra']} sin w1: se ignoró.")
                continue
            c["w1"] = float(r["w1"])
            for k in ("w2", "a", "b"):
                if _ok(r[k]):
                    c[k] = float(r[k])
        else:
            if not _ok(r["P"]):
                avisos.append(f"Carga puntual en barra {c['barra']} sin P: se ignoró.")
                continue
            c["P"] = float(r["P"])
            if _ok(r["a"]):
                c["a"] = float(r["a"])
        mod["cargas_barras"].append(c)
    return mod, avisos


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------
def ejemplos():
    viga = {"titulo": "Viga continua de tres claros", "E": 200e6, "A": 0.006, "I": 1.2e-4,
            "nodos": {1: (0, 0), 2: (5, 0), 3: (11, 0), 4: (16, 0)},
            "barras": {1: {"i": 1, "j": 2}, 2: {"i": 2, "j": 3}, 3: {"i": 3, "j": 4}},
            "apoyos": {1: (1, 1, 0), 2: (0, 1, 0), 3: (0, 1, 0), 4: (0, 1, 0)},
            "cargas_barras": [{"barra": 1, "tipo": "distribuida", "w1": -15, "dir": "global_y"},
                              {"barra": 2, "tipo": "distribuida", "w1": -15, "dir": "global_y"},
                              {"barra": 3, "tipo": "puntual", "P": -40, "a": 2.5, "dir": "global_y"}]}
    nod = {1: (0, 0), 2: (4, 0), 3: (8, 0), 4: (2, 3), 5: (6, 3)}
    br = {1: (1, 2), 2: (2, 3), 3: (4, 5), 4: (1, 4), 5: (4, 2), 6: (2, 5), 7: (5, 3), 8: (4, 3)}
    armadura = {"titulo": "Armadura plana", "E": 200e6, "A": 0.002, "I": 1e-6, "nodos": nod,
                "barras": {k: {"i": a, "j": b, "art_i": True, "art_j": True} for k, (a, b) in br.items()},
                "apoyos": {1: (1, 1, 0), 3: (0, 1, 0)}, "cargas_nodales": {4: (5, -10, 0), 5: (0, -10, 0)}}
    triart = {"titulo": "Marco triarticulado de dos aguas", "E": 200e6, "A": 0.006, "I": 1.2e-4,
              "nodos": {1: (0, 0), 2: (0, 4), 3: (5, 6), 4: (10, 4), 5: (10, 0)},
              "barras": {1: {"i": 1, "j": 2}, 2: {"i": 2, "j": 3, "art_j": True},
                         3: {"i": 3, "j": 4, "art_i": True}, 4: {"i": 5, "j": 4}},
              "apoyos": {1: (1, 1, 0), 5: (1, 1, 0)},
              "cargas_barras": [{"barra": 2, "tipo": "distribuida", "w1": -6, "dir": "global_y_proy"},
                                {"barra": 3, "tipo": "distribuida", "w1": -6, "dir": "global_y_proy"},
                                {"barra": 1, "tipo": "distribuida", "w1": 3, "dir": "global_x"}]}
    vacio = {"titulo": "Estructura nueva", "E": 200e6, "A": 0.006, "I": 1.2e-4, "nodos": {}, "barras": {}}
    return {"Marco con voladizo": copy.deepcopy(motor.MODELO), "Viga continua": viga,
            "Armadura": armadura, "Marco triarticulado": triart, "Vacío": vacio}


# ---------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------
def _cargar_modelo(mod):
    ss = st.session_state
    guardar_historial()
    ss.tablas = modelo_a_tablas(mod)
    ss.base = copy.deepcopy(ss.tablas)
    ss.ver = ss.get("ver", 0) + 1
    ss.chart_ver = ss.get("chart_ver", 0) + 1
    ss.pendiente = None
    ss.res = None
    ss.titulo = mod.get("titulo", "Estructura")
    ss.E_def = float(mod.get("E") or 200e6)
    ss.A_def = float(mod.get("A") or 0.006)
    ss.I_def = float(mod.get("I") or 1.2e-4)
    ajustar_cuadricula()


def ajustar_cuadricula():
    ss = st.session_state
    nodos = ss.tablas["nodos"].dropna()
    if len(nodos):
        ss.gx0 = float(math.floor(nodos["x"].min()) - 2)
        ss.gx1 = float(math.ceil(nodos["x"].max()) + 2)
        ss.gy0 = float(math.floor(nodos["y"].min()) - 1)
        ss.gy1 = float(math.ceil(nodos["y"].max()) + 2)
    else:
        ss.gx0, ss.gx1, ss.gy0, ss.gy1 = -1.0, 12.0, -1.0, 8.0


def guardar_historial():
    ss = st.session_state
    if "tablas" in ss:
        ss.historial = (ss.get("historial", []) + [copy.deepcopy(ss.tablas)])[-30:]


def aplicar_tablas(nuevas):
    ss = st.session_state
    guardar_historial()
    ss.tablas = nuevas
    ss.base = copy.deepcopy(nuevas)
    ss.ver += 1
    ss.chart_ver += 1


def deshacer():
    ss = st.session_state
    if ss.get("historial"):
        ss.tablas = ss.historial.pop()
        ss.base = copy.deepcopy(ss.tablas)
        ss.ver += 1
        ss.chart_ver += 1
        ss.pendiente = None


def init():
    ss = st.session_state
    if "tablas" not in ss:
        ss.dx, ss.dy = 1.0, 1.0
        _cargar_modelo(ejemplos()["Marco con voladizo"])
        ss.historial = []


# ---------------------------------------------------------------------
# Acciones del dibujo (función pura, fácil de probar)
# ---------------------------------------------------------------------
def _siguiente_id(serie):
    s = pd.to_numeric(serie, errors="coerce").dropna()
    return int(s.max()) + 1 if len(s) else 1


def _nodo_en(t, x, y, tol=1e-6):
    n = t["nodos"].dropna(subset=["id", "x", "y"])
    hit = n[(abs(n["x"] - x) < tol) & (abs(n["y"] - y) < tol)]
    return int(hit.iloc[0]["id"]) if len(hit) else None


def aplicar_clic(tablas, punto, modo, op, pendiente):
    """punto = {"tipo": "g"|"n"|"b", "id", "x", "y"}. Devuelve (tablas, pendiente, mensaje, cambio)."""
    t = copy.deepcopy(tablas)
    tipo = punto["tipo"]
    nid = None
    if tipo in ("g", "n"):
        nid = int(punto["id"]) if tipo == "n" else _nodo_en(t, punto["x"], punto["y"])

    def crear_nodo():
        nuevo = _siguiente_id(t["nodos"]["id"])
        fila = pd.DataFrame([{"id": nuevo, "x": float(punto["x"]), "y": float(punto["y"])}])
        t["nodos"] = _tipar("nodos", pd.concat([t["nodos"], fila], ignore_index=True))
        return nuevo

    if modo == "Nodo":
        if nid is not None:
            return tablas, pendiente, f"Ya hay un nodo ({nid}) en ese punto.", False
        if tipo == "b":
            return tablas, pendiente, "Toca un punto de la cuadrícula para crear un nodo.", False
        n = crear_nodo()
        return t, pendiente, f"Nodo {n} creado en ({punto['x']:g}, {punto['y']:g}).", True

    if modo == "Barra":
        if tipo == "b":
            return tablas, pendiente, "Toca nodos o puntos de la cuadrícula para trazar barras.", False
        creado = False
        if nid is None:
            nid = crear_nodo()
            creado = True
        if pendiente is None or pendiente not in set(t["nodos"]["id"].dropna().astype(int)):
            return t, nid, f"Inicio en nodo {nid}. Toca el nodo final.", creado
        if pendiente == nid:
            return t, None, "Trazo cancelado.", creado
        b = t["barras"].dropna(subset=["i", "j"])
        if len(b[((b["i"] == pendiente) & (b["j"] == nid)) | ((b["i"] == nid) & (b["j"] == pendiente))]):
            return t, nid, f"Ya existe una barra entre {pendiente} y {nid}.", creado
        bid = _siguiente_id(t["barras"]["id"])
        fila = pd.DataFrame([{"id": bid, "i": pendiente, "j": nid, "E": np.nan, "A": np.nan, "I": np.nan,
                              "art_i": False, "art_j": False}])
        t["barras"] = _tipar("barras", pd.concat([t["barras"], fila], ignore_index=True))
        sig = nid if op.get("encadenar", True) else None
        return t, sig, f"Barra {bid} creada ({pendiente} a {nid}).", True

    if modo == "Apoyo":
        if nid is None:
            return tablas, pendiente, "Toca un nodo existente para asignarle apoyo.", False
        ap = t["apoyos"]
        ap = ap[ap["nodo"] != nid]
        preset = APOYOS_PRESET[op["apoyo"]]
        if preset is not None:
            ap = pd.concat([ap, pd.DataFrame([{"nodo": nid, "rx": bool(preset[0]), "ry": bool(preset[1]),
                                               "rz": bool(preset[2])}])], ignore_index=True)
        t["apoyos"] = _tipar("apoyos", ap)
        return t, pendiente, (f"Apoyo quitado del nodo {nid}." if preset is None
                              else f"{op['apoyo']} en nodo {nid}."), True

    if modo == "Carga en nodo":
        if nid is None:
            return tablas, pendiente, "Toca un nodo existente para cargarlo.", False
        cn = t["cargas_nodales"]
        cn = cn[cn["nodo"] != nid]
        if any(abs(v) > 0 for v in (op["Fx"], op["Fy"], op["M"])):
            cn = pd.concat([cn, pd.DataFrame([{"nodo": nid, "Fx": op["Fx"], "Fy": op["Fy"], "M": op["M"]}])],
                           ignore_index=True)
            msg = f"Carga asignada al nodo {nid}."
        else:
            msg = f"Cargas del nodo {nid} eliminadas."
        t["cargas_nodales"] = _tipar("cargas_nodales", cn)
        return t, pendiente, msg, True

    if modo == "Carga en barra":
        if tipo != "b":
            return tablas, pendiente, "Toca el rombo naranja al centro de una barra.", False
        bid = int(punto["id"])
        fila = {"barra": bid, "tipo": op["tipo"], "dir": op["dir"], "w1": np.nan, "w2": np.nan,
                "a": np.nan, "b": np.nan, "P": np.nan}
        if op["tipo"] == "distribuida":
            fila["w1"] = op["w1"]
            fila["w2"] = op["w2"]
        else:
            fila["P"] = op["P"]
            fila["a"] = op["a"]
        t["cargas_barras"] = _tipar("cargas_barras", pd.concat([t["cargas_barras"], pd.DataFrame([fila])],
                                                              ignore_index=True))
        return t, pendiente, f"Carga agregada a la barra {bid}.", True

    if modo == "Articulación":
        b = t["barras"]
        if tipo == "b":
            bid = int(punto["id"])
            k = b.index[b["id"] == bid]
            ext = op["extremo"]
            for col in (["art_i"] if ext == "Extremo i" else ["art_j"] if ext == "Extremo j" else ["art_i", "art_j"]):
                b.loc[k, col] = ~b.loc[k, col].astype(bool)
            t["barras"] = _tipar("barras", b)
            return t, pendiente, f"Articulación cambiada en barra {bid} ({ext.lower()}).", True
        if nid is None:
            return tablas, pendiente, "Toca un nodo o el rombo de una barra.", False
        mi, mj = b["i"] == nid, b["j"] == nid
        if not (mi.any() or mj.any()):
            return tablas, pendiente, f"El nodo {nid} no tiene barras.", False
        todas = bool(b.loc[mi, "art_i"].all() and b.loc[mj, "art_j"].all())
        b.loc[mi, "art_i"] = not todas
        b.loc[mj, "art_j"] = not todas
        t["barras"] = _tipar("barras", b)
        return t, pendiente, (f"Articulación quitada en nodo {nid}." if todas
                              else f"Articulación en nodo {nid} (todas las barras que llegan)."), True

    if modo == "Borrar":
        if tipo == "b":
            bid = int(punto["id"])
            t["barras"] = _tipar("barras", t["barras"][t["barras"]["id"] != bid])
            t["cargas_barras"] = _tipar("cargas_barras", t["cargas_barras"][t["cargas_barras"]["barra"] != bid])
            return t, pendiente, f"Barra {bid} borrada.", True
        if nid is None:
            return tablas, pendiente, "No hay nada que borrar en ese punto.", False
        b = t["barras"]
        quitar = set(b.loc[(b["i"] == nid) | (b["j"] == nid), "id"].dropna().astype(int))
        t["barras"] = _tipar("barras", b[~b["id"].isin(quitar)])
        t["cargas_barras"] = _tipar("cargas_barras", t["cargas_barras"][~t["cargas_barras"]["barra"].isin(quitar)])
        t["nodos"] = _tipar("nodos", t["nodos"][t["nodos"]["id"] != nid])
        for nombre in ("apoyos", "resortes", "cargas_nodales", "asentamientos"):
            t[nombre] = _tipar(nombre, t[nombre][t[nombre]["nodo"] != nid])
        return t, (None if pendiente == nid else pendiente), \
            f"Nodo {nid} borrado" + (f" con barras {sorted(quitar)}." if quitar else "."), True

    return tablas, pendiente, "", False


# ---------------------------------------------------------------------
# Figura interactiva
# ---------------------------------------------------------------------
def _gvec(dir_, c, s):
    if dir_ == "local_x":
        return c, s
    if dir_ == "local_y":
        return -s, c
    if str(dir_).startswith("global_x"):
        return 1.0, 0.0
    return 0.0, 1.0


def figura(t, pendiente, grid):
    x0, x1, y0, y1, dx, dy = grid
    fig = go.Figure()
    # 0) cuadrícula
    xs = np.arange(x0, x1 + dx / 2, dx)
    ys = np.arange(y0, y1 + dy / 2, dy)
    GX, GY = np.meshgrid(xs, ys)
    gx, gy = np.round(GX.ravel(), 6), np.round(GY.ravel(), 6)
    fig.add_trace(go.Scatter(x=gx, y=gy, mode="markers", marker=dict(size=7, color="rgba(31,59,87,0.16)"),
                             customdata=[["g", 0] for _ in gx], hovertemplate="(%{x}, %{y})<extra></extra>",
                             showlegend=False))
    nodos = {int(r.id): (r.x, r.y) for r in t["nodos"].dropna(subset=["id", "x", "y"]).itertuples()}
    barras = []
    for r in t["barras"].dropna(subset=["id", "i", "j"]).itertuples():
        if int(r.i) in nodos and int(r.j) in nodos:
            barras.append((int(r.id), int(r.i), int(r.j), bool(r.art_i), bool(r.art_j)))
    # 1) barras (líneas)
    lx, ly = [], []
    for _, i, j, _, _ in barras:
        lx += [nodos[i][0], nodos[j][0], None]
        ly += [nodos[i][1], nodos[j][1], None]
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=AZUL, width=4), hoverinfo="skip",
                             showlegend=False))
    # 2) rombos de barra
    mx = [(nodos[i][0] + nodos[j][0]) / 2 for _, i, j, _, _ in barras]
    my = [(nodos[i][1] + nodos[j][1]) / 2 for _, i, j, _, _ in barras]
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers+text", marker=dict(size=12, symbol="diamond", color=NARANJA,
                             line=dict(color="white", width=1)), text=[f"({b[0]})" for b in barras],
                             textposition="bottom center", textfont=dict(size=11, color="#7e5109"),
                             customdata=[["b", b[0]] for b in barras],
                             hovertemplate="Barra %{customdata[1]}<extra></extra>", showlegend=False))
    # 3) apoyos
    simbolos = {(1, 1, 1): ("square", 24), (1, 1, 0): ("triangle-up", 22), (0, 1, 0): ("triangle-up-open", 22),
                (1, 0, 0): ("triangle-right-open", 22)}
    for r in t["apoyos"].dropna(subset=["nodo"]).itertuples():
        n = int(r.nodo)
        if n not in nodos:
            continue
        clave = (int(r.rx), int(r.ry), int(r.rz))
        sym, size = simbolos.get(clave, ("diamond-open", 20))
        desp = 0.28 * dy if sym.startswith("triangle-up") or sym == "square" else 0
        fig.add_trace(go.Scatter(x=[nodos[n][0] - (0.28 * dx if sym.startswith("triangle-right") else 0)],
                                 y=[nodos[n][1] - desp], mode="markers",
                                 marker=dict(size=size, symbol=sym, color="#7f8c8d", line=dict(color="#2c3e50", width=2)),
                                 hoverinfo="skip", showlegend=False))
    # 4) articulaciones
    hx, hy = [], []
    for _, i, j, ai, aj in barras:
        (xi, yi), (xj, yj) = nodos[i], nodos[j]
        L = math.hypot(xj - xi, yj - yi) or 1
        d = min(0.35 * min(dx, dy), 0.2 * L)
        if ai:
            hx.append(xi + (xj - xi) / L * d)
            hy.append(yi + (yj - yi) / L * d)
        if aj:
            hx.append(xj - (xj - xi) / L * d)
            hy.append(yj - (yj - yi) / L * d)
    if hx:
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", marker=dict(size=10, color="white",
                                 line=dict(color="black", width=2)), hoverinfo="skip", showlegend=False))
    # 5) nodos (encima de todo)
    ids = list(nodos)
    colores = [ROJO if n == pendiente else AZUL for n in ids]
    fig.add_trace(go.Scatter(x=[nodos[n][0] for n in ids], y=[nodos[n][1] for n in ids], mode="markers+text",
                             marker=dict(size=[18 if n == pendiente else 13 for n in ids], color=colores,
                                         line=dict(color="white", width=1.5)),
                             text=[str(n) for n in ids], textposition="top right",
                             textfont=dict(size=13, color=AZUL), customdata=[["n", n] for n in ids],
                             hovertemplate="Nodo %{customdata[1]} (%{x}, %{y})<extra></extra>", showlegend=False))
    # cargas (anotaciones)
    anot = []
    for r in t["cargas_nodales"].dropna(subset=["nodo"]).itertuples():
        n = int(r.nodo)
        if n not in nodos:
            continue
        fx, fy, mz = [0.0 if pd.isna(v) else float(v) for v in (r.Fx, r.Fy, r.M)]
        F = math.hypot(fx, fy)
        if F > 0:
            anot.append(dict(x=nodos[n][0], y=nodos[n][1], ax=-fx / F * 60, ay=fy / F * 60, showarrow=True,
                             arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor=ROJO,
                             text=f"{F:g} kN" if fx == 0 or fy == 0 else f"({fx:g}, {fy:g})",
                             font=dict(color=ROJO, size=11), bgcolor="rgba(255,255,255,0.8)"))
        if mz:
            anot.append(dict(x=nodos[n][0], y=nodos[n][1], xshift=-28, yshift=20, showarrow=False,
                             text=f"{'↺' if mz > 0 else '↻'} {abs(mz):g}", font=dict(color=ROJO, size=12)))
    info_b = {b[0]: b for b in barras}
    for r in t["cargas_barras"].dropna(subset=["barra"]).itertuples():
        bid = int(r.barra)
        if bid not in info_b:
            continue
        _, i, j, _, _ = info_b[bid]
        (xi, yi), (xj, yj) = nodos[i], nodos[j]
        L = math.hypot(xj - xi, yj - yi)
        if L == 0:
            continue
        c, s = (xj - xi) / L, (yj - yi) / L
        gxv, gyv = _gvec(r.dir if isinstance(r.dir, str) else "global_y", c, s)
        if str(r.tipo).startswith("pun"):
            if pd.isna(r.P):
                continue
            a = L / 2 if pd.isna(r.a) else float(r.a)
            sg = 1 if r.P >= 0 else -1
            anot.append(dict(x=xi + c * a, y=yi + s * a, ax=-sg * gxv * 55, ay=sg * gyv * 55, showarrow=True,
                             arrowhead=2, arrowwidth=2, arrowcolor=ROJO, text=f"{r.P:g} kN",
                             font=dict(color=ROJO, size=11), bgcolor="rgba(255,255,255,0.8)"))
        else:
            if pd.isna(r.w1):
                continue
            w1 = float(r.w1)
            w2 = w1 if pd.isna(r.w2) else float(r.w2)
            a = 0.0 if pd.isna(r.a) else float(r.a)
            b = L if pd.isna(r.b) else float(r.b)
            wm = max(abs(w1), abs(w2), 1e-9)
            for k, xx in enumerate(np.linspace(a, b, 6)):
                w = w1 + (w2 - w1) * (xx - a) / max(b - a, 1e-9)
                sg = 1 if w >= 0 else -1
                lng = 12 + 26 * abs(w) / wm
                anot.append(dict(x=xi + c * xx, y=yi + s * xx, ax=-sg * gxv * lng, ay=sg * gyv * lng,
                                 showarrow=True, arrowhead=2, arrowwidth=1.3, arrowcolor=ROJO,
                                 text=((f"{w1:g}" if w1 == w2 else f"{w1:g} a {w2:g}") + " kN/m") if k == 3 else "",
                                 font=dict(color=ROJO, size=10)))
    fig.update_layout(annotations=anot, height=560, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white", dragmode=False,
                      xaxis=dict(range=[x0 - dx, x1 + dx], zeroline=False, gridcolor="#eef2f6", title="x (m)"),
                      yaxis=dict(range=[y0 - dy, y1 + dy], zeroline=False, gridcolor="#eef2f6", title="y (m)",
                                 scaleanchor="x", scaleratio=1))
    return fig


# ---------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------
init()
ss = st.session_state


def _cb_ejemplo():
    _cargar_modelo(ejemplos()[ss.sel_ejemplo])


def _cb_json():
    f = ss.get("archivo_json")
    if f is None:
        return
    try:
        _cargar_modelo(json.loads(f.getvalue().decode("utf-8")))
        ss.msg = ("ok", f"Modelo '{f.name}' cargado.")
    except Exception as e:
        ss.msg = ("error", f"No se pudo leer el archivo: {e}")


with st.sidebar:
    st.header("Modelo")
    st.text_input("Título", key="titulo")
    st.selectbox("Ejemplos", list(ejemplos()), key="sel_ejemplo")
    st.button("Cargar ejemplo", on_click=_cb_ejemplo, width="stretch")
    st.subheader("Propiedades por defecto")
    st.caption("Se usan en barras sin E, A o I propios. Unidades: kN, m, kPa.")
    st.number_input("E (kPa)", key="E_def", format="%.4e", min_value=0.0)
    st.number_input("A (m²)", key="A_def", format="%.4e", min_value=0.0)
    st.number_input("I (m⁴)", key="I_def", format="%.4e", min_value=0.0)
    st.subheader("Cuadrícula de dibujo")
    c1, c2 = st.columns(2)
    c1.number_input("Paso x (m)", key="dx", min_value=0.05, step=0.25)
    c2.number_input("Paso y (m)", key="dy", min_value=0.05, step=0.25)
    c1.number_input("x mín", key="gx0", step=1.0)
    c2.number_input("x máx", key="gx1", step=1.0)
    c1.number_input("y mín", key="gy0", step=1.0)
    c2.number_input("y máx", key="gy1", step=1.0)
    st.subheader("Archivo")
    st.file_uploader("Abrir modelo (.json)", type=["json"], key="archivo_json", on_change=_cb_json)
    zona_guardar = st.empty()

st.title("Análisis matricial de estructuras planas")
st.caption("Armaduras, vigas y marcos 2D por el método de rigidez directa. Dibuja, revisa las tablas y calcula.")

if ss.get("msg"):
    tipo_msg, texto = ss.msg
    (st.success if tipo_msg == "ok" else st.error if tipo_msg == "error" else st.info)(texto)
    ss.msg = None

tab_dib, tab_tab, tab_res = st.tabs(["Dibujar", "Tablas", "Resultados"])

# ---- Tablas (se ejecuta primero para que el dibujo use lo editado) ----
with tab_tab:
    st.caption("Edita valores exactos aquí. Deja E, A o I vacíos para usar los valores por defecto. "
               "Cargas: + en el sentido positivo del eje (global_y negativo = hacia abajo).")
    num = lambda lbl, fmt="%g": st.column_config.NumberColumn(lbl, format=fmt)
    ids = lambda lbl: st.column_config.NumberColumn(lbl, format="%d", step=1)
    config = {
        "nodos": {"id": ids("Nodo"), "x": num("x (m)"), "y": num("y (m)")},
        "barras": {"id": ids("Barra"), "i": ids("Nodo i"), "j": ids("Nodo j"), "E": num("E (kPa)", "%.4g"),
                   "A": num("A (m²)", "%.4g"), "I": num("I (m⁴)", "%.4g"),
                   "art_i": st.column_config.CheckboxColumn("Articulado en i"),
                   "art_j": st.column_config.CheckboxColumn("Articulado en j")},
        "apoyos": {"nodo": ids("Nodo"), "rx": st.column_config.CheckboxColumn("Restringe x"),
                   "ry": st.column_config.CheckboxColumn("Restringe y"),
                   "rz": st.column_config.CheckboxColumn("Restringe giro")},
        "resortes": {"nodo": ids("Nodo"), "kx": num("kx (kN/m)"), "ky": num("ky (kN/m)"), "kz": num("kz (kN·m/rad)")},
        "cargas_nodales": {"nodo": ids("Nodo"), "Fx": num("Fx (kN)"), "Fy": num("Fy (kN)"), "M": num("M (kN·m)")},
        "cargas_barras": {"barra": ids("Barra"),
                          "tipo": st.column_config.SelectboxColumn("Tipo", options=TIPOS_CARGA, default="distribuida"),
                          "dir": st.column_config.SelectboxColumn("Dirección", options=DIRS, default="global_y"),
                          "w1": num("w1 (kN/m)"), "w2": num("w2 (kN/m)"), "a": num("a (m)"), "b": num("b (m)"),
                          "P": num("P (kN)")},
        "asentamientos": {"nodo": ids("Nodo"), "dx": num("dx (m)"), "dy": num("dy (m)"), "giro": num("giro (rad)")},
    }
    titulos = {"nodos": "Nodos", "barras": "Barras", "apoyos": "Apoyos", "cargas_nodales": "Cargas en nodos",
               "cargas_barras": "Cargas en barras", "resortes": "Resortes", "asentamientos": "Asentamientos"}
    ca, cb = st.columns(2)
    for k, nombre in enumerate(["nodos", "apoyos", "barras", "cargas_nodales", "cargas_barras", "resortes",
                                "asentamientos"]):
        col = ca if nombre in ("nodos", "apoyos", "cargas_nodales", "resortes") else cb
        with col:
            st.markdown(f"**{titulos[nombre]}**")
            if nombre == "cargas_barras":
                st.caption("Distribuida: w1, w2 (opcional), a y b en m desde el nodo i (opcionales). "
                           "Puntual: P y a.")
            df = st.data_editor(ss.base[nombre], key=f"ed_{nombre}_{ss.ver}", num_rows="dynamic",
                                column_config=config[nombre], hide_index=True, width="stretch")
            ss.tablas[nombre] = _tipar(nombre, df)

# ---- Dibujar ----
with tab_dib:
    c_herr, c_fig = st.columns([1, 3])
    with c_herr:
        modo = st.radio("Herramienta", ["Nodo", "Barra", "Apoyo", "Carga en nodo", "Carga en barra",
                                        "Articulación", "Borrar"], key="modo")
        op = {}
        ayuda = {
            "Nodo": "Toca un punto de la cuadrícula.",
            "Barra": "Toca el nodo inicial y luego el final. Si tocas la cuadrícula se crea el nodo.",
            "Apoyo": "Elige el tipo y toca un nodo.",
            "Carga en nodo": "Define la carga y toca un nodo. Con todo en cero se borran sus cargas.",
            "Carga en barra": "Define la carga y toca el rombo naranja de la barra.",
            "Articulación": "Toca un nodo para articular todas las barras que llegan, o el rombo de una barra "
                            "para elegir el extremo.",
            "Borrar": "Toca un nodo (borra sus barras) o el rombo de una barra.",
        }
        st.caption(ayuda[modo])
        if modo == "Barra":
            op["encadenar"] = st.checkbox("Encadenar barras", value=True,
                                          help="El nodo final se vuelve el inicio de la siguiente barra.")
            if ss.pendiente is not None:
                st.info(f"Trazando desde el nodo {ss.pendiente}.")
                if st.button("Terminar trazo"):
                    ss.pendiente = None
                    ss.chart_ver += 1
                    st.rerun()
        elif modo == "Apoyo":
            op["apoyo"] = st.selectbox("Tipo de apoyo", list(APOYOS_PRESET))
        elif modo == "Carga en nodo":
            op["Fx"] = st.number_input("Fx (kN)", value=0.0)
            op["Fy"] = st.number_input("Fy (kN)", value=-10.0)
            op["M"] = st.number_input("M (kN·m, + antihorario)", value=0.0)
        elif modo == "Carga en barra":
            op["tipo"] = st.selectbox("Tipo", TIPOS_CARGA)
            op["dir"] = st.selectbox("Dirección", DIRS,
                                     help="global_y_proy: por metro de proyección horizontal (techos inclinados).")
            if op["tipo"] == "distribuida":
                op["w1"] = st.number_input("w1 (kN/m)", value=-10.0)
                op["w2"] = st.number_input("w2 (kN/m)", value=op["w1"])
            else:
                op["P"] = st.number_input("P (kN)", value=-10.0)
                op["a"] = st.number_input("a (m desde el nodo i)", value=1.0, min_value=0.0)
        elif modo == "Articulación":
            op["extremo"] = st.radio("Al tocar una barra", ["Extremo i", "Extremo j", "Ambos extremos"])
        if modo != "Barra" and ss.pendiente is not None:
            ss.pendiente = None
        st.divider()
        st.button("Deshacer", on_click=deshacer, disabled=not ss.get("historial"), width="stretch")

    with c_fig:
        grid = (ss.gx0, ss.gx1, ss.gy0, ss.gy1, ss.dx, ss.dy)
        npts = ((ss.gx1 - ss.gx0) / ss.dx + 1) * ((ss.gy1 - ss.gy0) / ss.dy + 1)
        if npts > 6000 or ss.gx1 <= ss.gx0 or ss.gy1 <= ss.gy0:
            st.error("La cuadrícula es demasiado densa o sus límites no son válidos. Ajusta el paso o los límites.")
        else:
            fig = figura(ss.tablas, ss.pendiente, grid)
            evento = st.plotly_chart(fig, key=f"dib_{ss.chart_ver}", on_select="rerun", selection_mode="points",
                                     theme=None, config={"displayModeBar": False, "scrollZoom": False})
            puntos = (evento.selection or {}).get("points", []) if evento else []
            if puntos:
                p = puntos[0]
                cd = p.get("customdata") or ["g", 0]
                punto = {"tipo": cd[0], "id": cd[1], "x": round(float(p["x"]), 6), "y": round(float(p["y"]), 6)}
                nuevas, pend, msg, cambio = aplicar_clic(ss.tablas, punto, modo, op, ss.pendiente)
                ss.pendiente = pend
                if cambio:
                    aplicar_tablas(nuevas)
                else:
                    ss.chart_ver += 1
                if msg:
                    ss.msg = ("info", msg)
                st.rerun()
        st.caption("Azul: nodos. Rombos naranjas: barras (tócalos para cargarlas, articularlas o borrarlas). "
                   "Círculos blancos: articulaciones. Nodo rojo: inicio del trazo actual.")

# ---- Resultados ----
with tab_res:
    modelo, avisos = tablas_a_modelo(ss.tablas, ss.E_def, ss.A_def, ss.I_def, ss.titulo)
    firma = json.dumps(modelo, sort_keys=True, default=str)
    for a in avisos:
        st.warning(a)
    if st.button("Calcular", type="primary"):
        try:
            with st.spinner("Resolviendo el sistema y armando el PDF..."):
                res = motor.analizar(modelo)
                buf = io.BytesIO()
                motor.generar_pdf(res, buf)
            ss.res = {"res": res, "pdf": buf.getvalue(), "firma": firma,
                      "figs": {"Modelo": motor.fig_modelo(res).getvalue(),
                               "Deformada": motor.fig_deformada(res).getvalue(),
                               "Axial N": motor.fig_diagrama(res, "N").getvalue(),
                               "Cortante V": motor.fig_diagrama(res, "V").getvalue(),
                               "Momento M": motor.fig_diagrama(res, "M").getvalue()}}
        except motor.ErrorEstructura as e:
            ss.res = None
            st.error(f"Revisa el modelo: {e}")
        except Exception as e:
            ss.res = None
            st.error(f"Error inesperado al calcular: {e}")

    if ss.get("res"):
        R = ss.res
        res = R["res"]
        if R["firma"] != firma:
            st.warning("El modelo cambió después del último cálculo. Presiona Calcular para actualizar.")
        GI = res["GI"]
        clas = "Determinada" if GI == 0 else f"Indeterminada (grado {GI})" if GI > 0 else "Revisar"
        umax = max(float(np.max(np.hypot(m["def_u"] * m["c"] - m["def_v"] * m["s"],
                                         m["def_u"] * m["s"] + m["def_v"] * m["c"]))) for m in res["miembros"]) * 1000
        sfx, sfy, sm = res["equilibrio"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Grado de indeterminación", f"GI = {GI}", help=clas)
        m2.metric("Desplazamiento máx.", f"{umax:.3f} mm", help="Incluye puntos intermedios de las barras.")
        m3.metric("|M| máximo", f"{max(max(abs(m['M_max'][0]), abs(m['M_min'][0])) for m in res['miembros']):.2f} kN·m")
        m4.metric("Equilibrio", "Cumple" if max(abs(sfx), abs(sfy), abs(sm)) < 1e-6 * res["escala_f"] * 100 else "Revisar")
        st.download_button("Descargar PDF con el desarrollo", R["pdf"], file_name="analisis_matricial.pdf",
                           mime="application/pdf", type="primary")

        tabs_fig = st.tabs(list(R["figs"]))
        for tf, (nombre, png) in zip(tabs_fig, R["figs"].items()):
            with tf:
                st.image(png, width="stretch")

        idx = res["idx"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Desplazamientos**")
            st.dataframe(pd.DataFrame([{"Nodo": n, "u (mm)": res["u"][3 * idx[n]] * 1000,
                                        "v (mm)": res["u"][3 * idx[n] + 1] * 1000,
                                        "giro (mrad)": res["u"][3 * idx[n] + 2] * 1000} for n in res["ids"]]),
                         hide_index=True, width="stretch")
        with c2:
            st.markdown("**Reacciones**")
            filas = []
            for n in res["ids"]:
                g = 3 * idx[n]
                if n in res["apoyos"]:
                    filas.append({"Nodo": str(n), **{k: (res["reac"][g + q] if res["restr"][g + q] else None)
                                                     for q, k in enumerate(["Rx (kN)", "Ry (kN)", "Mz (kN·m)"])}})
                if n in res["resortes"]:
                    filas.append({"Nodo": f"{n} (resorte)", **{k: (res["fres"][g + q] if res["kres"][g + q] else None)
                                                               for q, k in enumerate(["Rx (kN)", "Ry (kN)", "Mz (kN·m)"])}})
            st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
        st.markdown("**Elementos mecánicos por barra** (x desde el nodo i; N + tensión; M + tensiona el lado -y local)")
        st.dataframe(pd.DataFrame([{"Barra": m["id"], "N i": m["N"][0], "N j": m["N"][-1], "V i": m["V"][0],
                                    "V j": m["V"][-1], "M i": m["M"][0], "M j": m["M"][-1],
                                    "M máx": m["M_max"][0], "x M máx (m)": m["M_max"][1],
                                    "M mín": m["M_min"][0], "x M mín (m)": m["M_min"][1]} for m in res["miembros"]]),
                     hide_index=True, width="stretch")
    else:
        st.info("Presiona Calcular para resolver la estructura.")

# El botón de guardar va al final para incluir las ediciones hechas en esta misma ejecución
mod_actual, _ = tablas_a_modelo(ss.tablas, ss.E_def, ss.A_def, ss.I_def, ss.titulo)
zona_guardar.download_button("Guardar modelo (.json)", json.dumps(mod_actual, indent=2, ensure_ascii=False),
                             file_name="modelo_estructura.json", mime="application/json", width="stretch")
