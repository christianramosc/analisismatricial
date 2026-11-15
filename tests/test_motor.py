# -*- coding: utf-8 -*-
"""Pruebas del motor contra soluciones cerradas.  Ejecutar:  pytest"""
import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analisis_matricial_marcos as motor  # noqa: E402

E, A, I = 200e6, 0.006, 1.2e-4
EI, EA = E * I, E * A


def viga(L=6.0, apoyos=None, **extra):
    mod = {"E": E, "A": A, "I": I, "nodos": {1: (0, 0), 2: (L, 0)}, "barras": {1: {"i": 1, "j": 2}},
           "apoyos": apoyos or {1: (1, 1, 0), 2: (0, 1, 0)}}
    mod.update(extra)
    return mod


def reac(res, nodo, q):
    return res["reac"][3 * res["idx"][nodo] + q]


def revisar(res):
    assert motor.equilibrio_ok(res)
    escala = max(1.0, max(np.abs(m["f"]).max() for m in res["miembros"]))
    assert max(m["cierre"] for m in res["miembros"]) < 1e-8 * escala


def test_viga_simple_carga_uniforme():
    L, w = 6.0, 10.0
    res = motor.analizar(viga(L, cargas_barras=[{"barra": 1, "tipo": "distribuida", "w1": -w, "dir": "global_y"}]))
    revisar(res)
    m = res["miembros"][0]
    assert m["M_max"][0] == pytest.approx(w * L ** 2 / 8, rel=1e-6)
    assert m["M_max"][1] == pytest.approx(L / 2, abs=L / 200)
    assert reac(res, 1, 1) == pytest.approx(w * L / 2)
    assert m["flecha"]["valor"] == pytest.approx(-5 * w * L ** 4 / (384 * EI), rel=1e-4)
    assert res["GI"] == 0


def test_voladizo_carga_en_la_punta():
    L, P = 3.0, 12.0
    res = motor.analizar(viga(L, apoyos={1: (1, 1, 1)}, cargas_nodales={2: (0, -P, 0)}))
    revisar(res)
    assert res["u"][4] == pytest.approx(-P * L ** 3 / (3 * EI), rel=1e-9)
    assert reac(res, 1, 2) == pytest.approx(P * L)
    assert res["miembros"][0]["M"][0] == pytest.approx(-P * L)


def test_viga_empotrada_articulada():
    L, w = 5.0, 8.0
    res = motor.analizar(viga(L, apoyos={1: (1, 1, 1), 2: (0, 1, 0)},
                              cargas_barras=[{"barra": 1, "tipo": "distribuida", "w1": -w, "dir": "global_y"}]))
    revisar(res)
    assert reac(res, 2, 1) == pytest.approx(3 * w * L / 8)
    assert reac(res, 1, 2) == pytest.approx(w * L ** 2 / 8)
    assert res["GI"] == 1


def test_momento_concentrado_en_barra():
    L, M0, a = 8.0, 40.0, 3.0
    res = motor.analizar(viga(L, cargas_barras=[{"barra": 1, "tipo": "momento", "M": M0, "a": a}]))
    revisar(res)
    m = res["miembros"][0]
    # momento antihorario: reacciones -M0/L en el apoyo derecho... par de reacciones = M0/L
    assert reac(res, 2, 1) == pytest.approx(-M0 / L)
    assert reac(res, 1, 1) == pytest.approx(M0 / L)
    antes = np.interp(a - 1e-4, m["x"], m["M"])
    despues = np.interp(a + 1e-4, m["x"], m["M"])
    assert antes == pytest.approx(M0 * a / L, rel=1e-3)
    assert antes - despues == pytest.approx(M0, rel=1e-3)
    assert m["M"][0] == pytest.approx(0, abs=1e-9) and m["M"][-1] == pytest.approx(0, abs=1e-9)


def test_momento_en_barra_equivale_a_momento_en_nodo():
    """Un momento en barra justo en un nodo intermedio debe dar lo mismo que el momento nodal."""
    base = {"E": E, "A": A, "I": I, "nodos": {1: (0, 0), 2: (4, 0), 3: (9, 0)},
            "barras": {1: {"i": 1, "j": 2}, 2: {"i": 2, "j": 3}}, "apoyos": {1: (1, 1, 1), 3: (0, 1, 0)}}
    r1 = motor.analizar({**base, "cargas_nodales": {2: (0, 0, 25.0)}})
    r2 = motor.analizar({**base, "cargas_barras": [{"barra": 2, "tipo": "momento", "M": 25.0, "a": 0.0}]})
    assert np.allclose(r1["u"], r2["u"], atol=1e-12)
    assert np.allclose(r1["reac"], r2["reac"], atol=1e-9)


def test_temperatura_uniforme_restringida():
    L, dT, al = 5.0, 30.0, 1.2e-5
    res = motor.analizar(viga(L, apoyos={1: (1, 1, 1), 2: (1, 1, 1)},
                              cargas_barras=[{"barra": 1, "tipo": "termica", "dT": dT}]))
    m = res["miembros"][0]
    assert m["N"][0] == pytest.approx(-EA * al * dT)
    assert np.allclose(m["def_u"], 0, atol=1e-12)


def test_temperatura_uniforme_libre():
    L, dT, al = 5.0, 30.0, 1.2e-5
    res = motor.analizar(viga(L, cargas_barras=[{"barra": 1, "tipo": "termica", "dT": dT}]))
    m = res["miembros"][0]
    assert res["u"][3] == pytest.approx(al * dT * L)
    assert np.allclose(m["N"], 0, atol=1e-6)
    assert m["def_u"][-1] == pytest.approx(al * dT * L)


def test_gradiente_termico_viga_simple():
    L, dTg, h, al = 6.0, 20.0, 0.4, 1.2e-5       # cara superior (+y) más caliente -> se arquea hacia arriba
    res = motor.analizar(viga(L, cargas_barras=[{"barra": 1, "tipo": "termica", "dTg": dTg, "h": h}]))
    m = res["miembros"][0]
    assert np.allclose(m["M"], 0, atol=1e-6)
    assert m["flecha"]["valor"] == pytest.approx(al * dTg / h * L ** 2 / 8, rel=1e-4)


def test_gradiente_termico_doble_empotre():
    L, dTg, h, al = 6.0, 20.0, 0.4, 1.2e-5
    res = motor.analizar(viga(L, apoyos={1: (1, 1, 1), 2: (1, 1, 1)},
                              cargas_barras=[{"barra": 1, "tipo": "termica", "dTg": dTg, "h": h}]))
    m = res["miembros"][0]
    assert np.allclose(m["M"], EI * al * dTg / h, rtol=1e-9)    # tensión en la cara fría (-y)
    assert np.allclose(m["def_v"], 0, atol=1e-12)


def test_gradiente_sin_peralte_da_error():
    with pytest.raises(motor.ErrorEstructura):
        motor.analizar(viga(cargas_barras=[{"barra": 1, "tipo": "termica", "dTg": 10}]))


def test_peso_propio():
    L, gamma = 6.0, 77.0
    res = motor.analizar(viga(L, peso_propio=True, gamma=gamma))
    revisar(res)
    assert reac(res, 1, 1) + reac(res, 2, 1) == pytest.approx(gamma * A * L)


def test_combinaciones_y_superposicion():
    mod = viga(6.0,
               cargas_barras=[{"barra": 1, "tipo": "distribuida", "w1": -10, "dir": "global_y", "caso": "CM"},
                              {"barra": 1, "tipo": "puntual", "P": -20, "a": 2.0, "dir": "global_y", "caso": "cv"}],
               cargas_nodales=[{"nodo": 2, "Fx": 5, "Fy": 0, "M": 0, "caso": "VIENTO"}],
               combinaciones={"Servicio": {"CM": 1, "CV": 1}, "Resistencia": {"CM": 1.3, "CV": 1.5},
                              "Viento": {"CM": 1.1, "CV": 1.1, "VIENTO": 1.1}})
    todos = motor.analizar_combinaciones(mod)
    assert list(todos) == ["Servicio", "Resistencia", "Viento"]
    cm = motor.analizar(mod, {"CM": 1})
    cv = motor.analizar(mod, {"CV": 1})
    assert np.allclose(todos["Resistencia"]["u"], 1.3 * cm["u"] + 1.5 * cv["u"], atol=1e-12)
    assert todos["Servicio"]["casos"] == ["CM", "CV", "VIENTO"]
    assert reac(todos["Servicio"], 1, 0) == pytest.approx(0, abs=1e-9)
    assert reac(todos["Viento"], 1, 0) == pytest.approx(-5.5)
    for r in todos.values():
        revisar(r)
    env = motor.envolvente(todos)
    assert env["barras"][1]["M"]["combo_max"] == "Resistencia"
    assert env["barras"][1]["M"]["max"] == pytest.approx(todos["Resistencia"]["miembros"][0]["M_max"][0])
    assert env["reacciones"][(1, 1)][1] == "Resistencia"
    with pytest.raises(motor.ErrorEstructura):
        motor.analizar(mod, "No existe")


def test_sin_combinaciones_usa_todas_las_cargas():
    todos = motor.analizar_combinaciones(motor.MODELO)
    assert list(todos) == [motor.SIN_COMBINACION]


def test_modelo_de_ejemplo_no_cambia():
    """Valores de referencia del marco de ejemplo (regresión)."""
    res = motor.analizar(motor.MODELO)
    revisar(res)
    assert res["GI"] == 2
    total_y = sum(reac(res, n, 1) for n in (1, 4))
    assert total_y == pytest.approx(20 * 6 + 8 * 2 + 10)
    total_x = sum(reac(res, n, 0) for n in (1, 4))
    assert total_x == pytest.approx(-(15 + 5 * 4))


def test_armadura():
    nod = {1: (0, 0), 2: (4, 0), 3: (2, 3)}
    mod = {"E": E, "A": 0.002, "I": 1e-6, "nodos": nod,
           "barras": {k: {"i": a, "j": b, "art_i": True, "art_j": True} for k, (a, b) in
                      {1: (1, 2), 2: (1, 3), 3: (3, 2)}.items()},
           "apoyos": {1: (1, 1, 0), 2: (0, 1, 0)}, "cargas_nodales": {3: (0, -30, 0)}}
    res = motor.analizar(mod)
    revisar(res)
    L = np.hypot(2, 3)
    assert res["miembros"][1]["N"][0] == pytest.approx(-15 * L / 3)
    assert res["miembros"][0]["N"][0] == pytest.approx(10.0)
    assert res["GI"] == 0


def test_inestable():
    with pytest.raises(motor.ErrorEstructura):
        motor.analizar(viga(apoyos={1: (0, 1, 0), 2: (0, 1, 0)}, cargas_nodales={2: (0, -1, 0)}))


def test_estaciones():
    L, w = 6.0, 10.0
    res = motor.analizar(viga(L, cargas_barras=[{"barra": 1, "tipo": "distribuida", "w1": -w, "dir": "global_y"}]))
    est = motor.estaciones(res["miembros"][0], 7)
    assert len(est["x"]) == 7
    assert est["M"][3] == pytest.approx(w * L ** 2 / 8, rel=1e-6)
    assert est["V"][0] == pytest.approx(w * L / 2)


def test_secciones():
    r = motor.seccion_rectangular(0.3, 0.6)
    assert r["A"] == pytest.approx(0.18) and r["I"] == pytest.approx(0.3 * 0.6 ** 3 / 12)
    c = motor.seccion_circular(0.5)
    t = motor.seccion_tubo_circular(0.5, 0.25)
    assert t["A"] == pytest.approx(c["A"]) and t["I"] == pytest.approx(c["I"])
    caj = motor.seccion_cajon(0.2, 0.3, 0.1)
    assert caj["A"] == pytest.approx(0.06) and caj["I"] == pytest.approx(0.2 * 0.3 ** 3 / 12)
    i = motor.seccion_I(0.3, 0.15, 0.01, 0.15)          # alma tan ancha como el patín = rectángulo
    assert i["A"] == pytest.approx(0.045) and i["I"] == pytest.approx(0.15 * 0.3 ** 3 / 12)
    assert motor.E_concreto(25, "ACI") == pytest.approx(4700 * 5 * 1000)
    assert motor.E_concreto(25, "NTC") == pytest.approx(4400 * 5 * 1000)


def test_pdf_y_excel():
    mod = viga(6.0, peso_propio=True,
               cargas_barras=[{"barra": 1, "tipo": "distribuida", "w1": -10, "dir": "global_y", "caso": "CV"},
                              {"barra": 1, "tipo": "momento", "M": 15, "a": 2.0, "caso": "CV"},
                              {"barra": 1, "tipo": "termica", "dT": 20, "dTg": 10, "h": 0.3, "caso": "T"}],
               combinaciones={"Servicio": {"CM": 1, "CV": 1, "T": 1}, "Resistencia": {"CM": 1.3, "CV": 1.5}})
    todos = motor.analizar_combinaciones(mod)
    env = motor.envolvente(todos)
    pdf = io.BytesIO()
    motor.generar_pdf(todos["Servicio"], pdf, env=env)
    assert pdf.getvalue()[:4] == b"%PDF" and len(pdf.getvalue()) > 20000
    xls = io.BytesIO()
    motor.exportar_excel(todos, xls, env)
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(xls.getvalue()))
    assert {"Desplazamientos", "Reacciones", "Elementos mecánicos", "Estaciones", "Envolvente"} <= set(wb.sheetnames)
    assert wb["Reacciones"].max_row > 2
