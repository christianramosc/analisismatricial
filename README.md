# Análisis matricial de estructuras planas

Aplicación de Streamlit y motor en Python para analizar armaduras, vigas y marcos 2D por el
método de rigidez directa (3 grados de libertad por nodo). Genera un PDF con el desarrollo
matricial paso a paso y exporta los resultados a Excel.

## Qué resuelve

- Estructuras determinadas e indeterminadas (calcula el grado de indeterminación y detecta mecanismos).
- Apoyos de cualquier tipo, resortes, asentamientos y articulaciones internas.
- Cargas nodales y cargas en barras: distribuidas (uniformes, trapezoidales, parciales), puntuales
  y **momentos concentrados**, en ejes locales, globales o por proyección.
- **Temperatura**: cambio uniforme (ΔT) y gradiente entre caras (ΔTg con el peralte h).
- **Peso propio** automático (peso volumétrico × área), por defecto o por barra.
- **Casos de carga y combinaciones** (por ejemplo `1.3 CM + 1.5 CV`) con **envolvente** de N, V, M,
  reacciones y flechas, indicando la combinación que rige.
- **Flechas por barra** y su relación L/flecha contra un límite L/n que tú eliges.
- Detalle interactivo por barra (N, V, M y deflexión) y valores por estaciones.
- Calculadora de sección (rectangular, circular, tubo, cajón/PTR, perfil I) y módulo del concreto.

Unidades: kN, m, kPa, rad (E del acero = 200e6 kPa).

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usar el motor sin la app (Colab o scripts)

```python
import analisis_matricial_marcos as am

modelo = dict(am.MODELO)                      # edita nodos, barras, apoyos y cargas
modelo["combinaciones"] = {"Servicio": {"CM": 1, "CV": 1}, "Resistencia": {"CM": 1.3, "CV": 1.5}}

todos = am.analizar_combinaciones(modelo)     # {nombre: resultados}
env = am.envolvente(todos)
am.generar_pdf(todos["Resistencia"], "resultados.pdf", env=env, limite_flecha=240)
am.exportar_excel(todos, "resultados.xlsx", env)
```

El formato completo del modelo está comentado en `MODELO`, al inicio de `analisis_matricial_marcos.py`.

## Pruebas

```bash
pip install pytest
pytest
```

Las pruebas comparan el motor con soluciones cerradas (viga simple, voladizo, viga empotrada-articulada,
momento concentrado, temperatura uniforme y gradiente, peso propio, superposición de combinaciones, armadura).

## Despliegue en Streamlit Community Cloud

1. En https://share.streamlit.io elige **New app**.
2. Repositorio: `christianramosc/analisismatricial`, rama `main`.
3. Archivo principal: `app.py`.
