# Guía de contribución

Gracias por su interés. Este es un proyecto académico, así que las contribuciones deben mantener su principio central: **el núcleo numérico se implementa explícitamente con NumPy y cada fórmula está documentada y verificada.**

## Preparar el entorno

```bash
git clone https://github.com/maiwxzs/wdbc-optimizer.git
cd wdbc-optimizer/backend
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt

cd ../frontend
npm ci
```

## Antes de abrir un *pull request*

```bash
cd backend
ruff check . ../scripts ../examples  # estilo y análisis estático
python -m pytest                     # todas las pruebas deben pasar
cd ../frontend
npm run typecheck && npm run build
```

La integración continua (`.github/workflows/ci.yml`) ejecuta lo mismo en Python 3.10 a 3.13 y en Node 20.

## Reglas del proyecto

1. **Nada de bibliotecas de ML en el núcleo.** `app/math_engine`, `app/optimization` y `app/evaluation` solo pueden importar NumPy y la biblioteca estándar. `tests/test_architecture.py` lo verifica y fallará si no se cumple.
2. **Cada cambio matemático necesita:**
   - una prueba que lo contraste con una referencia **independiente** (diferencias finitas, forma cerrada, fuerza bruta…), no consigo mismo;
   - la documentación correspondiente en `docs/theory/` o `docs/implementation/`, con la fórmula en LaTeX y la correspondencia con el código.
3. **Sin avisos numéricos.** `pytest.ini` convierte `RuntimeWarning` en error. Si una ruta inestable es intencional (por ejemplo, el modo `naive`), silencie los avisos de forma explícita con `np.errstate` y documéntelo.
4. **JSON estricto.** Toda respuesta de la API debe pasar `json.dumps(..., allow_nan=False)`.
5. **No invente resultados.** Si un cambio afecta a cifras documentadas, regenere `results/` con `python scripts/reproduce_experiments.py` y actualice `docs/experiments/results.md`.
6. **Estilo:** PEP 8 verificado con Ruff (configuración en `backend/pyproject.toml`), *type hints* en las funciones públicas y *docstrings* en español.

## Mensajes de *commit*

Frase corta en imperativo que describa el cambio, por ejemplo: `Añade búsqueda de línea de Armijo al optimizador`.

## Informar de errores

Abra un *issue* con: versión de Python y sistema operativo, los pasos para reproducirlo, el resultado esperado y el obtenido y, si aplica, el valor de `stop_reason` y de `divergence_detail`. Para vulnerabilidades, consulte [SECURITY.md](SECURITY.md).

## Código de conducta

La participación en este proyecto se rige por el [Código de Conducta](CODE_OF_CONDUCT.md).
