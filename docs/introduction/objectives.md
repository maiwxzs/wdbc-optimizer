# Objetivos

## Objetivo general

Implementar y auditar, desde sus fundamentos matemáticos, un optimizador de gradiente descendente para la regresión logística regularizada sobre el conjunto WDBC, y demostrar empíricamente su convergencia, sus límites de estabilidad y la corrección de su gradiente.

## Objetivos específicos y su verificación

| # | Objetivo | Cómo se verifica en el repositorio |
|---|---|---|
| 1 | Formular la regresión logística como problema convexo y derivar $\nabla J$ y $L$ | [teoría](../theory/mathematical_foundations.md); `test_hessian_is_psd_and_bounded_by_lipschitz_constant`, `test_convexity_along_random_segments` |
| 2 | Implementar una sigmoide y una pérdida que no desborden ni produzcan `NaN` | `test_sigmoid_never_overflows`, `test_bce_handles_saturation`; `pytest.ini` convierte los `RuntimeWarning` en errores |
| 3 | Implementar el gradiente descendente con criterios de parada robustos | `tests/test_gradient_descent.py` (los cuatro motivos de parada) |
| 4 | Verificar el gradiente analítico con diferencias finitas | `test_gradient_check_*`; experimento E5 |
| 5 | Caracterizar el error de las diferencias finitas en función de $h$ | `test_step_sweep_has_v_shape`; experimento E5 |
| 6 | Estudiar el efecto de $\alpha$ y $\lambda$ en la convergencia | experimentos E2, E3 y E4 |
| 7 | Evaluar el clasificador con métricas clínicamente relevantes | experimento E1; `test_auc_matches_bruteforce` |
| 8 | Ofrecer una interfaz que haga visible todo lo anterior | `frontend/` (tres pestañas) |
| 9 | Hacer el trabajo reproducible y verificable por terceros | `scripts/reproduce_experiments.py`, CI en `.github/workflows/ci.yml` |
