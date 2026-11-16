# Bibliografía

Formato IEEE. Cada referencia indica el concepto de la documentación que respalda. Los DOI incluidos se verificaron en el registro del editor o en bases bibliográficas públicas. Donde no se indica DOI es porque no se verificó, no porque no exista.

## Conjunto de datos

[1] W. Wolberg, O. Mangasarian, N. Street y W. Street, "Breast Cancer Wisconsin (Diagnostic)," UCI Machine Learning Repository, 1993. [Online]. Disponible: https://doi.org/10.24432/C5DW2B. Licencia CC BY 4.0.
→ [dataset](../experiments/dataset.md)

[2] W. N. Street, W. H. Wolberg y O. L. Mangasarian, "Nuclear feature extraction for breast tumor diagnosis," en *Proc. SPIE 1905, Biomedical Image Processing and Biomedical Visualization*, San José, CA, EE. UU., 1993, pp. 861–870, doi: 10.1117/12.148698.
→ origen de las 30 características

## Regresión logística y estadística

[3] D. R. Cox, "The regression analysis of binary sequences," *J. Roy. Statist. Soc. Ser. B*, vol. 20, no. 2, pp. 215–242, 1958, doi: 10.1111/j.2517-6161.1958.tb00292.x.
→ modelo logístico ([probabilidad](../theory/probability_and_statistics.md))

[4] A. Albert y J. A. Anderson, "On the existence of maximum likelihood estimates in logistic regression models," *Biometrika*, vol. 71, no. 1, pp. 1–10, 1984.
→ separación y no existencia del estimador (experimento E4)

[5] J. A. Hanley y B. J. McNeil, "The meaning and use of the area under a receiver operating characteristic (ROC) curve," *Radiology*, vol. 143, no. 1, pp. 29–36, 1982, doi: 10.1148/radiology.143.1.7063747.
→ interpretación de la AUC

[6] H. B. Mann y D. R. Whitney, "On a test of whether one of two random variables is stochastically larger than the other," *Ann. Math. Statist.*, vol. 18, no. 1, pp. 50–60, 1947.
→ estadístico U, implementación de `roc_auc`

[7] T. Hastie, R. Tibshirani y J. Friedman, *The Elements of Statistical Learning*, 2.ª ed. Nueva York, NY, EE. UU.: Springer, 2009.
→ regresión logística, regularización, métricas

[8] C. M. Bishop, *Pattern Recognition and Machine Learning*. Nueva York, NY, EE. UU.: Springer, 2006.
→ entropía cruzada, interpretación MAP de L2

## Optimización

[9] J. Nocedal y S. J. Wright, *Numerical Optimization*, 2.ª ed. Nueva York, NY, EE. UU.: Springer, 2006.
→ gradiente descendente, Newton, búsqueda de línea, diferencias finitas ([optimización](../theory/optimization_algorithms.md))

[10] S. Boyd y L. Vandenberghe, *Convex Optimization*. Cambridge, Reino Unido: Cambridge Univ. Press, 2004.
→ convexidad, convexidad fuerte, cotas de los criterios de parada

[11] Y. Nesterov, *Introductory Lectures on Convex Optimization: A Basic Course*. Boston, MA, EE. UU.: Kluwer Academic, 2004.
→ funciones $L$-suaves, lema de descenso, velocidad $(1-\mu/L)^k$

## Análisis numérico y punto flotante

[12] IEEE, *IEEE Standard for Floating-Point Arithmetic*, IEEE Std 754-2019 (Revisión de IEEE Std 754-2008), 2019, doi: 10.1109/IEEESTD.2019.8766229.
→ formato `float64` ([punto flotante](../theory/floating_point.md))

[13] D. Goldberg, "What every computer scientist should know about floating-point arithmetic," *ACM Comput. Surv.*, vol. 23, no. 1, pp. 5–48, 1991, doi: 10.1145/103162.103163.
→ redondeo, cancelación, valores especiales

[14] N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2.ª ed. Filadelfia, PA, EE. UU.: SIAM, 2002.
→ análisis de errores, evaluación estable de funciones

[15] R. L. Burden y J. D. Faires, *Numerical Analysis*, 9.ª ed. Boston, MA, EE. UU.: Brooks/Cole, 2011.
→ diferenciación numérica, error de truncamiento frente a redondeo ([derivación numérica](../theory/numerical_differentiation.md))

## Cálculo y álgebra lineal

[16] J. E. Marsden y A. Tromba, *Vector Calculus*, 6.ª ed. Nueva York, NY, EE. UU.: W. H. Freeman, 2012.
→ gradiente, regla de la cadena, Hessiana, Taylor multivariable ([cálculo](../theory/calculus.md))

[17] G. Strang, *Introduction to Linear Algebra*, 5.ª ed. Wellesley, MA, EE. UU.: Wellesley-Cambridge Press, 2016.
→ matrices simétricas, autovalores, cociente de Rayleigh ([álgebra lineal](../theory/linear_algebra.md))

## Software

[18] C. R. Harris *et al.*, "Array programming with NumPy," *Nature*, vol. 585, no. 7825, pp. 357–362, 2020, doi: 10.1038/s41586-020-2649-2.
→ NumPy

[19] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, no. 85, pp. 2825–2830, 2011.
→ carga de datos, partición, `StandardScaler`, PCA

[20] NumPy Developers, "numpy.logaddexp," *NumPy Documentation*. [Online]. Disponible: https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
→ implementación de Log-Sum-Exp
