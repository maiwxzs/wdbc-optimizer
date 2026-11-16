# Conjunto de datos: Wisconsin Diagnostic Breast Cancer (WDBC)

## Procedencia

| Campo | Valor |
|---|---|
| Nombre | Breast Cancer Wisconsin (Diagnostic) |
| Creadores | W. H. Wolberg, O. L. Mangasarian, W. N. Street (Universidad de Wisconsin) |
| Repositorio | UCI Machine Learning Repository, *dataset* 17 |
| DOI | [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) |
| Licencia | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Artículo introductorio | Street, Wolberg y Mangasarian (1993), *Nuclear feature extraction for breast tumor diagnosis*, Proc. SPIE 1905 |
| Copia utilizada | la incluida en scikit-learn (`sklearn.datasets.load_breast_cancer`); no se descarga nada en tiempo de ejecución |

La licencia CC BY 4.0 permite redistribuir y adaptar los datos siempre que se dé el crédito correspondiente. El repositorio **no** guarda una copia propia de los datos (no hay carpeta `data/`): se cargan desde la instalación de scikit-learn. La cita recomendada está en la [bibliografía](../references/bibliography.md).

## Contenido

- **569 casos**: 212 malignos ($37.3\,\%$) y 357 benignos.
- **30 características reales**, calculadas a partir de imágenes digitalizadas de aspirados con aguja fina (PAAF) de masas mamarias. Para cada imagen se miden 10 propiedades de los núcleos celulares y se resumen con 3 estadísticos:

| Propiedad (10) | Descripción |
|---|---|
| radio | media de las distancias del centro al contorno |
| textura | desviación típica de los niveles de gris |
| perímetro, área | tamaño del núcleo |
| suavidad | variación local de las longitudes de radio |
| compacidad | $\text{perímetro}^2/\text{área}-1$ |
| concavidad | severidad de las porciones cóncavas del contorno |
| puntos cóncavos | número de porciones cóncavas |
| simetría | — |
| dimensión fractal | aproximación de la "línea de costa" $-1$ |

| Estadístico (3) | Prefijo/sufijo en scikit-learn | Traducción en la interfaz |
|---|---|---|
| media | `mean …` | (media) |
| error estándar | `… error` | (error est.) |
| peor valor (media de los 3 mayores) | `worst …` | (peor) |

- Sin valores faltantes.

## Preprocesamiento aplicado

1. **Recodificación de etiquetas:** $y=1-\text{target}$, de modo que $1$ = maligno.
2. **Partición estratificada** 80/20 con semilla 42: 455 casos de entrenamiento (170 malignos) y 114 de prueba (42 malignos).
3. **Estandarización** por columna con media y desviación típica del conjunto de entrenamiento.
4. **PCA a 2 componentes**, solo para visualización (varianza explicada: $44.6\,\%+18.5\,\%$).

Los índices originales de cada caso se conservan en `idx_train` e `idx_test`.

## Consideraciones éticas y de uso

- Son datos clínicos **anonimizados** y públicos desde 1995.
- Proceden de **una sola institución** y de una época concreta: un modelo entrenado con ellos no tiene por qué generalizar a otras poblaciones, equipos de imagen o protocolos.
- El proyecto es **académico**. No es un dispositivo médico ni sustituye el criterio clínico (ver [alcance y limitaciones](../introduction/scope_and_limitations.md)).
