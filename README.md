# Planificador SIAR

Implementación final del planificador de trayectorias desarrollado para el robot **SIAR** (*Sewer Inspection Autonomous Robot*). El objetivo del programa es generar trayectorias viables sobre un mapa 2D de alcantarillado, teniendo en cuenta la geometría del entorno, el *gutter* central, las restricciones kinodinámicas del robot y la estabilidad estática de las configuraciones generadas.

El planificador combina una referencia global obtenida mediante **A\*** con una planificación local basada en **RRT kinodinámico**. Una vez encontrada una primera solución válida, se activa una fase posterior de refinamiento mediante mecanismos propios de **RRT\***, como la selección de padre y el *rewiring*. De esta forma se busca obtener una solución inicial de forma rápida y, posteriormente, mejorarla sin aplicar el coste computacional de RRT\* durante toda la búsqueda.

## Estructura recomendada del repositorio

```text
Planificador_SIAR/
│
├── Planificador_A_RRT_estrella.py   # Código principal del planificador final
├── Pb4_ampliado.png                 # Mapa de alcantarillado utilizado
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Descripción del repositorio
```

El repositorio se centra únicamente en el resultado final del trabajo. Las versiones intermedias, pruebas de ablación y scripts antiguos no se incluyen como parte de la estructura principal, ya que su evolución queda reflejada en el historial de Git.

## Archivo principal

El archivo principal es:

```text
Planificador_A_RRT_estrella.py
```

Este script incluye:

- carga y segmentación del mapa,
- selección interactiva de configuración inicial y objetivo,
- planificación global mediante A\*,
- generación de *waypoints* y corredor de muestreo,
- expansión kinodinámica del árbol,
- validación física de configuraciones,
- refinamiento posterior mediante *rewiring*,
- visualización del árbol y de la trayectoria final,
- cálculo automático de métricas experimentales.

## Requisitos

El código está implementado en Python y utiliza principalmente OpenCV y NumPy.


El resto de módulos utilizados (`math`, `random`, `heapq`, `statistics`, `dataclasses`, `time` y `pathlib`) pertenecen a la biblioteca estándar de Python.

## Ejecución

Para ejecutar el planificador:

```bash
python Planificador_A_RRT_estrella.py
```

Al abrirse la ventana de OpenCV, se seleccionan la configuración inicial y final de forma interactiva.

Controles principales:

| Acción | Tecla / ratón |
|---|---|
| Seleccionar posición inicial | Clic izquierdo |
| Seleccionar orientación inicial | Segundo clic izquierdo |
| Seleccionar posición objetivo | Clic derecho |
| Seleccionar orientación objetivo | Segundo clic derecho |
| Ejecutar planificación | Espacio |
| Reiniciar selección | `r` |
| Salir | `q` o `Esc` |
| Acercar / alejar zoom | `+` / `-` |
| Mover vista | `W`, `A`, `S`, `D` |
| Reajustar vista | `f` |

## Representación del mapa

El mapa utilizado por defecto es:

```python
MAP_PATH = "Pb4_ampliado.png"
```

El mapa se carga en escala de grises y se segmenta internamente en tres regiones:

| Región | Criterio en el código | Significado |
|---|---|---|
| Zona transitable | `gray < 100` | Suelo por el que puede circular el robot |
| Pared | `100 <= gray < 180` | Obstáculo no transitable |
| *Gutter* | `gray >= 180` | Canal central del alcantarillado |

El planificador distingue entre zonas transitables, paredes y *gutter* para evitar colisiones y descartar configuraciones inestables.

## Modelo del robot

El estado del robot se representa como:

```text
q = (x, y, θ, w)
```

donde:

- `x`, `y`: posición del robot en píxeles,
- `θ`: orientación del robot,
- `w`: anchura actual del robot en metros.

Parámetros principales del modelo:

| Parámetro | Valor actual | Efecto |
|---|---:|---|
| `ROBOT_LEN` | `0.88 m` | Longitud nominal del robot utilizada para calcular la geometría del cuerpo y la posición aproximada de las ruedas. |
| `ROBOT_W_MIN` | `0.52 m` | Anchura mínima permitida. Configuraciones por debajo de este valor se descartan. |
| `ROBOT_W_MAX` | `0.85 m` | Anchura máxima permitida. Configuraciones por encima de este valor se descartan. |
| `ROBOT_W0` | `0.70 m` | Anchura inicial usada al seleccionar el inicio y el objetivo. |
| `PIXELS_PER_M` | `75 px/m` | Factor de escala entre píxeles y metros. Afecta a distancias, longitudes de trayectoria, tolerancias y métricas. |
| `TABLA_CONFIGURACIONES` | tabla discreta de anchuras y desplazamientos del centro de masas | Relaciona el ancho del robot con el desplazamiento longitudinal del centro de masas usado en la verificación de estabilidad. |
| `calcular_largo(w)` | `-0.675*w + 1.3175` | Aproxima la longitud efectiva del robot en función de su anchura. |

## Parámetros principales del planificador

| Parámetro | Valor actual | Efecto |
|---|---:|---|
| `MAX_ITERS` | `50000` | Número máximo de iteraciones antes de detener la búsqueda si no se encuentra solución. |
| `TREE_TIMEOUT` | `60.0 s` | Tiempo máximo por árbol antes de reiniciar la búsqueda. Solo afecta si `USE_TIMEOUT_RESTARTS = True`. |
| `POST_SOLUTION_IMPROVEMENT_TIME` | `10 s` | Tiempo adicional de refinamiento tras encontrar la primera solución válida. |
| `GOAL_SAMPLING_RATE` | `0.1` | Probabilidad de muestrear directamente el objetivo. Aumentarlo puede acelerar la llegada, pero reduce exploración. |
| `DT` | `0.05 s` | Paso temporal de integración de la dinámica. Valores menores aumentan precisión y coste computacional. |
| `TPROP` | `0.5 s` | Horizonte temporal de cada propagación dinámica. Define cuánto avanza el robot por expansión. |
| `CHECK_STEP` | `2` | Frecuencia con la que se valida la trayectoria durante la propagación. Valores menores aumentan seguridad y coste. |
| `NEAR_GOAL_DIST` | `0.30 m` | Tolerancia de posición para considerar alcanzado el objetivo. |
| `NEAR_GOAL_DTH` | `20°` | Tolerancia angular para aceptar la orientación final. |
| `NEAR_GOAL_DW` | `0.10 m` | Tolerancia de anchura para aceptar la configuración final. |
| `TAU_H` | `0.6` | Constante de tiempo del filtro de cambio de ancho. Controla la velocidad con la que el robot alcanza la anchura objetivo. |
| `REP_SIM` | `10` | Número de ejecuciones independientes realizadas para calcular métricas. |
| `RNG_SEED_BASE` | `12345` | Semilla base para reproducibilidad de los experimentos. |

## Parámetros de guiado global con A\*

| Parámetro | Valor actual | Efecto |
|---|---:|---|
| `USE_ASTAR_GUIDANCE` | `True` | Activa el uso de A\* como referencia global. |
| `N_WAYPOINTS` | `10` | Número de puntos intermedios generados a partir de la trayectoria de A\*. |
| `WAYPOINT_BIAS` | `0.2` | Probabilidad de muestrear el *waypoint* activo. Ayuda a guiar el árbol hacia la referencia global. |
| `CORRIDOR_WIDTH_M` | `0.6 m` | Anchura del corredor alrededor de la referencia de A\*. Limita la exploración a zonas relevantes. |
| `CENTER_BIAS_WEIGHT` | `0.05` | Penalización usada por A\* para favorecer zonas más centradas y alejadas de paredes. |

## Parámetros de RRT\* y refinamiento

| Parámetro | Valor actual | Efecto |
|---|---:|---|
| `USE_REWIRING` | `False` | Desactiva el *rewiring* desde el inicio de la búsqueda. |
| `USE_POST_SOLUTION_RRT_STAR` | `True` | Activa el refinamiento tipo RRT\* solo después de encontrar una primera solución. |
| `NEIGHBOR_RADIUS_M` | `0.1 m` | Radio de búsqueda de vecinos para selección de padre y *rewiring*. |
| `HEADING_COST_WEIGHT` | `0.3 m` convertido a píxeles | Penaliza cambios de orientación en el coste de conexión. |
| `REWIRE_POS_TOL` | `0.20 m` | Tolerancia de posición para aceptar una conexión durante el *rewiring*. |
| `REWIRE_TH_TOL` | `20°` | Tolerancia angular para aceptar una conexión durante el *rewiring*. |
| `REWIRE_W_TOL` | `0.08 m` | Tolerancia de anchura para aceptar una conexión durante el *rewiring*. |

## Parámetros de búsqueda de vecinos

| Parámetro | Valor actual | Efecto |
|---|---:|---|
| `K_SUBSET` | `64` | Número de nodos muestreados en la búsqueda aproximada del vecino más cercano. Reduce coste frente a búsqueda exacta. |
| `EXACT_EVERY` | `20` | Cada cuántas iteraciones se realiza una búsqueda exacta del vecino más cercano. |

## Conjunto de controles

El planificador utiliza un conjunto discreto de acciones de la forma:

```text
(v, ω, Δw)
```

donde:

- `v`: velocidad lineal en m/s,
- `ω`: velocidad angular en rad/s,
- `Δw`: comando discreto de cambio de anchura.

Con `USE_SWITCHABLE_CONTROL_SETS = False`, el planificador utiliza `CONTROL_SET_ALL`:

```python
CONTROL_SET_ALL = [
    (0.25,  0.0,   0),
    (0.25,  0.2,   0),
    (0.25, -0.2,   0),
    (0.35,  0.0,   0),
    (0.00,  0.0,  -1),
    (0.00,  0.0,  +1),
    (0.00,  0.2,   0),
    (0.00, -0.2,   0),
    (0.10,  0.6,   0),
    (0.10, -0.6,   0),
    (0.10,  0.3,   0),
    (0.10, -0.3,   0),
    (0.12,  0.8,   0),
    (0.12, -0.8,   0),
]
```

Este conjunto permite combinar avance recto, giros suaves, giros más cerrados, rotaciones en el sitio y cambios discretos de anchura.

## Parámetros de visualización

| Parámetro | Valor actual | Efecto |
|---|---:|---|
| `DRAW_EVERY` | `100` | Frecuencia de actualización visual del árbol. |
| `N_FRAMES` | `6` | Número de configuraciones del robot dibujadas sobre la trayectoria final. |
| `VIEW_MAX_W` | `1600` | Anchura máxima de la ventana de visualización. |
| `VIEW_MAX_H` | `900` | Altura máxima de la ventana de visualización. |
| `VIEW_ZOOM_STEP` | `1.25` | Factor de incremento del zoom. |
| `VIEW_MAX_ZOOM` | `12.0` | Zoom máximo permitido. |
| `SHOW_STATE_LABELS` | `True` | Muestra etiquetas con la anchura del robot en las configuraciones dibujadas. |
| `DEBUG` | `True` | Activa mensajes de depuración en consola. |
| `DEBUG_EVERY` | `200` | Frecuencia de impresión de mensajes de depuración. |

## Validación de configuraciones

Cada configuración generada por el planificador se valida mediante el módulo `valid_configuration`. Una configuración se acepta únicamente si cumple:

1. El centro del robot está dentro de los límites del mapa.
2. La anchura está dentro del rango permitido.
3. El cuerpo del robot no colisiona con paredes.
4. Las ruedas no están sobre paredes.
5. Existen al menos tres puntos de apoyo.
6. El polígono de soporte tiene área suficiente.
7. La proyección del centro de masas queda dentro del polígono de soporte.

Esta validación permite rechazar configuraciones que podrían provocar pérdida de estabilidad al pasar cerca o sobre el *gutter*.

## Métricas calculadas

El programa calcula automáticamente métricas de ejecución y calidad de trayectoria:

| Métrica | Significado |
|---|---|
| `elapsed_s` | Tiempo total de ejecución. |
| `iters` | Número de iteraciones realizadas. |
| `propagations` | Número total de propagaciones dinámicas evaluadas. |
| `invalid_pct` | Porcentaje de propagaciones descartadas por no ser válidas. |
| `node_count` | Número de nodos del árbol. |
| `path_length_m` | Longitud de la trayectoria final en metros. |
| `width_change_count` | Número de cambios de anchura a lo largo de la solución. |
| `mean_ref_dev_m` | Desviación media respecto a la referencia global generada por A\*. |
| `success` | Indica si se ha encontrado una solución válida. |

## Resultado esperado

Tras seleccionar inicio y objetivo, el planificador muestra:

- el mapa segmentado,
- la referencia global generada por A\*,
- el árbol explorado por el planificador local,
- la trayectoria final,
- varias configuraciones intermedias del robot,
- el centro de masas y el polígono de soporte en las configuraciones dibujadas.

La trayectoria final se obtiene considerando tanto la geometría del mapa como la estabilidad estática del robot.

## Limitaciones

La implementación actual trabaja sobre un mapa 2D segmentado. No incluye todavía integración completa en ROS/Gazebo ni validación sobre la plataforma física real. Además, la estabilidad considerada es estática, por lo que no se modelan de forma completa efectos dinámicos como aceleraciones, inercias, deslizamientos, pendientes o pérdidas de contacto durante el movimiento.

## Trabajos futuros

Algunas posibles líneas de mejora son:

- integración del planificador en ROS/Gazebo,
- validación sobre el robot SIAR real,
- uso de mapas 3D o nubes de puntos,
- incorporación de criterios de estabilidad dinámica,
- optimización automática del conjunto de controles,
- uso de aprendizaje por refuerzo para mejorar la selección de acciones.
