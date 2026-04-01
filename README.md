# Planificador_SIAR

Implementación de un planificador de trayectorias para el robot SIAR utilizando el algoritmo **RRT (Rapidly-exploring Random Tree)** y variantes adaptadas a sus necesidades de navegación.

##  Descripción

Este proyecto desarrolla y prueba un planificador de rutas optimizado para el robot SIAR en entornos simulados o reales.  
Se basa en **RRT** y parámetros configurables que permiten ajustar su rendimiento a diferentes escenarios.  
El planificador genera una secuencia de puntos de paso que el sistema de control del robot puede seguir para evitar obstáculos y alcanzar un objetivo.

El objetivo del proyecto es generar trayectorias viables teniendo en cuenta:
- la geometría del entorno,
- la presencia del *gutter* central,
- la estabilidad del robot,
- y las restricciones cinemáticas y kinodinámicas de la plataforma.


## Estructura actual del repositorio

### `Conf_est/`
Contiene material orientado al análisis de estabilidad del robot SIAR sobre el entorno:

- `Conf_Estable_medidas.py`: script para evaluar configuraciones estables a partir de medidas del robot.
- `Conf_est_mapa.py`: validación de configuraciones estables sobre el mapa.
- `SIARSimulator2D.py`: simulador 2D del robot SIAR.
- `Pb4.png`: mapa utilizado en este bloque.
- Imágenes asociadas a resultados y ejemplos:
  - `Conf_Estable_medidad_estable.png`
  - `Conf_Estable_medidad_no_estable.png`
  - `Conf_est_mapa.png`
  - `SIARSimulador2D_anchomax.png`
  - `SIARSimulador2D_anchomin.png`

### `RRt_Conf/`
Contiene el núcleo de los algoritmos de planificación y su evolución experimental:

- `rrt_SIAR_v0.py`: primera versión básica del planificador.
- `rrt_siar.py`: versión temprana del RRT adaptado al robot SIAR.
- `RRT_kinodinamico.py`: implementación kinodinámica base.
- `RRT_kinodinamico_astar.py`: variante guiada mediante A*.
- `RRT_kinodinamico_rrtstar.py`: variante basada en RRT*.
- `RRT_biRRt.py`: versión bidireccional.
- `control_set_demo_standalone.py`: demostración aislada del conjunto de controles.
- `Pb4.png`: mapa empleado en las simulaciones de esta carpeta.
- `RRT_kinodinamico_resumen.pdf`: documento resumen asociado a esta parte.
- Imágenes de apoyo:
  - `rrt_SIAR_v0_1.png`
  - `rrt_SIAR_v0_2.png`
  - `rrt_SIAR_v0_log.png`

---

## Archivo principal recomendado

Actualmente, el repositorio público mantiene una estructura centrada en versiones intermedias y experimentales. Dentro del bloque `RRt_Conf/`, las implementaciones más representativas del desarrollo son:

- `RRT_kinodinamico.py`
- `RRT_kinodinamico_astar.py`
- `RRT_kinodinamico_rrtstar.py`
- `RRT_biRRt.py`

---

## Requisitos

El proyecto está implementado en Python 3 y utiliza principalmente:

- `numpy`
- `matplotlib`
- `opencv-python`
- `networkx`
- `shapely`

por lo que conviene instalarlas si se van a reutilizar scripts anteriores o ampliar el proyecto.

Instalación orientativa:

```bash
pip install numpy matplotlib networkx shapely opencv-python
```

---

## Nota

El repositorio conserva distintas versiones del trabajo porque reflejan la evolución del TFG, desde el estudio de configuraciones estables hasta variantes más avanzadas del planificador basadas en RRT.


