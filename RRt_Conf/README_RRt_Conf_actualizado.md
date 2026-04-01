# RRt_Conf

Esta carpeta reúne el bloque principal de algoritmos de planificación de trayectoria desarrollados para el robot SIAR.

Aquí se agrupan las distintas versiones del planificador basadas en RRT, desde implementaciones iniciales hasta variantes kinodinámicas, guiadas y bidireccionales.

---

## Jerarquía interna de los códigos

La evolución aproximada de esta carpeta puede entenderse así:

1. **Prototipos iniciales**
   - `rrt_SIAR_v0.py`
   - `rrt_siar.py`

   Estas versiones representan los primeros enfoques del planificador. Su objetivo principal es generar configuraciones válidas del robot y construir un árbol básico sobre el mapa.

2. **Planificación kinodinámica**
   - `RRT_kinodinamico.py`

   En esta fase se introduce un modelo más realista del movimiento del robot, de modo que la expansión del árbol no se basa solo en geometría, sino también en acciones y propagación dinámica.

3. **Guiado mediante referencia global**
   - `RRT_kinodinamico_astar.py`

   Esta versión añade una guía global basada en A*, utilizada para orientar la exploración del árbol y mejorar la convergencia hacia el objetivo.

4. **Optimización tipo RRT***
   - `RRT_kinodinamico_rrtstar.py`

   Introduce la idea de optimización de caminos mediante la lógica de RRT*, buscando trayectorias de mayor calidad que las obtenidas por un RRT estándar.

5. **Expansión bidireccional**
   - `RRT_biRRt.py`

   Implementa una variante bidireccional, en la que el crecimiento se realiza desde el inicio y desde el objetivo para intentar reducir el tiempo de búsqueda.

6. **Herramientas auxiliares**
   - `control_set_demo_standalone.py`

   Script de apoyo para visualizar y analizar el conjunto de controles empleado en algunas versiones del planificador.

---

## Archivos presentes en la carpeta

### Scripts principales
- `rrt_SIAR_v0.py`  
  Primera versión básica del planificador para SIAR.

- `rrt_siar.py`  
  Evolución inicial del RRT adaptado al robot y al mapa.

- `RRT_kinodinamico.py`  
  Implementación kinodinámica base.

- `RRT_kinodinamico_astar.py`  
  Versión kinodinámica guiada mediante A*.

- `RRT_kinodinamico_rrtstar.py`  
  Variante RRT* para mejorar la calidad de la trayectoria.

- `RRT_biRRt.py`  
  Variante bidireccional del algoritmo.

- `control_set_demo_standalone.py`  
  Demo aislada del conjunto de acciones o controles.

### Recursos auxiliares
- `Pb4.png`  
  Mapa principal usado en esta carpeta.

- `RRT_kinodinamico_resumen.pdf`  
  Documento resumen relacionado con este bloque.

### Imágenes de apoyo
- `rrt_SIAR_v0_1.png`
- `rrt_SIAR_v0_2.png`
- `rrt_SIAR_v0_log.png`

Estas imágenes parecen asociadas a resultados, ejemplos o registros visuales de las primeras pruebas del planificador.

### Carpeta interna generada automáticamente
- `__pycache__/`  
  Carpeta generada por Python con archivos compilados. No forma parte del desarrollo conceptual del proyecto.

---

## Relación entre los scripts

La estructura conceptual de esta carpeta puede resumirse así:

```text
rrt_SIAR_v0.py
   ↓
rrt_siar.py
   ↓
RRT_kinodinamico.py
   ↓
RRT_kinodinamico_astar.py
   ↓
RRT_kinodinamico_rrtstar.py
   ↘
    RRT_biRRt.py
```

No implica necesariamente una dependencia directa entre todos los archivos, pero sí refleja una jerarquía razonable de evolución del trabajo.

---

## Objetivo de esta carpeta

El propósito de `RRt_Conf/` es concentrar la parte del TFG dedicada a la planificación de trayectorias, diferenciándola del bloque de estabilidad del robot almacenado en `Conf_est/`.

En resumen:
- `Conf_est/` → estabilidad del robot
- `RRt_Conf/` → algoritmos de planificación
