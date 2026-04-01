# RRt_Conf

Esta carpeta contiene el desarrollo completo de los algoritmos de planificación de trayectorias para el robot SIAR, basados en RRT y sus distintas variantes.

El contenido refleja la evolución del trabajo desde los primeros prototipos hasta la versión final del planificador.

---

## Relación entre scripts

La evolución del planificador sigue una progresión clara:

```
Prototipos RRT
   ↓
RRT con validación geométrica
   ↓
RRT kinodinámico
   ↓
RRT kinodinámico guiado (A*)
   ↓
RRT* kinodinámico
   ↓
Versión final robusta (timeout + reinicio)
```

Donde el código final del proyecto es:

```
rrt_estrella_kinodinamico_final_timeout.py
```

Este archivo integra las mejoras desarrolladas en el resto de scripts:
- modelo kinodinámico
- validación de configuraciones estables
- optimización tipo RRT*
- mejoras prácticas (timeout y reinicio del árbol)

---

##  Descripción de los códigos

### 🔹 Prototipos iniciales
- `rrt_SIAR_v0.py`  
  Primer prototipo del planificador basado en RRT.

- `rrt_siar.py`  
  Evolución inicial con adaptación al robot SIAR y al mapa.

---

### 🔹 Planificación kinodinámica
- `RRT_kinodinamico.py`  
  Introduce el modelo de movimiento del robot mediante acciones.

---

### 🔹 Guiado del planificador
- `RRT_kinodinamico_astar.py`  
  Añade una referencia global (A*) para orientar el crecimiento del árbol.

---

### 🔹 Optimización de trayectorias
- `RRT_kinodinamico_rrtstar.py`  
  Implementa RRT* para mejorar la calidad del camino mediante rewiring.

---

### 🔹 Variantes del algoritmo
- `RRT_biRRt.py`  
  Variante bidireccional para acelerar la búsqueda.

---

### 🔹 Utilidades
- `control_set_demo_standalone.py`  
  Demo para visualizar y analizar el conjunto de controles del robot.

---

### 🔹 Código final (fuera de esta carpeta)

- `rrt_estrella_kinodinamico_final_timeout.py`

Versión más completa del planificador, que incorpora:
- planificación kinodinámica
- validación de estabilidad del robot
- optimización tipo RRT*
- control de tiempo de ejecución (timeout)
- reinicio del árbol para mejorar la exploración
- métricas de evaluación

---



Esta carpeta contiene versiones intermedias del algoritmo que permiten entender la evolución del desarrollo hasta la versión final del planificador.

