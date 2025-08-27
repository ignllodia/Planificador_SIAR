# Planificador_SIAR

Implementación de un planificador de trayectorias para el robot SIAR utilizando el algoritmo **RRT (Rapidly-exploring Random Tree)** y variantes adaptadas a sus necesidades de navegación.

##  Descripción

Este proyecto desarrolla y prueba un planificador de rutas optimizado para el robot SIAR en entornos simulados o reales.  
Se basa en **RRT** y parámetros configurables que permiten ajustar su rendimiento a diferentes escenarios.  
El planificador genera una secuencia de puntos de paso que el sistema de control del robot puede seguir para evitar obstáculos y alcanzar un objetivo.

##  Contenido del repositorio

- **`RRt_Conf/`** → Configuraciones del algoritmo RRT (distancia de expansión, número de muestras, límites del entorno, etc.).
- **`Conf_est/`** → Configuraciones experimentales utilizadas en pruebas (semillas aleatorias, tolerancias, etc.).
- **`.gitattributes`** → Atributos de Git para estandarizar el formato de los archivos.

##  Requisitos

Este planificador está implementado en **Python 3.x** y requiere las siguientes dependencias:

```bash
pip install numpy matplotlib networkx shapely
```
Estas librerías se usan para:

-numpy → Operaciones numéricas y manejo de vectores/matrices.

-matplotlib → Visualización de los resultados y trayectorias.

-networkx → Representación y manipulación de grafos.

-shapely → Cálculos geométricos con polígonos, puntos y líneas.

 Ejecución
Clona el repositorio:

```bash
git clone https://github.com/ignllodia/Planificador_SIAR.git
cd Planificador_SIAR
```
Configura los parámetros del planificador editando los archivos en **`RRt_Conf/`**.

Ejecuta el script principal del planificador (será el script con una versión, ejm: _v0)

```bash

python3 planificador.py
