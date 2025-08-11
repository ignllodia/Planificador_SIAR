
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from rrt_siar import RRT_SIAR, tabla_configuraciones, calcular_largo

# Parámetros
escala_px_por_m = 62
wheel_size = 4
max_iteraciones = 2000
epsilon = 10

# Inicializar entorno
rrt = RRT_SIAR("Pb4.png")
inicio = rrt.generar_configuracion_valida()
objetivo = rrt.generar_configuracion_valida()
nodos = [inicio]
encontrado = False

# RRT loop
for _ in range(max_iteraciones):
    muestra = rrt.generar_configuracion_valida()
    if not muestra:
        continue
    nodo_cercano = min(nodos, key=lambda n: n.distancia(muestra))
    if rrt.conectar_nodos(nodo_cercano, muestra):
        muestra.padre = nodo_cercano
        nodos.append(muestra)
        if muestra.distancia(objetivo) < epsilon:
            if rrt.conectar_nodos(muestra, objetivo):
                objetivo.padre = muestra
                nodos.append(objetivo)
                encontrado = True
                break

# Mostrar solo el árbol inicial con conexiones
img_arbol = rrt.mapa.copy()
for nodo in nodos:
    cv2.circle(img_arbol, (int(nodo.x), int(nodo.y)), 2, (0, 0, 255), -1)
    if nodo.padre:
        cv2.line(img_arbol, (int(nodo.x), int(nodo.y)),
                 (int(nodo.padre.x), int(nodo.padre.y)), (255, 0, 0), 1)

cv2.circle(img_arbol, (int(inicio.x), int(inicio.y)), 5, (0, 255, 0), -1)
cv2.circle(img_arbol, (int(objetivo.x), int(objetivo.y)), 5, (0, 255, 255), -1)

plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(cv2.cvtColor(img_arbol, cv2.COLOR_BGR2RGB))
ax.set_title("Árbol RRT con nodos generados")
ax.axis("off")
plt.pause(2)

# Visualización secuencial de configuraciones
if encontrado:
    img_base = img_arbol.copy()
    actual = objetivo
    trayecto = []
    while actual and actual.padre:
        trayecto.append(actual)
        actual = actual.padre
    trayecto.append(inicio)
    trayecto = trayecto[::-1]

    for i, nodo in enumerate(trayecto):
        imagen = img_base.copy()
        ancho, offset = tabla_configuraciones[nodo.sensor_id // 10][1:]
        largo = calcular_largo(ancho)
        print(f"Paso {i}: x={int(nodo.x)}, y={int(nodo.y)}, θ={nodo.theta:.2f} rad, sensor={nodo.sensor_id}, ancho={ancho:.2f} m")

        x, y, theta = nodo.x, nodo.y, nodo.theta
        width_px = ancho * escala_px_por_m
        length_px = largo * escala_px_por_m
        offset_px = offset * escala_px_por_m
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        # Ruedas
        wheel_dx = width_px / 2
        wheel_dy = length_px / 2.5
        wheel_offsets = [
            (-wheel_dx, -wheel_dy), (-wheel_dx, 0), (-wheel_dx, +wheel_dy),
            (+wheel_dx, -wheel_dy), (+wheel_dx, 0), (+wheel_dx, +wheel_dy)
        ]
        support_pts = []
        for wx_off, wy_off in wheel_offsets:
            wx = int(x + (wx_off * cos_t - wy_off * sin_t))
            wy = int(y + (wx_off * sin_t + wy_off * cos_t))
            if 0 <= wx < rrt.width and 0 <= wy < rrt.height and rrt.mask_gutter[wy, wx] != 255:
                support_pts.append([wx, wy])
                cv2.rectangle(imagen, (wx - wheel_size, wy - wheel_size),
                              (wx + wheel_size, wy + wheel_size), (0, 255, 0), -1)
            else:
                cv2.rectangle(imagen, (wx - wheel_size, wy - wheel_size),
                              (wx + wheel_size, wy + wheel_size), (0, 0, 255), -1)

        # CM
        cm_x = int(x)
        cm_y = int(y + offset_px * cos_t)
        cv2.circle(imagen, (cm_x, cm_y), 4, (0, 165, 255), -1)

        # Polígono de soporte
        if len(support_pts) >= 3:
            hull = cv2.convexHull(np.array(support_pts))
            cv2.polylines(imagen, [hull], isClosed=True, color=(255, 0, 255), thickness=1)

        # Robot body
        dx = width_px / 2
        dy = length_px / 2
        corners = np.array([
            [-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]
        ])
        rotated = np.array([
            [x + (c[0] * cos_t - c[1] * sin_t), y + (c[0] * sin_t + c[1] * cos_t)]
            for c in corners
        ], dtype=np.int32)
        cv2.polylines(imagen, [rotated], isClosed=True, color=(0, 255, 255), thickness=2)

        ax.clear()
        ax.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Configuración paso {i}")
        ax.axis("off")
        plt.pause(1.0)

    plt.ioff()
    plt.show()
else:
    print("⚠️ No se encontró una trayectoria válida.")
