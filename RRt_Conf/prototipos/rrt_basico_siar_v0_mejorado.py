import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from rrt_basico_siar_validacion_mejorada import (
    RRT_SIAR,
    calcular_largo,
    tabla_configuraciones,
)


escala_px_por_m = 62
wheel_size = 4
max_iteraciones = 300
epsilon = 10
RUTA_MAPA = Path(__file__).with_name("Pb4.png")


def construir_arbol(rrt, inicio, objetivo):
    if inicio is None or objetivo is None:
        return [], False

    nodos = [inicio]
    encontrado = False

    for _ in range(max_iteraciones):
        muestra = rrt.generar_configuracion_valida()
        if not muestra:
            continue

        vecino = min(nodos, key=lambda nodo: nodo.distancia(muestra))
        if not rrt.conectar_nodos(vecino, muestra):
            continue

        muestra.padre = vecino
        nodos.append(muestra)

        if muestra.distancia(objetivo) < epsilon and rrt.conectar_nodos(muestra, objetivo):
            objetivo.padre = muestra
            nodos.append(objetivo)
            encontrado = True
            break

    return nodos, encontrado


def dibujar_arbol(rrt, nodos, inicio, objetivo):
    imagen = rrt.mapa.copy()
    for nodo in nodos:
        cv2.circle(imagen, (int(nodo.x), int(nodo.y)), 2, (0, 0, 255), -1)
        if nodo.padre:
            cv2.line(
                imagen,
                (int(nodo.x), int(nodo.y)),
                (int(nodo.padre.x), int(nodo.padre.y)),
                (255, 0, 0),
                1,
            )

    cv2.circle(imagen, (int(inicio.x), int(inicio.y)), 5, (0, 255, 0), -1)
    cv2.circle(imagen, (int(objetivo.x), int(objetivo.y)), 5, (0, 255, 255), -1)
    return imagen


def reconstruir_trayecto(inicio, objetivo):
    trayecto = []
    actual = objetivo
    while actual and actual.padre:
        trayecto.append(actual)
        actual = actual.padre

    trayecto.append(inicio)
    trayecto.reverse()
    return trayecto


def dibujar_configuracion(rrt, base, nodo):
    imagen = base.copy()
    ancho, offset = tabla_configuraciones[nodo.sensor_id // 10][1:]
    largo = calcular_largo(ancho)

    x = nodo.x
    y = nodo.y
    theta = nodo.theta
    width_px = ancho * escala_px_por_m
    length_px = largo * escala_px_por_m
    offset_px = offset * escala_px_por_m
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    wheel_dx = width_px / 2
    wheel_dy = length_px / 2.5
    wheel_offsets = [
        (-wheel_dx, -wheel_dy),
        (-wheel_dx, 0),
        (-wheel_dx, +wheel_dy),
        (+wheel_dx, -wheel_dy),
        (+wheel_dx, 0),
        (+wheel_dx, +wheel_dy),
    ]

    support_pts = []
    for wx_off, wy_off in wheel_offsets:
        wx = int(x + (wx_off * cos_t - wy_off * sin_t))
        wy = int(y + (wx_off * sin_t + wy_off * cos_t))
        if 0 <= wx < rrt.width and 0 <= wy < rrt.height and rrt.mask_gutter[wy, wx] != 255:
            support_pts.append([wx, wy])
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(
            imagen,
            (wx - wheel_size, wy - wheel_size),
            (wx + wheel_size, wy + wheel_size),
            color,
            -1,
        )

    cm_x = int(x)
    cm_y = int(y + offset_px * cos_t)
    cv2.circle(imagen, (cm_x, cm_y), 4, (0, 165, 255), -1)

    if len(support_pts) >= 3:
        hull = cv2.convexHull(np.array(support_pts))
        cv2.polylines(imagen, [hull], isClosed=True, color=(255, 0, 255), thickness=1)

    dx = width_px / 2
    dy = length_px / 2
    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    body = np.array(
        [
            [x + (px * cos_t - py * sin_t), y + (px * sin_t + py * cos_t)]
            for px, py in corners
        ],
        dtype=np.int32,
    )
    cv2.polylines(imagen, [body], isClosed=True, color=(0, 255, 255), thickness=2)

    return imagen, ancho


def main():
    if not RUTA_MAPA.exists():
        raise FileNotFoundError(f"No se encuentra el mapa: {RUTA_MAPA}")

    print(f"Cargando mapa desde: {RUTA_MAPA}")
    rrt = RRT_SIAR(str(RUTA_MAPA))
    inicio = rrt.generar_configuracion_valida()
    objetivo = rrt.generar_configuracion_valida()
    if inicio is None or objetivo is None:
        print("No se pudo generar una configuracion inicial u objetivo valida.")
        return

    nodos, encontrado = construir_arbol(rrt, inicio, objetivo)
    if not nodos:
        print("No se pudo inicializar el arbol RRT.")
        return

    print(f"Nodos generados: {len(nodos)}")

    img_arbol = dibujar_arbol(rrt, nodos, inicio, objetivo)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(img_arbol, cv2.COLOR_BGR2RGB))
    ax.set_title("Arbol RRT con validacion mejorada")
    ax.axis("off")
    plt.pause(2)

    if not encontrado:
        print("No se encontro una trayectoria valida.")
        plt.ioff()
        plt.show()
        return

    img_base = img_arbol.copy()
    trayecto = reconstruir_trayecto(inicio, objetivo)
    for i, nodo in enumerate(trayecto):
        imagen, ancho = dibujar_configuracion(rrt, img_base, nodo)
        print(
            f"Paso {i}: x={int(nodo.x)}, y={int(nodo.y)}, "
            f"theta={nodo.theta:.2f} rad, sensor={nodo.sensor_id}, ancho={ancho:.2f} m"
        )

        ax.clear()
        ax.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Configuracion mejorada paso {i}")
        ax.axis("off")
        plt.pause(1.0)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
