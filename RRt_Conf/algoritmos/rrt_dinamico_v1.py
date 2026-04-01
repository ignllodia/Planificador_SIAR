import math
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


max_iter = 20000
epsilon = 40
dt = 0.1
T = 1.0
bias_freq = 10
tau = 0.5
escala_px_por_m = 62

v_set = [16.0, 24.0, 32.0]
w_set = [-1.0, -0.5, 0.0, 0.5, 1.0]
h_deseadas = [0.52, 0.58, 0.64, 0.70]

tabla_configuraciones = [
    (0, 0.51, 0.14),
    (10, 0.58, 0.12),
    (20, 0.64, 0.09),
    (30, 0.68, 0.06),
    (40, 0.70, 0.04),
    (50, 0.71, 0.02),
    (60, 0.71, 0.00),
    (70, 0.71, -0.01),
    (80, 0.70, -0.03),
    (90, 0.69, -0.05),
    (100, 0.68, -0.07),
    (110, 0.66, -0.08),
    (120, 0.64, -0.10),
    (130, 0.61, -0.11),
    (140, 0.58, -0.12),
    (150, 0.51, -0.14),
]


def resolve_map_path(map_name):
    base = Path(map_name)
    candidatos = [
        base,
        Path.cwd() / map_name,
        Path(__file__).resolve().parent / map_name,
        Path(__file__).resolve().parent.parent / map_name,
    ]
    for ruta in candidatos:
        if ruta.is_file():
            return str(ruta)
    raise FileNotFoundError(f"No se pudo cargar el archivo {map_name}")


def calcular_largo(h):
    return -0.675 * h + 1.3175


def sensor_id_desde_h(h):
    sensor_id, _, _ = min(tabla_configuraciones, key=lambda fila: abs(fila[1] - h))
    return sensor_id


def offset_cm_desde_h(h):
    _, _, offset = min(tabla_configuraciones, key=lambda fila: abs(fila[1] - h))
    return offset


class NodoDinamico:
    def __init__(self, x, y, theta, h, sensor_id=None, padre=None, trayectoria=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.h = h
        self.sensor_id = sensor_id if sensor_id is not None else sensor_id_desde_h(h)
        self.padre = padre
        self.trayectoria = [] if trayectoria is None else trayectoria

    def distancia(self, otro):
        return math.hypot(self.x - otro.x, self.y - otro.y)


def cuerpo_robot(x, y, theta, h):
    largo_px = calcular_largo(h) * escala_px_por_m
    ancho_px = h * escala_px_por_m
    dx = largo_px / 2.0
    dy = ancho_px / 2.0
    c = math.cos(theta)
    s = math.sin(theta)
    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    return np.array(
        [[x + px * c - py * s, y + px * s + py * c] for px, py in corners],
        dtype=np.int32,
    )


def centros_ruedas(x, y, theta, h):
    largo_px = calcular_largo(h) * escala_px_por_m
    ancho_px = h * escala_px_por_m
    x_front, x_mid, x_rear = +0.35 * largo_px, 0.0, -0.35 * largo_px
    y_left, y_right = +0.5 * ancho_px, -0.5 * ancho_px
    c = math.cos(theta)
    s = math.sin(theta)

    def tf(x_local, y_local):
        return int(round(x + x_local * c - y_local * s)), int(round(y + x_local * s + y_local * c))

    return [
        tf(x_front, y_left),
        tf(x_front, y_right),
        tf(x_mid, y_left),
        tf(x_mid, y_right),
        tf(x_rear, y_left),
        tf(x_rear, y_right),
    ]


def punto_cm(x, y, theta, h):
    off_px = offset_cm_desde_h(h) * escala_px_por_m
    return int(round(x + off_px * math.cos(theta))), int(round(y + off_px * math.sin(theta)))


def integrar_dinamica_completa(x, y, theta, h, v, w, h_objetivo, tiempo, paso):
    trayectoria = []
    t = 0.0
    while t < tiempo:
        x += v * math.cos(theta) * paso
        y += v * math.sin(theta) * paso
        theta += w * paso
        h += ((h_objetivo - h) / tau) * paso
        h = min(max(h, min(h_deseadas)), max(h_deseadas))
        trayectoria.append((x, y, theta, h))
        t += paso
    return trayectoria


mapa = cv2.imread(resolve_map_path("Pb4.png"))
alto, ancho, _ = mapa.shape
gris = cv2.cvtColor(mapa, cv2.COLOR_BGR2GRAY)
paredes = cv2.inRange(gris, 90, 110)
gutter = cv2.inRange(gris, 160, 200)
puntos_validos = np.column_stack(np.where(gutter > 0))


def generar_nodo_valido():
    for _ in range(1000):
        y, x = puntos_validos[random.randint(0, len(puntos_validos) - 1)]
        theta = random.uniform(-math.pi, math.pi)
        h = random.choice(h_deseadas)
        if not verificar_configuracion(x, y, theta, h):
            continue
        return NodoDinamico(x, y, theta, h, sensor_id_desde_h(h))
    return None


def verificar_colision(x, y):
    if x < 0 or x >= ancho or y < 0 or y >= alto:
        return True
    return paredes[int(y), int(x)] != 0


def verificar_configuracion(x, y, theta, h):
    if verificar_colision(x, y):
        return False

    body = cuerpo_robot(x, y, theta, h)
    if np.any(body[:, 0] < 0) or np.any(body[:, 0] >= ancho):
        return False
    if np.any(body[:, 1] < 0) or np.any(body[:, 1] >= alto):
        return False

    robot_mask = np.zeros_like(gris)
    cv2.fillPoly(robot_mask, [body], 255)
    if cv2.countNonZero(cv2.bitwise_and(robot_mask, paredes)) > 0:
        return False

    soportes = []
    for cx, cy in centros_ruedas(x, y, theta, h):
        if cx < 0 or cx >= ancho or cy < 0 or cy >= alto:
            return False
        if paredes[cy, cx] != 0:
            return False
        if gutter[cy, cx] == 0:
            soportes.append([cx, cy])

    if len(soportes) < 3:
        return False

    hull = cv2.convexHull(np.array(soportes, dtype=np.int32))
    cm = punto_cm(x, y, theta, h)
    return cv2.pointPolygonTest(hull.astype(np.float32), (float(cm[0]), float(cm[1])), False) >= 0


inicio = generar_nodo_valido()
objetivo = generar_nodo_valido()
nodos = [inicio]
camino = []
encontrado = False

for i in range(max_iter):
    if i % bias_freq == 0:
        q_rand = NodoDinamico(objetivo.x, objetivo.y, objetivo.theta, objetivo.h, objetivo.sensor_id)
    else:
        q_rand = generar_nodo_valido()
        if not q_rand:
            continue

    q_near = min(nodos, key=lambda n: n.distancia(q_rand))
    mejor_nuevo = None
    mejor_dist = float("inf")

    for v in v_set:
        for w in w_set:
            for h_objetivo in h_deseadas:
                tray = integrar_dinamica_completa(
                    q_near.x, q_near.y, q_near.theta, q_near.h, v, w, h_objetivo, T, dt
                )
                if not tray:
                    continue

                if any(verificar_colision(px, py) for px, py, _, _ in tray):
                    continue

                xf, yf, thetaf, hf = tray[-1]
                if not verificar_configuracion(xf, yf, thetaf, hf):
                    continue

                nuevo = NodoDinamico(
                    xf,
                    yf,
                    thetaf,
                    hf,
                    sensor_id_desde_h(hf),
                    padre=q_near,
                    trayectoria=tray,
                )
                dist = nuevo.distancia(q_rand)
                if dist < mejor_dist:
                    mejor_nuevo = nuevo
                    mejor_dist = dist

    if mejor_nuevo is None:
        continue

    nodos.append(mejor_nuevo)
    if mejor_nuevo.distancia(objetivo) < epsilon:
        objetivo.padre = mejor_nuevo
        encontrado = True
        break


fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(cv2.cvtColor(mapa, cv2.COLOR_BGR2RGB))
ax.set_title("Arbol RRT dinamico con modelo cinemático")
ax.axis("off")

for nodo in nodos:
    if nodo.padre and nodo.trayectoria:
        xs = [p[0] for p in nodo.trayectoria]
        ys = [p[1] for p in nodo.trayectoria]
        ax.plot(xs, ys, color="gray", linewidth=0.5)

ax.plot(inicio.x, inicio.y, "go", label="Inicio")
ax.plot(objetivo.x, objetivo.y, "yo", label="Objetivo")
ax.legend()
plt.show()

if encontrado:
    actual = objetivo
    while actual:
        camino.append(actual)
        actual = actual.padre
    camino.reverse()

    print("Camino encontrado. Configuraciones:")
    for i, nodo in enumerate(camino):
        largo = calcular_largo(nodo.h)
        _, ancho, offset = min(tabla_configuraciones, key=lambda fila: abs(fila[1] - nodo.h))
        print(
            f"Paso {i}: x={int(nodo.x)}, y={int(nodo.y)}, "
            f"theta={nodo.theta:.2f} rad, sensor={nodo.sensor_id}, "
            f"ancho={ancho:.2f} m, largo={largo:.2f} m, offset CM={offset:.2f} m"
        )
else:
    print("No se encontró camino.")
