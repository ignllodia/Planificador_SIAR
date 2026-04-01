import math
import random

import cv2
import numpy as np


seccion_tunel_m = 1
escala_px_por_m = 62 / seccion_tunel_m
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


def calcular_largo(ancho):
    return -0.675 * ancho + 1.3175


def _datos_sensor(sensor_id):
    return tabla_configuraciones[sensor_id // 10][1:]


class Nodo:
    def __init__(self, x, y, theta, sensor_id, padre=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.sensor_id = sensor_id
        self.padre = padre

    def distancia(self, otro):
        return math.hypot(self.x - otro.x, self.y - otro.y)


class RRT_SIAR:
    def __init__(self, mapa_path):
        self.mapa = cv2.imread(mapa_path)
        self.gray = cv2.cvtColor(self.mapa, cv2.COLOR_BGR2GRAY)
        self.mask_walls = cv2.inRange(self.gray, 90, 110)
        self.mask_gutter = cv2.inRange(self.gray, 160, 200)
        self.height, self.width = self.gray.shape
        self.nodos = []

    def _cuerpo_rotado(self, x, y, theta, width_px, length_px):
        dx = width_px / 2
        dy = length_px / 2
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        esquinas = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
        poligono = [
            [x + (px * cos_t - py * sin_t), y + (px * sin_t + py * cos_t)]
            for px, py in esquinas
        ]
        return np.array(poligono, dtype=np.int32), cos_t, sin_t

    def validar_configuracion(self, x, y, theta, sensor_id):
        ancho, offset = _datos_sensor(sensor_id)
        largo = calcular_largo(ancho)
        width_px = ancho * escala_px_por_m
        length_px = largo * escala_px_por_m
        offset_px = offset * escala_px_por_m

        cuerpo, cos_t, sin_t = self._cuerpo_rotado(x, y, theta, width_px, length_px)
        if np.any(cuerpo[:, 0] < 0) or np.any(cuerpo[:, 0] >= self.width):
            return False
        if np.any(cuerpo[:, 1] < 0) or np.any(cuerpo[:, 1] >= self.height):
            return False

        mascara_robot = np.zeros_like(self.gray)
        cv2.fillPoly(mascara_robot, [cuerpo], 255)
        if cv2.countNonZero(cv2.bitwise_and(mascara_robot, self.mask_walls)) > 0:
            return False

        rueda_dx = width_px / 2
        rueda_dy = length_px / 2.5
        offsets_ruedas = [
            (-rueda_dx, -rueda_dy),
            (-rueda_dx, 0),
            (-rueda_dx, +rueda_dy),
            (+rueda_dx, -rueda_dy),
            (+rueda_dx, 0),
            (+rueda_dx, +rueda_dy),
        ]

        puntos_apoyo = []
        for off_x, off_y in offsets_ruedas:
            wx = int(x + (off_x * cos_t - off_y * sin_t))
            wy = int(y + (off_x * sin_t + off_y * cos_t))
            dentro = 0 <= wx < self.width and 0 <= wy < self.height
            if dentro and self.mask_gutter[wy, wx] != 255:
                puntos_apoyo.append([wx, wy])

        if len(puntos_apoyo) < 3:
            return False

        centro_masas = (x, int(y + offset_px * cos_t))
        casco = cv2.convexHull(np.array(puntos_apoyo))
        dentro_soporte = cv2.pointPolygonTest(
            casco.astype(np.float32),
            (float(centro_masas[0]), float(centro_masas[1])),
            False,
        )
        return dentro_soporte >= 0

    def generar_configuracion_valida(self):
        for _ in range(1000):
            y = np.random.randint(0, self.height)
            x = np.random.randint(0, self.width)
            if self.mask_gutter[y, x] != 255:
                continue

            theta = random.uniform(0, 2 * math.pi)
            sensor_id, _, _ = random.choice(tabla_configuraciones)
            if self.validar_configuracion(x, y, theta, sensor_id):
                return Nodo(x, y, theta, sensor_id)

        return None

    def conectar_nodos(self, n1, n2, pasos=10):
        for i in range(1, pasos + 1):
            alpha = i / pasos
            x = (1 - alpha) * n1.x + alpha * n2.x
            y = (1 - alpha) * n1.y + alpha * n2.y
            theta = (1 - alpha) * n1.theta + alpha * n2.theta
            if not self.validar_configuracion(x, y, theta, n2.sensor_id):
                return False

        return True
