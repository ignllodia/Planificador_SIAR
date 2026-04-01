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
        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.height, self.width = self.gray.shape
        self.nodos = []

    def _rotar_cuerpo(self, x, y, theta, width_px, length_px):
        dx = width_px / 2.0
        dy = length_px / 2.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        esquinas = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)

        cuerpo = np.empty_like(esquinas)
        for i, (px, py) in enumerate(esquinas):
            cuerpo[i] = [
                x + (px * cos_t - py * sin_t),
                y + (px * sin_t + py * cos_t),
            ]

        return cuerpo, cos_t, sin_t

    def validar_configuracion(self, x, y, theta, sensor_id):
        ancho, offset = _datos_sensor(sensor_id)
        largo = calcular_largo(ancho)
        width_px = max(1.0, ancho * escala_px_por_m)
        length_px = max(1.0, largo * escala_px_por_m)
        offset_px = offset * escala_px_por_m

        cuerpo, cos_t, sin_t = self._rotar_cuerpo(x, y, theta, width_px, length_px)
        cuerpo_i = cuerpo.astype(np.int32)
        if (
            (cuerpo[:, 0].min() < 0)
            or (cuerpo[:, 0].max() >= self.width)
            or (cuerpo[:, 1].min() < 0)
            or (cuerpo[:, 1].max() >= self.height)
        ):
            return False

        margin = int(max(1, round(width_px * 0.5 + 2)))
        ksz = max(3, margin | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        paredes_dilatadas = cv2.dilate(self.mask_walls, kernel)

        mascara_robot = np.zeros_like(self.gray)
        cv2.fillPoly(mascara_robot, [cuerpo_i], 255)
        if cv2.countNonZero(cv2.bitwise_and(mascara_robot, paredes_dilatadas)) > 0:
            return False

        wheel_dx = width_px / 2.0 * 0.95
        wheel_dy = length_px / 2.5
        wheel_offsets = [
            (-wheel_dx, -wheel_dy),
            (-wheel_dx, 0.0),
            (-wheel_dx, +wheel_dy),
            (+wheel_dx, -wheel_dy),
            (+wheel_dx, 0.0),
            (+wheel_dx, +wheel_dy),
        ]

        support_pts = []
        for wx_off, wy_off in wheel_offsets:
            wx = int(round(x + (wx_off * cos_t - wy_off * sin_t)))
            wy = int(round(y + (wx_off * sin_t + wy_off * cos_t)))
            if 0 <= wx < self.width and 0 <= wy < self.height:
                if paredes_dilatadas[wy, wx] == 0 and self.mask_gutter[wy, wx] != 255:
                    support_pts.append([wx, wy])

        if len(support_pts) < 3:
            return False

        cmx = int(round(x))
        cmy = int(round(y + offset_px * math.cos(theta)))
        hull = cv2.convexHull(np.array(support_pts, dtype=np.int32))
        if cv2.pointPolygonTest(hull.astype(np.float32), (float(cmx), float(cmy)), False) < 0:
            return False

        return True

    def generar_configuracion_valida(self):
        gutter_band = cv2.dilate(self.mask_gutter, self.kernel3)
        for _ in range(2000):
            y = np.random.randint(0, self.height)
            x = np.random.randint(0, self.width)
            if gutter_band[y, x] != 255:
                continue

            theta = random.uniform(0, 2 * math.pi)
            sensor_id, _, _ = random.choice(tabla_configuraciones)
            if self.validar_configuracion(x, y, theta, sensor_id):
                return Nodo(x, y, theta, sensor_id)

        return None

    def conectar_nodos(self, n1, n2, pasos=20):
        for i in range(1, pasos + 1):
            alpha = i / pasos
            x = (1 - alpha) * n1.x + alpha * n2.x
            y = (1 - alpha) * n1.y + alpha * n2.y
            theta = (1 - alpha) * n1.theta + alpha * n2.theta
            if not self.validar_configuracion(x, y, theta, n2.sensor_id):
                return False

        return True
