
import numpy as np
import math
import random
import cv2

# === Parámetros del entorno y robot ===
seccion_tunel_m = 1
escala_px_por_m = 62 / seccion_tunel_m
tabla_configuraciones = [
    (0, 0.51, 0.14), (10, 0.58, 0.12), (20, 0.64, 0.09), (30, 0.68, 0.06),
    (40, 0.70, 0.04), (50, 0.71, 0.02), (60, 0.71, 0.00), (70, 0.71, -0.01),
    (80, 0.70, -0.03), (90, 0.69, -0.05), (100, 0.68, -0.07), (110, 0.66, -0.08),
    (120, 0.64, -0.10), (130, 0.61, -0.11), (140, 0.58, -0.12), (150, 0.51, -0.14),
]

def calcular_largo(ancho):
    return -0.675 * ancho + 1.3175

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

    def validar_configuracion(self, x, y, theta, sensor_id):
        ancho, offset = tabla_configuraciones[sensor_id // 10][1:]
        largo = calcular_largo(ancho)
        width_px = ancho * escala_px_por_m
        length_px = largo * escala_px_por_m
        offset_px = offset * escala_px_por_m

        dx, dy = width_px / 2, length_px / 2
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        corners = np.array([
            [-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]
        ])
        rotated = np.array([
            [x + (c[0]*cos_t - c[1]*sin_t), y + (c[0]*sin_t + c[1]*cos_t)]
            for c in corners
        ], dtype=np.int32)

        if np.any(rotated[:,0] < 0) or np.any(rotated[:,0] >= self.width) or            np.any(rotated[:,1] < 0) or np.any(rotated[:,1] >= self.height):
            return False

        robot_mask = np.zeros_like(self.gray)
        cv2.fillPoly(robot_mask, [rotated], 255)
        if cv2.countNonZero(cv2.bitwise_and(robot_mask, self.mask_walls)) > 0:
            return False

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
            if 0 <= wx < self.width and 0 <= wy < self.height and self.mask_gutter[wy, wx] != 255:
                support_pts.append([wx, wy])

        if len(support_pts) < 3:
            return False

        cm = (x, int(y + offset_px * cos_t))
        hull = cv2.convexHull(np.array(support_pts))
        inside = cv2.pointPolygonTest(hull.astype(np.float32), (float(cm[0]), float(cm[1])), False) >= 0
        return inside

    def generar_configuracion_valida(self):
        for _ in range(1000):
            y, x = np.random.randint(0, self.height), np.random.randint(0, self.width)
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
            sensor = n2.sensor_id
            if not self.validar_configuracion(x, y, theta, sensor):
                return False
        return True
