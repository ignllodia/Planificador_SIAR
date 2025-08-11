import cv2
import numpy as np
import random
import math

# ----------------------
# ⚙ Parámetros principales
seccion_tunel_m = 1
escala_px_por_m = 62 / seccion_tunel_m
# ----------------------

# Tabla del SIAR
tabla_configuraciones = [
    (0,    0.51,  0.14),
    (10,   0.58,  0.12),
    (20,   0.64,  0.09),
    (30,   0.68,  0.06),
    (40,   0.70,  0.04),
    (50,   0.71,  0.02),
    (60,   0.71,  0.00),
    (70,   0.71, -0.01),
    (80,   0.70, -0.03),
    (90,   0.69, -0.05),
    (100,  0.68, -0.07),
    (110,  0.66, -0.08),
    (120,  0.64, -0.10),
    (130,  0.61, -0.11),
    (140,  0.58, -0.12),
    (150,  0.51, -0.14),
]

# Función corregida: longitud en función del ancho
def calcular_largo(ancho):
    return -0.675 * ancho + 1.3175

# Cargar imagen y preparar máscaras
img_original = cv2.imread('Pb4.png')
if img_original is None:
    print("No se pudo cargar la imagen.")
    exit()
height, width, _ = img_original.shape
gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
mask_walls = cv2.inRange(gray, 90, 110)
mask_gutter = cv2.inRange(gray, 160, 200)
valid_positions = np.column_stack(np.where(mask_gutter > 0))

found_any = False

while not found_any:
    y, x = valid_positions[random.randint(0, len(valid_positions)-1)]
    print(f"\nProbando centro del gutter en (x={x}, y={y})")
    found_any_config = False

    for (sensor, ancho_m, offset_cm_m) in tabla_configuraciones:
        largo_m = calcular_largo(ancho_m)
        width_px = int(ancho_m * escala_px_por_m)
        length_px = int(largo_m * escala_px_por_m)
        offset_px = offset_cm_m * escala_px_por_m

        wheel_dx = width_px / 2
        wheel_dy = length_px / 2.5
        wheel_size = 8

        for angle in range(0, sensor + 1, 15):
            theta = math.radians(angle)
            cos_t, sin_t = math.cos(theta), math.sin(theta)

            dx = width_px / 2
            dy = length_px / 2
            corners = np.array([
                [-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]
            ])
            rotated_corners = np.array([
                [x + (c[0] * cos_t - c[1] * sin_t),
                 y + (c[0] * sin_t + c[1] * cos_t)] for c in corners
            ], dtype=np.int32)

            if np.any(rotated_corners[:, 0] < 0) or np.any(rotated_corners[:, 0] >= width) \
               or np.any(rotated_corners[:, 1] < 0) or np.any(rotated_corners[:, 1] >= height):
                continue

            robot_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(robot_mask, [rotated_corners], 255)
            collision = cv2.countNonZero(cv2.bitwise_and(robot_mask, mask_walls))
            if collision != 0:
                continue

            cm_x = x
            cm_y = int(y + offset_px * cos_t)

            wheel_offsets = [
                (-wheel_dx, -wheel_dy), (-wheel_dx, 0), (-wheel_dx, +wheel_dy),
                (+wheel_dx, -wheel_dy), (+wheel_dx, 0), (+wheel_dx, +wheel_dy)
            ]
            support_points = []
            wheel_colors = []
            wheel_positions = []
            for (wx_off, wy_off) in wheel_offsets:
                wx = int(x + (wx_off * cos_t - wy_off * sin_t))
                wy = int(y + (wx_off * sin_t + wy_off * cos_t))
                wheel_positions.append((wx, wy))
                if mask_gutter[wy, wx] != 255:
                    support_points.append([wx, wy])
                    wheel_colors.append((0, 255, 0))
                else:
                    wheel_colors.append((0, 0, 255))

            img_result = img_original.copy()

            stable = False
            if len(support_points) >= 3:
                hull = cv2.convexHull(np.array(support_points))
                inside = cv2.pointPolygonTest(hull.astype(np.float32), (float(cm_x), float(cm_y)), False) >= 0
                color_poly = (0, 255, 0) if inside else (0, 0, 255)
                cv2.polylines(img_result, [hull], isClosed=True, color=color_poly, thickness=1)
                stable = inside

            cv2.polylines(img_result, [rotated_corners], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.circle(img_result, (cm_x, cm_y), 4, (0, 255, 0) if stable else (0, 0, 255), -1)

            for ((wx, wy), c) in zip(wheel_positions, wheel_colors):
                cv2.rectangle(img_result, (wx - wheel_size//2, wy - wheel_size//2),
                                             (wx + wheel_size//2, wy + wheel_size//2), c, -1)

            # Etiquetas
            cv2.putText(img_result, f"Sensor: {sensor} deg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.putText(img_result, f"Ancho: {ancho_m:.2f} m", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.putText(img_result, f"Largo: {largo_m:.2f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.putText(img_result, f"Offset CM: {offset_cm_m:.2f} m", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.putText(img_result, f"Angulo: {angle} deg", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            cv2.imshow("Simulación de estabilidad del SIAR", img_result)
            cv2.waitKey(500)

            found_any_config = True

    if found_any_config:
        found_any = True

cv2.destroyAllWindows()
