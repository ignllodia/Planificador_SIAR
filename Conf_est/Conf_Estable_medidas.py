import pygame
import math
import time
import random

pygame.init()

# Configuración de pantalla
width, height = 1000, 700
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Simulación de Estabilidad del SIAR")

# Colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)

# Datos reales de configuración (sensor, ancho en m, desplazamiento centro de masas en m)
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

class Robot:
    def __init__(self, x, y, width_m, length_m, cm_offset_m):
        self.x = x
        self.y = y
        scale = 600  # nuevo factor de escala: 1 m = 600 px
        self.width_px = width_m * scale
        self.length_px = length_m * scale
        self.cm_offset_px = cm_offset_m * scale
        self.uncertainty_radius = 20
        self.set_wheels()
        self.set_center_of_mass()

    def set_wheels(self):
        dx = self.width_px / 2
        dy = self.length_px / 2.5
        self.wheel_positions = [
            (self.x - dx, self.y - dy),  # izquierda frontal
            (self.x - dx, self.y),       # izquierda media
            (self.x - dx, self.y + dy),  # izquierda trasera
            (self.x + dx, self.y - dy),  # derecha frontal
            (self.x + dx, self.y),       # derecha media
            (self.x + dx, self.y + dy),  # derecha trasera
        ]
        # Ruedas activas aleatoriamente
        self.wheel_contact = [random.choice([True, False]) for _ in self.wheel_positions]

    def set_center_of_mass(self):
        self.center_of_mass = [self.x, self.y + self.cm_offset_px]

    def draw(self, screen):
        for i, (wx, wy) in enumerate(self.wheel_positions):
            color = GREEN if self.wheel_contact[i] else RED
            pygame.draw.circle(screen, color, (int(wx), int(wy)), 10)

        polygon = self.stability_polygon()
        if polygon:
            pygame.draw.polygon(screen, BLUE, polygon, 2)

        stable, danger = self.is_stable()
        if danger:
            cm_color = ORANGE
        elif stable:
            cm_color = GREEN
        else:
            cm_color = RED
        pygame.draw.circle(screen, cm_color, (int(self.center_of_mass[0]), int(self.center_of_mass[1])), 10)

    def stability_polygon(self):
        points = [self.wheel_positions[i] for i in range(6) if self.wheel_contact[i]]
        if len(points) < 3:
            return None
        return self.convex_hull(points)

    def is_stable(self):
        polygon = self.stability_polygon()
        if not polygon:
            return False, False
        danger = self.polygon_crosses_area(self.center_of_mass, self.uncertainty_radius, polygon)
        inside = self.point_in_polygon(self.center_of_mass, polygon)
        return inside, danger

    @staticmethod
    def point_in_polygon(point, polygon):
        x, y = point
        inside = False
        p1x, p1y = polygon[0]
        for i in range(len(polygon) + 1):
            p2x, p2y = polygon[i % len(polygon)]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    @staticmethod
    def polygon_crosses_area(point, radius, polygon):
        x, y = point
        for i in range(len(polygon)):
            p1x, p1y = polygon[i]
            p2x, p2y = polygon[(i + 1) % len(polygon)]
            if Robot.line_intersects_circle(p1x, p1y, p2x, p2y, x, y, radius):
                return True
        return False

    @staticmethod
    def line_intersects_circle(x1, y1, x2, y2, cx, cy, radius):
        acx = cx - x1
        acy = cy - y1
        abx = x2 - x1
        aby = y2 - y1
        ab2 = abx ** 2 + aby ** 2
        acab = acx * abx + acy * aby
        t = max(0, min(1, acab / ab2))
        hx = acx - abx * t
        hy = acy - aby * t
        return hx ** 2 + hy ** 2 <= radius ** 2

    @staticmethod
    def convex_hull(points):
        points = sorted(set(points))
        if len(points) <= 1:
            return points
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower, upper = [], []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

# Bucle secuencial de simulación
font = pygame.font.SysFont(None, 28)
for sensor_val, ancho, offset_cm in tabla_configuraciones:
    robot = Robot(x=500, y=350, width_m=ancho, length_m=0.98, cm_offset_m=offset_cm)
    screen.fill(BLACK)
    robot.draw(screen)
    info = f"Sensor={sensor_val} | Ancho={ancho:.2f}m | CM offset={offset_cm:.2f}m"
    txt = font.render(info, True, WHITE)
    screen.blit(txt, (20, 20))
    pygame.display.flip()
    time.sleep(2)

pygame.quit()
