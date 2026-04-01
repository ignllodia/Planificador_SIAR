# -*- coding: utf-8 -*-

import math
import random

import cv2
import numpy as np


MAP_PATH = "Pb4.png"

ROBOT_LEN = 0.88
ROBOT_W_MIN = 0.52
ROBOT_W_MAX = 0.85
ROBOT_W0 = 0.70

PIXELS_PER_M = 65

TABLA_CONFIGURACIONES = [
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
WS = [fila[1] for fila in TABLA_CONFIGURACIONES]

DT = 0.05
TPROP = 0.5
CHECK_STEP = 3
TAU_H = 0.2

CONTROL_SET = [
    (0.25, 0.0, 0),
    (0.25, 0.2, 0),
    (0.25, -0.2, 0),
    (0.35, 0.0, 0),
]
CONTROL_SET_TURN = [
    (0.00, 0.0, -1),
    (0.00, 0.0, 1),
    (0.10, 0.6, 0),
    (0.10, -0.6, 0),
    (0.12, 0.3, 0),
    (0.12, -0.3, 0),
    (0.25, 0.8, 0),
    (0.25, -0.8, 0),
]


def m2px(m):
    return int(round(m * PIXELS_PER_M))


def px2m(px):
    return float(px) / PIXELS_PER_M


def calcular_largo(w_m):
    return -0.675 * w_m + 1.3175


def wrap_angle(angulo):
    while angulo > math.pi:
        angulo -= 2 * math.pi
    while angulo < -math.pi:
        angulo += 2 * math.pi
    return angulo


def ang_dist(a, b):
    return abs(wrap_angle(a - b))


def _indice_ancho_cercano(w):
    return min(range(len(WS)), key=lambda i: abs(WS[i] - w))


class State:
    def __init__(self, x, y, th, w):
        self.x = x
        self.y = y
        self.th = th
        self.w = w


class SewerMap:
    def __init__(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"No se pudo abrir: {path}")

        self.gray = img
        self.h, self.w = img.shape[:2]

        self.free_mask = np.zeros_like(img, dtype=np.uint8)
        self.wall_mask = np.zeros_like(img, dtype=np.uint8)
        self.gutter_mask = np.zeros_like(img, dtype=np.uint8)

        self.free_mask[self.gray < 100] = 255
        self.wall_mask[(self.gray >= 100) & (self.gray < 180)] = 255
        self.gutter_mask[self.gray >= 180] = 255

        ys, xs = np.where(self.free_mask > 0)
        self.free_points = np.column_stack((xs, ys))

    def inside_bounds(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def sample_free(self, rng):
        if len(self.free_points) == 0:
            raise RuntimeError("No hay pixeles libres en el mapa.")

        idx = rng.randrange(0, len(self.free_points))
        x, y = self.free_points[idx]
        return int(x), int(y)

    def draw_overlay(self):
        vis = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        vis[self.free_mask > 0] = (200, 200, 200)
        vis[self.wall_mask > 0] = (0, 0, 0)
        vis[self.gutter_mask > 0] = (0, 0, 255)
        return vis


def robot_polygon(state):
    largo = m2px(calcular_largo(state.w))
    ancho = m2px(state.w)
    hx = largo / 2.0
    hy = ancho / 2.0

    puntos = np.array([[+hx, +hy], [+hx, -hy], [-hx, -hy], [-hx, +hy]], dtype=np.float32)
    c = math.cos(state.th)
    s = math.sin(state.th)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    poly = (rot @ puntos.T).T
    poly[:, 0] += state.x
    poly[:, 1] += state.y
    return poly.astype(np.int32)


def wheel_centers_pixels(state):
    largo = m2px(ROBOT_LEN)
    ancho = m2px(state.w)
    xf, xm, xr = +0.35 * largo, 0.0, -0.35 * largo
    yl, yr = +0.5 * ancho, -0.5 * ancho
    c = math.cos(state.th)
    s = math.sin(state.th)

    def transformar(x_local, y_local):
        return (
            int(round(c * x_local - s * y_local + state.x)),
            int(round(s * x_local + c * y_local + state.y)),
        )

    return {
        "FL": transformar(xf, yl),
        "FR": transformar(xf, yr),
        "ML": transformar(xm, yl),
        "MR": transformar(xm, yr),
        "RL": transformar(xr, yl),
        "RR": transformar(xr, yr),
    }


def support_points_pixels(smap, state):
    puntos = []
    for cx, cy in wheel_centers_pixels(state).values():
        if 0 <= cx < smap.w and 0 <= cy < smap.h and smap.gutter_mask[cy, cx] == 0:
            puntos.append([cx, cy])
    return puntos


def cog_pixel_from_table(state):
    _, _, offset_m = min(TABLA_CONFIGURACIONES, key=lambda fila: abs(fila[1] - state.w))
    offset_px = m2px(offset_m)
    c = math.cos(state.th)
    s = math.sin(state.th)
    return int(round(state.x + offset_px * c)), int(round(state.y + offset_px * s))


def _polygon_area(contour_int32):
    contorno = contour_int32.reshape(-1, 2).astype(np.float32)
    return abs(cv2.contourArea(contorno))


def valid_configuration(smap, state):
    x = int(state.x)
    y = int(state.y)
    if not smap.inside_bounds(x, y):
        return False
    if not (ROBOT_W_MIN <= state.w <= ROBOT_W_MAX):
        return False

    poly = robot_polygon(state).reshape((-1, 1, 2))
    x0 = max(int(np.min(poly[:, :, 0])) - 2, 0)
    y0 = max(int(np.min(poly[:, :, 1])) - 2, 0)
    x1 = min(int(np.max(poly[:, :, 0])) + 3, smap.w)
    y1 = min(int(np.max(poly[:, :, 1])) + 3, smap.h)
    if x1 <= x0 or y1 <= y0:
        return False

    roi_w = x1 - x0
    roi_h = y1 - y0
    rmask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    poly_roi = poly.copy()
    poly_roi[:, :, 0] -= x0
    poly_roi[:, :, 1] -= y0
    cv2.fillPoly(rmask, [poly_roi], 255)

    wall_roi = smap.wall_mask[y0:y1, x0:x1]
    if cv2.countNonZero(cv2.bitwise_and(rmask, wall_roi)) > 0:
        return False

    supports = []
    for cx, cy in wheel_centers_pixels(state).values():
        if not smap.inside_bounds(cx, cy):
            return False
        if smap.wall_mask[cy, cx] != 0:
            return False
        if smap.gutter_mask[cy, cx] == 0:
            supports.append([cx, cy])

    if len(supports) < 3:
        return False

    hull = cv2.convexHull(np.array(supports, dtype=np.int32))
    if _polygon_area(hull) < 1.0:
        return False

    cx, cy = cog_pixel_from_table(state)
    if cv2.pointPolygonTest(hull.astype(np.float32), (float(cx), float(cy)), False) < 0:
        return False

    return True


def propagate(state, control, smap):
    v, w_z, step_ref = control
    steps = max(1, int(round(TPROP / DT)))

    x = state.x
    y = state.y
    th = state.th
    w = state.w
    path_pts = []
    traj_states = []

    px_step = v * DT * PIXELS_PER_M
    i0 = _indice_ancho_cercano(w)
    iref = max(0, min(len(WS) - 1, i0 + step_ref))
    wref = WS[iref]

    for k in range(steps):
        c = math.cos(th)
        s = math.sin(th)
        x += px_step * c
        y += px_step * s
        th = wrap_angle(th + w_z * DT)

        w += ((wref - w) / TAU_H) * DT
        w = min(max(w, ROBOT_W_MIN), ROBOT_W_MAX)

        st = State(x, y, th, w)
        if (k % CHECK_STEP) == 0:
            if not smap.inside_bounds(int(x), int(y)):
                return None, None, None
            if not valid_configuration(smap, st):
                return None, None, None

        path_pts.append((int(x), int(y)))
        traj_states.append(st)

    new_state = State(x, y, th, w)
    if not valid_configuration(smap, new_state):
        return None, None, None

    return new_state, path_pts, traj_states


def propagate_no_check(state, control):
    v, w_z, step_ref = control
    steps = max(1, int(round(TPROP / DT)))

    x = state.x
    y = state.y
    th = state.th
    w = state.w
    path_pts = []
    traj_states = []
    px_step = v * DT * PIXELS_PER_M

    for _ in range(steps):
        c = math.cos(th)
        s = math.sin(th)
        x += px_step * c
        y += px_step * s
        th = wrap_angle(th + w_z * DT)

        i0 = _indice_ancho_cercano(w)
        iref = max(0, min(len(WS) - 1, i0 + step_ref))
        wref = WS[iref]
        w += ((wref - w) / TAU_H) * DT
        w = min(max(w, ROBOT_W_MIN), ROBOT_W_MAX)

        st = State(x, y, th, w)
        path_pts.append((int(x), int(y)))
        traj_states.append(st)

    return State(x, y, th, w), path_pts, traj_states


def draw_state(img, state, color, thickness=2, smap=None):
    poly = robot_polygon(state).reshape((-1, 1, 2))
    cv2.polylines(img, [poly], True, color, thickness)

    p0 = (int(state.x), int(state.y))
    p1 = (
        int(state.x + 0.35 * math.cos(state.th) * m2px(1.0)),
        int(state.y + 0.35 * math.sin(state.th) * m2px(1.0)),
    )
    cv2.arrowedLine(img, p0, p1, color, 2, tipLength=0.25)

    if smap is not None:
        for cx, cy in wheel_centers_pixels(state).values():
            if not smap.inside_bounds(cx, cy):
                continue
            if smap.wall_mask[cy, cx] != 0:
                col = (0, 0, 255)
            elif smap.gutter_mask[cy, cx] != 0:
                col = (0, 0, 0)
            else:
                col = (0, 255, 0)
            cv2.circle(img, (cx, cy), 3, col, -1)

    cv2.putText(
        img,
        f"w={state.w:.3f} m",
        (int(state.x) + 10, int(state.y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def pick_valid_state(smap, rng, max_tries=800):
    for _ in range(max_tries):
        x, y = smap.sample_free(rng)
        th = rng.uniform(-math.pi, math.pi)
        st = State(x, y, th, ROBOT_W0)
        if valid_configuration(smap, st):
            return st

    ys, xs = np.where(smap.gutter_mask > 0)
    puntos = list(zip(xs, ys))
    for _ in range(max_tries):
        if not puntos:
            break
        x, y = puntos[rng.randrange(0, len(puntos))]
        th = rng.uniform(-math.pi, math.pi)
        st = State(x, y, th, ROBOT_W0)
        if valid_configuration(smap, st):
            return st

    raise RuntimeError("No se pudo encontrar un estado valido para la demo.")


def animate_control(smap, base_img, state, control, label, win="Demo controles"):
    ns, path_pts, traj_states = propagate(state, control, smap)
    collision = False
    if ns is None or not path_pts:
        collision = True
        ns, path_pts, traj_states = propagate_no_check(state, control)

    frame = base_img.copy()
    for i in range(1, len(path_pts)):
        cv2.line(frame, path_pts[i - 1], path_pts[i], (120, 180, 255), 2)

    draw_state(frame, state, (0, 200, 0), 2, smap)
    draw_state(frame, ns, (255, 0, 0), 2, smap)

    txt = label + ("  (colision/invalid)" if collision else "")
    cv2.putText(
        frame,
        txt,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if collision else (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(win, frame)
    cv2.waitKey(600)

    for st in traj_states:
        step_img = base_img.copy()
        draw_state(step_img, st, (0, 165, 255), 2, smap)
        cv2.putText(
            step_img,
            txt,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if collision else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win, step_img)
        if cv2.waitKey(80) & 0xFF == ord("q"):
            break

    return state if collision else ns


class HomePicker:
    def __init__(self):
        self.pos = None
        self.th = None
        self.stage = 0

    def mouse(self, event, x, y, flags, param):
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.stage == 0:
            self.pos = (x, y)
            self.stage = 1
            return

        if self.stage == 1:
            dx = x - self.pos[0]
            dy = y - self.pos[1]
            self.th = math.atan2(dy, dx)
            self.stage = 2


def pick_home(smap, base):
    picker = HomePicker()
    cv2.namedWindow("Demo controles", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Demo controles", picker.mouse)

    while True:
        img = base.copy()
        cv2.putText(
            img,
            "Click 1: posicion home | Click 2: orientacion | ESPACIO para confirmar",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if picker.pos is not None:
            cv2.circle(img, picker.pos, 4, (0, 255, 0), -1)
        if picker.pos is not None and picker.th is not None:
            draw_state(img, State(picker.pos[0], picker.pos[1], picker.th, ROBOT_W0), (0, 255, 0), 2, smap)

        cv2.imshow("Demo controles", img)
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            raise SystemExit
        if key == ord(" ") and picker.pos is not None and picker.th is not None:
            return State(picker.pos[0], picker.pos[1], picker.th, ROBOT_W0)


def _imprimir_controles():
    print("Controles disponibles:")
    for idx, ctrl in enumerate(CONTROL_SET):
        print(f"  n{idx}: v={ctrl[0]:.2f} w_z={ctrl[1]:.2f} step_ref={ctrl[2]}")
    for idx, ctrl in enumerate(CONTROL_SET_TURN):
        print(f"  t{idx}: v={ctrl[0]:.2f} w_z={ctrl[1]:.2f} step_ref={ctrl[2]}")
    print("Comandos: n<idx> / t<idx> / r (reset home) / q (salir)")


def _resolver_comando(cmd):
    mode = cmd[0].lower()
    try:
        idx = int(cmd[1:])
    except ValueError:
        return None, None, "Formato invalido. Usa n<idx> o t<idx>."

    if mode == "n" and 0 <= idx < len(CONTROL_SET):
        ctrl = CONTROL_SET[idx]
        label = f"Modo NORMAL | v={ctrl[0]:.2f} w_z={ctrl[1]:.2f} step_ref={ctrl[2]}"
        return ctrl, label, None

    if mode == "t" and 0 <= idx < len(CONTROL_SET_TURN):
        ctrl = CONTROL_SET_TURN[idx]
        label = f"Modo TURN | v={ctrl[0]:.2f} w_z={ctrl[1]:.2f} step_ref={ctrl[2]}"
        return ctrl, label, None

    return None, None, "Indice fuera de rango."


def main():
    smap = SewerMap(MAP_PATH)
    base = smap.draw_overlay()
    home = pick_home(smap, base)
    state = home

    cv2.namedWindow("Demo controles", cv2.WINDOW_NORMAL)
    _imprimir_controles()

    while True:
        cmd = input("Selecciona control: ").strip()
        if not cmd:
            continue

        if cmd.lower() == "q":
            break

        if cmd.lower() == "r":
            state = home
            frame = base.copy()
            draw_state(frame, state, (0, 200, 0), 2, smap)
            cv2.imshow("Demo controles", frame)
            continue

        ctrl, label, error = _resolver_comando(cmd)
        if error is not None:
            print(error)
            continue

        state = animate_control(smap, base, state, ctrl, label, win="Demo controles")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
