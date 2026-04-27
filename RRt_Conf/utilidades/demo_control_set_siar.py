# -*- coding: utf-8 -*-

import math
import random

import cv2
import numpy as np


MAP_PATH = "sewer_map_x3.png"

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

CONTROL_SET_ALL = [
    (0.25, 0.0, 0),
    (0.25, 0.2, 0),
    (0.25, -0.2, 0),
    (0.35, 0.0, 0),
    (0.00, 0.0, -1),
    (0.00, 0.0, 1),
    (0.10, 0.6, 0),
    (0.10, -0.6, 0),
    (0.10, 0.3, 0),
    (0.10, -0.3, 0),
    (0.12, 0.8, 0),
    (0.12, -0.8, 0),
]

WINDOW_NAME = "Demo controles"
VIEW_MAX_W = 1600
VIEW_MAX_H = 900
VIEW_ZOOM_STEP = 1.25
VIEW_MAX_ZOOM = 12.0
VIEW_PAN_FRAC = 0.18
VISUAL_REF_MAX_DIM = 1400
DEBUG_INVALID_CONTROLS = True


def m2px(m):
    return int(round(m * PIXELS_PER_M))


def px2m(px):
    return float(px) / PIXELS_PER_M


def visual_scale(smap):
    max_dim = max(1, smap.w, smap.h)
    return min(1.0, VISUAL_REF_MAX_DIM / float(max_dim))


def visual_thickness(smap, base=2, min_px=1):
    return max(min_px, int(round(base * visual_scale(smap))))


def visual_marker_size(smap, base=7, min_px=4):
    return max(min_px, int(round(base * max(0.7, visual_scale(smap)))))


def draw_marker(img, pt, color, smap, base_size=7, base_thickness=1, marker_type=cv2.MARKER_CROSS):
    cv2.drawMarker(
        img,
        (int(pt[0]), int(pt[1])),
        color,
        markerType=marker_type,
        markerSize=visual_marker_size(smap, base=base_size),
        thickness=visual_thickness(smap, base_thickness),
        line_type=cv2.LINE_AA,
    )


class Viewer:
    def __init__(self, smap):
        self.map_w = smap.w
        self.map_h = smap.h
        fit = min(VIEW_MAX_W / max(1, self.map_w), VIEW_MAX_H / max(1, self.map_h), 1.0)
        self.display_w = max(320, int(round(self.map_w * fit)))
        self.display_h = max(240, int(round(self.map_h * fit)))
        self.zoom = 1.0
        self.cx = self.map_w / 2.0
        self.cy = self.map_h / 2.0
        self._last_view = (0, 0, self.map_w, self.map_h)

    def reset(self):
        self.zoom = 1.0
        self.cx = self.map_w / 2.0
        self.cy = self.map_h / 2.0

    def _view_size(self):
        vw = max(40, min(self.map_w, int(round(self.map_w / self.zoom))))
        vh = max(40, min(self.map_h, int(round(self.map_h / self.zoom))))
        return vw, vh

    def _clamp_center(self):
        vw, vh = self._view_size()
        half_w = vw / 2.0
        half_h = vh / 2.0
        self.cx = min(max(self.cx, half_w), self.map_w - half_w)
        self.cy = min(max(self.cy, half_h), self.map_h - half_h)

    def current_view(self):
        self._clamp_center()
        vw, vh = self._view_size()
        x0 = int(round(self.cx - vw / 2.0))
        y0 = int(round(self.cy - vh / 2.0))
        x0 = min(max(0, x0), self.map_w - vw)
        y0 = min(max(0, y0), self.map_h - vh)
        self._last_view = (x0, y0, vw, vh)
        return self._last_view

    def render(self, img):
        x0, y0, vw, vh = self.current_view()
        crop = img[y0:y0 + vh, x0:x0 + vw]
        interp = cv2.INTER_AREA if (vw > self.display_w or vh > self.display_h) else cv2.INTER_LINEAR
        return cv2.resize(crop, (self.display_w, self.display_h), interpolation=interp)

    def show(self, img, extra_lines=None):
        disp = self.render(img)
        hud = [f"zoom x{self.zoom:.2f} | +/- zoom | WASD mover | f reajustar"]
        if extra_lines:
            hud.extend(extra_lines)
        y = 18
        for line in hud:
            cv2.putText(disp, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            y += 16
        cv2.imshow(WINDOW_NAME, disp)

    def display_to_map(self, x, y):
        x0, y0, vw, vh = self.current_view()
        if not (0 <= x < self.display_w and 0 <= y < self.display_h):
            return None
        mx = x0 + (x / max(1, self.display_w)) * vw
        my = y0 + (y / max(1, self.display_h)) * vh
        mx = min(max(0, int(round(mx))), self.map_w - 1)
        my = min(max(0, int(round(my))), self.map_h - 1)
        return mx, my

    def zoom_in(self):
        self.zoom = min(VIEW_MAX_ZOOM, self.zoom * VIEW_ZOOM_STEP)

    def zoom_out(self):
        self.zoom = max(1.0, self.zoom / VIEW_ZOOM_STEP)

    def pan(self, dx_sign, dy_sign):
        vw, vh = self._view_size()
        self.cx += dx_sign * vw * VIEW_PAN_FRAC
        self.cy += dy_sign * vh * VIEW_PAN_FRAC
        self._clamp_center()


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


def evaluate_configuration(smap, state):
    x = int(state.x)
    y = int(state.y)
    if not smap.inside_bounds(x, y):
        return False, {"reason": "center_out_of_bounds", "center": (x, y)}
    if not (ROBOT_W_MIN <= state.w <= ROBOT_W_MAX):
        return False, {"reason": "width_out_of_range", "w": state.w}

    poly = robot_polygon(state).reshape((-1, 1, 2))
    x0 = max(int(np.min(poly[:, :, 0])) - 2, 0)
    y0 = max(int(np.min(poly[:, :, 1])) - 2, 0)
    x1 = min(int(np.max(poly[:, :, 0])) + 3, smap.w)
    y1 = min(int(np.max(poly[:, :, 1])) + 3, smap.h)
    if x1 <= x0 or y1 <= y0:
        return False, {"reason": "empty_roi"}

    roi_w = x1 - x0
    roi_h = y1 - y0
    rmask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    poly_roi = poly.copy()
    poly_roi[:, :, 0] -= x0
    poly_roi[:, :, 1] -= y0
    cv2.fillPoly(rmask, [poly_roi], 255)

    wall_roi = smap.wall_mask[y0:y1, x0:x1]
    if cv2.countNonZero(cv2.bitwise_and(rmask, wall_roi)) > 0:
        return False, {"reason": "body_hits_wall"}

    supports = []
    wheel_positions = wheel_centers_pixels(state)
    wheel_states = {}
    for wheel_name, (cx, cy) in wheel_positions.items():
        if not smap.inside_bounds(cx, cy):
            return False, {"reason": "wheel_out_of_bounds", "wheel": wheel_name, "point": (cx, cy)}
        if smap.wall_mask[cy, cx] != 0:
            return False, {"reason": "wheel_on_wall", "wheel": wheel_name, "point": (cx, cy)}
        if smap.gutter_mask[cy, cx] == 0:
            supports.append([cx, cy])
            wheel_states[wheel_name] = "support"
        else:
            wheel_states[wheel_name] = "gutter"

    if len(supports) < 3:
        return False, {
            "reason": "insufficient_supports",
            "support_count": len(supports),
            "wheel_states": wheel_states,
        }

    hull = cv2.convexHull(np.array(supports, dtype=np.int32))
    if _polygon_area(hull) < 1.0:
        return False, {"reason": "degenerate_support_polygon", "wheel_states": wheel_states}

    cx, cy = cog_pixel_from_table(state)
    if cv2.pointPolygonTest(hull.astype(np.float32), (float(cx), float(cy)), False) < 0:
        return False, {
            "reason": "cog_outside_support_polygon",
            "cog": (cx, cy),
            "support_count": len(supports),
            "wheel_states": wheel_states,
        }

    return True, {
        "reason": "ok",
        "support_count": len(supports),
        "wheel_states": wheel_states,
        "cog": (cx, cy),
    }


def valid_configuration(smap, state):
    ok, _ = evaluate_configuration(smap, state)
    return ok


def format_debug_info(info):
    if not info:
        return "sin detalle"

    reason = info.get("reason", "unknown")
    if reason == "center_out_of_bounds":
        return f"centro fuera de mapa en {info['center']}"
    if reason == "width_out_of_range":
        return f"ancho fuera de rango w={info['w']:.3f}"
    if reason == "empty_roi":
        return "ROI vacia"
    if reason == "body_hits_wall":
        return "cuerpo del robot intersecta wall"
    if reason == "wheel_out_of_bounds":
        return f"rueda {info['wheel']} fuera de mapa en {info['point']}"
    if reason == "wheel_on_wall":
        return f"rueda {info['wheel']} sobre wall en {info['point']}"
    if reason == "insufficient_supports":
        return f"apoyos insuficientes ({info['support_count']}) | {info.get('wheel_states', {})}"
    if reason == "degenerate_support_polygon":
        return f"poligono de soporte degenerado | {info.get('wheel_states', {})}"
    if reason == "cog_outside_support_polygon":
        return (
            f"CoM fuera del poligono de soporte en {info['cog']} | "
            f"apoyos={info['support_count']} | {info.get('wheel_states', {})}"
        )
    return str(info)


def propagate(state, control, smap):
    v, w_z, step_ref = control
    steps = max(1, int(round(TPROP / DT)))

    x = state.x
    y = state.y
    th = state.th
    w = state.w
    path_pts = []
    traj_states = []
    debug_info = None

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
                debug_info = {
                    "stage": "intermediate",
                    "step": k,
                    "point": (int(x), int(y)),
                    "reason": "center_out_of_bounds",
                }
                return None, None, None, debug_info
            ok, details = evaluate_configuration(smap, st)
            if not ok:
                debug_info = {
                    "stage": "intermediate",
                    "step": k,
                    "state": st,
                    **details,
                }
                return None, None, None, debug_info

        path_pts.append((int(x), int(y)))
        traj_states.append(st)

    new_state = State(x, y, th, w)
    ok, details = evaluate_configuration(smap, new_state)
    if not ok:
        debug_info = {
            "stage": "final",
            "step": steps - 1,
            "state": new_state,
            **details,
        }
        return None, None, None, debug_info

    return new_state, path_pts, traj_states, None


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
    if smap is not None:
        thickness = visual_thickness(smap, thickness)
        arrow_thickness = visual_thickness(smap, 2)
    else:
        arrow_thickness = 2

    poly = robot_polygon(state).reshape((-1, 1, 2))
    cv2.polylines(img, [poly], True, color, thickness)

    p0 = (int(state.x), int(state.y))
    p1 = (
        int(state.x + 0.35 * math.cos(state.th) * m2px(1.0)),
        int(state.y + 0.35 * math.sin(state.th) * m2px(1.0)),
    )
    cv2.arrowedLine(img, p0, p1, color, arrow_thickness, tipLength=0.25)

    if smap is not None:
        supports = support_points_pixels(smap, state)
        for cx, cy in wheel_centers_pixels(state).values():
            if not smap.inside_bounds(cx, cy):
                continue
            if smap.wall_mask[cy, cx] != 0:
                col = (0, 0, 255)
            elif smap.gutter_mask[cy, cx] != 0:
                col = (0, 0, 0)
            else:
                col = (0, 255, 0)
            draw_marker(img, (cx, cy), col, smap, base_size=6, base_thickness=1, marker_type=cv2.MARKER_CROSS)

        if len(supports) >= 3:
            hull = cv2.convexHull(np.array(supports, dtype=np.int32))
            cv2.polylines(img, [hull], True, (0, 255, 0), visual_thickness(smap, 1))
            cx, cy = cog_pixel_from_table(state)
            inside = cv2.pointPolygonTest(hull.astype(np.float32), (float(cx), float(cy)), False) >= 0
        else:
            cx, cy = cog_pixel_from_table(state)
            inside = False

        draw_marker(
            img,
            (cx, cy),
            (0, 255, 0) if inside else (0, 0, 255),
            smap,
            base_size=7,
            base_thickness=2,
            marker_type=cv2.MARKER_TILTED_CROSS,
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


def handle_viewer_key(viewer, key):
    if key in (ord("+"), ord("=")):
        viewer.zoom_in()
        return True
    if key in (ord("-"), ord("_")):
        viewer.zoom_out()
        return True
    if key == ord("f"):
        viewer.reset()
        return True
    if key == ord("w"):
        viewer.pan(0, -1)
        return True
    if key == ord("s"):
        viewer.pan(0, 1)
        return True
    if key == ord("a"):
        viewer.pan(-1, 0)
        return True
    if key == ord("d"):
        viewer.pan(1, 0)
        return True
    return False


def instruction_lines():
    return [
        "Selecciona control por terminal: c<idx>",
        "r: reset home | q: salir",
    ]


def animate_control(smap, base_img, state, control, label, viewer):
    ns, path_pts, traj_states, debug_info = propagate(state, control, smap)
    collision = False
    if ns is None or not path_pts:
        collision = True
        ns, path_pts, traj_states = propagate_no_check(state, control)

    frame = base_img.copy()
    for i in range(1, len(path_pts)):
        cv2.line(frame, path_pts[i - 1], path_pts[i], (120, 180, 255), visual_thickness(smap, 2))

    draw_state(frame, state, (0, 200, 0), 2, smap)
    draw_state(frame, ns, (255, 0, 0), 2, smap)

    txt = label + ("  (colision/invalid)" if collision else "")
    debug_txt = None
    if collision and debug_info is not None:
        debug_txt = f"DEBUG [{debug_info.get('stage', '?')} step={debug_info.get('step', '?')}]: {format_debug_info(debug_info)}"
        if DEBUG_INVALID_CONTROLS:
            print(debug_txt)
    for _ in range(30):
        extra_lines = [txt]
        if debug_txt is not None:
            extra_lines.append(debug_txt)
        viewer.show(frame, extra_lines=extra_lines)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            return state
        handle_viewer_key(viewer, key)

    for st in traj_states:
        step_img = base_img.copy()
        draw_state(step_img, st, (0, 165, 255), 2, smap)
        extra_lines = [txt]
        if debug_txt is not None:
            extra_lines.append(debug_txt)
        viewer.show(step_img, extra_lines=extra_lines)
        key = cv2.waitKey(80) & 0xFF
        if key == ord("q"):
            break
        handle_viewer_key(viewer, key)

    return state if collision else ns


class HomePicker:
    def __init__(self, viewer):
        self.viewer = viewer
        self.pos = None
        self.th = None
        self.stage = 0

    def mouse(self, event, x, y, flags, param):
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        mapped = self.viewer.display_to_map(x, y)
        if mapped is None:
            return
        x, y = mapped

        if self.stage == 0:
            self.pos = (x, y)
            self.stage = 1
            return

        if self.stage == 1:
            dx = x - self.pos[0]
            dy = y - self.pos[1]
            self.th = math.atan2(dy, dx)
            self.stage = 2


def pick_home(smap, base, viewer):
    picker = HomePicker(viewer)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, viewer.display_w, viewer.display_h)
    cv2.setMouseCallback(WINDOW_NAME, picker.mouse)

    while True:
        img = base.copy()
        if picker.pos is not None:
            draw_marker(img, picker.pos, (0, 255, 0), smap, base_size=8, base_thickness=2, marker_type=cv2.MARKER_CROSS)
        if picker.pos is not None and picker.th is not None:
            draw_state(img, State(picker.pos[0], picker.pos[1], picker.th, ROBOT_W0), (0, 255, 0), 2, smap)

        viewer.show(img, extra_lines=[
            "Click 1: posicion home | Click 2: orientacion",
            "ESPACIO: confirmar | q: salir",
        ])
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            raise SystemExit
        handle_viewer_key(viewer, key)
        if key == ord(" ") and picker.pos is not None and picker.th is not None:
            return State(picker.pos[0], picker.pos[1], picker.th, ROBOT_W0)


def _imprimir_controles():
    print("Controles disponibles:")
    for idx, ctrl in enumerate(CONTROL_SET_ALL):
        print(f"  c{idx}: v={ctrl[0]:.2f} w_z={ctrl[1]:.2f} step_ref={ctrl[2]}")
    print("Comandos: c<idx> / r (reset home) / q (salir)")


def _resolver_comando(cmd):
    mode = cmd[0].lower()
    try:
        idx = int(cmd[1:])
    except ValueError:
        return None, None, "Formato invalido. Usa c<idx>."

    if mode == "c" and 0 <= idx < len(CONTROL_SET_ALL):
        ctrl = CONTROL_SET_ALL[idx]
        label = f"Control c{idx} | v={ctrl[0]:.2f} w_z={ctrl[1]:.2f} step_ref={ctrl[2]}"
        return ctrl, label, None

    return None, None, "Indice fuera de rango."


def main():
    smap = SewerMap(MAP_PATH)
    base = smap.draw_overlay()
    viewer = Viewer(smap)
    home = pick_home(smap, base, viewer)
    state = home

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, viewer.display_w, viewer.display_h)
    _imprimir_controles()

    while True:
        frame = base.copy()
        draw_state(frame, state, (0, 200, 0), 2, smap)
        viewer.show(frame, extra_lines=instruction_lines())
        cv2.waitKey(1)

        cmd = input("Selecciona control: ").strip()
        if not cmd:
            continue

        if cmd.lower() == "q":
            break

        if cmd.lower() == "r":
            state = home
            frame = base.copy()
            draw_state(frame, state, (0, 200, 0), 2, smap)
            viewer.show(frame, extra_lines=instruction_lines())
            continue

        ctrl, label, error = _resolver_comando(cmd)
        if error is not None:
            print(error)
            continue

        state = animate_control(smap, base, state, ctrl, label, viewer)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
