import math
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


MAPA_PATH = "Pb4.png"
ESCALA_PX_POR_M = 62.0

N_ITER = 3500
GOAL_BIAS = 0.18
M_CANDIDATOS = 10
DT = 0.08
T_SEG = 0.6
PASOS_VALIDACION = 5
MAX_FRAC_GUTTER = 0.35
DIST_GOAL_PX = 14.0
OFFSET_M = 0.0
TAU_H = 0.25

V_SET = [22.0, 30.0, 40.0]
W_SET = [-1.0, -0.5, 0.0, 0.5, 1.0]
H_SET = [0.52, 0.58, 0.64, 0.70]

SHOW_LIVE_TREE = True
LIVE_REFRESH_EVERY = 120
DRAW_ROBOTS_AT_END = True

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

random.seed(7)
np.random.seed(7)


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
    raise FileNotFoundError(f"No se encontró {map_name}")


def calcular_largo(h_m):
    return -0.675 * h_m + 1.3175


def offset_cm_para_h(h_m):
    _, _, offset = min(TABLA_CONFIGURACIONES, key=lambda fila: abs(fila[1] - h_m))
    return offset


def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class NodoDinamico:
    def __init__(self, x, y, theta, h, offset_m=0.0, padre=None, trayectoria_local=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.h = h
        self.offset_m = offset_m
        self.padre = padre
        self.trayectoria = [] if trayectoria_local is None else list(trayectoria_local)

    def distancia(self, otro):
        return math.hypot(self.x - otro.x, self.y - otro.y)


def segmentar_tres_clases(gray: np.ndarray):
    vals, counts = np.unique(gray, return_counts=True)
    if len(vals) < 3:
        raise RuntimeError("Se esperaban 3 niveles principales de gris.")

    idx = np.argsort(counts)[-3:]
    modos = np.sort(vals[idx])

    def banda(valor, tol=6):
        return max(0, valor - tol), min(255, valor + tol)

    flo_lo, flo_hi = banda(int(modos[0]))
    wal_lo, wal_hi = banda(int(modos[1]))
    gut_lo, gut_hi = banda(int(modos[2]))

    mask_floor = ((gray >= flo_lo) & (gray <= flo_hi)).astype(np.uint8)
    mask_pared = ((gray >= wal_lo) & (gray <= wal_hi)).astype(np.uint8)
    mask_gutter = ((gray >= gut_lo) & (gray <= gut_hi)).astype(np.uint8)
    return mask_floor, mask_pared, mask_gutter, modos


def _closest_on_mask(x, y, mask_bin):
    h, w = mask_bin.shape
    xi = max(0, min(w - 1, int(round(x))))
    yi = max(0, min(h - 1, int(round(y))))
    if mask_bin[yi, xi] > 0:
        return xi, yi

    inv = np.where(mask_bin > 0, 0, 1).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lab = labels[yi, xi]
    ys, xs = np.where(labels == lab)
    if len(xs) == 0:
        return xi, yi

    idx = np.argmin((xs - xi) ** 2 + (ys - yi) ** 2)
    return int(xs[idx]), int(ys[idx])


def seleccionar_inicio_fin_interactivo(mapa_bgr, mask_gutter):
    vis = mapa_bgr.copy()
    puntos = []
    win = "Selecciona inicio (verde) y fin (rojo) en el gutter"

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN or len(puntos) >= 2:
            return

        xg, yg = _closest_on_mask(x, y, (mask_gutter > 0).astype(np.uint8))
        puntos.append((xg, yg))
        color = (0, 255, 0) if len(puntos) == 1 else (0, 0, 255)
        cv2.circle(vis, (xg, yg), 6, color, -1)
        cv2.imshow(win, vis)

    cv2.namedWindow(win)
    cv2.imshow(win, vis)
    cv2.setMouseCallback(win, on_mouse)

    while len(puntos) < 2:
        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyWindow(win)
            raise SystemExit("Selección cancelada.")

    cv2.destroyWindow(win)
    return puntos[0], puntos[1]


def distancia_px(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)


def nodo_mas_cercano(nodos: List[NodoDinamico], x, y) -> NodoDinamico:
    return min(nodos, key=lambda n: distancia_px(n.x, n.y, x, y))


def reconstruir_camino(n: NodoDinamico):
    camino = []
    cur = n
    while cur is not None:
        if cur.trayectoria:
            camino = cur.trayectoria + camino
        else:
            camino = [(cur.x, cur.y, cur.theta, cur.h)] + camino
        cur = cur.padre
    return camino


def footprint_chasis_px(x_px, y_px, th, h_m, offset_m, escala_px_por_m):
    largo_px = calcular_largo(h_m) * escala_px_por_m
    ancho_px = h_m * escala_px_por_m
    dx = largo_px / 2.0
    dy = ancho_px / 2.0
    c = math.cos(th)
    s = math.sin(th)
    esquinas = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
    if offset_m != 0.0:
        esquinas[:, 0] += offset_m * escala_px_por_m
    return np.array(
        [[x_px + px * c - py * s, y_px + px * s + py * c] for px, py in esquinas],
        dtype=np.int32,
    )


def ruedas_centros_y_radio_px(x_px, y_px, th, h_m, escala_px_por_m):
    largo_px = calcular_largo(h_m) * escala_px_por_m
    ancho_px = h_m * escala_px_por_m
    x_front, x_mid, x_rear = +0.35 * largo_px, 0.0, -0.35 * largo_px
    y_left, y_right = +0.5 * ancho_px, -0.5 * ancho_px
    radio = max(2, int(round(0.08 * escala_px_por_m)))
    c = math.cos(th)
    s = math.sin(th)

    def tf(x_local, y_local):
        return (
            int(round(x_px + x_local * c - y_local * s)),
            int(round(y_px + x_local * s + y_local * c)),
            radio,
        )

    return [
        tf(x_front, y_left),
        tf(x_front, y_right),
        tf(x_mid, y_left),
        tf(x_mid, y_right),
        tf(x_rear, y_left),
        tf(x_rear, y_right),
    ]


def _centro_masas_px(x_px, y_px, th, h_m):
    offset_px = (OFFSET_M + offset_cm_para_h(h_m)) * ESCALA_PX_POR_M
    return (
        int(round(x_px + offset_px * math.cos(th))),
        int(round(y_px + offset_px * math.sin(th))),
    )


def integrar_dinamica_completa(x, y, theta, h, v, w, h_obj, T_total, dt):
    tray = []
    t = 0.0
    while t < T_total:
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta = wrap_angle(theta + w * dt)
        h += ((h_obj - h) / TAU_H) * dt
        h = min(max(h, min(H_SET)), max(H_SET))
        tray.append((x, y, theta, h))
        t += dt
    return tray


def es_configuracion_valida(
    x,
    y,
    th,
    h,
    offset_m,
    escala_px_por_m,
    mask_pared,
    mask_gutter,
    max_frac_rueda_en_gutter,
    dist_obst_px=None,
    margen_min_px=0,
):
    h_img, w_img = mask_pared.shape
    if not (0 <= int(round(x)) < w_img and 0 <= int(round(y)) < h_img):
        return False

    body = footprint_chasis_px(x, y, th, h, offset_m, escala_px_por_m)
    if np.any(body[:, 0] < 0) or np.any(body[:, 0] >= w_img):
        return False
    if np.any(body[:, 1] < 0) or np.any(body[:, 1] >= h_img):
        return False

    if dist_obst_px is not None and margen_min_px > 0:
        x_int = int(round(x))
        y_int = int(round(y))
        if dist_obst_px[y_int, x_int] < margen_min_px:
            return False

    wall_mask_u8 = mask_pared.astype(np.uint8) * 255 if mask_pared.dtype != np.uint8 else mask_pared
    robot_mask = np.zeros_like(wall_mask_u8)
    cv2.fillPoly(robot_mask, [body], 255)
    if cv2.countNonZero(cv2.bitwise_and(robot_mask, wall_mask_u8)) > 0:
        return False

    ruedas = ruedas_centros_y_radio_px(x, y, th, h, escala_px_por_m)
    ruedas_en_gutter = 0
    soportes = []
    for cx, cy, _ in ruedas:
        if not (0 <= cx < w_img and 0 <= cy < h_img):
            return False
        if bool(mask_pared[cy, cx]):
            return False
        if bool(mask_gutter[cy, cx]):
            ruedas_en_gutter += 1
        else:
            soportes.append([cx, cy])

    if ruedas_en_gutter / max(1, len(ruedas)) > max_frac_rueda_en_gutter:
        return False
    if len(soportes) < 3:
        return False

    hull = cv2.convexHull(np.array(soportes, dtype=np.int32))
    cm = _centro_masas_px(x, y, th, h)
    return cv2.pointPolygonTest(hull.astype(np.float32), (float(cm[0]), float(cm[1])), False) >= 0


def es_trayectoria_valida_a_lo_largo(
    x0,
    y0,
    th0,
    h0,
    x1,
    y1,
    th1,
    h1,
    offset_m,
    escala_px_por_m,
    mask_pared,
    mask_gutter,
    pasos,
    max_frac_rueda_en_gutter,
    dist_obst_px=None,
    margen_min_px=0,
):
    for i in range(1, pasos + 1):
        alpha = i / pasos
        x = (1 - alpha) * x0 + alpha * x1
        y = (1 - alpha) * y0 + alpha * y1
        th = wrap_angle((1 - alpha) * th0 + alpha * th1)
        h = (1 - alpha) * h0 + alpha * h1
        if not es_configuracion_valida(
            x,
            y,
            th,
            h,
            offset_m,
            escala_px_por_m,
            mask_pared,
            mask_gutter,
            max_frac_rueda_en_gutter,
            dist_obst_px=dist_obst_px,
            margen_min_px=margen_min_px,
        ):
            return False
    return True


def dibujar_robot(
    img_bgr,
    x_px,
    y_px,
    th,
    h_m,
    escala_px_por_m,
    offset_m=0.0,
    color_cuerpo=(0, 255, 255),
    color_rueda=(0, 255, 0),
):
    poly = footprint_chasis_px(x_px, y_px, th, h_m, offset_m, escala_px_por_m)
    cv2.polylines(img_bgr, [poly], True, color_cuerpo, 2, cv2.LINE_AA)
    for cx, cy, radio in ruedas_centros_y_radio_px(x_px, y_px, th, h_m, escala_px_por_m):
        if 0 <= cx < img_bgr.shape[1] and 0 <= cy < img_bgr.shape[0]:
            cv2.circle(img_bgr, (cx, cy), radio, color_rueda, -1, lineType=cv2.LINE_AA)


def mostrar_resultado(
    mapa_gray,
    segmentos_arbol_px,
    camino_px,
    inicio_px,
    fin_px,
    escala_px_por_m,
    offset_m=0.0,
):
    vis = cv2.cvtColor(mapa_gray, cv2.COLOR_GRAY2BGR)
    for p0, p1 in segmentos_arbol_px:
        cv2.line(vis, p0, p1, (255, 255, 0), 2, cv2.LINE_AA)

    if len(camino_px) >= 2:
        pts = np.array([(int(round(x)), int(round(y))) for x, y, _, _ in camino_px], dtype=np.int32)
        cv2.polylines(vis, [pts], False, (0, 0, 255), 3, cv2.LINE_AA)

    if len(camino_px) > 1:
        step = max(1, len(camino_px) // 40)
        for x, y, th, h in camino_px[::step]:
            dibujar_robot(
                vis,
                x,
                y,
                th,
                h,
                escala_px_por_m,
                offset_m=offset_m,
                color_cuerpo=(0, 255, 255),
                color_rueda=(0, 200, 0),
            )

    cv2.circle(vis, (int(inicio_px[0]), int(inicio_px[1])), 6, (0, 255, 0), -1)
    cv2.circle(vis, (int(fin_px[0]), int(fin_px[1])), 6, (0, 255, 255), -1)
    return vis


def rrt_dinamico(
    gray,
    mask_pared_bool,
    mask_gutter_bool,
    dist_obst_px,
    margen_min_px,
    q_start,
    q_goal,
    live_canvas=None,
    live_every=120,
):
    segmentos_px: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    arbol: List[NodoDinamico] = [q_start]
    h_img, w_img = gray.shape

    if live_canvas is None:
        live_canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for it in range(N_ITER):
        if random.random() < GOAL_BIAS:
            xr, yr = q_goal.x, q_goal.y
        else:
            xr = random.uniform(0, w_img - 1)
            yr = random.uniform(0, h_img - 1)
            if mask_pared_bool[int(yr), int(xr)]:
                continue

        qn = nodo_mas_cercano(arbol, xr, yr)
        mejor = None
        mejor_tray = None
        mejor_dist = float("inf")

        for _ in range(M_CANDIDATOS):
            v = random.choice(V_SET)
            w = random.choice(W_SET)
            h_obj = random.choice(H_SET)
            tray = integrar_dinamica_completa(qn.x, qn.y, qn.theta, qn.h, v, w, h_obj, T_SEG, DT)
            if not tray:
                continue

            x0, y0, th0, h0 = qn.x, qn.y, qn.theta, qn.h
            valida = True
            for x1, y1, th1, h1 in tray:
                if not es_trayectoria_valida_a_lo_largo(
                    x0,
                    y0,
                    th0,
                    h0,
                    x1,
                    y1,
                    th1,
                    h1,
                    offset_m=OFFSET_M,
                    escala_px_por_m=ESCALA_PX_POR_M,
                    mask_pared=mask_pared_bool,
                    mask_gutter=mask_gutter_bool,
                    pasos=PASOS_VALIDACION,
                    max_frac_rueda_en_gutter=MAX_FRAC_GUTTER,
                    dist_obst_px=dist_obst_px,
                    margen_min_px=margen_min_px,
                ):
                    valida = False
                    break
                x0, y0, th0, h0 = x1, y1, th1, h1

            if not valida:
                continue

            xf, yf, thf, hf = tray[-1]
            if not es_configuracion_valida(
                xf,
                yf,
                thf,
                hf,
                OFFSET_M,
                ESCALA_PX_POR_M,
                mask_pared=mask_pared_bool,
                mask_gutter=mask_gutter_bool,
                max_frac_rueda_en_gutter=MAX_FRAC_GUTTER,
                dist_obst_px=dist_obst_px,
                margen_min_px=margen_min_px,
            ):
                continue

            dist = distancia_px(xf, yf, xr, yr)
            if dist < mejor_dist:
                mejor_dist = dist
                mejor = (xf, yf, thf, hf)
                mejor_tray = tray

        if mejor is None:
            if SHOW_LIVE_TREE and (it % live_every == 0):
                vis = live_canvas.copy()
                for p0, p1 in segmentos_px:
                    cv2.line(vis, p0, p1, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("RRT - Arbol (en vivo)", vis)
                cv2.waitKey(1)
            continue

        q_new = NodoDinamico(*mejor, offset_m=OFFSET_M, padre=qn, trayectoria_local=mejor_tray)
        arbol.append(q_new)
        p0 = (int(round(qn.x)), int(round(qn.y)))
        p1 = (int(round(q_new.x)), int(round(q_new.y)))
        segmentos_px.append((p0, p1))

        if SHOW_LIVE_TREE and (len(segmentos_px) % live_every == 0):
            vis = live_canvas.copy()
            for s0, s1 in segmentos_px:
                cv2.line(vis, s0, s1, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("RRT - Arbol (en vivo)", vis)
            cv2.waitKey(1)

        if distancia_px(q_new.x, q_new.y, q_goal.x, q_goal.y) <= DIST_GOAL_PX:
            q_goal.padre = q_new
            arbol.append(q_goal)
            return reconstruir_camino(q_goal), segmentos_px

    return None, segmentos_px


if __name__ == "__main__":
    mapa_bgr = cv2.imread(resolve_map_path(MAPA_PATH), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(mapa_bgr, cv2.COLOR_BGR2GRAY)

    _, mask_pared, mask_gutter, modos = segmentar_tres_clases(gray)
    print(f"Modos (gris): suelo={modos[0]}, pared={modos[1]}, gutter={modos[2]}")

    (x_ini, y_ini), (x_fin, y_fin) = seleccionar_inicio_fin_interactivo(mapa_bgr, mask_gutter)
    print(f"Inicio (px): {(x_ini, y_ini)} | Fin (px): {(x_fin, y_fin)}")

    libre_para_dt = np.where(mask_pared > 0, 0, 255).astype(np.uint8)
    dist_obst_px = cv2.distanceTransform(libre_para_dt, cv2.DIST_L2, 5)

    half_min_w_px = 0.5 * min(H_SET) * ESCALA_PX_POR_M
    margen_min_px = max(2, int(round(0.6 * half_min_w_px)))

    q_start = NodoDinamico(x_ini, y_ini, 0.0, 0.64, offset_m=OFFSET_M, padre=None, trayectoria_local=[])
    q_goal = NodoDinamico(x_fin, y_fin, 0.0, 0.64, offset_m=OFFSET_M, padre=None, trayectoria_local=[])

    base_live = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(base_live, (x_ini, y_ini), 6, (0, 255, 0), -1)
    cv2.circle(base_live, (x_fin, y_fin), 6, (0, 255, 255), -1)

    camino, segmentos = rrt_dinamico(
        gray,
        mask_pared_bool=mask_pared.astype(bool),
        mask_gutter_bool=mask_gutter.astype(bool),
        dist_obst_px=dist_obst_px,
        margen_min_px=margen_min_px,
        q_start=q_start,
        q_goal=q_goal,
        live_canvas=base_live,
        live_every=LIVE_REFRESH_EVERY,
    )

    vis_tree_only = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for p0, p1 in segmentos:
        cv2.line(vis_tree_only, p0, p1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(vis_tree_only, (x_ini, y_ini), 6, (0, 255, 0), -1)
    cv2.circle(vis_tree_only, (x_fin, y_fin), 6, (0, 255, 255), -1)
    cv2.imshow("Fase A - Arbol (final, sin ruta ni robots)", vis_tree_only)
    cv2.waitKey(0)

    if camino is None:
        print("No se encontró camino.")
        cv2.destroyAllWindows()
        raise SystemExit(0)

    vis_tree_route = vis_tree_only.copy()
    if len(camino) >= 2:
        pts = np.array([(int(round(x)), int(round(y))) for x, y, _, _ in camino], np.int32)
        cv2.polylines(vis_tree_route, [pts], False, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow("Fase B - Arbol + Ruta (sin robots)", vis_tree_route)
    cv2.waitKey(0)

    if DRAW_ROBOTS_AT_END:
        vis_full = mostrar_resultado(
            gray,
            segmentos,
            camino,
            (x_ini, y_ini),
            (x_fin, y_fin),
            escala_px_por_m=ESCALA_PX_POR_M,
            offset_m=OFFSET_M,
        )
        cv2.imshow("Fase C - Arbol + Ruta + Robots", vis_full)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
