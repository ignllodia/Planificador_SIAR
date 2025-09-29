# -*- coding: utf-8 -*-
"""
Planificador RRT kinodinámico para SIAR (modelo uniciclo + ajuste de ancho).
- Búsqueda nearest híbrida (subset + exacta periódica)
- Validación por ROI del polígono del robot
- Chequeo incremental en propagate
- Visualización del árbol y estados intermedios
- Estabilidad con apoyos en ruedas y CoG dentro del hull
- Cambio de conjunto de controles por "atasco" (histéresis y cooldown).
"""

import cv2
import numpy as np
import math
import random
from dataclasses import dataclass

# --------------------- Mapa ---------------------
MAP_PATH = "Pb4.png"

# --------------------- Robot (aprox. SIAR) ---------------------
ROBOT_LEN   = 0.88
ROBOT_W_MIN = 0.52
ROBOT_W_MAX = 0.85
ROBOT_W0    = 0.70
W_DOT_MAX   = 0.20  # no se usa explícitamente aquí

# --------------------- Escala ---------------------
PIXELS_PER_M = 65
def m2px(m): return int(round(m * PIXELS_PER_M))
def px2m(px): return float(px) / PIXELS_PER_M

# --------------------- Tabla configuración estable ---------------------
TABLA_CONFIGURACIONES = [
    (0,    0.51,  0.14),(10,   0.58,  0.12),(20,   0.64,  0.09),
    (30,   0.68,  0.06),(40,   0.70,  0.04),(50,   0.71,  0.02),
    (60,   0.71,  0.00),(70,   0.71, -0.01),(80,   0.70, -0.03),
    (90,   0.69, -0.05),(100,  0.68, -0.07),(110,  0.66, -0.08),
    (120,  0.64, -0.10),(130,  0.61, -0.11),(140,  0.58, -0.12),
    (150,  0.51, -0.14),
]
def calcular_largo(w_m): return -0.675*w_m + 1.3175
WS = [row[1] for row in TABLA_CONFIGURACIONES]
def config_from_w(w): return min(TABLA_CONFIGURACIONES, key=lambda r: abs(r[1]-w))

# --------------------- Planificador ---------------------
MAX_ITERS = 60000
GOAL_SAMPLING_RATE = 0.10
DT = 0.09
TPROP = 0.2
CHECK_STEP = 3
NEAR_GOAL_DIST = m2px(0.30)
NEAR_GOAL_DTH  = math.radians(20.0)
NEAR_GOAL_DW   = 0.10
TAU_H = 0.6

# Nearest
K_SUBSET    = 64
EXACT_EVERY = 40
RNG_SEED    = 12345

# Dibujo
DRAW_EVERY = 200

# Histéresis del cambio de modo (sin sensado geométrico)
ENTER_TURN        = 20   # fallos seguidos para entrar en modo giros
EXIT_STREAK       = 300  # éxitos seguidos para volver a modo normal
MODE_FREEZE       = 500  # enfriamiento: iter mínimas entre cambios

# --------------------- Control sets ---------------------
CONTROL_SET = [
    (0.25,  0.0,   0),   # recto
    (0.25,  0.4,   0),   # suave izq
    (0.25, -0.4,   0),   # suave dcha
]
CONTROL_SET_TURN = [
    (0.00,  0.0,  -1),   # estrechar parado
    (0.00,  0.0,  +1),   # ensanchar parado
    (0.10,  0.6,   0),   # giro fuerte izq lento
    (0.10, -0.6,   0),   # giro fuerte dcha lento
    (0.12,  0.3,   0),   # giro medio izq
    (0.12, -0.3,   0),   # giro medio dcha
]

# --------------------- Utilidades ---------------------
def wrap_angle(a):
    while a > math.pi:  a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def ang_dist(a, b): return abs(wrap_angle(a - b))

@dataclass
class State:
    x: float  # px
    y: float  # px
    th: float # rad
    w: float  # m

@dataclass
class Node:
    state: State
    parent: int
    path: list          # [(x,y), ...]
    traj_states: list   # [State, ...]

class SewerMap:
    def __init__(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"No se pudo abrir: {path}")
        self.gray = img
        self.h, self.w = img.shape[:2]

        self.free_mask   = np.zeros_like(img, dtype=np.uint8)
        self.wall_mask   = np.zeros_like(img, dtype=np.uint8)
        self.gutter_mask = np.zeros_like(img, dtype=np.uint8)

        self.free_mask[self.gray < 100]                      = 255
        self.wall_mask[(self.gray >=100) & (self.gray <180)] = 255
        self.gutter_mask[self.gray >=180]                    = 255

        ys, xs = np.where(self.free_mask > 0)
        self.free_points = np.column_stack((xs, ys))

    def inside_bounds(self, x, y): return 0 <= x < self.w and 0 <= y < self.h

    def sample_free(self, rng):
        if len(self.free_points) == 0:
            raise RuntimeError("No hay píxeles libres en el mapa.")
        i = rng.randrange(0, len(self.free_points))
        x, y = self.free_points[i]
        return int(x), int(y)

    def draw_overlay(self):
        vis = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        vis[self.free_mask > 0]   = (200,200,200)
        vis[self.wall_mask > 0]   = (0,0,0)
        vis[self.gutter_mask > 0] = (0,0,255)
        return vis

# --------------------- Geometría robot ---------------------
def robot_polygon(state: State):
    L = m2px(calcular_largo(state.w))
    W = m2px(state.w)
    hx, hy = L/2.0, W/2.0
    pts = np.array([[ +hx, +hy],[ +hx, -hy],[-hx, -hy],[-hx, +hy]], dtype=np.float32)
    c, s = math.cos(state.th), math.sin(state.th)
    R = np.array([[c, -s],[s, c]], dtype=np.float32)
    rot = (R @ pts.T).T
    rot[:,0] += state.x; rot[:,1] += state.y
    return rot.astype(np.int32)

def wheel_centers_pixels(state: State):
    L = m2px(ROBOT_LEN); W = m2px(state.w)
    xf, xm, xr = +0.35*L, 0.0, -0.35*L
    yl, yr = +0.5*W, -0.5*W
    c, s = math.cos(state.th), math.sin(state.th)
    def tf(xl, yl):
        return (int(round(c*xl - s*yl + state.x)),
                int(round(s*xl + c*yl + state.y)))
    return {'FL': tf(xf, yl), 'FR': tf(xf, yr),
            'ML': tf(xm, yl), 'MR': tf(xm, yr),
            'RL': tf(xr, yl), 'RR': tf(xr, yr)}

def support_points_pixels(smap: "SewerMap", state: State):
    pts = []
    for (cx, cy) in wheel_centers_pixels(state).values():
        if 0 <= cx < smap.w and 0 <= cy < smap.h and smap.gutter_mask[cy, cx] == 0:
            pts.append([cx, cy])
    return pts

def cog_pixel_from_table(state: State):
    _, _, off_m = config_from_w(state.w)
    off_px = m2px(off_m)
    c, s = math.cos(state.th), math.sin(state.th)
    return int(round(state.x + off_px*c)), int(round(state.y + off_px*s))

# --------------------- Validación ---------------------
def _polygon_area(contour_int32):
    c = contour_int32.reshape(-1,2).astype(np.float32)
    return abs(cv2.contourArea(c))

def valid_configuration(smap: "SewerMap", state: State):
    x = int(state.x); y = int(state.y)
    if not smap.inside_bounds(x, y): return False
    if not (ROBOT_W_MIN <= state.w <= ROBOT_W_MAX): return False

    poly = robot_polygon(state).reshape((-1,1,2))
    x0 = max(int(np.min(poly[:,:,0]))-2, 0)
    y0 = max(int(np.min(poly[:,:,1]))-2, 0)
    x1 = min(int(np.max(poly[:,:,0]))+3, smap.w)
    y1 = min(int(np.max(poly[:,:,1]))+3, smap.h)
    if x1<=x0 or y1<=y0: return False

    roi_w, roi_h = x1-x0, y1-y0
    rmask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    poly_roi = poly.copy()
    poly_roi[:,:,0] -= x0; poly_roi[:,:,1] -= y0
    cv2.fillPoly(rmask, [poly_roi], 255)

    wall_roi = smap.wall_mask[y0:y1, x0:x1]
    if cv2.countNonZero(cv2.bitwise_and(rmask, wall_roi)) > 0: return False

    centers = wheel_centers_pixels(state)
    supports = []
    wm, gm = smap.wall_mask, smap.gutter_mask
    for _, (cx, cy) in centers.items():
        if not smap.inside_bounds(cx, cy): return False
        if wm[cy, cx] != 0: return False
        if gm[cy, cx] == 0: supports.append([cx, cy])

    if len(supports) < 3: return False
    hull = cv2.convexHull(np.array(supports, dtype=np.int32))
    if _polygon_area(hull) < 1.0: return False
    cx, cy = cog_pixel_from_table(state)
    if cv2.pointPolygonTest(hull.astype(np.float32), (float(cx), float(cy)), False) < 0: return False
    return True

# --------------------- Dinámica ---------------------
def propagate(state: State, control, smap: SewerMap):
    v, w_z, step_ref = control
    steps = max(1, int(round(TPROP / DT)))

    x, y, th, w = state.x, state.y, state.th, state.w
    path_pts, traj_states = [], []

    cos, sin, wrap = math.cos, math.sin, wrap_angle
    inside = smap.inside_bounds

    for k in range(steps):
        c, s = cos(th), sin(th)
        x  += m2px(v*c*DT)
        y  += m2px(v*s*DT)
        th  = wrap(th + w_z*DT)

        i0   = min(range(len(WS)), key=lambda i: abs(WS[i]-w))
        iref = max(0, min(len(WS)-1, i0 + step_ref))
        wref = WS[iref]
        w   += ((wref - w)/TAU_H) * DT
        w    = min(max(w, ROBOT_W_MIN), ROBOT_W_MAX)

        st = State(x, y, th, w)
        if (k % CHECK_STEP) == 0:
            if not inside(int(x), int(y)): return None, None, None
            if not valid_configuration(smap, st): return None, None, None

        path_pts.append((int(x), int(y)))
        traj_states.append(st)

    new_state = State(x, y, th, w)
    if not valid_configuration(smap, new_state): return None, None, None
    return new_state, path_pts, traj_states

# --------------------- Métrica de estado ---------------------
def state_distance(a: State, b: State):
    dx, dy = a.x - b.x, a.y - b.y
    dpos = math.hypot(dx, dy)
    dth  = ang_dist(a.th, b.th)
    dw   = abs(a.w - b.w)
    return dpos + m2px(0.5)*dth + m2px(0.2)*dw

# --------------------- UI ---------------------
class Picker:
    def __init__(self, smap: SewerMap):
        self.smap = smap; self.reset()
    def reset(self):
        self.start_pos = None; self.start_th = None
        self.goal_pos  = None; self.goal_th  = None
        self.tmp_mode  = None
    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.start_pos is None:
                self.start_pos = (x, y); self.tmp_mode = 'start'
            elif self.start_th is None and self.tmp_mode == 'start':
                dx, dy = x - self.start_pos[0], y - self.start_pos[1]
                self.start_th = math.atan2(dy, dx); self.tmp_mode = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.goal_pos is None:
                self.goal_pos = (x, y); self.tmp_mode = 'goal'
            elif self.goal_th is None and self.tmp_mode == 'goal':
                dx, dy = x - self.goal_pos[0], y - self.goal_pos[1]
                self.goal_th = math.atan2(dy, dx); self.tmp_mode = None
    def have_start(self): return self.start_pos is not None and self.start_th is not None
    def have_goal(self):  return self.goal_pos  is not None and self.goal_th  is not None

def draw_state(img, state: State, color, thickness=2, smap: "SewerMap" = None):
    poly = robot_polygon(state).reshape((-1,1,2))
    cv2.polylines(img, [poly], True, color, thickness)
    p0 = (int(state.x), int(state.y))
    p1 = (int(state.x + 0.35*math.cos(state.th)*m2px(1.0)),
          int(state.y + 0.35*math.sin(state.th)*m2px(1.0)))
    cv2.arrowedLine(img, p0, p1, color, 2, tipLength=0.25)
    if smap is not None:
        for (_, (cx, cy)) in wheel_centers_pixels(state).items():
            if not smap.inside_bounds(cx, cy): continue
            if smap.wall_mask[cy, cx] != 0: col = (0,0,255)
            elif smap.gutter_mask[cy, cx] != 0: col = (0,0,0)
            else: col = (0,255,0)
            cv2.circle(img, (cx, cy), 3, col, -1)

def draw_instructions(img):
    texts = [
        "Clic izq: start (pos+orient)",
        "Clic der: goal  (pos+orient)",
        "ESPACIO: planificar | r: reset | q: salir",
    ]
    y = 22
    for t in texts:
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        y += 22

# --------------------- RRT ---------------------
def plan_rrt(smap: SewerMap, start: State, goal: State, vis_img):
    rng = random.Random(RNG_SEED)
    nodes = [Node(start, parent=-1, path=[], traj_states=[])]
    base_img = vis_img.copy()
    goal_px  = (int(goal.x), int(goal.y))

    segs = []

    # conmutación de conjuntos de control por rachas (sin sensores frontales/laterales)
    mode = 'NORMAL'      # 'NORMAL' / 'TURN'
    streak_fail = 0
    streak_ok   = 0
    cooldown    = 0
    print(f"[SET] {mode} (inicio)")

    for it in range(MAX_ITERS):
        if cooldown > 0:
            cooldown -= 1

        # --- sampleo del objetivo
        if rng.random() < GOAL_SAMPLING_RATE:
            tx, ty, tth, tw = goal.x, goal.y, goal.th, goal.w
        else:
            tx, ty = smap.sample_free(rng)
            tth = rng.uniform(-math.pi, math.pi)
            tw  = rng.uniform(ROBOT_W_MIN, ROBOT_W_MAX)
        target = State(tx, ty, tth, tw)

        # --- nearest (aprox/exacto periódico)
        if (it % EXACT_EVERY) == 0 or len(nodes) <= K_SUBSET:
            dists = [state_distance(n.state, target) for n in nodes]
            idx_near = int(np.argmin(dists))
        else:
            idxs = rng.sample(range(len(nodes)), k=min(K_SUBSET, len(nodes)))
            best, bestd = None, 1e18
            for idx in idxs:
                d = state_distance(nodes[idx].state, target)
                if d < bestd: bestd, best = d, idx
            idx_near = best

        near = nodes[idx_near].state

        # --- selección del conjunto de controles según modo actual
        controls = CONTROL_SET if mode == 'NORMAL' else CONTROL_SET_TURN

        # --- intentar propagar con los controles activos
        best_state = None; best_path = None; best_traj = None; best_score = float('inf')
        for u in controls:
            ns, path_pts, traj_states = propagate(near, u, smap)
            if ns is None: 
                continue
            d = state_distance(ns, target)
            if d < best_score:
                best_score, best_state = d, ns
                best_path, best_traj = path_pts, traj_states

        # --- actualizar rachas y conmutación
        if best_state is None:
            # no se añadió nodo
            streak_fail += 1
            streak_ok = 0
            if mode == 'NORMAL' and cooldown == 0 and streak_fail >= ENTER_TURN:
                mode = 'TURN'; cooldown = MODE_FREEZE
                print(f"[SET] NORMAL -> TURN   @ iter={it} (fallos={streak_fail})")
            # pintar amortiguado
            if (it+1) % DRAW_EVERY == 0 and segs:
                for (a,b) in segs: cv2.line(base_img, a, b, (120,180,255), 1)
                segs.clear()
                tmp = base_img.copy()
                draw_state(tmp, start, (0,200,0), 2, smap)
                draw_state(tmp, goal,  (255,0,0), 2, smap)
                cv2.circle(tmp, goal_px, 5, (255,0,0), -1)
                cv2.imshow("RRT kinodinamico", tmp); cv2.waitKey(1)
            continue
        else:
            # se añadió nodo válido
            nodes.append(Node(best_state, parent=idx_near, path=best_path, traj_states=best_traj))
            streak_ok += 1
            streak_fail = max(0, streak_fail - 1)
            if mode == 'TURN' and cooldown == 0 and streak_ok >= EXIT_STREAK:
                mode = 'NORMAL'; cooldown = MODE_FREEZE
                print(f"[SET] TURN   -> NORMAL @ iter={it} (exitos={streak_ok})")

        # dibujo amortiguado del árbol
        if best_path is not None and len(best_path) > 1:
            for i in range(1, len(best_path)):
                segs.append((best_path[i-1], best_path[i]))
        if (it+1) % DRAW_EVERY == 0 and segs:
            for (a,b) in segs: cv2.line(base_img, a, b, (120,180,255), 1)
            segs.clear()
            tmp = base_img.copy()
            draw_state(tmp, start, (0,200,0), 2, smap)
            draw_state(tmp, goal,  (255,0,0), 2, smap)
            cv2.circle(tmp, goal_px, 5, (255,0,0), -1)
            cv2.imshow("RRT kinodinamico", tmp); cv2.waitKey(1)

        # ¿hemos llegado?
        if (math.hypot(best_state.x - goal.x, best_state.y - goal.y) < NEAR_GOAL_DIST and
            ang_dist(best_state.th, goal.th) < NEAR_GOAL_DTH and
            abs(best_state.w - goal.w) < NEAR_GOAL_DW):

            # reconstrucción de trayectoria
            idx = len(nodes) - 1
            sol_points, sol_states = [], []
            while idx != -1:
                n = nodes[idx]
                if n.path:        sol_points.extend(n.path[::-1])
                if n.traj_states: sol_states.extend(n.traj_states[::-1])
                sol_states.append(n.state)
                idx = n.parent
            sol_points = sol_points[::-1]
            sol_states = sol_states[::-1]

            # dibujo final
            out = base_img.copy()
            if sol_points and len(sol_points) > 1:
                for i in range(1, len(sol_points)):
                    cv2.line(out, sol_points[i-1], sol_points[i], (0,255,0), 2)

            if len(sol_states) >= 7:
                idxs = np.linspace(1, len(sol_states)-2, 5, dtype=int)
                for j in idxs:
                    st = sol_states[j]
                    draw_state(out, st, (180,90,10), 2, smap)
                    supports = support_points_pixels(smap, st)
                    if len(supports) >= 3:
                        hull = cv2.convexHull(np.array(supports, dtype=np.int32))
                        cv2.polylines(out, [hull], True, (0,255,0), 1)
                        cx, cy = cog_pixel_from_table(st)
                        inside = cv2.pointPolygonTest(hull.astype(np.float32), (float(cx), float(cy)), False) >= 0
                    else:
                        cx, cy = cog_pixel_from_table(st); inside = False
                    cv2.circle(out, (cx,cy), 3, (0,255,0) if inside else (0,0,255), -1)

            draw_state(out, start, (0,200,0), 2, smap)
            draw_state(out, goal,  (255,0,0), 2, smap)
            cv2.circle(out, goal_px, 5, (255,0,0), -1)

            supports = support_points_pixels(smap, goal)
            if len(supports) >= 3:
                hull = cv2.convexHull(np.array(supports, dtype=np.int32))
                cv2.polylines(out, [hull], True, (0,255,0), 2)
                cx, cy = cog_pixel_from_table(goal)
                inside = cv2.pointPolygonTest(hull.astype(np.float32), (float(cx), float(cy)), False) >= 0
            else:
                cx, cy = cog_pixel_from_table(goal); inside = False
            cv2.circle(out, (cx,cy), 4, (0,255,0) if inside else (0,0,255), -1)

            return out, sol_points, nodes, it+1

    # sin plan
    out = base_img.copy()
    draw_state(out, start, (0,200,0), 2, smap)
    draw_state(out, goal,  (0,0,255), 2, smap)
    cv2.putText(out, "No se encontró trayectoria", (10, smap.h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
    return out, None, nodes, MAX_ITERS

# --------------------- main ---------------------
def main():
    smap = SewerMap(MAP_PATH)
    walls_rgb = smap.draw_overlay()

    picker = Picker(smap)
    cv2.namedWindow("RRT kinodinamico", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("RRT kinodinamico", picker.mouse)

    base = walls_rgb.copy()
    draw_instructions(base)
    cur_img = base.copy()
    solution_img = None

    while True:
        cur_img = base.copy()

        if picker.start_pos is not None:
            cv2.circle(cur_img, picker.start_pos, 4, (0,255,0), -1)
            if picker.start_th is not None:
                s = State(picker.start_pos[0], picker.start_pos[1], picker.start_th, ROBOT_W0)
                draw_state(cur_img, s, (0,255,0), 2, smap)

        if picker.goal_pos is not None:
            cv2.circle(cur_img, picker.goal_pos, 4, (255,0,0), -1)
            if picker.goal_th is not None:
                g = State(picker.goal_pos[0], picker.goal_pos[1], picker.goal_th, ROBOT_W0)
                draw_state(cur_img, g, (255,0,0), 2, smap)

        if solution_img is not None:
            cv2.addWeighted(solution_img, 0.7, cur_img, 0.3, 0, cur_img)

        draw_instructions(cur_img)
        cv2.imshow("RRT kinodinamico", cur_img)
        k = cv2.waitKey(20) & 0xFF

        if k in (27, ord('q')):
            break

        if k == ord('r'):
            picker.reset(); solution_img = None
            base = walls_rgb.copy(); draw_instructions(base)

        if k == 32:  # SPACE
            if not picker.have_start() or not picker.have_goal():
                base2 = base.copy()
                cv2.putText(base2, "Selecciona START y GOAL (pos+orient)", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT kinodinamico", base2)
                cv2.waitKey(900)
                continue

            start_state = State(picker.start_pos[0], picker.start_pos[1], picker.start_th, ROBOT_W0)
            goal_state  = State(picker.goal_pos[0],  picker.goal_pos[1],  picker.goal_th,  ROBOT_W0)

            if not valid_configuration(smap, start_state):
                tmp = base.copy()
                draw_state(tmp, start_state, (0,255,255), 2, smap)
                cv2.putText(tmp, "Start no válido", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT kinodinamico", tmp); cv2.waitKey(900)
                continue

            if not valid_configuration(smap, goal_state):
                tmp = base.copy()
                draw_state(tmp, goal_state, (0,255,255), 2, smap)
                cv2.putText(tmp, "Goal no válido", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT kinodinamico", tmp); cv2.waitKey(900)
                continue

            plan_img, sol_points, nodes, iters = plan_rrt(smap, start_state, goal_state, walls_rgb)
            solution_img = plan_img

            msg = f"Iteraciones: {iters}"
            if sol_points is None: msg += " | sin solución"
            else: msg += f" | nodos: {len(nodes)}"
            out = plan_img.copy()
            cv2.putText(out, msg, (10, smap.h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("RRT kinodinamico", out)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
