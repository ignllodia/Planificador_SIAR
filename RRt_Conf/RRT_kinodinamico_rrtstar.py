# -*- coding: utf-8 -*-
"""
RRT* kinodinámico guiado por A* (corredor) para SIAR.
Basado en RRT_kinodinamico_astar.py pero con rewiring (RRT*):
- Coste acumulado por nodo (longitud de trayectorias en px)
- Selección de padre óptimo en vecindario
- Rewire de vecinos si la nueva ruta reduce su coste
"""

import cv2
import numpy as np
import math
import random
import heapq
from dataclasses import dataclass

# --------------------- Mapa ---------------------
MAP_PATH = "Pb4.png"

# --------------------- Robot (aprox. SIAR) ---------------------
ROBOT_LEN   = 0.88
ROBOT_W_MIN = 0.52
ROBOT_W_MAX = 0.85
ROBOT_W0    = 0.70
W_DOT_MAX   = 0.20

# --------------------- Escala ---------------------
PIXELS_PER_M = 75
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
MAX_ITERS = 50000
GOAL_SAMPLING_RATE = 0.20
DT = 0.05
TPROP = 0.5
CHECK_STEP = 2
NEAR_GOAL_DIST = m2px(0.30)
NEAR_GOAL_DTH  = math.radians(20.0)
NEAR_GOAL_DW   = 0.10
TAU_H = 0.6

# Nearest
K_SUBSET    = 64
EXACT_EVERY = 20
RNG_SEED    = 12345

# Dibujo
DRAW_EVERY = 100
N_FRAMES = 20
N_WAYPOINTS = 10

# Sesgo a waypoints y corredor
WAYPOINT_BIAS = 0.5
CORRIDOR_WIDTH_M = 0.6
STAGNATION_ITERS = 300
STAGNATION_DELTA = 1.0
CENTER_BIAS_WEIGHT = 0.05

# Debug
DEBUG = True
DEBUG_EVERY = 200

# Histeresis y rewiring
ENTER_TURN        = 200
EXIT_STREAK       = 500
MODE_FREEZE       = 1000
NEIGHBOR_RADIUS_M = 0.3  # vecindario para rewiring (m) 
HEADING_COST_WEIGHT = m2px(0.3)  # penaliza cambios de heading en aristas

# --------------------- Control sets ---------------------
CONTROL_SET = [
    (0.25,  0.0,   0),
    (0.25,  0.2,   0),
    (0.25, -0.2,   0),
    (0.35,  0.0,   0),
]
CONTROL_SET_TURN = [
    (0.00,  0.0,  -1),
    (0.00,  0.0,  +1),
    (0.10,  0.6,   0),
    (0.10, -0.6,   0),
    (0.10,  0.3,   0),
    (0.10, -0.3,   0),
    (0.12,  0.8,   0),
    (0.12, -0.8,   0),
]

# --------------------- Utilidades ---------------------
def wrap_angle(a):
    while a > math.pi:  a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def ang_dist(a, b): return abs(wrap_angle(a - b))

@dataclass
class State:
    x: float
    y: float
    th: float
    w: float

@dataclass
class Node:
    state: State
    parent: int
    path: list
    traj_states: list
    cost: float

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
            raise RuntimeError("No hay pixeles libres en el mapa.")
        i = rng.randrange(0, len(self.free_points))
        x, y = self.free_points[i]
        return int(x), int(y)

    def draw_overlay(self):
        vis = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        vis[self.free_mask > 0]   = (200,200,200)
        vis[self.wall_mask > 0]   = (0,0,0)
        vis[self.gutter_mask > 0] = (0,0,255)
        return vis

# --------------------- Geometría ---------------------
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
    px_step = v * DT * PIXELS_PER_M

    i0   = min(range(len(WS)), key=lambda i: abs(WS[i]-w))
    iref = max(0, min(len(WS)-1, i0 + step_ref))
    wref = WS[iref]

    for k in range(steps):
        c, s = cos(th), sin(th)
        x  += px_step * c
        y  += px_step * s
        th  = wrap(th + w_z*DT)

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

# --------------------- Métrica ---------------------
def state_distance(a: State, b: State):
    dx, dy = a.x - b.x, a.y - b.y
    dpos = math.hypot(dx, dy)
    dth  = ang_dist(a.th, b.th)
    dw   = abs(a.w - b.w)
    return dpos + m2px(0.5)*dth + m2px(0.2)*dw

# --------------------- A* gutter centrado ---------------------
def astar_path_gutter(smap: "SewerMap", start_xy, goal_xy):
    sx, sy = start_xy; gx, gy = goal_xy
    h, w = smap.h, smap.w
    gutter = smap.gutter_mask

    gutter_bin = (gutter > 0).astype(np.uint8)
    dist_field = cv2.distanceTransform(gutter_bin, cv2.DIST_L2, 3)
    maxd = dist_field.max() if dist_field.size > 0 else 0.0
    def cell_cost(x, y):
        if maxd <= 0: return 1.0
        return 1.0 + CENTER_BIAS_WEIGHT * (maxd - dist_field[y, x])

    def inside(x, y): return 0 <= x < w and 0 <= y < h
    def neigh(x, y):
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x+dx, y+dy
            if inside(nx, ny) and gutter[ny, nx] != 0:
                yield nx, ny

    start = (int(sx), int(sy)); goal = (int(gx), int(gy))
    if not inside(*start) or not inside(*goal): return None
    if gutter[start[1], start[0]] == 0 or gutter[goal[1], goal[0]] == 0: return None

    openh = []
    heapq.heappush(openh, (abs(sx-gx)+abs(sy-gy), 0, start))
    came = {start: None}
    gscore = {start: 0}

    while openh:
        _, gc, cur = heapq.heappop(openh)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur); cur = came[cur]
            return path[::-1]
        for n in neigh(*cur):
            ng = gc + cell_cost(n[0], n[1])
            if ng < gscore.get(n, 1e18):
                gscore[n] = ng
                came[n] = cur
                f = ng + abs(n[0]-gx) + abs(n[1]-gy)
                heapq.heappush(openh, (f, ng, n))
    return None

def resample_points(path_px, n_points=30):
    if not path_px or len(path_px) < 2: return []
    pts = np.array(path_px, dtype=np.float32)
    segs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    dist = np.concatenate(([0.0], np.cumsum(segs)))
    total = dist[-1]
    if total < 1e-6: return [tuple(map(int, pts[0]))]*n_points
    targets = np.linspace(0, total, n_points)
    res = []
    for t in targets:
        i = np.searchsorted(dist, t) - 1
        i = max(0, min(len(segs)-1, i))
        alpha = (t - dist[i]) / max(segs[i], 1e-6)
        p = pts[i] + alpha * (pts[i+1] - pts[i])
        res.append((float(p[0]), float(p[1])))
    return res

def build_waypoint_states(smap: "SewerMap", path_px, w_default=ROBOT_W0):
    waypoints = []
    resampled = resample_points(path_px, n_points=N_WAYPOINTS)
    for i, (x, y) in enumerate(resampled):
        if i < len(resampled)-1:
            dx = resampled[i+1][0] - x
            dy = resampled[i+1][1] - y
        else:
            dx = resampled[i][0] - resampled[i-1][0]
            dy = resampled[i][1] - resampled[i-1][1]
        th = math.atan2(dy, dx + 1e-6)
        st = State(x, y, th, w_default)
        if valid_configuration(smap, st):
            waypoints.append(st)
    return waypoints

def point_polyline_dist(pt, poly):
    if poly is None or len(poly) < 2:
        return 1e18
    px, py = float(pt[0]), float(pt[1])
    best = 1e18
    for i in range(len(poly)-1):
        ax, ay = poly[i]; bx, by = poly[i+1]
        vx, vy = bx-ax, by-ay
        wx, wy = px-ax, py-ay
        denom = vx*vx + vy*vy
        t = 0.0 if denom == 0 else max(0.0, min(1.0, (vx*wx + vy*wy)/denom))
        projx = ax + t*vx; projy = ay + t*vy
        d = math.hypot(px - projx, py - projy)
        if d < best: best = d
    return best

def minimize_width_changes(sol_states, smap: "SewerMap"):
    """Intenta mantener el ancho constante: cada estado adopta el ancho del anterior si sigue siendo válido."""
    if not sol_states: return sol_states
    adjusted = [sol_states[0]]
    prev_w = sol_states[0].w
    for st in sol_states[1:]:
        cand = State(st.x, st.y, st.th, prev_w)
        if valid_configuration(smap, cand):
            adjusted.append(cand)
            prev_w = prev_w
        else:
            adjusted.append(st)
            prev_w = st.w
    return adjusted

# --------------------- Picker/UI ---------------------
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
    cv2.putText(img, f"w={state.w:.3f} m", (int(state.x)+10, int(state.y)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

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

# --------------------- RRT* ---------------------
def path_cost(path_pts):
    if not path_pts or len(path_pts) < 2: return 0.0
    d = 0.0
    for i in range(1, len(path_pts)):
        a, b = path_pts[i-1], path_pts[i]
        d += math.hypot(a[0]-b[0], a[1]-b[1])
    return d

def edge_cost(parent_state: State, child_state: State, path_pts):
    base = path_cost(path_pts)
    dheading = ang_dist(parent_state.th, child_state.th)
    return base + HEADING_COST_WEIGHT * dheading

def neighborhood(nodes, new_state, radius_px):
    res = []
    for idx, n in enumerate(nodes):
        if state_distance(n.state, new_state) <= radius_px:
            res.append(idx)
    return res

def plan_rrt_star(smap: SewerMap, start: State, goal: State, vis_img, waypoints=None, corridor_poly=None):
    rng = random.Random(RNG_SEED)
    nodes = [Node(start, parent=-1, path=[], traj_states=[], cost=0.0)]
    base_img = vis_img.copy()
    goal_px  = (int(goal.x), int(goal.y))

    wp_idx = 0
    wp_tolerance = NEAR_GOAL_DIST

    segs = []

    mode = 'NORMAL'
    streak_fail = 0
    streak_ok   = 0
    cooldown    = 0
    stagnation  = 0
    last_best   = None
    neighbor_radius_px = m2px(NEIGHBOR_RADIUS_M)
    print(f"[SET] {mode} (inicio)")

    for it in range(MAX_ITERS):
        if cooldown > 0:
            cooldown -= 1
        valid_controls = 0

        use_wp = waypoints is not None and wp_idx < len(waypoints)
        cur_wp = waypoints[wp_idx] if use_wp else None
        if use_wp and rng.random() < WAYPOINT_BIAS:
            tx, ty, tth, tw = cur_wp.x, cur_wp.y, cur_wp.th, cur_wp.w
        elif rng.random() < GOAL_SAMPLING_RATE:
            tx, ty, tth, tw = goal.x, goal.y, goal.th, goal.w
        else:
            corridor_px = m2px(CORRIDOR_WIDTH_M)
            attempts = 0
            while True:
                tx, ty = smap.sample_free(rng)
                if corridor_poly is None:
                    break
                if point_polyline_dist((tx, ty), corridor_poly) <= corridor_px:
                    break
                attempts += 1
                if attempts >= 30:
                    break
            tth = rng.uniform(-math.pi, math.pi)
            tw  = rng.uniform(ROBOT_W_MIN, ROBOT_W_MAX)
        target = State(tx, ty, tth, tw)

        if (it % EXACT_EVERY) == 0 or len(nodes) <= K_SUBSET:
            dists = [state_distance(n.state, target) for n in nodes]
            idx_near = int(np.argmin(dists))
        else:
            idxs = rng.sample(range(len(nodes)), k=min(K_SUBSET, len(nodes)))
            best, bestd = None, float('inf')
            for idx in idxs:
                d = state_distance(nodes[idx].state, target)
                if d < bestd: bestd, best = d, idx
            idx_near = best if best is not None else idxs[0]

        near = nodes[idx_near].state
        if near is None:
            if DEBUG: print(f"[DBG it={it}] nodo nearest sin estado idx={idx_near}, se salta iter")
            continue

        controls = CONTROL_SET if mode == 'NORMAL' else CONTROL_SET_TURN

        best_state = None; best_path = None; best_traj = None; best_score = float('inf')
        for u in controls:
            ns, path_pts, traj_states = propagate(near, u, smap)
            if ns is None:
                continue
            valid_controls += 1
            d = state_distance(ns, target)
            if d < best_score:
                best_score, best_state = d, ns
                best_path, best_traj = path_pts, traj_states

        if best_state is not None:
            if last_best is None or (last_best - best_score) > STAGNATION_DELTA:
                last_best = best_score
                stagnation = 0
            else:
                stagnation += 1
                if mode == 'NORMAL' and cooldown == 0 and stagnation >= STAGNATION_ITERS:
                    mode = 'TURN'; cooldown = MODE_FREEZE
                    stagnation = 0
                    print(f"[SET] NORMAL -> TURN   @ iter={it} (estancamiento)")

        if DEBUG and (it % DEBUG_EVERY) == 0:
            wp_str = f"wp={wp_idx}/{len(waypoints)}" if cur_wp is not None else "wp=None"
            if best_state is None:
                print(f"[DBG it={it}] mode={mode} nearest={idx_near} valids={valid_controls} {wp_str}")
            else:
                print(f"[DBG it={it}] mode={mode} nearest={idx_near} valids={valid_controls} d_target={best_score:.1f} pos=({best_state.x:.1f},{best_state.y:.1f}) {wp_str}")

        if best_state is None:
            streak_fail += 1
            streak_ok = 0
            if mode == 'NORMAL' and cooldown == 0 and streak_fail >= ENTER_TURN:
                mode = 'TURN'; cooldown = MODE_FREEZE
                print(f"[SET] NORMAL -> TURN   @ iter={it} (fallos={streak_fail})")
        else:
            # RRT*: elegir padre óptimo
            edge_cost_near = edge_cost(nodes[idx_near].state, best_state, best_path)
            new_cost = nodes[idx_near].cost + edge_cost_near
            best_parent = idx_near
            neighbors = neighborhood(nodes, best_state, neighbor_radius_px)
            for nidx in neighbors:
                cand_edge = edge_cost(nodes[nidx].state, best_state, best_path)
                cand_cost = nodes[nidx].cost + cand_edge
                if cand_cost < new_cost:
                    best_parent = nidx
                    new_cost = cand_cost

            node_new = Node(best_state, parent=best_parent, path=best_path, traj_states=best_traj, cost=new_cost)
            nodes.append(node_new)
            new_idx = len(nodes) - 1

            # Rewire vecinos
            for nidx in neighbors:
                if nidx == best_parent:
                    continue
                n = nodes[nidx]
                alt_cost = node_new.cost + edge_cost(node_new.state, n.state, n.path if n.path else best_path)
                if alt_cost + 1e-6 < n.cost:
                    nodes[nidx].parent = new_idx
                    nodes[nidx].cost = alt_cost

            streak_ok += 1
            streak_fail = max(0, streak_fail - 1)
            if mode == 'TURN' and cooldown == 0 and streak_ok >= EXIT_STREAK:
                mode = 'NORMAL'; cooldown = MODE_FREEZE
                print(f"[SET] TURN   -> NORMAL @ iter={it} (exitos={streak_ok})")

            if use_wp and wp_idx < len(waypoints):
                if math.hypot(best_state.x - waypoints[wp_idx].x, best_state.y - waypoints[wp_idx].y) < wp_tolerance:
                    wp_idx += 1

        if best_path is not None and len(best_path) > 1:
            for i in range(1, len(best_path)):
                segs.append((best_path[i-1], best_path[i]))
        if (it+1) % DRAW_EVERY == 0 and segs:
            for (a,b) in segs: cv2.line(base_img, a, b, (120,180,255), 2)
            segs.clear()
            tmp = base_img.copy()
            draw_state(tmp, start, (0,200,0), 2, smap)
            draw_state(tmp, goal,  (255,0,0), 2, smap)
            cv2.circle(tmp, goal_px, 5, (255,0,0), -1)
            if cur_wp is not None:
                cv2.circle(tmp, (int(cur_wp.x), int(cur_wp.y)), 5, (0,255,0), 2)
            cv2.imshow("RRT* kinodinamico", tmp); cv2.waitKey(1)

        if best_state is not None and (
            math.hypot(best_state.x - goal.x, best_state.y - goal.y) < NEAR_GOAL_DIST and
            ang_dist(best_state.th, goal.th) < NEAR_GOAL_DTH and
            abs(best_state.w - goal.w) < NEAR_GOAL_DW):

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
            sol_states = minimize_width_changes(sol_states, smap)
            # reporta si hubo cambios de ancho tras el ajuste
            for i in range(1, len(sol_states)):
                if abs(sol_states[i].w - sol_states[i-1].w) > 1e-6:
                    print(f"[MIN_WIDTH] cambio w en idx {i}: {sol_states[i-1].w:.3f} -> {sol_states[i].w:.3f}")

            out = base_img.copy()
            if sol_points and len(sol_points) > 1:
                for i in range(1, len(sol_points)):
                    cv2.line(out, sol_points[i-1], sol_points[i], (0,255,0), 2)

            if sol_states and N_FRAMES > 0:
                idxs = np.linspace(0, len(sol_states)-1, N_FRAMES, dtype=int)
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

            return out, sol_points, nodes, len(nodes)

    out = base_img.copy()
    draw_state(out, start, (0,200,0), 2, smap)
    draw_state(out, goal,  (0,0,255), 2, smap)
    cv2.putText(out, "No se encontro la trayectoria", (10, smap.h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
    return out, None, nodes, MAX_ITERS

# --------------------- main ---------------------
def main():
    smap = SewerMap(MAP_PATH)
    walls_rgb = smap.draw_overlay()

    picker = Picker(smap)
    cv2.namedWindow("RRT* kinodinamico", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("RRT* kinodinamico", picker.mouse)

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
        cv2.imshow("RRT* kinodinamico", cur_img)
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
                cv2.imshow("RRT* kinodinamico", base2)
                cv2.waitKey(900)
                continue

            start_state = State(picker.start_pos[0], picker.start_pos[1], picker.start_th, ROBOT_W0)
            goal_state  = State(picker.goal_pos[0],  picker.goal_pos[1],  picker.goal_th,  ROBOT_W0)

            if not valid_configuration(smap, start_state):
                tmp = base.copy()
                draw_state(tmp, start_state, (0,255,255), 2, smap)
                cv2.putText(tmp, "Start no valido", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT* kinodinamico", tmp); cv2.waitKey(900)
                continue

            if not valid_configuration(smap, goal_state):
                tmp = base.copy()
                draw_state(tmp, goal_state, (0,255,255), 2, smap)
                cv2.putText(tmp, "Goal no valido", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT* kinodinamico", tmp); cv2.waitKey(900)
                continue

            astar_path = astar_path_gutter(smap, picker.start_pos, picker.goal_pos)
            if astar_path is None or len(astar_path) < 2:
                tmp = base.copy()
                cv2.putText(tmp, "A* no encontro camino en gutter", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT* kinodinamico", tmp); cv2.waitKey(900)
                continue

            waypoints = build_waypoint_states(smap, astar_path, w_default=ROBOT_W0)
            if len(waypoints) < 2:
                tmp = base.copy()
                cv2.putText(tmp, "Waypoints insuficientes", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT* kinodinamico", tmp); cv2.waitKey(900)
                continue

            base = walls_rgb.copy()
            draw_instructions(base)
            for i in range(1, len(astar_path)):
                cv2.line(base, astar_path[i-1], astar_path[i], (0,120,255), 1)
            for wp in waypoints:
                cv2.circle(base, (int(wp.x), int(wp.y)), 2, (0,100,255), -1)

            plan_img, sol_points, nodes, iters = plan_rrt_star(smap, start_state, goal_state, base, waypoints=waypoints, corridor_poly=[(wp.x, wp.y) for wp in waypoints])
            solution_img = plan_img

            msg = f"Iteraciones: {iters}"
            if sol_points is None: msg += " | sin solucion"
            else: msg += f" | nodos: {len(nodes)}"
            out = plan_img.copy()
            cv2.putText(out, msg, (10, smap.h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("RRT* kinodinamico", out)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
