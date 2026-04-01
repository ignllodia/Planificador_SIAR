# -*- coding: utf-8 -*-
"""
RRT* kinodinÃ¡mico guiado por A* para SIAR.
Basado en rrt_kinodinamico_guiado_astar.py pero con rewiring (RRT*):
"""

import cv2
import numpy as np
import math
import random
import heapq
import statistics
from dataclasses import dataclass
import time
from pathlib import Path

# --------------------- Mapa ---------------------
MAP_PATH = "Pb4.png"          # ruta del mapa en escala de grises

def resolve_map_path(map_path):
    p = Path(map_path)
    candidates = [
        p,
        Path.cwd() / map_path,
        Path(__file__).resolve().parent / map_path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    # Fallback: buscar por nombre en directorios de trabajo habituales.
    search_roots = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path.home() / "OneDrive - UNIVERSIDAD DE SEVILLA",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        try:
            for match in root.rglob(map_path):
                if match.is_file():
                    return str(match)
        except (PermissionError, OSError):
            continue

    searched = "\n - ".join(str(c.resolve()) for c in candidates + search_roots)
    raise FileNotFoundError(
        f"No se encontro el mapa '{map_path}'. Rutas probadas:\n - {searched}"
    )

# --------------------- Robot (aprox. SIAR) ---------------------
ROBOT_LEN   = 0.88            # largo nominal (m)
ROBOT_W_MIN = 0.52            # ancho minimo permitido (m)
ROBOT_W_MAX = 0.85            # ancho maximo permitido (m)
ROBOT_W0    = 0.70            # ancho inicial (m)

# --------------------- Escala ---------------------
PIXELS_PER_M = 75             # pixeles por metro
def m2px(m): return int(round(m * PIXELS_PER_M))
def px2m(px): return float(px) / PIXELS_PER_M

# --------------------- Tabla configuraciÃ³n estable ---------------------
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
MAX_ITERS = 50000             # iteraciones maximas
TREE_TIMEOUT = 10.0           # segundos antes de reiniciar el arbol si no hay solucion
GOAL_SAMPLING_RATE = 0.1     # prob. de muestrear el goal
DT = 0.05                     # paso de integracion (s)
TPROP = 0.5                   # horizonte de propagacion (s)
CHECK_STEP = 2                # validar cada N pasos
NEAR_GOAL_DIST = m2px(0.30)   # tolerancia posicion goal (px)
NEAR_GOAL_DTH  = math.radians(20.0)  # tolerancia orientacion goal
NEAR_GOAL_DW   = 0.10         # tolerancia ancho goal (m)
TAU_H = 0.6                   # constante de tiempo del filtro de ancho

# Nearest
K_SUBSET    = 64              # vecinos muestreados en nearest aproximado
EXACT_EVERY = 20              # frecuencia de nearest exacto
RNG_SEED_BASE = 12345         # semilla base; se deriva por corrida e intento
REP_SIM = 1               # numero de simulaciones validas a ejecutar

# Dibujo
DRAW_EVERY = 100             # refresco de dibujo del arbol
N_FRAMES = 6                 # fotogramas de la solucion a dibujar
N_WAYPOINTS = 10              # waypoints generados desde A*

# Sesgo a waypoints y corredor
WAYPOINT_BIAS = 0.5           # prob. de muestrear waypoint activo
CORRIDOR_WIDTH_M = 0.6        # ancho de banda para muestreo libre (m)
CENTER_BIAS_WEIGHT = 0.05     # penaliza pegarse a pared en A*

# Debug
DEBUG = True                  # habilita logs debug
DEBUG_EVERY = 200             # frecuencia de logs debug

# RRT*
NEIGHBOR_RADIUS_M = 0.3       # radio de vecinos para rewiring (m)
HEADING_COST_WEIGHT = m2px(0.3)  # penalizacion por cambio de heading en coste
REWIRE_POS_TOL = m2px(0.20)   # tolerancia posicion para rewire (px)
REWIRE_TH_TOL = math.radians(20.0)  # tolerancia heading para rewire
REWIRE_W_TOL = 0.08           # tolerancia de ancho para rewire (m)

# --------------------- Control set ---------------------
CONTROL_SET_ALL = [
    (0.25,  0.0,   0),
    (0.25,  0.2,   0),
    (0.25, -0.2,   0),
    (0.35,  0.0,   0),
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
        self.traversable_mask = cv2.bitwise_or(self.free_mask, self.gutter_mask)
        self.has_gutter = cv2.countNonZero(self.gutter_mask) > 0
        non_wall = (self.wall_mask == 0).astype(np.uint8)
        self.wall_dist_px = cv2.distanceTransform(non_wall, cv2.DIST_L2, 3)

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

# --------------------- GeometrÃ­a ---------------------
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

# --------------------- ValidaciÃ³n ---------------------
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

# --------------------- DinÃ¡mica ---------------------
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

# --------------------- MÃ©trica ---------------------
def state_distance(a: State, b: State):
    dx, dy = a.x - b.x, a.y - b.y
    dpos = math.hypot(dx, dy)
    dth  = ang_dist(a.th, b.th)
    dw   = abs(a.w - b.w)
    return dpos + m2px(0.5)*dth + m2px(0.2)*dw

# --------------------- A* guiado ---------------------
def astar_path_guided(smap: "SewerMap", start_xy, goal_xy):
    sx, sy = start_xy; gx, gy = goal_xy
    h, w = smap.h, smap.w
    start = (int(sx), int(sy))
    goal = (int(gx), int(gy))

    def inside(x, y): return 0 <= x < w and 0 <= y < h
    if not inside(*start) or not inside(*goal):
        return None, None

    use_gutter = (
        smap.has_gutter and
        smap.gutter_mask[start[1], start[0]] != 0 and
        smap.gutter_mask[goal[1], goal[0]] != 0
    )
    nav_mask = smap.gutter_mask if use_gutter else smap.traversable_mask
    nav_label = "gutter" if use_gutter else "abierto"

    nav_bin = (nav_mask > 0).astype(np.uint8)
    dist_field = cv2.distanceTransform(nav_bin, cv2.DIST_L2, 3)
    maxd = dist_field.max() if dist_field.size > 0 else 0.0
    def cell_cost(x, y):
        if maxd <= 0: return 1.0
        return 1.0 + CENTER_BIAS_WEIGHT * (maxd - dist_field[y, x])

    def neigh(x, y):
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x+dx, y+dy
            if inside(nx, ny) and nav_mask[ny, nx] != 0:
                yield nx, ny

    if nav_mask[start[1], start[0]] == 0 or nav_mask[goal[1], goal[0]] == 0:
        return None, nav_label

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
            return path[::-1], nav_label
        for n in neigh(*cur):
            ng = gc + cell_cost(n[0], n[1])
            if ng < gscore.get(n, 1e18):
                gscore[n] = ng
                came[n] = cur
                f = ng + abs(n[0]-gx) + abs(n[1]-gy)
                heapq.heappush(openh, (f, ng, n))
    return None, nav_label

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

def width_change_stats(sol_states):
    if sol_states is None or len(sol_states) == 0:
        return None
    n_changes = 0
    ws = [st.w for st in sol_states]
    for i in range(1, len(ws)):
        if abs(ws[i] - ws[i-1]) > 1e-6:
            n_changes += 1
    return n_changes

def reference_deviation_mean_m(path_pts, ref_poly):
    if path_pts is None or len(path_pts) == 0 or ref_poly is None or len(ref_poly) < 2:
        return None
    devs_px = [point_polyline_dist(pt, ref_poly) for pt in path_pts]
    if not devs_px:
        return None
    return px2m(sum(devs_px) / len(devs_px))

def compute_run_metrics(sol_points, sol_states, elapsed, iterations,
                        n_propagations, n_invalid, node_count, ref_poly=None, seed=None, success=False,
                        timed_out=False, timeout_retries=0, attempts=1):
    fail_pct = 100.0 * (n_invalid / max(1, n_propagations))
    metrics = {
        "seed": seed,
        "success": success,
        "timed_out": timed_out,
        "elapsed_s": elapsed,
        "iters": iterations,
        "propagations": n_propagations,
        "_invalid_total": n_invalid,
        "invalid_pct": fail_pct,
        "node_count": node_count,
        "timeout_retries": timeout_retries,
        "attempts": attempts,
        "path_length_m": None,
        "width_change_count": None,
        "mean_ref_dev_m": None,
    }
    if not success:
        return metrics

    length_m = px2m(path_cost(sol_points))
    width_changes = width_change_stats(sol_states)
    mean_ref_dev_m = reference_deviation_mean_m(sol_points, ref_poly)

    metrics.update({
        "path_length_m": length_m,
        "width_change_count": width_changes,
        "mean_ref_dev_m": mean_ref_dev_m,
    })
    return metrics

def mean_std(values):
    if not values:
        return None, None
    mu = statistics.mean(values)
    sigma = statistics.stdev(values) if len(values) >= 2 else 0.0
    return mu, sigma

def fmt_metric(value, fmt_spec):
    if value is None:
        return "NA"
    return format(value, fmt_spec)

def print_timeout_retry(run_idx, total_runs, attempt_idx, max_restarts=None):
    msg = f"[TIMEOUT run={run_idx+1}/{total_runs}] intento={attempt_idx}"
    if max_restarts is not None:
        msg += f" | reinicios={attempt_idx}/{max_restarts}"
    print(msg)

def print_run_metrics(metrics, run_idx=None, total_runs=None):
    prefix = "[METRICS]"
    if run_idx is not None and total_runs is not None:
        prefix = f"[METRICS run={run_idx+1}/{total_runs}]"

    if not metrics["success"]:
        status = "TIMEOUT" if metrics["timed_out"] else "NO_SOLUTION"
        print(f"{prefix} seed={metrics['seed']} | elapsed={metrics['elapsed_s']:.3f}s | "
              f"iters={metrics['iters']} | {status} | prop={metrics['propagations']} | "
              f"invalid_pct={metrics['invalid_pct']:.1f}% | "
              f"nodes={metrics['node_count']} | attempts={metrics['attempts']} | "
              f"timeout_retries={metrics['timeout_retries']}")
        return

    print(
        f"{prefix} seed={metrics['seed']} | elapsed={metrics['elapsed_s']:.3f}s | "
        f"iters={metrics['iters']} | length={fmt_metric(metrics['path_length_m'], '.2f')}m | "
        f"prop={metrics['propagations']} | invalid_pct={metrics['invalid_pct']:.1f}% | "
        f"nodes={metrics['node_count']} | attempts={metrics['attempts']} | "
        f"timeout_retries={metrics['timeout_retries']} | "
        f"dref_mean={fmt_metric(metrics['mean_ref_dev_m'], '.3f')}m | "
        f"w_changes={fmt_metric(metrics['width_change_count'], '.0f')}"
    )

def print_experiment_summary(all_metrics):
    total = len(all_metrics)
    successes = [m for m in all_metrics if m["success"]]
    n_success = len(successes)
    success_rate = 100.0 * n_success / max(1, total)
    total_timeout_retries = sum(m["timeout_retries"] for m in all_metrics)
    print(f"[SUMMARY] runs={total} | success={n_success}/{total} ({success_rate:.1f}%) | "
          f"timeout_retries={total_timeout_retries}")

    for key, label, unit, success_only in [
        ("elapsed_s", "elapsed", "s", False),
        ("iters", "iters", "", False),
        ("invalid_pct", "invalid_pct", "%", False),
        ("node_count", "nodes", "", False),
        ("attempts", "attempts", "", False),
        ("timeout_retries", "timeout_retries", "", False),
        ("path_length_m", "length", "m", True),
        ("mean_ref_dev_m", "dref_mean", "m", True),
        ("width_change_count", "w_changes", "", True),
    ]:
        src = successes if success_only else all_metrics
        values = [m[key] for m in src if m[key] is not None]
        mu, sigma = mean_std(values)
        if mu is None:
            continue
        if unit:
            print(f"[SUMMARY] {label}={mu:.3f} +- {sigma:.3f} {unit}")
        else:
            print(f"[SUMMARY] {label}={mu:.3f} +- {sigma:.3f}")

def minimize_width_changes(sol_states, smap: "SewerMap"):
    """Intenta mantener el ancho constante: cada estado adopta el ancho del anterior si sigue siendo vÃ¡lido."""
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

def draw_state(img, state: State, color, thickness=2, smap: "SewerMap" = None, show_details=True):
    """Dibuja el polÃ­gono del robot y su orientaciÃ³n."""
    poly = robot_polygon(state).reshape((-1,1,2))
    cv2.polylines(img, [poly], True, color, thickness)
    p0 = (int(state.x), int(state.y))
    p1 = (int(state.x + 0.35*math.cos(state.th)*m2px(1.0)),
          int(state.y + 0.35*math.sin(state.th)*m2px(1.0)))
    cv2.arrowedLine(img, p0, p1, color, 2, tipLength=0.25)

    if not show_details:
        return

    if smap is not None:
        for (_, (cx, cy)) in wheel_centers_pixels(state).items():
            if not smap.inside_bounds(cx, cy):
                continue
            if smap.wall_mask[cy, cx] != 0:
                col = (0,0,255)
            elif smap.gutter_mask[cy, cx] != 0:
                col = (0,0,0)
            else:
                col = (0,255,0)
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

def can_connect_states(a: State, b: State):
    return (
        math.hypot(a.x - b.x, a.y - b.y) <= REWIRE_POS_TOL and
        ang_dist(a.th, b.th) <= REWIRE_TH_TOL and
        abs(a.w - b.w) <= REWIRE_W_TOL
    )

def propagate_subtree_cost_delta(nodes, children, root_idx, delta):
    if abs(delta) <= 1e-9:
        return
    stack = list(children[root_idx])
    while stack:
        idx = stack.pop()
        nodes[idx].cost += delta
        stack.extend(children[idx])

def plan_rrt_star(smap: SewerMap, start: State, goal: State, vis_img, waypoints=None, corridor_poly=None,
                  ref_poly=None, seed=None):
    rng = random.Random(RNG_SEED_BASE if seed is None else seed)
    nodes = [Node(start, parent=-1, path=[], traj_states=[], cost=0.0)]
    children = [[]]
    base_img = vis_img.copy()
    goal_px = (int(goal.x), int(goal.y))

    wp_idx = 0
    wp_tolerance = NEAR_GOAL_DIST
    segs = []

    neighbor_radius_px = m2px(NEIGHBOR_RADIUS_M)
    t0 = time.perf_counter()
    n_propagations = 0
    n_invalid = 0
    controls = CONTROL_SET_ALL

    def best_control_towards(parent_state: State, target_state: State):
        nonlocal n_propagations, n_invalid
        best_state = None
        best_path = None
        best_traj = None
        best_score = float('inf')
        valid_count = 0

        for u in controls:
            n_propagations += 1
            ns, path_pts, traj_states = propagate(parent_state, u, smap)
            if ns is None:
                n_invalid += 1
                continue
            valid_count += 1
            d = state_distance(ns, target_state)
            if d < best_score:
                best_score = d
                best_state = ns
                best_path = path_pts
                best_traj = traj_states
        return best_state, best_path, best_traj, best_score, valid_count

    for it in range(MAX_ITERS):
        # timeout para reiniciar si se estanca
        if (time.perf_counter() - t0) > TREE_TIMEOUT:
            elapsed = time.perf_counter() - t0
            metrics = compute_run_metrics(
                None, None, elapsed, it, n_propagations, n_invalid, len(nodes),
                ref_poly=ref_poly, seed=seed, success=False, timed_out=True
            )
            return base_img, None, nodes, it, len(nodes), True, metrics

        valid_controls = 0

        # muestreo objetivo / waypoint / libre en corredor
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
            tw = rng.uniform(ROBOT_W_MIN, ROBOT_W_MAX)
        target = State(tx, ty, tth, tw)

        # nearest (aprox/exacto)
        if (it % EXACT_EVERY) == 0 or len(nodes) <= K_SUBSET:
            dists = [state_distance(n.state, target) for n in nodes]
            idx_near = int(np.argmin(dists))
        else:
            idxs = rng.sample(range(len(nodes)), k=min(K_SUBSET, len(nodes)))
            best, bestd = None, float('inf')
            for idx in idxs:
                d = state_distance(nodes[idx].state, target)
                if d < bestd:
                    bestd, best = d, idx
            idx_near = best if best is not None else idxs[0]

        near = nodes[idx_near].state
        if near is None:
            if DEBUG:
                print(f"[DBG it={it}] nodo nearest sin estado idx={idx_near}, se salta iter")
            continue

        # primer intento desde nearest para centrar vecindario
        seed_state, _, _, seed_score, valid_controls = best_control_towards(near, target)

        if DEBUG and (it % DEBUG_EVERY) == 0:
            wp_str = f"wp={wp_idx}/{len(waypoints)}" if cur_wp is not None else "wp=None"
            if seed_state is None:
                print(f"[DBG it={it}] valids={valid_controls} {wp_str}")
            else:
                print(f"[DBG it={it}] valids={valid_controls} d_target={seed_score:.1f} pos=({seed_state.x:.1f},{seed_state.y:.1f}) {wp_str}")

        if seed_state is None:
            continue

        # elegir padre usando conexion dinamica recalculada para cada candidato
        parent_candidates = set(neighborhood(nodes, seed_state, neighbor_radius_px))
        parent_candidates.add(idx_near)

        best_parent = None
        best_state = None
        best_path = None
        best_traj = None
        best_target_dist = float('inf')
        best_total_cost = float('inf')

        for pidx in parent_candidates:
            cand_state, cand_path, cand_traj, cand_dist, _ = best_control_towards(nodes[pidx].state, target)
            if cand_state is None:
                continue

            cand_edge = edge_cost(nodes[pidx].state, cand_state, cand_path)
            cand_total = nodes[pidx].cost + cand_edge

            better_target = cand_dist + 1e-9 < best_target_dist
            same_target_better_cost = abs(cand_dist - best_target_dist) <= 1e-9 and cand_total < best_total_cost
            if better_target or same_target_better_cost:
                best_parent = pidx
                best_state = cand_state
                best_path = cand_path
                best_traj = cand_traj
                best_target_dist = cand_dist
                best_total_cost = cand_total

        if best_parent is None or best_state is None:
            continue

        node_new = Node(best_state, parent=best_parent, path=best_path, traj_states=best_traj, cost=best_total_cost)
        nodes.append(node_new)
        new_idx = len(nodes) - 1
        children.append([])
        children[best_parent].append(new_idx)

        # evitar ciclos: no rewire de ancestros del nuevo nodo
        ancestors_of_new = set()
        cur = best_parent
        while cur != -1:
            ancestors_of_new.add(cur)
            cur = nodes[cur].parent

        # rewire con conexion dinamica valida
        neighbors = neighborhood(nodes, best_state, neighbor_radius_px)
        for nidx in neighbors:
            if nidx in (best_parent, new_idx):
                continue
            if nidx in ancestors_of_new:
                continue

            neighbor_state = nodes[nidx].state
            conn_state, conn_path, conn_traj, _, _ = best_control_towards(node_new.state, neighbor_state)
            if conn_state is None or not can_connect_states(conn_state, neighbor_state):
                continue

            rewire_path = list(conn_path) if conn_path is not None else []
            target_pt = (int(round(neighbor_state.x)), int(round(neighbor_state.y)))
            if not rewire_path:
                rewire_path = [target_pt]
            elif math.hypot(rewire_path[-1][0] - target_pt[0], rewire_path[-1][1] - target_pt[1]) > 0.5:
                rewire_path.append(target_pt)

            rewire_traj = list(conn_traj) if conn_traj is not None else []
            rewire_traj.append(neighbor_state)

            alt_cost = node_new.cost + edge_cost(node_new.state, neighbor_state, rewire_path)
            if alt_cost + 1e-6 < nodes[nidx].cost:
                old_parent = nodes[nidx].parent
                old_cost = nodes[nidx].cost

                nodes[nidx].parent = new_idx
                nodes[nidx].path = rewire_path
                nodes[nidx].traj_states = rewire_traj
                nodes[nidx].cost = alt_cost

                if old_parent != -1:
                    try:
                        children[old_parent].remove(nidx)
                    except ValueError:
                        pass
                if nidx not in children[new_idx]:
                    children[new_idx].append(nidx)

                propagate_subtree_cost_delta(nodes, children, nidx, alt_cost - old_cost)

        # avance de waypoint
        if use_wp and wp_idx < len(waypoints):
            if math.hypot(best_state.x - waypoints[wp_idx].x, best_state.y - waypoints[wp_idx].y) < wp_tolerance:
                wp_idx += 1

        # acumular segmentos del arbol
        if best_path is not None and len(best_path) > 1:
            for i in range(1, len(best_path)):
                segs.append((best_path[i - 1], best_path[i]))

        # refresco de dibujo
        if (it + 1) % DRAW_EVERY == 0 and segs:
            for (a, b) in segs:
                cv2.line(base_img, a, b, (120, 180, 255), 2)
            segs.clear()

            tmp = base_img.copy()
            draw_state(tmp, start, (0, 200, 0), 2, smap, show_details=False)
            draw_state(tmp, goal, (255, 0, 0), 2, smap, show_details=False)
            cv2.circle(tmp, goal_px, 5, (255, 0, 0), -1)
            if cur_wp is not None:
                cv2.circle(tmp, (int(cur_wp.x), int(cur_wp.y)), 5, (0, 255, 0), 2)
            cv2.imshow("RRT* kinodinamico", tmp)
            cv2.waitKey(1)

        # condicion de llegada
        if (
            math.hypot(best_state.x - goal.x, best_state.y - goal.y) < NEAR_GOAL_DIST and
            ang_dist(best_state.th, goal.th) < NEAR_GOAL_DTH and
            abs(best_state.w - goal.w) < NEAR_GOAL_DW
        ):
            idx = len(nodes) - 1
            sol_points, sol_states = [], []
            while idx != -1:
                n = nodes[idx]
                if n.path:
                    sol_points.extend(n.path[::-1])
                if n.traj_states:
                    sol_states.extend(n.traj_states[::-1])
                sol_states.append(n.state)
                idx = n.parent
            sol_points = sol_points[::-1]
            sol_states = sol_states[::-1]
            sol_states = minimize_width_changes(sol_states, smap)
            for i in range(1, len(sol_states)):
                if abs(sol_states[i].w - sol_states[i - 1].w) > 1e-6:
                    print(f"[MIN_WIDTH] cambio w en idx {i}: {sol_states[i-1].w:.3f} -> {sol_states[i].w:.3f}")
            out = base_img.copy()

            # camino final
            if sol_points and len(sol_points) > 1:
                for i in range(1, len(sol_points)):
                    cv2.line(out, sol_points[i - 1], sol_points[i], (0, 255, 0), 2)

            # configuraciones
            if sol_states and N_FRAMES > 0:
                idxs = np.linspace(0, len(sol_states) - 1, N_FRAMES, dtype=int)
                for j in idxs:
                    st = sol_states[j]
                    draw_state(out, st, (180, 90, 10), 2, smap, show_details=True)
                    supports = support_points_pixels(smap, st)
                    if len(supports) >= 3:
                        hull = cv2.convexHull(np.array(supports, dtype=np.int32))
                        cv2.polylines(out, [hull], True, (0, 255, 0), 1)
                        cx, cy = cog_pixel_from_table(st)
                        inside = cv2.pointPolygonTest(hull.astype(np.float32), (float(cx), float(cy)), False) >= 0
                    else:
                        cx, cy = cog_pixel_from_table(st)
                        inside = False
                    cv2.circle(out, (cx, cy), 3, (0, 255, 0) if inside else (0, 0, 255), -1)

            draw_state(out, start, (0, 200, 0), 2, smap, show_details=False)
            draw_state(out, goal, (255, 0, 0), 2, smap, show_details=False)
            cv2.circle(out, goal_px, 5, (255, 0, 0), -1)

            t1 = time.perf_counter()
            elapsed = t1 - t0

            metrics = compute_run_metrics(
                sol_points, sol_states, elapsed, it+1, n_propagations, n_invalid, len(nodes),
                ref_poly=ref_poly, seed=seed, success=True, timed_out=False
            )
            return out, sol_points, nodes, it + 1, len(nodes), False, metrics

    # sin solucion
    out = base_img.copy()
    draw_state(out, start, (0, 200, 0), 2, smap, show_details=False)
    draw_state(out, goal, (0, 0, 255), 2, smap, show_details=False)
    cv2.putText(out, "No se encontro la trayectoria", (10, smap.h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    metrics = compute_run_metrics(
        None, None, elapsed, MAX_ITERS, n_propagations, n_invalid, len(nodes),
        ref_poly=ref_poly, seed=seed, success=False, timed_out=False
    )
    return out, None, nodes, MAX_ITERS, len(nodes), False, metrics

def accumulate_attempt_metrics(total_metrics, attempt_metrics):
    if total_metrics is None:
        total_metrics = attempt_metrics.copy()
        total_metrics["attempts"] = 1
        total_metrics["timeout_retries"] = 1 if attempt_metrics["timed_out"] else 0
        return total_metrics

    total_metrics["elapsed_s"] += attempt_metrics["elapsed_s"]
    total_metrics["iters"] += attempt_metrics["iters"]
    total_metrics["propagations"] += attempt_metrics["propagations"]
    total_metrics["_invalid_total"] += attempt_metrics["_invalid_total"]
    total_metrics["invalid_pct"] = 100.0 * (
        total_metrics["_invalid_total"] / max(1, total_metrics["propagations"])
    )
    total_metrics["node_count"] = attempt_metrics["node_count"]
    total_metrics["seed"] = attempt_metrics["seed"]
    total_metrics["timed_out"] = attempt_metrics["timed_out"]
    total_metrics["attempts"] += 1
    if attempt_metrics["timed_out"]:
        total_metrics["timeout_retries"] += 1
    return total_metrics

def merge_successful_attempt_metrics(total_metrics, success_metrics):
    merged = success_metrics.copy()
    if total_metrics is None:
        return merged
    merged["elapsed_s"] = total_metrics["elapsed_s"]
    merged["iters"] = total_metrics["iters"]
    merged["propagations"] = total_metrics["propagations"]
    merged["_invalid_total"] = total_metrics["_invalid_total"]
    merged["invalid_pct"] = 100.0 * (
        merged["_invalid_total"] / max(1, merged["propagations"])
    )
    merged["attempts"] = total_metrics["attempts"]
    merged["timeout_retries"] = total_metrics["timeout_retries"]
    return merged

def run_single_simulation_with_retries(smap: SewerMap, walls_rgb, picker,
                                       start_state: State, goal_state: State, run_idx: int, total_runs: int):
    timeout_retries = 0
    attempt_idx = 0
    cumulative_metrics = None
    last_result = None

    while True:
        astar_path, _ = astar_path_guided(smap, picker.start_pos, picker.goal_pos)
        if astar_path is None or len(astar_path) < 2:
            return None, None, None, None, None, {
                "type": "astar_fail",
                "message": "A* no encontro camino transitable",
            }

        waypoints = build_waypoint_states(smap, astar_path, w_default=ROBOT_W0)
        if len(waypoints) < 2:
            return None, None, None, None, None, {
                "type": "waypoint_fail",
                "message": "Waypoints insuficientes",
            }

        base = walls_rgb.copy()
        for i in range(1, len(astar_path)):
            cv2.line(base, astar_path[i-1], astar_path[i], (0,100,255), 1)
        for wp in waypoints:
            cv2.circle(base, (int(wp.x), int(wp.y)), 2, (0,100,255), -1)

        seed = RNG_SEED_BASE + run_idx * 100000 + attempt_idx * 997
        plan_img, sol_points, nodes, iters, node_count, timed_out, attempt_metrics = plan_rrt_star(
            smap, start_state, goal_state, base,
            waypoints=waypoints,
            corridor_poly=[(wp.x, wp.y) for wp in waypoints],
            ref_poly=astar_path,
            seed=seed
        )
        cumulative_metrics = accumulate_attempt_metrics(cumulative_metrics, attempt_metrics)
        last_result = (plan_img, sol_points, nodes, iters, node_count)

        if sol_points is not None:
            final_metrics = merge_successful_attempt_metrics(cumulative_metrics, attempt_metrics)
            print_run_metrics(final_metrics, run_idx=run_idx, total_runs=total_runs)
            return plan_img, sol_points, nodes, iters, node_count, {
                "type": "success",
                "metrics": final_metrics,
            }

        if timed_out:
            timeout_retries += 1
            attempt_idx += 1
            cumulative_metrics["timeout_retries"] = timeout_retries
            print_timeout_retry(run_idx, total_runs, attempt_idx, max_restarts=None)
            continue

        final_metrics = cumulative_metrics.copy()
        final_metrics["timeout_retries"] = timeout_retries
        print_run_metrics(final_metrics, run_idx=run_idx, total_runs=total_runs)
        plan_img, sol_points, nodes, iters, node_count = last_result
        return plan_img, sol_points, nodes, iters, node_count, {
            "type": "failure",
            "metrics": final_metrics,
        }

def run_repeated_experiments(smap: SewerMap, walls_rgb, picker,
                             start_state: State, goal_state: State, repetitions):
    all_metrics = []
    best_result = None
    best_length_m = float('inf')
    last_result = None

    for run_idx in range(repetitions):
        plan_img, sol_points, nodes, iters, node_count, run_info = run_single_simulation_with_retries(
            smap, walls_rgb, picker, start_state, goal_state, run_idx, repetitions
        )

        if run_info["type"] in ("astar_fail", "waypoint_fail"):
            return (plan_img, sol_points, nodes, iters, node_count, run_info), all_metrics

        metrics = run_info["metrics"]
        all_metrics.append(metrics)
        last_result = (plan_img, sol_points, nodes, iters, node_count, run_info)

        if metrics["success"] and metrics["path_length_m"] is not None and metrics["path_length_m"] < best_length_m:
            best_length_m = metrics["path_length_m"]
            best_result = (plan_img, sol_points, nodes, iters, node_count, run_info)

    if all_metrics:
        print_experiment_summary(all_metrics)
    return best_result if best_result is not None else last_result, all_metrics

# --------------------- main ---------------------
def main():
    smap = SewerMap(resolve_map_path(MAP_PATH))
    walls_rgb = smap.draw_overlay()

    picker = Picker(smap)
    cv2.namedWindow("RRT* kinodinamico", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("RRT* kinodinamico", picker.mouse)

    show_instructions = True
    base = walls_rgb.copy()
    if show_instructions:
        draw_instructions(base)
    cur_img = base.copy()
    solution_img = None

    while True:
        cur_img = base.copy()

        if picker.start_pos is not None:
            cv2.circle(cur_img, picker.start_pos, 4, (0,255,0), -1)
            if picker.start_th is not None:
                s = State(picker.start_pos[0], picker.start_pos[1], picker.start_th, ROBOT_W0)
                draw_state(cur_img, s, (0,255,0), 2, smap, show_details=True)

        if picker.goal_pos is not None:
            cv2.circle(cur_img, picker.goal_pos, 4, (255,0,0), -1)
            if picker.goal_th is not None:
                g = State(picker.goal_pos[0], picker.goal_pos[1], picker.goal_th, ROBOT_W0)
                draw_state(cur_img, g, (255,0,0), 2, smap, show_details=True)

        if solution_img is not None:
            cv2.addWeighted(solution_img, 0.7, cur_img, 0.3, 0, cur_img)

        if show_instructions:
            draw_instructions(cur_img)
        cv2.imshow("RRT* kinodinamico", cur_img)
        k = cv2.waitKey(20) & 0xFF

        if k in (27, ord('q')):
            break

        if k == ord('r'):
            picker.reset(); solution_img = None
            show_instructions = True
            base = walls_rgb.copy()
            if show_instructions:
                draw_instructions(base)

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
                draw_state(tmp, start_state, (0,255,255), 2, smap, show_details=True)
                cv2.putText(tmp, "Start no valido", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT* kinodinamico", tmp); cv2.waitKey(900)
                continue

            if not valid_configuration(smap, goal_state):
                tmp = base.copy()
                draw_state(tmp, goal_state, (0,255,255), 2, smap, show_details=True)
                cv2.putText(tmp, "Goal no valido", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT* kinodinamico", tmp); cv2.waitKey(900)
                continue
            show_instructions = False
            base = walls_rgb.copy()
            best_result, all_metrics = run_repeated_experiments(
                smap, walls_rgb, picker, start_state, goal_state, max(1, REP_SIM)
            )

            if best_result is None:
                out = base.copy()
                cv2.putText(out, "No se pudo iniciar el experimento", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT* kinodinamico", out)
                continue

            plan_img, sol_points, nodes, iters, node_count, run_info = best_result
            solution_img = plan_img

            if run_info["type"] == "astar_fail":
                tmp = base.copy()
                cv2.putText(tmp, run_info["message"], (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT* kinodinamico", tmp); cv2.waitKey(900)
                continue

            if run_info["type"] == "waypoint_fail":
                tmp = base.copy()
                cv2.putText(tmp, run_info["message"], (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT* kinodinamico", tmp); cv2.waitKey(900)
                continue

            n_success = sum(1 for m in all_metrics if m["success"])
            best_metrics = run_info["metrics"]
            msg = f"Rep: {max(1, REP_SIM)} | exito: {n_success}/{len(all_metrics)} | iters: {iters}"
            if sol_points is None:
                msg += " | sin solucion"
            else:
                msg += (
                    f" | nodos: {node_count} | L={best_metrics['path_length_m']:.2f}m | "
                    f"reint_timeout={best_metrics['timeout_retries']}"
                )

            out = (plan_img if plan_img is not None else base).copy()
            cv2.putText(out, msg, (10, smap.h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("RRT* kinodinamico", out)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
