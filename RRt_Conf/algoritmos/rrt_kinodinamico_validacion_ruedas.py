# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import random
from dataclasses import dataclass

# ===================== Parámetros del mapa / umbrales =====================
# Ajusta la ruta a tu mapa:
MAP_PATH = "Pb4.png"    # <-- cámbialo a tu fichero

# Umbrales para detectar paredes (obstáculos positivos) y (opcional) gutter (negativos)
# Suponemos: paredes claras (alto valor), suelo más oscuro. Ajusta a tu mapa real.
TH_WALL_MIN = 180   # píxeles >= TH_WALL_MIN se consideran PARED
# Si tienes gutter marcado oscuro (rojo en tu pipeline), puedes activar algo extra si lo necesitas
USE_GUTTER_MASK = False
TH_GUTTER_MAX = 60  # píxeles <= TH_GUTTER_MAX se consideran "gutter" (negativo)

# ===================== Parámetros del robot (SIAR aprox) =====================
ROBOT_LEN = 0.88     # m (longitud a ancho máx aprox, escala se fija con PIXELS_PER_M)
ROBOT_W_MIN = 0.52   # m
ROBOT_W_MAX = 0.85   # m
ROBOT_W0   = 0.70    # m (ancho inicial por defecto si no se decide otra cosa)
W_DOT_MAX  = 0.20    # m/s velocidad de cambio de ancho

# Escala píxel-metro (si tu mapa está en píxeles sin escala, usa 1.0; si conoces resolución, ajústala)
PIXELS_PER_M = 70
def m2px(m): return int(round(m * PIXELS_PER_M))
def px2m(px): return float(px) / PIXELS_PER_M

# ===================== Parámetros del planificador =====================
MAX_ITERS = 60000
GOAL_SAMPLING_RATE = 0.10  # prob. de muestrear el goal directamente
DT = 0.05                   # s paso integración
TPROP = 0.4                 # s horizonte de propagación por control
CHECK_STEP = 2              # validar cada N steps de integración
NEAR_GOAL_DIST = m2px(0.30) # cercanía en píxeles
NEAR_GOAL_DTH  = math.radians(20.0)
NEAR_GOAL_DW   = 0.10       # m

# Conjunto de acciones (v, omega, u_w). v en m/s, omega en rad/s, u_w en m/s
# Puedes ampliar/reducir
CONTROL_SET = [
    (0.25,  0.0,   0.0),
    (0.25,  0.8,   0.0),
    (0.25, -0.8,   0.0),
    (0.25,  0.4,   0.0),
    (0.25, -0.4,   0.0),
    (0.18,  0.0,  +W_DOT_MAX),
    (0.18,  0.0,  -W_DOT_MAX),
]

# ===================== Utilidades =====================

def wrap_angle(a):
    while a > math.pi:  a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def ang_dist(a, b):
    return abs(wrap_angle(a - b))

@dataclass
class State:
    x: float  # píxeles
    y: float  # píxeles
    th: float # rad
    w: float  # m (ancho del robot)

@dataclass
class Node:
    state: State
    parent: int
    path: list          # lista de (x,y) en píxeles desde parent hasta este nodo (para dibujar)
    traj_states: list   # lista de State a lo largo de la propagación (incluye intermedios)

class SewerMap:
    def __init__(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"No se pudo abrir: {path}")
        self.gray = img
        self.h, self.w = img.shape[:2]

        # Clasificación de píxeles por intensidad (adaptada a Pb4.png)
        self.wall_mask   = np.zeros_like(img, dtype=np.uint8)
        self.gutter_mask = np.zeros_like(img, dtype=np.uint8)
        self.free_mask   = np.zeros_like(img, dtype=np.uint8)

        # Tres rangos de gris:
        # suelo oscuro ≈ <100
        # pared gris medio ≈ [100,180)
        # gutter claro ≈ >=180
        self.free_mask[self.gray < 100]   = 255   # suelo transitable
        self.wall_mask[(self.gray >=100) & (self.gray <180)] = 255  # pared
        self.gutter_mask[self.gray >=180] = 255  # gutter

        # Puntos de muestreo = suelo libre
        ys, xs = np.where(self.free_mask > 0)
        self.free_points = np.column_stack((xs, ys))  # (N,2)


    def inside_bounds(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def sample_free(self):
        if len(self.free_points) == 0:
            raise RuntimeError("No hay píxeles libres en el mapa (revisa TH_WALL_MIN).")
        idx = random.randrange(0, len(self.free_points))
        x, y = self.free_points[idx]
        return int(x), int(y)
    
    def draw_overlay(self):
        vis = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        vis[self.free_mask > 0]   = (200,200,200)  # gris claro
        vis[self.wall_mask > 0]   = (0,0,0)        # negro
        vis[self.gutter_mask > 0] = (0,0,255)      # rojo
        return vis

# Polígono del footprint del robot (rectángulo centrado en (x,y) con orientación th)
def robot_polygon(state: State):
    L = m2px(ROBOT_LEN)
    W = m2px(state.w)
    # rect centrado, eje x hacia delante:
    hx = L/2.0
    hy = W/2.0
    pts = np.array([
        [ +hx,  +hy],
        [ +hx,  -hy],
        [ -hx,  -hy],
        [ -hx,  +hy],
    ], dtype=np.float32)
    # rotación + traslación
    c, s = math.cos(state.th), math.sin(state.th)
    R = np.array([[c, -s],[s, c]], dtype=np.float32)
    rot = (R @ pts.T).T
    rot[:,0] += state.x
    rot[:,1] += state.y
    return rot.astype(np.int32)
# --- Ruedas: centros y máscaras ---
def wheel_centers_pixels(state: State):
    """
    Devuelve diccionario con centros de las 6 ruedas en píxeles.
    Etiquetas: FL, ML, RL (izquierda: front/middle/rear) y FR, MR, RR (derecha)
    """
    L = m2px(ROBOT_LEN)
    W = m2px(state.w)
    # offsets en el marco del robot (x hacia delante, y izquierda positiva)
    # longitudinales: tres ejes (front, middle, rear)
    x_front = +0.35 * L
    x_mid   = 0.0
    x_rear  = -0.35 * L
    # transversal: izquierda/derecha a ±W/2
    y_left  = +0.5 * W
    y_right = -0.5 * W

    # rotación + traslación
    c, s = math.cos(state.th), math.sin(state.th)
    def tf(xl, yl):
        xr = c*xl - s*yl + state.x
        yr = s*xl + c*yl + state.y
        return (int(round(xr)), int(round(yr)))

    return {
        'FL': tf(x_front, y_left),   'FR': tf(x_front, y_right),
        'ML': tf(x_mid,   y_left),   'MR': tf(x_mid,   y_right),
        'RL': tf(x_rear,  y_left),   'RR': tf(x_rear,  y_right),
    }

def wheel_masks(smap: "SewerMap", state: State, radius_px: int = None):
    """
    Máscara por rueda (círculo) y máscara de 'footprint' circular para estimar área en gutter/pared.
    radius_px: radio de rueda en píxeles (aprox). Si None, usa 0.5 * (0.26 m * PIXELS_PER_M).
    """
    if radius_px is None:
        # diámetro real ≈ 0.26 m -> radio ≈ 0.13 m
        radius_px = max(2, int(round(0.13 * PIXELS_PER_M)))

    centers = wheel_centers_pixels(state)
    masks = {}
    for k, (cx, cy) in centers.items():
        m = np.zeros((smap.h, smap.w), dtype=np.uint8)
        cv2.circle(m, (cx, cy), radius_px, 255, -1)
        masks[k] = m
    return masks

# Rasteriza el polígono del robot a una máscara uint8
def footprint_mask(smap: SewerMap, state: State):
    mask = np.zeros((smap.h, smap.w), dtype=np.uint8)
    poly = robot_polygon(state).reshape((-1,1,2))
    cv2.fillPoly(mask, [poly], 255)
    return mask
# --- RUEDAS: círculos por rueda y pintura verde/negro si pisa gutter ---
WHEEL_DIAM_M = 0.26   # ~26 cm
WHEEL_RAD_M  = WHEEL_DIAM_M/2
WHEEL_W_M    = 0.07   # ancho aprox, solo para máscara fina si quisieras rectángulos
WHEEL_GUTTER_MAX_FRAC = 0.33  # si excede esta fracción -> rueda en negro

def wheel_centers_body(w_m: float):
    """Centros de 6 ruedas en coords cuerpo (m). 3 por lado, espaciadas en largo."""
    L = ROBOT_LEN
    xs = [ +L/2 - L*0.15, 0.0, -L/2 + L*0.15 ]  # front, mid, rear
    half_w = w_m/2.0
    centers = []
    for x in xs: centers.append( (x, +half_w) )
    for x in xs: centers.append( (x, -half_w) )
    return centers  # [Lfront,Lmid,Lrear,Rfront,Rmid,Rrear]

def wheel_circles_px(state: State):
    """Devuelve [(cx,cy,r_px), ...] en píxeles para las 6 ruedas."""
    ca, sa = math.cos(state.th), math.sin(state.th)
    circles = []
    r_px = m2px(WHEEL_RAD_M)
    for xb, yb in wheel_centers_body(state.w):
        wx = state.x + m2px(xb*ca - yb*sa)
        wy = state.y + m2px(xb*sa + yb*ca)
        circles.append( (int(round(wx)), int(round(wy)), r_px) )
    return circles

def wheel_gutter_fraction(smap: SewerMap, circle):
    """Fracción del área del círculo que cae en gutter."""
    cx, cy, r = circle
    x0 = max(cx - r, 0); x1 = min(cx + r + 1, smap.w)
    y0 = max(cy - r, 0); y1 = min(cy + r + 1, smap.h)
    if x0>=x1 or y0>=y1: return 0.0
    w = x1-x0; h=y1-y0
    Y, X = np.ogrid[0:h, 0:w]
    dx = (X + x0 - cx); dy = (Y + y0 - cy)
    circ = ((dx*dx + dy*dy) <= (r*r)).astype(np.uint8)*255
    gut_roi = smap.gutter_mask[y0:y1, x0:x1]
    inter = cv2.countNonZero(cv2.bitwise_and(circ, gut_roi))
    area  = cv2.countNonZero(circ)
    return 0.0 if area==0 else inter/float(area)

def draw_wheels(smap: SewerMap, img, state: State):
    """Pinta ruedas: VERDE si fracción en gutter <= umbral, NEGRO si > umbral."""
    for c in wheel_circles_px(state):
        frac = wheel_gutter_fraction(smap, c)
        color = (0,255,0) if frac <= WHEEL_GUTTER_MAX_FRAC else (0,0,0)  # verde / negro
        cv2.circle(img, (c[0],c[1]), c[2], color, thickness=2)

# Validación de configuración: no colisiona con paredes y está dentro de los límites.
# (Opcional) puedes reforzar con reglas de gutter si lo deseas (aquí solo paredes).
def valid_configuration(smap: "SewerMap", state: State):
    # 1) Límites y ancho
    if not smap.inside_bounds(int(state.x), int(state.y)):
        return False
    if not (ROBOT_W_MIN <= state.w <= ROBOT_W_MAX):
        return False

    # 2) No colisión del cuerpo con PAREDES
    robot_mask = footprint_mask(smap, state)
    if cv2.countNonZero(cv2.bitwise_and(robot_mask, smap.wall_mask)) > 0:
        return False

    # 3) Reglas por RUEDAS con gutter/pared
    wmasks = wheel_masks(smap, state)
    wheels_on_gutter = []
    wheels_on_wall   = []

    # porcentaje mínimo de solapamiento para considerar que "pisa" (ajustable)
    MIN_TOUCH = 0.20

    for tag, m in wmasks.items():
        area = cv2.countNonZero(m) + 1e-6
        gut_area = cv2.countNonZero(cv2.bitwise_and(m, smap.gutter_mask))
        wall_area = cv2.countNonZero(cv2.bitwise_and(m, smap.wall_mask))
        if wall_area / area > MIN_TOUCH:
            wheels_on_wall.append(tag)
        if gut_area / area > MIN_TOUCH:
            wheels_on_gutter.append(tag)

    # 3.a) Una rueda no puede pisar PARED
    if len(wheels_on_wall) > 0:
        return False

    # 3.b) No permitir 4 o más ruedas en el gutter (tu caso observado)
    if len(wheels_on_gutter) >= 4:
        return False

    # 3.c) No permitir que las 3 ruedas de un mismo lado estén en el gutter
    left  = {'FL','ML','RL'}
    right = {'FR','MR','RR'}
    gut_set = set(wheels_on_gutter)
    if len(gut_set & left) == 3 or len(gut_set & right) == 3:
        return False

    # 3.d) Si hay 2 y 2 (dos por lado), sólo válidas 'esquinas opuestas'
    #     Permitimos {FL, RR} o {FR, RL}. Cualquier otro patrón 2+2 se invalida.
    if len(gut_set & left) >= 2 and len(gut_set & right) >= 2:
        allowed_pairs = [ {'FL','RR'}, {'FR','RL'} ]
        # coger exactamente 2 por lado
        if len(gut_set) == 4:
            if not any(p.issubset(gut_set) for p in allowed_pairs):
                return False
        else:
            # patrones raros (5 ruedas ya rechazadas arriba), o 3 total repartidas 2+1
            # si cae aquí con 2+2 pero >4 ya está filtrado; por seguridad invalidamos
            return False

    return True


# Propagación del sistema (unicycle + cambio de ancho): integra durante TPROP con DT
def propagate(state: State, control, smap: SewerMap):
    v, w_z, wdot = control
    steps = max(1, int(round(TPROP / DT)))
    x = state.x; y = state.y; th = state.th; w = state.w
    path_pts = []
    traj_states = []
    for k in range(steps):
        dx_m = v * math.cos(th) * DT
        dy_m = v * math.sin(th) * DT
        x += m2px(dx_m)
        y += m2px(dy_m)
        th = wrap_angle(th + w_z*DT)
        w = max(ROBOT_W_MIN, min(ROBOT_W_MAX, w + wdot*DT))
        st = State(x, y, th, w)
        if (k % CHECK_STEP) == 0:
            if not valid_configuration(smap, st):
                return None, None, None  # colisión
        path_pts.append((int(x), int(y)))
        traj_states.append(st)
    new_state = State(x, y, th, w)
    if not valid_configuration(smap, new_state):
        return None, None, None
    return new_state, path_pts, traj_states


# Distancia entre estados (píxeles y pesos angulares/ancho)
def state_distance(a: State, b: State):
    dx = a.x - b.x
    dy = a.y - b.y
    dpos = math.hypot(dx, dy)
    dth = ang_dist(a.th, b.th)
    dw  = abs(a.w - b.w)
    # Pondera (ajusta si quieres)
    return dpos + m2px(0.5)*dth + m2px(0.2)*dw

# Selección interactiva de start/goal con el ratón
class Picker:
    def __init__(self, smap: SewerMap):
        self.smap = smap
        self.reset()

    def reset(self):
        self.start_pos = None
        self.start_th  = None
        self.goal_pos  = None
        self.goal_th   = None
        self.tmp_point = None
        self.tmp_mode  = None # 'start' or 'goal'

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.start_pos is None:
                self.start_pos = (x, y)
                self.tmp_mode  = 'start'
            elif self.start_th is None and self.tmp_mode == 'start':
                dx = x - self.start_pos[0]
                dy = y - self.start_pos[1]
                self.start_th = math.atan2(dy, dx)
                self.tmp_mode = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.goal_pos is None:
                self.goal_pos = (x, y)
                self.tmp_mode  = 'goal'
            elif self.goal_th is None and self.tmp_mode == 'goal':
                dx = x - self.goal_pos[0]
                dy = y - self.goal_pos[1]
                self.goal_th = math.atan2(dy, dx)
                self.tmp_mode = None

    def have_start(self):
        return self.start_pos is not None and self.start_th is not None

    def have_goal(self):
        return self.goal_pos is not None and self.goal_th is not None

# Dibujo auxiliar
def draw_state(img, state: State, color, thickness=2, smap: "SewerMap" = None):
    # cuerpo
    poly = robot_polygon(state).reshape((-1,1,2))
    cv2.polylines(img, [poly], True, color, thickness)

    # flecha orientación
    p0 = (int(state.x), int(state.y))
    p1 = (int(state.x + 0.35*math.cos(state.th)*m2px(1.0)),
          int(state.y + 0.35*math.sin(state.th)*m2px(1.0)))
    cv2.arrowedLine(img, p0, p1, color, 2, tipLength=0.25)

    # ruedas: verde si libre, negro si en gutter, rojo si en pared (no debería ocurrir porque invalidamos)
    if smap is not None:
        wm = wheel_masks(smap, state)
        for tag, m in wm.items():
            # centro para dibujar círculo
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                continue
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            area = cv2.countNonZero(m) + 1e-6
            gut_area = cv2.countNonZero(cv2.bitwise_and(m, smap.gutter_mask))
            wall_area = cv2.countNonZero(cv2.bitwise_and(m, smap.wall_mask))
            if wall_area / area > 0.20:
                col = (0,0,255)         # rojo pared (debug)
            elif gut_area / area > 0.20:
                col = (0,0,0)           # negro gutter
            else:
                col = (0,255,0)         # verde libre
            # radio visual ~ 70% del usado en máscara
            r_vis = max(2, int(round(0.9 * (0.13 * PIXELS_PER_M))))
            cv2.circle(img, (cx, cy), r_vis, col, 2)

def draw_instructions(img):
    lines = [
        "Clic izq: start (1 pos, 2 orient.)",
        "Clic der: goal  (1 pos, 2 orient.)",
        "ESPACIO: planificar | r: reset | q: salir",
    ]
    y0 = 22
    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, y0 + 22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

# Planificador RRT kinodinámico
def plan_rrt(smap: SewerMap, start: State, goal: State, vis_img):
    nodes = [Node(start, parent=-1, path=[],traj_states=[])]
    rng = random.Random()
    base_img = vis_img.copy()
    goal_px  = (int(goal.x), int(goal.y))

    for it in range(MAX_ITERS):
        # Muestreo
        if rng.random() < GOAL_SAMPLING_RATE:
            tx, ty, tth, tw = goal.x, goal.y, goal.th, goal.w
        else:
            tx, ty = smap.sample_free()
            tth = rng.uniform(-math.pi, math.pi)
            tw  = rng.uniform(ROBOT_W_MIN, ROBOT_W_MAX)
        target = State(tx, ty, tth, tw)

        # Nearest
        dists = [state_distance(n.state, target) for n in nodes]
        idx_near = int(np.argmin(dists))
        near = nodes[idx_near].state

        # Probar todos los controles y elegir el que más acerque al target
        best_state = None
        best_path  = None
        best_dist  = float('inf')
        for u in CONTROL_SET:
            new_state, path_pts, traj_states = propagate(near, u, smap)

            if new_state is None:
                continue
            d = state_distance(new_state, target)
            if d < best_dist:
                best_dist = d
                best_state = new_state
                best_path  = path_pts

        if best_state is None:
            # nodo muerto, saltamos
            continue

        nodes.append(Node(best_state, parent=idx_near, path=best_path, traj_states=traj_states))


        # Dibuja rama
        for i in range(1, len(best_path)):
            cv2.line(base_img, best_path[i-1], best_path[i], (120,180,255), 1)

        # ¿Cerca del goal?
        if (math.hypot(best_state.x - goal.x, best_state.y - goal.y) < NEAR_GOAL_DIST and
            ang_dist(best_state.th, goal.th) < NEAR_GOAL_DTH and
            abs(best_state.w - goal.w) < NEAR_GOAL_DW):
            # Intento de “conexión” directa con una acción recta (opcional)
            # Aquí aceptamos ya como solución
            # Reconstruir trayectoria
# Reconstruir trayectoria (puntos y estados)
            path_idx = len(nodes) - 1
            sol_points = []
            sol_states = []
            while path_idx != -1:
                n = nodes[path_idx]
                if n.path:
                    sol_points.extend(n.path[::-1])
                if n.traj_states:
                    sol_states.extend(n.traj_states[::-1])
                sol_states.append(n.state)  # incluir el estado del nodo
                path_idx = n.parent
            sol_points = sol_points[::-1]
            sol_states = sol_states[::-1]

            # ----- LOG de cambios de ancho -----
            if len(sol_states) > 1:
                w0 = sol_states[0].w
                hubo_cambio = False
                for i, st in enumerate(sol_states[1:], start=1):
                    if abs(st.w - w0) > 1e-3:
                        print(f"[LOG] Cambio de ancho en la ruta: de {w0:.3f} m a {st.w:.3f} m en el estado {i}")
                        hubo_cambio = True
                        w0 = st.w
                if not hubo_cambio:
                    print("[LOG] No hubo cambios de ancho entre el inicio y el goal.")

            out = base_img.copy()

            # ----- Dibujo de la ruta -----
            for i in range(1, len(sol_points)):
                cv2.line(out, sol_points[i-1], sol_points[i], (0,255,0), 2)

            # ----- Dibujar 5 estados intermedios (uniformemente espaciados) -----
            if len(sol_states) >= 7:  # inicio y final + 5 intermedios
                idxs = np.linspace(1, len(sol_states)-2, 5, dtype=int)  # evita sobrescribir inicio/fin
                for idx in idxs:
                    st = sol_states[idx]
                    draw_state(out, st, (180,90,10), 2)   # cuerpo intermedio
                    draw_wheels(smap, out, st)            # ruedas verdes/negro según gutter

            # ----- Inicio / Goal (con ruedas también) -----
            draw_state(out, start, (0,200,0), 2)
            draw_wheels(smap, out, start)

            draw_state(out, goal,  (255,0,0), 2)  # AZUL
            cv2.circle(out, (int(goal.x), int(goal.y)), 5, (255,0,0), -1)
            draw_wheels(smap, out, goal)

            return out, sol_points, nodes, it+1


        if (it+1) % 200 == 0:
            # refresco visual periódico
            tmp = base_img.copy()
            draw_state(tmp, start, (0,200,0), 2)
            draw_state(tmp, goal,  (255,0,0), 2)
            cv2.circle(tmp, goal_px, 5, (255,0,0), -1)
            cv2.imshow("RRT kinodinamico", tmp)
            cv2.waitKey(1)

    # Sin solución
    out = base_img.copy()
    draw_state(out, start, (0,200,0), 2)
    draw_state(out, goal,  (0,0,255), 2)
    cv2.putText(out, "No se encontro trayectoria", (10, smap.h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
    return out, None, nodes, MAX_ITERS

def main():
    smap = SewerMap(MAP_PATH)

    walls_rgb = smap.draw_overlay()  # paredes=negro, gutter=rojo, transitable=gris claro

    picker = Picker(smap)
    cv2.namedWindow("RRT kinodinamico", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("RRT kinodinamico", picker.mouse)

    base = walls_rgb.copy()
    draw_instructions(base)
    cur_img = base.copy()
    solution_img = None

    start_state = None
    goal_state  = None

    while True:
        cur_img = base.copy()

        # Dibujo provisional de puntos / flechas mientras eliges
        if picker.start_pos is not None:
            cv2.circle(cur_img, picker.start_pos, 4, (0,255,0), -1)
            if picker.start_th is not None:
                s = State(picker.start_pos[0], picker.start_pos[1], picker.start_th, ROBOT_W0)
                draw_state(cur_img, s, (0,255,0), 2)

        if picker.goal_pos is not None:
            cv2.circle(cur_img, picker.goal_pos, 4, (255,0,0), -1)
            if picker.goal_th is not None:
                g = State(picker.goal_pos[0], picker.goal_pos[1], picker.goal_th, ROBOT_W0)
                draw_state(cur_img, g, (255,0,0), 2)

        if solution_img is not None:
            # mostrar la última solución
            cv2.addWeighted(solution_img, 0.7, cur_img, 0.3, 0, cur_img)

        draw_instructions(cur_img)
        cv2.imshow("RRT kinodinamico", cur_img)
        k = cv2.waitKey(20) & 0xFF

        if k in (27, ord('q')):  # ESC o q
            break

        if k == ord('r'):
            picker.reset()
            solution_img = None
            start_state = None
            goal_state  = None
            base = walls_rgb.copy()
            draw_instructions(base)

        if k == 32:  # SPACE -> planificar si hay start/goal
            if not picker.have_start() or not picker.have_goal():
                # indica en pantalla
                base2 = base.copy()
                cv2.putText(base2, "Selecciona START y GOAL (pos+orient)", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT kinodinamico", base2)
                cv2.waitKey(900)
                continue

            start_state = State(picker.start_pos[0], picker.start_pos[1], picker.start_th, ROBOT_W0)
            goal_state  = State(picker.goal_pos[0],  picker.goal_pos[1],  picker.goal_th,  ROBOT_W0)

            # Validar SOLO ahora (antes no abortamos)
            if not valid_configuration(smap, start_state):
                tmp = base.copy()
                draw_state(tmp, start_state, (0,255,255), 2)
                cv2.putText(tmp, "Start no valido: ajusta posicion/orientacion", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT kinodinamico", tmp)
                cv2.waitKey(1200)
                continue

            if not valid_configuration(smap, goal_state):
                tmp = base.copy()
                draw_state(tmp, goal_state, (0,255,255), 2)
                cv2.putText(tmp, "Goal no valido: ajusta posicion/orientacion", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("RRT kinodinamico", tmp)
                cv2.waitKey(1200)
                continue

            # Lanza RRT
            plan_img, sol_points, nodes, iters = plan_rrt(smap, start_state, goal_state, walls_rgb)
            solution_img = plan_img

            # Mensaje
            msg = f"Iteraciones: {iters}"
            if sol_points is None:
                msg += " | sin solución"
            else:
                msg += f" | nodos: {len(nodes)}"
            out = plan_img.copy()
            cv2.putText(out, msg, (10, smap.h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("RRT kinodinamico", out)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
