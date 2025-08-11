import matplotlib.pyplot as plt
import numpy as np

# Datos de la tabla 6.1
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

def longitud_en_funcion_del_ancho(ancho):
    # Interpolación lineal inversa basada en:
    # Ancho 0.52 m => Longitud 1.08 m
    # Ancho 0.85 m => Longitud 0.88 m
    ancho_min = 0.52
    ancho_max = 0.85
    largo_min = 0.88
    largo_max = 1.08

    if ancho < ancho_min:
        return largo_max
    elif ancho > ancho_max:
        return largo_min
    else:
        pendiente = (largo_min - largo_max) / (ancho_max - ancho_min)
        return largo_max + pendiente * (ancho - ancho_min)

class SIARSimulator2D:
    def __init__(self, width, electronics_offset):
        self.width = width
        self.length = longitud_en_funcion_del_ancho(width)
        self.electronics_offset = electronics_offset

        self.masses = {
            'front_left_wheel': 5,
            'front_right_wheel': 5,
            'rear_left_wheel': 5,
            'rear_right_wheel': 5,
            'left_wheel': 5,
            'right_wheel': 5,
            'electronics': 18,
            'estructura': 10
        }

        self.set_positions()

    def set_positions(self):
        hw = self.width / 2
        hl = self.length / 2

        self.positions = {
            'front_left_wheel': np.array([-hw, hl]),
            'front_right_wheel': np.array([hw, hl]),
            'rear_left_wheel': np.array([-hw, -hl]),
            'rear_right_wheel': np.array([hw, -hl]),
            'left_wheel': np.array([-hw, 0]),
            'right_wheel': np.array([hw, 0]),
            'electronics': np.array([0, self.electronics_offset]),
            'estructura': np.array([0, 0])
        }

    def compute_total_cm(self):
        total_mass = sum(self.masses.values())
        cm = sum(self.positions[k] * self.masses[k] for k in self.masses) / total_mass
        return cm

    def plot_robot(self, title_suffix=""):
        fig, ax = plt.subplots()

        lower_left = np.array([-self.width / 2, -self.length / 2])
        rect = plt.Rectangle(lower_left, self.width, self.length,
                             linewidth=1.5, edgecolor='blue', facecolor='lightblue', alpha=0.3)
        ax.add_patch(rect)

        for part, pos in self.positions.items():
            ax.plot(pos[0], pos[1], 'kx')
            ax.text(pos[0], pos[1], part, fontsize=7, ha='right', va='bottom')

        ax.plot(self.positions['electronics'][0], self.positions['electronics'][1],
                'go', label='CM electrónica')

        cm = self.compute_total_cm()
        ax.plot(cm[0], cm[1], 'ro', label='CM total')

        ax.set_aspect('equal')
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.7, 0.7)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"SIAR 2D - {title_suffix}")
        ax.grid(True)
        ax.legend()
        plt.show()

if __name__ == '__main__':
    for sensor, ancho, offset_cm in tabla_configuraciones:
        sim = SIARSimulator2D(width=ancho, electronics_offset=offset_cm)
        sim.plot_robot(f"Sensor={sensor}, Ancho={ancho:.2f} m, Largo={sim.length:.2f} m")
