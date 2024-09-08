# library imports
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# our imports
from configs import *
from pendulum_dynamics import *


class PendulumRendering:

    def __init__(self):

        self._simulation_init()

    def _simulation_init(self):

        # System dynamic init
        self.x_n = np.array([np.pi / 2, 0])  # Initial state (theta, theta_dot)
        self.t = 0

        # Create figure and axis
        self.fig, self.ax = plt.subplots()

        # Set axis limits
        self.ax.set_xlim(-L - 1, L + 1)
        self.ax.set_ylim(-L - 1, 1)

        # Create pendulum arm and bob (pre-allocate)
        self.line, = self.ax.plot([], [], color='blue', label='Pendulum Arm')  # Line for the arm
        self.circle = plt.Circle((0, -L), 0.0875, color='red', label='Pendulum Bob')  # Circle for the bob
        self.ax.add_patch(self.circle)

        self.ax.set_aspect('equal')
        self.ax.legend(loc='upper left')

    def _pendulum_rendering_update(self):

        # Rendering position update
        self.x = L * np.sin(self.x_n[0])
        self.y = L * np.cos(self.x_n[0]) 

    def _system_step(self, _):

        # Update the pendulum state
        self.x_n = pendulum_next_state(self.x_n, dt)
        self._pendulum_rendering_update()

        # Update the line (pendulum arm)
        self.line.set_data([0, self.x], [0, self.y])

        # Update the circle (pendulum bob)
        self.circle.center = (self.x, self.y)

    def render(self):

        # Animate without clearing everything on each frame
        _ = FuncAnimation(self.fig, self._system_step, interval=50)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    rendering = PendulumRendering()
    rendering.render()
