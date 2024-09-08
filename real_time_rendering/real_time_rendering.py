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

        # system dynamic init
        self.x_n = np.array([np.pi/2, 0])
        self.t = 0

    def _pendulum_rendering_update(self):

        # rendering position init
        self.x = L * np.sin(self.x_n[0])
        self.y = L * np.cos(self.x_n[0])

    def _system_step(self, _):

        self.x_n = pendulum_next_state(self.x_n, dt)
        self._pendulum_rendering_update()

        plt.cla()

        # Plot the pendulum rod (line)
        plt.plot([0, self.x], [0, self.y], color='blue', label='Pendulum Arm')

        # Plot the pendulum bob (circle)
        circle = plt.Circle((self.x, self.y), 0.1, color='red', label='Pendulum Ball')
        plt.gca().add_patch(circle)

        # Set axis limits to keep the pendulum in view
        plt.xlim(-L - 1, L + 1)
        plt.ylim(-L - 1, 1)

    def render(self):

        _ = FuncAnimation(plt.gcf(), self._system_step, interval=50)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    rendering = PendulumRendering()
    rendering.render()
