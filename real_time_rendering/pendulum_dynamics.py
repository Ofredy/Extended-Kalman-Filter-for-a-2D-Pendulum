import numpy as np

from configs import *

def pendulum_next_state(x_n, dt):

    x_n[0] += x_n[1] * dt
    x_n[1] -= -(g/L) * np.sin(x_n[0]) * dt

    return x_n
