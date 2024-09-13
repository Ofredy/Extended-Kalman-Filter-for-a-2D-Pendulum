# library imports
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# our imports
from pendulum_simulation import *


# monte constants
NUM_MONTE_RUNS = 50


def generate_monte_runs(theta_range=[np.pi/2, -np.pi/2]):

    x_0 = np.zeros(shape=(NUM_MONTE_RUNS, 2))
    x_0[:, 0] = np.linspace(theta_range[0], theta_range[1], num=NUM_MONTE_RUNS)

    monte_runs = []

    # Time array
    t = np.arange(0, 10, 0.05)

    for idx in range(NUM_MONTE_RUNS):

        solution_n = odeint(pendulum_dynamics, x_0[idx], t, args=(g, L, gamma))
        monte_runs.append(solution_n)

    return monte_runs

def add_noise_to_monte_runs(monte_runs):

    for idx in range(NUM_MONTE_RUNS):

        solution_n = monte_runs[idx]

        solution_n[:, 0] = solution_n[:, 0] + np.random.normal(0, process_noise_std, len(solution_n[:, 0]))
        solution_n[:, 1] = solution_n[:, 1] + np.random.normal(0, process_noise_std, len(solution_n[:, 1]))

        monte_runs[idx] = solution_n

    return monte_runs

def plot_monte_runs(monte_runs):

    # Time array
    t = np.arange(0, 10, 0.05)

    for idx in range(NUM_MONTE_RUNS):

        plt.plot(t, monte_runs[idx][:, 0])

    plt.xlabel('time')
    plt.ylabel('theta')
    plt.title('Monte Runs')
    plt.legend()
    plt.grid(True)
    plt.show()

def kalman_filter_simulation(monte_runs):

    pass


if __name__ == "__main__":

    monte_runs = generate_monte_runs()
    monte_runs = add_noise_to_monte_runs(monte_runs)
    plot_monte_runs(monte_runs)
