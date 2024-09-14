# library imports
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# our imports
from pendulum_simulation import *
from pendulum_ekf import *


# monte constants
NUM_MONTE_RUNS = 50


def generate_monte_runs(theta_range=[np.pi/2, -np.pi/2]):

    x_0 = np.zeros(shape=(NUM_MONTE_RUNS, 2))
    x_0[:, 0] = np.linspace(theta_range[0], theta_range[1], num=NUM_MONTE_RUNS)

    monte_runs = []

    # Time array
    t = np.arange(0, simulation_time, 0.05)

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
    t = np.arange(0, simulation_time, dt)

    for idx in range(NUM_MONTE_RUNS):

        plt.plot(t, monte_runs[idx][:, 0])

    plt.xlabel('time')
    plt.ylabel('theta')
    plt.title('Monte Runs')
    plt.legend()
    plt.grid(True)
    plt.show()

def simulation_init():

    return np.array([0, 0]).reshape(-1, 1), np.array([[ 5, 0 ],
                                                      [ 0, 5 ]])

def kalman_filter_simulation(monte_runs):

    monte_kalman_estimates = np.zeros(shape=(NUM_MONTE_RUNS, int(simulation_time/dt)))
    monte_measurement_time_steps = np.zeros(shape=(NUM_MONTE_RUNS, int(simulation_time/dt)))
    monte_covaraince_time_steps = np.zeros(shape=(NUM_MONTE_RUNS, int(simulation_time/dt)))

    for run_idx in range(NUM_MONTE_RUNS):

        theta = monte_runs[run_idx][:, 0]

        x_n, P_n = simulation_init()

        for time_step_idx in range(int(simulation_time//dt)):

            # make prediction 
            x_prediction, P_n = ekf_predict_t(x_n, P_n)

            # state update
            measurement = theta[time_step_idx] + np.random.normal(0, measurement_noise_std)
            z = L * np.sin(measurement)

            x_n, P_n = ekf_update_t(x_prediction, P_n, z)

            monte_kalman_estimates[run_idx][time_step_idx] = x_n[0][0]
            monte_measurement_time_steps[run_idx][time_step_idx] = measurement
            monte_covaraince_time_steps[run_idx][time_step_idx] = P_n[0][0]

    ekf_simulation_summary = { 'ekf_estimates': monte_kalman_estimates,
                               'measurements': monte_measurement_time_steps,
                               'covariance': monte_covaraince_time_steps }

    return ekf_simulation_summary

def plot_kalman_results(monte_runs, ekf_simulation_summary):

    # Time array
    t = np.arange(0, simulation_time, dt)
    plt.figure(1)

    monte_kalman_estimates = ekf_simulation_summary['ekf_estimates']
    monte_measurement_time_steps = ekf_simulation_summary['measurements']

    for run_idx in range(NUM_MONTE_RUNS):

        theta = monte_runs[run_idx][:, 0]
        kalman_estimates = monte_kalman_estimates[run_idx][:]
        measurement_time_steps = monte_measurement_time_steps[run_idx][:]

        # Calculate confidence intervals
        confidence_interval_upper = kalman_estimates + 3 * measurement_noise_std
        confidence_interval_lower = kalman_estimates - 3 * measurement_noise_std

        if run_idx == 0:
            plt.plot(t, theta, 'g', label='True Theta')
            plt.plot(t, kalman_estimates, 'r-', label='EKF Estimates')
            plt.plot(t, measurement_time_steps, 'b.', label='Measurements')
            plt.fill_between(t, confidence_interval_lower, confidence_interval_upper, color='y', alpha=0.5, label='95% Confidence Interval')

        else:
            plt.plot(t, theta, 'g')
            plt.plot(t, kalman_estimates, 'r-')
            plt.plot(t, measurement_time_steps, 'b.')
            plt.fill_between(t, confidence_interval_lower, confidence_interval_upper, color='y', alpha=0.5)

    # Add labels and legend
    plt.xlabel('time [s]')
    plt.ylabel('theta [rad]')
    plt.title('Pendulum EKF Monte Results Estimates with Confidence Interval')
    plt.legend()
    plt.grid(True)

    plt.savefig("ekf_monte_results.png")

def plot_kalman_error(monte_runs, ekf_simulation_summary):

    # Time array
    t = np.arange(0, simulation_time, dt)
    plt.figure(2)

    monte_kalman_estimates = ekf_simulation_summary['ekf_estimates']

    for run_idx in range(NUM_MONTE_RUNS):

        theta = monte_runs[run_idx][:, 0]
        kalman_estimates = monte_kalman_estimates[run_idx][:]

        error = theta - kalman_estimates

        if run_idx == 0:
            plt.plot(t, error, 'k', label='EKF Estimate Error')

        else:
            plt.plot(t, error, 'k')

    # Add labels and legend
    plt.xlabel('time [s]')
    plt.ylabel('Error [rad]')
    plt.title('EKF Error vs Time')
    plt.legend()
    plt.grid(True)

    plt.savefig("ekf_monte_error.png")


if __name__ == "__main__":

    # create monte data
    monte_runs = generate_monte_runs()
    monte_runs = add_noise_to_monte_runs(monte_runs)

    # do ekf monte sim
    ekf_simulation_summary = kalman_filter_simulation(monte_runs)

    # show results
    plot_kalman_results(monte_runs, ekf_simulation_summary)
    plot_kalman_error(monte_runs, ekf_simulation_summary)
    plt.show()
