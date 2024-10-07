# system imports
import math

# library imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# our imports
from pendulum_simulation import *
from pendulum_ekf import *


# monte constants
NUM_MONTE_RUNS = 50


def generate_monte_runs(theta_range=[np.pi/2, -np.pi/2], external_force=False):

    x_0 = np.zeros(shape=(NUM_MONTE_RUNS, 2))
    x_0[:, 0] = np.linspace(theta_range[0], theta_range[1], num=NUM_MONTE_RUNS)

    monte_runs = []

    # Time array
    t_span = [0, simulation_time] if not external_force else [0, force_simulation_time]
    t = np.arange(0, simulation_time, dt) if not external_force else np.arange(0, force_simulation_time, dt)

    for idx in range(NUM_MONTE_RUNS):

        if not external_force:
            solution_n = solve_ivp(lambda t, y: pendulum_dynamics(y, t, g, pendulum_model['length'], gamma, None, None, pendulum_model['inertia'], pendulum_model['total_mass']), t_span, x_0[idx], t_eval=t)

        else:
            solution_n = solve_ivp(lambda t, y: pendulum_dynamics(y, t, g, pendulum_model['length'], gamma, force_mag, force_frequency, pendulum_model['inertia'], pendulum_model['total_mass'], True), t_span, x_0[idx], t_eval=t, rtol=1e-8, atol=1e-10)

        monte_runs.append(np.column_stack((solution_n.y[0], solution_n.y[1])))

    return monte_runs

def add_noise_to_monte_runs(monte_runs):

    for idx in range(NUM_MONTE_RUNS):

        solution_n = monte_runs[idx]

        solution_n[:, 0] = solution_n[:, 0] + np.random.normal(0, math.sqrt(process_noise_variance), len(solution_n[:, 0]))
        solution_n[:, 1] = solution_n[:, 1] + np.random.normal(0, math.sqrt(process_noise_variance), len(solution_n[:, 1]))

        monte_runs[idx] = solution_n

    return monte_runs

def plot_monte_runs(monte_runs, external_force=False):

    # Time array
    t = np.arange(0, simulation_time, dt) if not external_force else np.arange(0, force_simulation_time, dt)

    for idx in range(NUM_MONTE_RUNS):

        plt.plot(t, monte_runs[idx][:, 0])

    plt.xlabel('time')
    plt.ylabel('theta')
    plt.title('Monte Runs')
    plt.legend()
    plt.grid(True)

def simulation_init():

    return np.array([np.random.normal(0, math.sqrt(x_0_guess_variance)), 0]).reshape(-1, 1), \
           np.array([[ x_0_guess_variance, 0 ],
                     [ 0, x_0_guess_variance ]])

def kalman_filter_simulation(monte_runs, external_force=False):

    if not external_force:
        monte_kalman_estimates = np.zeros(shape=(NUM_MONTE_RUNS, int(simulation_time/(1/prediction_hz))))
        monte_measurement_time_steps = np.zeros(shape=(NUM_MONTE_RUNS, int(simulation_time/(1/measurement_hz))))
        monte_covaraince_time_steps = np.zeros(shape=(NUM_MONTE_RUNS, int(simulation_time/(1/prediction_hz))))
        monte_k_n = np.zeros(shape=(NUM_MONTE_RUNS, int(simulation_time/(1/prediction_hz))))
        state_predictions = np.zeros(shape=(NUM_MONTE_RUNS, int(simulation_time/(1/prediction_hz))))

    else:
        monte_kalman_estimates = np.zeros(shape=(NUM_MONTE_RUNS, int(force_simulation_time/(1/prediction_hz))))
        monte_measurement_time_steps = np.zeros(shape=(NUM_MONTE_RUNS, int(force_simulation_time/(1/measurement_hz))))
        monte_covaraince_time_steps = np.zeros(shape=(NUM_MONTE_RUNS, int(force_simulation_time/(1/prediction_hz))))
        monte_k_n = np.zeros(shape=(NUM_MONTE_RUNS, int(force_simulation_time/(1/prediction_hz))))
        state_predictions = np.zeros(shape=(NUM_MONTE_RUNS, int(force_simulation_time/(1/prediction_hz))))

    for run_idx in range(NUM_MONTE_RUNS):

        theta = monte_runs[run_idx][:, 0]

        x_n, P_n = simulation_init()

        loop_range = int(simulation_time//dt) if not external_force else int(force_simulation_time//dt)

        prediction_counter = 0
        measurement_counter = 0

        for time_step_idx in range(loop_range):
            
            prediction_rate_idx = (1/prediction_hz) / dt 
            measurement_rate_idx = (1/measurement_hz) / dt 

            if time_step_idx % prediction_rate_idx == 0:

                # make prediction 
                x_prediction, P_n = ekf_predict_t(x_n, P_n)

                if time_step_idx % measurement_rate_idx == 0:
                    
                    # state update
                    measurement = theta[time_step_idx] + np.random.normal(0, math.sqrt(measurement_noise_variance))
                    monte_measurement_time_steps[run_idx][measurement_counter] = measurement
                    measurement_counter += 1

                    z = pendulum_model['length'] * np.sin(measurement)

                    x_n, P_n, k_n = ekf_update_t(x_prediction, P_n, z)

                else:
                    x_n = x_prediction

                monte_kalman_estimates[run_idx][prediction_counter] = x_n[0][0]
                monte_covaraince_time_steps[run_idx][prediction_counter] = P_n[0][0]
                monte_k_n[run_idx][prediction_counter] = np.linalg.norm(k_n)
                state_predictions[run_idx][prediction_counter] = x_prediction[0][0]

                prediction_counter += 1

    ekf_simulation_summary = { 'ekf_estimates': monte_kalman_estimates,
                               'measurements': monte_measurement_time_steps,
                               'covariance': np.sqrt(monte_covaraince_time_steps),
                               'kalman_gain_norm': monte_k_n,
                               'state_predictions': state_predictions }

    return ekf_simulation_summary

def assess_kalman_accuracy(monte_runs, ekf_simulation_summary, external_force=False):

    total_mae = 0

    time_indices = np.arange(int(simulation_time/dt)) % int((1/prediction_hz)/dt) == 0 if not external_force else np.arange(int(force_simulation_time/dt)) % int((1/prediction_hz)/dt) == 0

    for run_idx in range(NUM_MONTE_RUNS):

        theta = monte_runs[run_idx][:, 0][time_indices]  # True state (theta)
        kalman_estimates = ekf_simulation_summary['ekf_estimates'][run_idx][:]  # Kalman estimates

        mae = np.mean(np.abs(theta-kalman_estimates))

        total_mae += mae

    avg_mae = total_mae / NUM_MONTE_RUNS

    ekf_simulation_summary['avg_mae'] = avg_mae

    return ekf_simulation_summary


def plot_kalman_results(monte_runs, ekf_simulation_summary, external_force=False):

    # Time array
    t = np.arange(0, simulation_time, (1/prediction_hz)) if not external_force else np.arange(0, force_simulation_time, (1/prediction_hz))
    plt.figure(1)

    time_indices = np.arange(int(simulation_time/dt)) % int((1/prediction_hz)/dt)  == 0 if not external_force else np.arange(int(force_simulation_time/dt)) % int((1/prediction_hz)/dt)  == 0
    monte_time = simulation_time if not external_force else force_simulation_time

    monte_kalman_estimates = ekf_simulation_summary['ekf_estimates']
    monte_measurement_time_steps = ekf_simulation_summary['measurements']

    for run_idx in range(NUM_MONTE_RUNS):

        theta = monte_runs[run_idx][:, 0][time_indices]
        kalman_estimates = monte_kalman_estimates[run_idx][:]
        measurement_time_steps = monte_measurement_time_steps[run_idx][:]

        # Calculate confidence intervals
        confidence_interval_upper = kalman_estimates + 3 * math.sqrt(process_noise_variance)
        confidence_interval_lower = kalman_estimates - 3 * math.sqrt(process_noise_variance)

        if run_idx == 0:
            plt.plot(t, theta, 'g', label='True Theta')
            plt.plot(t, kalman_estimates, 'r-', label='EKF Estimates')
            plt.plot(np.arange(0, monte_time, (1/measurement_hz)), measurement_time_steps, 'b.', label='Measurements')
            plt.fill_between(t, confidence_interval_lower, confidence_interval_upper, color='y', alpha=0.5, label='95% Confidence Interval')

        else:
            plt.plot(t, theta, 'g')
            plt.plot(t, kalman_estimates, 'r-')
            plt.plot(np.arange(0, monte_time, (1/measurement_hz)), measurement_time_steps, 'b.')
            plt.fill_between(t, confidence_interval_lower, confidence_interval_upper, color='y', alpha=0.5)

    # Add labels and legend
    plt.xlabel('time [s]')
    plt.ylabel('theta [rad]')

    if not external_force:
        plt.title('EKF Monte Results, %d [Hz] Pred, %d [Hz] Meas, Avg MAE: %.3f' % (prediction_hz, measurement_hz, ekf_simulation_summary['avg_mae']))

    else:
        plt.title('EKF External Force Monte Results, %d [Hz] Pred, %d [Hz] Meas, Avg MAE: %.3f' % (prediction_hz, measurement_hz, ekf_simulation_summary['avg_mae']))

    plt.legend()
    plt.grid(True)

    if not external_force:
        plt.savefig("ekf_monte_results_%d_hz.png" % prediction_hz)

    else:
        plt.savefig("ekf_external_force_monte_results_%d_hz.png" %prediction_hz)

def plot_kalman_error(monte_runs, ekf_simulation_summary, external_force=False):

    # Time array
    t = np.arange(0, simulation_time, (1/prediction_hz)) if not external_force else np.arange(0, force_simulation_time, (1/prediction_hz))
    plt.figure(2)

    time_indices = np.arange(int(simulation_time/dt)) % int((1/prediction_hz)/dt)  == 0 if not external_force else np.arange(int(force_simulation_time/dt)) % int((1/prediction_hz)/dt)  == 0
    monte_kalman_estimates = ekf_simulation_summary['ekf_estimates']
    monte_covariance_time_steps = ekf_simulation_summary['covariance']

    for run_idx in range(NUM_MONTE_RUNS):

        theta = monte_runs[run_idx][:, 0][time_indices]
        kalman_estimates = monte_kalman_estimates[run_idx][:]
        covariance = monte_covariance_time_steps[run_idx][:]

        error = theta - kalman_estimates

        # Calculate confidence intervals
        confidence_interval_upper = error + 3 * covariance
        confidence_interval_lower = error - 3 * covariance

        if run_idx == 0:
            plt.plot(t, error, 'k', label='EKF Estimate Error')
            plt.fill_between(t, confidence_interval_lower, confidence_interval_upper, color='y', alpha=0.5, label='3-σ Uncertainty in Theta')

        else:
            plt.plot(t, error, 'k')
            plt.fill_between(t, confidence_interval_lower, confidence_interval_upper, color='y', alpha=0.5)

    # Add labels and legend
    plt.xlabel('time [s]')
    plt.ylabel('Error [rad]')

    if not external_force:
        plt.title('EKF Monte Error vs Time with %d [Hz] Pred, %d [Hz] Meas,' % (prediction_hz, measurement_hz))

    else:
        plt.title('EKF With External Force Monte Error vs Time  with %d [Hz] Pred, %d [Hz] Meas,' % (prediction_hz, measurement_hz))
    
    plt.legend()
    plt.grid(True)

    if not external_force:
        plt.savefig("ekf_monte_error_%d_hz.png" % prediction_hz)

    else:
        plt.savefig("ekf_external_force_monte_error_%d_hz.png" % prediction_hz)

def plot_kalman_gain(ekf_simulation_summary, figure_num=1, external_force=False):

    plt.figure(figure_num)
    t = np.arange(0, simulation_time, (1/prediction_hz)) if not external_force else np.arange(0, force_simulation_time, (1/prediction_hz))    
    
    for run_idx in range(NUM_MONTE_RUNS):

        k_n = ekf_simulation_summary['kalman_gain_norm'][run_idx][:]

        if run_idx == 0:
            plt.plot(t, k_n, 'k', label='Kalman Gain Norm')

        else:
            plt.plot(t, k_n, 'k')

    # Add labels and legend
    plt.xlabel('time [s]')
    plt.ylabel('Kalman Gain Norm')

    if not external_force:
        plt.title('Kalman Gain Norm vs Time, %d [Hz] Mesurements' % prediction_hz)

    else:
        plt.title('External Force Kalman Gain Norm vs Time, %d [Hz] Mesurements' % prediction_hz)
    
    plt.legend()
    plt.grid(True)

    if not external_force:
        plt.savefig("kalman_gain_summary_%d_hz.png" % prediction_hz)

    else:
        plt.savefig("ex_force_kalman_gain_summary_%d_hz.png" % prediction_hz)

def plot_theta_vs_prediction_and_gain(monte_runs, ekf_simulation_summary, fig_num=1, external_force=False, save_figs=False):

    # Create a new figure with figure number 6
    plt.figure(fig_num)

    t = np.arange(0, simulation_time, (1/prediction_hz)) if not external_force else np.arange(0, force_simulation_time, (1/prediction_hz))
    time_indices = np.arange(int(simulation_time/dt)) % int((1/prediction_hz)/dt)  == 0 if not external_force else np.arange(int(force_simulation_time/dt)) % int((1/prediction_hz)/dt)  == 0

    theta = monte_runs[0][:, 0][time_indices]
    theta_prediction = ekf_simulation_summary['state_predictions'][0][:]
    kalman_gains = ekf_simulation_summary['kalman_gain_norm'][0][:]

    ########### true theta vs state prediction ###########
    plt.plot(t, theta, 'g', label='True Theta')
    plt.plot(t, theta_prediction, 'p', label='State Predictions')

    # Add labels and legend
    plt.xlabel('time [s]')
    plt.ylabel('theta [rad]')
    plt.legend()
    plt.grid(True)
    plt.title('True theta vs theta predictions: %d [Hz] Measurements' % prediction_hz)

    if save_figs:
        plt.savefig("theta_vs_theta_predictions_%d_hz.png" % prediction_hz)

    ########### true theta vs state prediction ###########
    plt.figure(fig_num+1)
    plt.plot(t, kalman_gains, label='Kalman Gain Norms')
    plt.xlabel('time [s]')
    plt.ylabel('Kalman Gain Norm')
    plt.legend()
    plt.grid(True)
    plt.title('Kalman Gain Norm vs Time: %d [Hz] Measurements' % prediction_hz)

    if save_figs:
        plt.savefig("kalman_norm_summary_%d_hz.png" % prediction_hz)


if __name__ == "__main__":

    # create monte data
    monte_runs = generate_monte_runs()
    monte_runs = add_noise_to_monte_runs(monte_runs)

    # do ekf monte sim
    ekf_simulation_summary = kalman_filter_simulation(monte_runs)

    # show results
    ekf_simulation_summary = assess_kalman_accuracy(monte_runs, ekf_simulation_summary)
    plot_kalman_results(monte_runs, ekf_simulation_summary)
    plot_kalman_error(monte_runs, ekf_simulation_summary)
    plt.show()


    ################# external force simulation #################
    # create monte data
    ex_force_monte_runs = generate_monte_runs(external_force=True)
    ex_force_monte_runs = add_noise_to_monte_runs(ex_force_monte_runs)

    # do ekf monte sim
    ekf_ex_force_simulation_summary = kalman_filter_simulation(ex_force_monte_runs, external_force=True)

    # show results
    ekf_ex_force_simulation_summary = assess_kalman_accuracy(ex_force_monte_runs, ekf_ex_force_simulation_summary, external_force=True)
    plot_kalman_results(ex_force_monte_runs, ekf_ex_force_simulation_summary, external_force=True)
    plot_kalman_error(ex_force_monte_runs, ekf_ex_force_simulation_summary, external_force=True)
    plt.show()
