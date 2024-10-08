# library imports
import numpy as np
import matplotlib.pyplot as plt


######### constants #########
t0 = 0
x_0 = np.array([ -2, -5, 0, 0, 0, 0 ])
x_dot = np.array([0, 0, 0, 0, 0, 0])

simulation_time = 4
simulation_hz = 20
measurement_hz = 20
dt = 1 / simulation_hz

measurement_std = 0.01

beacons = np.array([[ -1, 1.25, 0 ],
                    [ 0, 0, 2.25 ],
                    [ 1, 1.25, 0 ]])

gd_convergence = 0.1
gd_learning_rate = 0.1
gd_max_iteration = 50


def tracking_object_ode(simulation_time, dt, x_0, x_dot):

    states = np.zeros(shape=(int(simulation_time/dt),  x_0.shape[0]))
    states[0] = x_0

    for idx in range(1, int(simulation_time/dt)):

        states[idx] = states[idx-1] + x_dot * dt

    return states

def update_estimate_radius(estimate_n):

    return np.linalg.norm(beacons - estimate_n, axis=1)

def update_delta_r_vector(measurement_buffer, estimate_radius):

    return  measurement_buffer - estimate_radius

def update_measurement_matrix(estimate_n, estimate_radius):

    return np.array([[ ( beacons[0][0] - estimate_n[0]) / estimate_radius[0], ( beacons[0][1] - estimate_n[1]) / estimate_radius[0], ( beacons[0][2] - estimate_n[2]) / estimate_radius[0] ],
                     [ ( beacons[1][0] - estimate_n[0]) / estimate_radius[1], ( beacons[1][1] - estimate_n[1]) / estimate_radius[1], ( beacons[1][2] - estimate_n[2]) / estimate_radius[1] ],
                     [ ( beacons[2][0] - estimate_n[0]) / estimate_radius[2], ( beacons[2][1] - estimate_n[1]) / estimate_radius[2], ( beacons[2][2] - estimate_n[2]) / estimate_radius[2] ]])

def update_estimate(estimate_n, delta_x):

    return delta_x + estimate_n

def estimate_position_least_squares(estimate_n, measurement_buffer):

    estimate_radius = update_estimate_radius(estimate_n)
    delta_r = update_delta_r_vector(measurement_buffer, estimate_radius)
    measurement_matrix = update_measurement_matrix(estimate_n, estimate_radius)

    delta_x_n = np.linalg.inv( np.transpose(measurement_matrix) @ measurement_matrix ) @ np.transpose(measurement_matrix) @ delta_r

    return update_estimate(estimate_n, delta_x_n)

def estimate_position_gradient_descent(estimate_n, measurement_buffer):

    error = 100
    iteration = 0

    estimate_radius = update_estimate_radius(estimate_n)
    delta_r = update_delta_r_vector(measurement_buffer, estimate_radius)
    measurement_matrix = update_measurement_matrix(estimate_n, estimate_radius)
    delta_x_n = np.array([0.1, 0.1, 0.1])

    while error > gd_convergence and iteration <= gd_max_iteration:

        delta_x_n = delta_x_n - gd_learning_rate * measurement_matrix @ ( delta_r - measurement_matrix @ delta_x_n )
        
        estimate_n = update_estimate(estimate_n, delta_x_n)
        estimate_radius = update_estimate_radius(estimate_n)
        delta_r = update_delta_r_vector(measurement_buffer, estimate_radius)
        measurement_matrix = update_measurement_matrix(estimate_n, estimate_radius)

        error = np.linalg.norm( delta_r - measurement_matrix @ delta_x_n )
        iteration += 1

    return estimate_n

def getting_measurements_sim(states, dt):

    curr_time = 0
    curr_beacon_idx = 0
    estimate_idx = 0

    max_estimates = int( simulation_time / ( 3 /  measurement_hz ) )
    time_for_estimate_idx = simulation_hz / measurement_hz

    ls_estimates = np.zeros(shape=(max_estimates, 3))
    gd_estimates = np.zeros(shape=(max_estimates, 3))
    true_positions = np.zeros(shape=(max_estimates, 3))

    ls_estimate_n = np.array([-1.9999, -4.999999, 0])
    gd_estimate_n = np.array([0, -1, 0])

    measurement_buffer = np.zeros(3)

    for idx in range(states.shape[0]):

        if idx % time_for_estimate_idx == 0:

            measurement_buffer[curr_beacon_idx] = np.linalg.norm( beacons[curr_beacon_idx] - states[idx][:3] ) + np.random.normal(0, measurement_std)

            curr_beacon_idx = ( curr_beacon_idx + 1 ) % 3

            if curr_beacon_idx == 0:

                ls_estimate_n = estimate_position_least_squares(ls_estimate_n, measurement_buffer)
                gd_estimate_n = estimate_position_gradient_descent(gd_estimate_n, measurement_buffer)

                ls_estimates[estimate_idx] = ls_estimate_n
                gd_estimates[estimate_idx] = gd_estimate_n
                true_positions[estimate_idx] = states[idx][:3]

                estimate_idx += 1

        curr_time += dt

    return true_positions, ls_estimates, gd_estimates


if __name__ == "__main__":

    states = tracking_object_ode(simulation_time, dt, x_0, x_dot)

    true_positions, ls_estimates, gd_estimates = getting_measurements_sim(states, dt)

    plt.plot(np.arange(0, 26), true_positions[:, 0], 'g')
    plt.plot(np.arange(0, 26), gd_estimates[:, 0], 'y')
    plt.show()