# library imports
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Define the pendulum dynamics function
def pendulum_dynamics(y, t, g, L, gamma):

    theta, omega = y
    dtheta_dt = omega
    domega_dt = - (g / L) * np.sin(theta) - (gamma/mass) * omega

    return [ dtheta_dt, domega_dt ]

# Initial conditions
y0 = [np.pi / 4, 0]  # Initial angle (45 degrees) and initial angular velocity (0)

# dynamic constants
mass = 1 # mass of ball g   
g = 9.81  # gravitational acceleration (m/s^2)
L = 1.0   # length of the pendulum (m)
gamma = 1 # damping coefficient
process_noise_std = 0.01

# Time array
t = np.arange(0, 10, 0.05)

# Solve the ODE 
solution = odeint(pendulum_dynamics, y0, t, args=(g, L, gamma))

# Extract the results
theta = solution[:, 0]  # Angle theta
theta_dot = solution[:, 1]  # Angular velocity

# Add process noise to the results
theta_noisy = theta + np.random.normal(0, process_noise_std, len(theta))
theta_dot_noisy = theta_dot + np.random.normal(0, process_noise_std, len(theta_dot))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(t, theta, label='Theta (angle)')
plt.plot(t, theta_dot, label='Theta_dot (angular velocity)', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('Pendulum: Theta and Theta Dot')
plt.legend()
plt.grid()
plt.show()

# Convert theta to Cartesian coordinates for animation
x = L * np.sin(theta)
y = -L * np.cos(theta)

# Create figure and axis
fig, ax = plt.subplots()

# Set axis limits
ax.set_xlim(-L - 1, L + 1)
ax.set_ylim(-L - 1, 1)

# Create pendulum arm and bob (pre-allocate)
line, = ax.plot([], [], color='blue', label='Pendulum Arm')  # Line for the arm
circle = plt.Circle((0, -L), 0.075, color='red', label='Pendulum Ball')  # Circle for the bob
ax.add_patch(circle)
ax.legend(loc='upper left')

# Initialize the plot elements (rod and bob)
def init():
    line.set_data([], [])
    circle.set_center((0, -L))  # Set initial position of the circle (bob)
    return line, circle

# Update function for animation
def update(i):
    line.set_data([0, x[i]], [0, y[i]])  # Update the pendulum arm
    circle.set_center((x[i], y[i]))  # Update the pendulum bob
    return line, circle

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=50, repeat=True)

# Display the animation
plt.show()
