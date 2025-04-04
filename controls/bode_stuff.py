import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import control as ctrl

# Create the figure for Bode plots
plt.figure(figsize=(12, 10))

# Magnitude subplot
plt.subplot(2, 1, 1)
plt.grid(True, which='both')
plt.title('Log-Log Magnitude')
plt.xlabel('ω (rad/s)')
plt.ylabel('|G(jω)|')

# Phase subplot
plt.subplot(2, 1, 2)
plt.grid(True, which='both')
plt.title('Log-Lin Phase')
plt.xlabel('ω (rad/s)')
plt.ylabel('∠G(jω) [degrees]')

# Frequency range for all plots
w = np.logspace(-1, 2, 1000)

# Define gain values and create a color map for better visualization
k_values = np.linspace(0.01, 0.10, 10)
colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))

# Loop through different gain values
for i, gain in enumerate(k_values):
    # Define transfer function G(s) = 101 / (s^2 + (2 + 101*gain)s + 101)
    num = [101]
    den = [1, (2 + 101 * gain), 101]
    G = ctrl.TransferFunction(num, den)

    # Get Bode plot data
    mag, phase, omega = ctrl.bode(G, w, dB=False, deg=True, plot=False)

    # Plot magnitude on the first subplot
    plt.subplot(2, 1, 1)
    plt.loglog(omega, mag, color=colors[i], label=f'k = {gain:.2f}')

    # Plot phase on the second subplot
    plt.subplot(2, 1, 2)
    plt.semilogx(omega, phase, color=colors[i], label=f'k = {gain:.2f}')

# Add legends to both subplots
plt.subplot(2, 1, 1)
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.legend(loc='best')

plt.tight_layout()

# Create a new figure for step responses
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.title('Step Response Comparison')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

# Time range for step response
t = np.linspace(0, 5, 1000)

# Case 1: The system with k = 0.1 (damped)
num_damped = [101]
den_damped = [1, (2 + 101 * 0.10), 101]  # k = 0.1
G_damped = ctrl.TransferFunction(num_damped, den_damped)

# Case 2: The undamped system (setting damping term to minimum)
num_undamped = [101]
den_undamped = [1, 2, 101]  # Original system without additional damping
G_undamped = ctrl.TransferFunction(num_undamped, den_undamped)

# Compute and plot step responses
t_damped, y_damped = ctrl.step_response(G_damped, T=t)
t_undamped, y_undamped = ctrl.step_response(G_undamped, T=t)

plt.plot(t_damped, y_damped, 'b-', linewidth=2, label='Damped (k = 0.10)')
plt.plot(t_undamped, y_undamped, 'r-', linewidth=2, label='Undamped (k = 0)')

plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plt.show()