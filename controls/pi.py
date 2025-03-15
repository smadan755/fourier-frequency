import numpy as np
import control as ctrl

import matplotlib

matplotlib.use('TkAgg')  # Set the backend before importing pyplot


import matplotlib.pyplot as plt

# (a) Define the original system matrices without feedback
# We'll create the system in the form ẋ = Ax + Bu
A_open = np.array([
    [0, 1, 0],
    [-1, -1, 0],
    [-1, 0, 0]
])

B_open = np.array([
    [0],
    [1],
    [0]
])

# Define the output matrix
C = np.array([[1, 0, 0]])
D = np.array([[0]])

# (b) Use control.place() for eigenvalues -1+1i, -1-1i, and -10
desired_poles_b = [-1+1j, -1-1j, -10]

# Calculate the feedback gain matrix K using pole placement
K_b = ctrl.place(A_open, B_open, desired_poles_b)
print("Feedback gain matrix K for system (b):", K_b)

# In our problem, the feedback structure is defined by the matrices A and B
# We need to map the gains from control.place() to our kp, ki, kd parameters

# Based on our system structure, the relationship would be:
# K = [kp+1, kd+1, -ki]
kp_b = K_b[0][0] - 1
kd_b = K_b[0][1] - 1
ki_b = -K_b[0][2]

print(f"System (b) gains: kp = {kp_b}, ki = {ki_b}, kd = {kd_b}")

# Create the closed-loop system matrices for (b)
A_b = np.array([
    [0, 1, 0],
    [-(1+kp_b), -(1+kd_b), ki_b],
    [-1, 0, 0]
])

B_b = np.array([[0], [kp_b], [1]])

# Verify eigenvalues for system (b)
eigs_b = np.linalg.eigvals(A_b)
print("Eigenvalues of closed-loop system (b):", eigs_b)

# (c) Use control.place() for eigenvalues -1+1i, -1-1i, and -15
desired_poles_c = [-1+1j, -1-1j, -15]

# Calculate feedback gain matrix K for system (c)
K_c = ctrl.place(A_open, B_open, desired_poles_c)
print("\nFeedback gain matrix K for system (c):", K_c)

# Map to kp, ki, kd
kp_c = K_c[0][0] - 1
kd_c = K_c[0][1] - 1
ki_c = -K_c[0][2]

print(f"System (c) gains: kp = {kp_c}, ki = {ki_c}, kd = {kd_c}")

# Create the closed-loop system matrices for (c)
A_c = np.array([
    [0, 1, 0],
    [-(1+kp_c), -(1+kd_c), ki_c],
    [-1, 0, 0]
])

B_c = np.array([[0], [kp_c], [1]])

# Verify eigenvalues for system (c)
eigs_c = np.linalg.eigvals(A_c)
print("Eigenvalues of closed-loop system (c):", eigs_c)

# (d) Create state-space models and simulate responses
sys_b = ctrl.ss(A_b, B_b, C, D)
sys_c = ctrl.ss(A_c, B_c, C, D)

# Step response
t = np.linspace(0, 10, 1000)
y_b, t_b = ctrl.step_response(sys_b, T=t)
y_c, t_c = ctrl.step_response(sys_c, T=t)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(y_b.T,t_b,  'b-', label='System (b): λ₃ = -10')
plt.plot( y_c.T,t_c, 'r--', label='System (c): λ₃ = -15')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Output y(t)')
plt.title('Step Response Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('step_response_comparison.png')  # Save figure for viewing
plt.show()

# Additional analysis of the differences
print("\n--- Analysis of differences in step responses ---")
print(f"Peak value (b): {np.max(y_b)}")
print(f"Peak value (c): {np.max(y_c)}")
print(f"Settling time (b): {t_b[np.where(np.abs(y_b-1) < 0.02)[0][0]] if any(np.abs(y_b-1) < 0.02) else 'N/A'}")
print(f"Settling time (c): {t_c[np.where(np.abs(y_c-1) < 0.02)[0][0]] if any(np.abs(y_c-1) < 0.02) else 'N/A'}")