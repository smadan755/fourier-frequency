import numpy as np
import control as ct

import matplotlib
matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
a = 1.0
bg = 1.0
b = 1.0
t = np.linspace(0, 40, 1000)
d = np.piecewise(t, [t < 5, t >= 5], [0, 6])

A_a = np.array([[-a]])
B_a = np.array([[-bg]])
C_a = np.array([[1]])
D_a = 0
sys_a = ct.ss(A_a, B_a, C_a, D_a)

t_out, y_a = ct.forced_response(sys_a, T=t, U=d)
plt.figure(figsize=(10, 6))

plt.plot(t_out, y_a, label="(a) w=0")

A_b = np.array([[-(a + 0.5*b)]])
B_b = np.array([[-bg]])
C_b = np.array([[1]])
D_b = 0
sys_b = ct.ss(A_b, B_b, C_b, D_b)

t_out, y_b = ct.forced_response(sys_b, T=t, U=d)

plt.plot(t_out, y_b, label="(b) w=-0.5*x")

A_c = np.array([
    [-(a + 0.5*b), -0.1*b],
    [          1.0,    0.0]
])
B_c = np.array([
    [-bg],
    [  0 ]
])
C_c = np.array([[1, 0]])
D_c = 0
sys_c = ct.ss(A_c, B_c, C_c, D_c)

t_out, y_c = ct.forced_response(sys_c, T=t, U=d)

plt.plot(t_out, y_c, label="(c) w=-0.5*x -0.1*z")

plt.title("Cruise Control: Velocity Deviation x(t)")
plt.xlabel("Time [s]")
plt.ylabel("x(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()