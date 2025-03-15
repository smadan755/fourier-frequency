import numpy as np
import control

import matplotlib
matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt

A = np.array([
    [0,    1,    0 ],
    [-1,  -1.8,  0 ],
    [-1,   0,    0 ]
])
B = np.array([
    [0],
    [1],
    [0]
])

p1 = -0.9 + 0.4359j
p2 = -0.9 - 0.4359j
p3 = -10

K = control.place(A, B, [p1, p2, p3])
print("Gain vector K =", K)

A_cl = A - B @ K
C = np.array([[1, 0, 0]])
D = 0
sys_cl = control.ss(A_cl, B, C, D)

A_2 = np.array([[0, 1],
                [-1, -1.8]])
B_2 = np.array([[0],[1]])
C_2 = np.array([[1, 0]])
D_2 = 0
sys_2 = control.ss(A_2, B_2, C_2, D_2)

T = np.linspace(0, 10, 300)
t, y = control.step_response(sys_cl, T)
t2, y2 = control.step_response(sys_2, T)

plt.figure(figsize=(10, 6))
plt.plot(t2, y2, label='2nd-order system', linestyle='--')
plt.plot(t, y, label='3rd-order with integrator')
plt.xlabel("Time [s]")
plt.ylabel("y(t)")
plt.title("Comparison of 2nd-order vs. 3rd-order (with integral)")
plt.grid(True)
plt.legend()
plt.show()