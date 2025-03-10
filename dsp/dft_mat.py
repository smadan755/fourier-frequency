import numpy as np
import matplotlib as plt
from complex import clean_complex


x = np.array([1,1,1,0])

w = np.exp(-1j * (2*np.pi)/len(x))

DFT_mat = np.array([
    [1, 1, 1, 1],
    [1, w, w**2, w**3],
    [1, w**2, w**4, w**6],
    [1, w**3, w**6, w**9]
])

print(f"This is the input sequence: {x}")


dft_x = clean_complex(np.dot(DFT_mat,x))


print(f"This is the DFT of x: {dft_x}")


print("This is cool, but lets go back")


w = np.exp(1j * (2*np.pi)/len(x))

iDFT_mat = np.array([
    [1, 1, 1, 1],
    [1, w, w**2, w**3],
    [1, w**2, w**4, w**6],
    [1, w**3, w**6, w**9]
])


print(f"This is the iDFT of x: {clean_complex(np.dot(iDFT_mat,dft_x))/4}")

