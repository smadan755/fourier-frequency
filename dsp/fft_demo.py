import matplotlib

matplotlib.use('TkAgg')  # Set the backend before importing pyplot

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Input sequence
    seq = [1, 0, -1, 0, 1, 0, -1, 0, 1, -1,0,1,0,-1,0,1,0,-1]
    NUM_SAMPLES = len(seq)

    # Create the DFT matrix
    dft_matrix = np.zeros((NUM_SAMPLES, NUM_SAMPLES), dtype=complex)
    for k in range(NUM_SAMPLES):
        for n in range(NUM_SAMPLES):
            # Correct DFT matrix calculation
            dft_matrix[k][n] = np.exp(-1j * 2 * np.pi * k * n / NUM_SAMPLES)

    # Apply the DFT matrix to the sequence
    dft_output = np.dot(dft_matrix, seq)

    # Create separate plots for magnitude and phase
    plt.figure(figsize=(12, 8))

    # Plot magnitude
    plt.subplot(2, 1, 1)
    plt.stem(range(NUM_SAMPLES), np.abs(dft_output))
    plt.title('DFT Magnitude')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    plt.grid(True)

    # Plot phase
    plt.subplot(2, 1, 2)
    plt.stem(range(NUM_SAMPLES), np.angle(dft_output))
    plt.title('DFT Phase')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Phase (radians)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Optional: Print the DFT values
    print("DFT Output:")
    for i, val in enumerate(dft_output):
        print(f"X[{i}] = {val:.4f}")


if __name__ == "__main__":
    main()