import matplotlib

matplotlib.use('TkAgg')  # Set the backend before importing pyplot

import numpy as np
import matplotlib.pyplot as plt


def main():
    rect = [0.25,0.5,0.75, 1, 1.25, 1.5,1.75]
    impulse_train = []

    for i in range(100):
        if (i % 10 == 0):
            impulse_train.append(1)
        else:
            impulse_train.append(0)

    pad_len = len(impulse_train) + len(rect) - 1
    diff_rect = abs(len(rect) - pad_len)
    diff_impulse_train = abs(len(impulse_train) - pad_len)

    rect = np.pad(rect, pad_width=(0, diff_rect), mode='constant', constant_values=0)
    impulse_train = np.pad(impulse_train, pad_width=(0, diff_impulse_train), mode='constant', constant_values=0)

    NUM_SAMPLES = len(rect)

    # Plot the original signals
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.stem(np.arange(NUM_SAMPLES), rect)
    plt.title('Rectangle Signal')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.stem(np.arange(NUM_SAMPLES), impulse_train)
    plt.title('Impulse Train')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Compute DFT
    dft_matrix = np.zeros((NUM_SAMPLES, NUM_SAMPLES), dtype=complex)
    w = np.exp(-1j * ((2 * np.pi) / NUM_SAMPLES))
    for k in range(NUM_SAMPLES):
        for n in range(NUM_SAMPLES):
            dft_matrix[k][n] = w ** (k * n)

    dft_rect = np.dot(dft_matrix, rect)
    dft_impulse_train = np.dot(dft_matrix, impulse_train)

    # Plot magnitude of DFTs
    plt.subplot(3, 2, 3)
    plt.stem(np.arange(NUM_SAMPLES), np.abs(dft_rect))
    plt.title('Magnitude of Rectangle DFT')
    plt.xlabel('k')
    plt.ylabel('|X(k)|')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.stem(np.arange(NUM_SAMPLES), np.abs(dft_impulse_train))
    plt.title('Magnitude of Impulse Train DFT')
    plt.xlabel('k')
    plt.ylabel('|X(k)|')
    plt.grid(True)

    # Compute product in frequency domain (convolution in time)
    dft_product = []
    for i in range(NUM_SAMPLES):
        dft_product.append(dft_rect[i] * dft_impulse_train[i])

    # Compute IDFT
    idft_matrix = np.zeros((NUM_SAMPLES, NUM_SAMPLES), dtype=complex)
    neg_w = np.exp(1j * ((2 * np.pi) / NUM_SAMPLES))
    for k in range(NUM_SAMPLES):
        for n in range(NUM_SAMPLES):
            idft_matrix[k][n] = neg_w ** (k * n)

    output = np.dot(idft_matrix, dft_product) / NUM_SAMPLES

    # Plot result of convolution
    plt.subplot(3, 2, 5)
    plt.stem(np.arange(NUM_SAMPLES), np.abs(dft_product))
    plt.title('Magnitude of DFT Product')
    plt.xlabel('k')
    plt.ylabel('|X(k)Y(k)|')
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.stem(np.arange(NUM_SAMPLES), np.real(output))
    plt.title('Convolution Result (IDFT)')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('convolution_results.png')
    plt.show()


if __name__ == "__main__":
    main()