import matplotlib

matplotlib.use('TkAgg')  # Set the backend before importing pyplot

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Let's analyze a sine function sampled at discrete intervals

    # Create an array of time points
    num_samples = 400
    t = np.linspace(0, 50, num_samples)  # 100 points from 0 to 1

    # Choose a frequency for the samples

    # Generate the composite sine wave with two frequency components
    sin_sampled = np.sin(np.pi * 1/8 * t) + np.sin(np.pi * 21/50 * t)

    # Compute the DFT manually
    w = np.exp(-1j * ((2 * np.pi) / len(sin_sampled)))
    dft_matrix = np.zeros((num_samples, num_samples), dtype=complex)

    for k in range(num_samples):
        for n in range(num_samples):
            dft_matrix[k][n] = w ** (k * n)

    dft_x = np.dot(dft_matrix, sin_sampled)
    dft_magnitude = abs(dft_x)  # Get magnitude spectrum

    # Compute frequency bins and shift for better visualization
    freqs = np.fft.fftfreq(len(sin_sampled), d=t[1] - t[0])
    freqs_shifted = np.fft.fftshift(freqs)
    dft_magnitude_shifted = np.fft.fftshift(dft_magnitude)

    # Create a figure with two subplots (one above the other)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot the time domain signal
    ax1.plot(t, sin_sampled, 'b-')
    ax1.set_title(f'Time Domain: Composite Sine Wave ')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)

    # Plot the frequency domain (DFT result)
    ax2.stem(freqs_shifted, dft_magnitude_shifted, 'r', markerfmt='ro', basefmt='k-')
    ax2.set_title('Frequency Domain: DFT Magnitude Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True)



    # Adjust the layout to prevent overlapping
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()