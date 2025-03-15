import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
from time import time


def main():
    # Parameters
    num_samples = 100

    # Create a binary data sequence (e.g., 0101...)
    data = np.zeros(num_samples)
    for i in range(0, num_samples, 10):
        data[i:i + 5] = 1  # Create pulse with width 5

    # Create different pulse shapes

    # 1. Rectangular pulse
    rect_pulse = np.ones(10)

    # 2. Raised cosine pulse
    t = np.linspace(-1, 1, 20)
    rc_pulse = 0.5 * (1 + np.cos(np.pi * t))

    # 3. Gaussian pulse
    t = np.linspace(-2, 2, 20)
    gaussian_pulse = np.exp(-t ** 2)

    # Plot the data sequence
    plt.figure(figsize=(12, 12))
    plt.subplot(5, 1, 1)
    plt.stem(range(len(data)), data)
    plt.title("Original Data Sequence")

    # Plot pulse shapes
    plt.subplot(5, 1, 2)
    plt.stem(range(len(rect_pulse)), rect_pulse, 'r-')
    plt.title("Rectangular Pulse")

    plt.subplot(5, 1, 3)
    plt.stem(range(len(rc_pulse)), rc_pulse, 'g-')
    plt.title("Raised Cosine Pulse")

    plt.subplot(5, 1, 4)
    plt.stem(range(len(gaussian_pulse)), gaussian_pulse, 'm-')
    plt.title("Gaussian Pulse")

    # Choose one pulse shape for convolution (let's use Gaussian)
    pulse = gaussian_pulse

    # Measure performance
    start_time = time()

    # Perform convolution via DFT
    # 1. Zero-pad signals to avoid circular wrap-around effects
    N = len(data) + len(pulse) - 1
    data_padded = np.pad(data, (0, N - len(data)), 'constant')
    pulse_padded = np.pad(pulse, (0, N - len(pulse)), 'constant')

    # 2. Compute DFTs
    data_dft = np.fft.fft(data_padded)
    pulse_dft = np.fft.fft(pulse_padded)

    # 3. Multiply in frequency domain
    product_dft = data_dft * pulse_dft

    # 4. Inverse DFT to get the convolution result
    convolution_result = np.fft.ifft(product_dft).real

    end_time = time()
    dft_time = end_time - start_time

    # For comparison, calculate using direct convolution
    start_time = time()
    direct_conv = np.convolve(data, pulse, 'full')
    end_time = time()
    direct_time = end_time - start_time

    # Plot the result
    plt.subplot(5, 1, 5)
    plt.plot(range(len(convolution_result)), convolution_result)
    plt.title("Pulse-Shaped Signal (Convolution via DFT)")
    plt.tight_layout()
    plt.show()

    # Compare results and performance
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(len(convolution_result)), convolution_result, 'b-', label=f'DFT Method ({dft_time:.6f}s)')
    plt.plot(range(len(direct_conv)), direct_conv, 'r--', label=f'Direct Method ({direct_time:.6f}s)')
    plt.title("Comparison of Convolution Methods")
    plt.legend()

    # Difference between methods
    plt.subplot(2, 1, 2)
    min_len = min(len(convolution_result), len(direct_conv))
    diff = convolution_result[:min_len] - direct_conv[:min_len]
    plt.plot(range(min_len), diff)
    plt.title(f"Difference (Max Error: {np.max(np.abs(diff)):.10f})")
    plt.tight_layout()
    plt.show()

    # Visualize the spectrum
    plt.figure(figsize=(12, 9))

    # Original data spectrum
    plt.subplot(3, 1, 1)
    data_spectrum = np.abs(np.fft.fftshift(np.fft.fft(data_padded)))
    freq = np.fft.fftshift(np.fft.fftfreq(len(data_padded)))
    plt.plot(freq, data_spectrum)
    plt.title("Spectrum of Data Signal")

    # Pulse spectrum
    plt.subplot(3, 1, 2)
    pulse_spectrum = np.abs(np.fft.fftshift(np.fft.fft(pulse_padded)))
    plt.plot(freq, pulse_spectrum)
    plt.title("Spectrum of Pulse Shape")

    # Result spectrum
    plt.subplot(3, 1, 3)
    result_spectrum = np.abs(np.fft.fftshift(np.fft.fft(convolution_result)))
    plt.plot(freq, result_spectrum)
    plt.title("Spectrum of Convolved Signal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()