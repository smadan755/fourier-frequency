import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import control as ctrl

# Time range for step response
t = np.linspace(0, 50, 1000)

# Performance constraints and targets
target_rise_time = 0.5  # Target rise time (seconds)
max_settling_time = 2.0  # Maximum allowed settling time (seconds)
max_overshoot = 10.0  # Maximum allowed percent overshoot

# Define ranges for parameter sweeps
Kp_values = np.linspace(10, 50, 21)  # Test Kp from 10 to 50 with 21 points
Ki_values = np.linspace(1, 20, 20)  # Test Ki from 1 to 20 with 20 points
Kd_values = np.linspace(0, 20, 21)  # Test Kd from 0 to 20 with 21 points

# Create figure for final results
plt.figure(figsize=(16, 12))


# Define utility functions
def calculate_rise_time(t, y):
    steady_state = y[-1]
    rise_10_idx = np.where(y >= 0.1 * steady_state)[0]
    rise_90_idx = np.where(y >= 0.9 * steady_state)[0]

    if len(rise_10_idx) > 0 and len(rise_90_idx) > 0:
        return t[rise_90_idx[0]] - t[rise_10_idx[0]]
    else:
        return float('inf')  # Return infinity if rise time can't be calculated


def calculate_settling_time(t, y, threshold=0.02):
    steady_state = y[-1]
    settling_idx = np.where(np.abs(y - steady_state) <= threshold * steady_state)[0]
    if len(settling_idx) > 0:
        return t[settling_idx[0]]
    else:
        return float('inf')  # Return infinity if settling time can't be calculated


def calculate_overshoot(y):
    steady_state = y[-1]
    peak = np.max(y)
    if steady_state > 0:
        return ((peak - steady_state) / steady_state) * 100
    else:
        return 0


def calculate_ss_error(y, target=1.0):
    steady_state = y[-1]
    error = abs(target - steady_state)
    return error


def evaluate_response(t_response, y_response):
    rise_time = calculate_rise_time(t_response, y_response)
    settling_time = calculate_settling_time(t_response, y_response)
    overshoot = calculate_overshoot(y_response)
    ss_error = calculate_ss_error(y_response)
    return {
        'rise_time': rise_time,
        'settling_time': settling_time,
        'overshoot': overshoot,
        'ss_error': ss_error
    }


def plot_response(t, y, ax, title, metrics=None):
    ax.plot(t, y, 'b-', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Target')

    # If we have metrics, add annotations
    if metrics:
        steady_state = y[-1]
        ax.axhline(y=steady_state, color='k', linestyle='--', alpha=0.5, label='Steady State')

        # Mark peak overshoot if there is any
        if metrics['overshoot'] > 0:
            peak = np.max(y)
            peak_idx = np.argmax(y)
            ax.plot(t[peak_idx], peak, 'ro', markersize=6)

        # Add info text
        info_text = f"Rise Time: {metrics['rise_time']:.4f}s\n"
        info_text += f"Settling Time: {metrics['settling_time']:.4f}s\n"
        info_text += f"Overshoot: {metrics['overshoot']:.2f}%\n"
        info_text += f"SS Error: {metrics['ss_error']:.6f}"

        ax.annotate(info_text, xy=(0.65, 0.25), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend(loc='best')


# Stage 1: Sweep Kp (with Ki=0, Kd=0) to get closest to target rise time
print("Stage 1: Sweeping Kp (with Ki=0, Kd=0)...")

# Initialize variables for Stage 1
Kp_rise_times = []
best_kp = None
best_kp_rise_time_diff = float('inf')
best_kp_sys = None
best_kp_response = None
best_kp_time = None

for Kp in Kp_values:
    # Set Ki and Kd to 0 for this stage
    Ki = 0
    Kd = 0

    # Update the transfer function with the current Kp
    num = [Kd, Kp, Ki]  # Kd*s² + Kp*s + Ki
    den = [1, 6, 11 + Kd, 6 + Kp, Ki]  # s⁴ + 6s³ + (11 + Kd)s² + (6 + Kp)s + Ki

    # Create the transfer function
    sys = ctrl.TransferFunction(num, den)

    # Compute the step response
    t_response, y_response = ctrl.step_response(sys, T=t)

    # Calculate rise time
    rise_time = calculate_rise_time(t_response, y_response)
    Kp_rise_times.append(rise_time)

    # Check if this is closer to our target rise time
    rise_time_diff = abs(rise_time - target_rise_time)
    if rise_time_diff < best_kp_rise_time_diff:
        best_kp_rise_time_diff = rise_time_diff
        best_kp = Kp
        best_kp_sys = sys
        best_kp_response = y_response
        best_kp_time = t_response

# Plot Stage 1 results
plt.subplot(3, 2, 1)
plt.plot(Kp_values, Kp_rise_times, 'bo-', linewidth=2)
plt.axhline(y=target_rise_time, color='r', linestyle='--', label=f'Target ({target_rise_time}s)')
plt.scatter([best_kp], [Kp_rise_times[np.where(Kp_values == best_kp)[0][0]]],
            color='red', s=100, zorder=5, label=f'Best Kp = {best_kp:.2f}')
plt.grid(True)
plt.xlabel('Kp Value')
plt.ylabel('Rise Time (s)')
plt.title('Stage 1: Rise Time vs. Kp (Ki=0, Kd=0)')
plt.legend()

plt.subplot(3, 2, 2)
metrics_stage1 = evaluate_response(best_kp_time, best_kp_response)
plot_response(best_kp_time, best_kp_response, plt.gca(),
              f'Stage 1: Response with Best Kp={best_kp:.2f} (Ki=0, Kd=0)', metrics_stage1)

print(f"Stage 1 Complete. Best Kp: {best_kp:.2f}, Rise Time: {metrics_stage1['rise_time']:.4f}s")

# Stage 2: Sweep Ki (with best Kp from Stage 1, Kd=0) to minimize steady-state error
print(f"\nStage 2: Sweeping Ki (with Kp={best_kp:.2f}, Kd=0)...")

# Initialize variables for Stage 2
Ki_ss_errors = []
Ki_rise_times = []
best_ki = None
best_combined_metric = float('inf')
best_ki_sys = None
best_ki_response = None
best_ki_time = None

for Ki in Ki_values:
    # Use best Kp from Stage 1, Kd=0
    Kp = best_kp
    Kd = 0

    # Update the transfer function with the current Ki
    num = [Kd, Kp, Ki]  # Kd*s² + Kp*s + Ki
    den = [1, 6, 11 + Kd, 6 + Kp, Ki]  # s⁴ + 6s³ + (11 + Kd)s² + (6 + Kp)s + Ki

    # Create the transfer function
    sys = ctrl.TransferFunction(num, den)

    # Compute the step response
    t_response, y_response = ctrl.step_response(sys, T=t)

    # Calculate steady-state error and rise time
    ss_error = calculate_ss_error(y_response)
    rise_time = calculate_rise_time(t_response, y_response)

    Ki_ss_errors.append(ss_error)
    Ki_rise_times.append(rise_time)

    # Create a combined metric that weights both SS error and rise time
    rise_time_error = abs(rise_time - target_rise_time)
    combined_metric = ss_error * 10 + rise_time_error  # Weighting SS error more

    # Check if this gives the best combined performance
    if combined_metric < best_combined_metric:
        best_combined_metric = combined_metric
        best_ki = Ki
        best_ki_sys = sys
        best_ki_response = y_response
        best_ki_time = t_response

# Plot Stage 2 results
plt.subplot(3, 2, 3)
plt.plot(Ki_values, Ki_ss_errors, 'go-', linewidth=2)
plt.scatter([best_ki], [Ki_ss_errors[np.where(Ki_values == best_ki)[0][0]]],
            color='red', s=100, zorder=5, label=f'Best Ki = {best_ki:.2f}')
plt.grid(True)
plt.xlabel('Ki Value')
plt.ylabel('Steady-State Error')
plt.title(f'Stage 2: SS Error vs. Ki (Kp={best_kp:.2f}, Kd=0)')
plt.legend()

plt.subplot(3, 2, 4)
metrics_stage2 = evaluate_response(best_ki_time, best_ki_response)
plot_response(best_ki_time, best_ki_response, plt.gca(),
              f'Stage 2: Response with Kp={best_kp:.2f}, Ki={best_ki:.2f}, Kd=0', metrics_stage2)

print(f"Stage 2 Complete. Best Ki: {best_ki:.2f}, SS Error: {metrics_stage2['ss_error']:.6f}")

# Stage 3: Sweep Kd (with best Kp from Stage 1, best Ki from Stage 2) to satisfy constraints
print(f"\nStage 3: Sweeping Kd (with Kp={best_kp:.2f}, Ki={best_ki:.2f})...")

# Initialize variables for Stage 3
Kd_settling_times = []
Kd_overshoots = []
Kd_rise_times = []  # Added to track rise times
valid_kds = []
valid_metrics = []
best_kd = None
best_performance_metric = float('inf')
best_kd_sys = None
best_kd_response = None
best_kd_time = None

# Define maximum allowed deviation from target rise time (in seconds)
max_rise_time_deviation = 0.1  # Allow rise time between target ± max_deviation

for Kd in Kd_values:
    # Use best Kp from Stage 1, best Ki from Stage 2
    Kp = best_kp
    Ki = best_ki

    # Update the transfer function with the current Kd
    num = [Kd, Kp, Ki]  # Kd*s² + Kp*s + Ki
    den = [1, 6, 11 + Kd, 6 + Kp, Ki]  # s⁴ + 6s³ + (11 + Kd)s² + (6 + Kp)s + Ki

    # Create the transfer function
    sys = ctrl.TransferFunction(num, den)

    # Compute the step response
    t_response, y_response = ctrl.step_response(sys, T=t)

    # Calculate metrics
    settling_time = calculate_settling_time(t_response, y_response)
    overshoot = calculate_overshoot(y_response)
    rise_time = calculate_rise_time(t_response, y_response)

    Kd_settling_times.append(settling_time)
    Kd_overshoots.append(overshoot)
    Kd_rise_times.append(rise_time)

    # Check if this Kd meets ALL our constraints including rise time
    rise_time_deviation = abs(rise_time - target_rise_time)
    if (settling_time <= max_settling_time and
            overshoot <= max_overshoot and
            rise_time_deviation <= max_rise_time_deviation):

        valid_kds.append(Kd)

        # Create a performance metric that balances all three criteria
        # Normalize each metric to its max allowed value for fair weighting
        normalized_settling = settling_time / max_settling_time
        normalized_overshoot = overshoot / max_overshoot
        normalized_rise_deviation = rise_time_deviation / max_rise_time_deviation

        # Sum the normalized metrics with weights
        performance_metric = (normalized_settling +
                              normalized_overshoot +
                              2 * normalized_rise_deviation)  # Weight rise time more

        valid_metrics.append(performance_metric)

        # Check if this gives the best performance among valid Kds
        if performance_metric < best_performance_metric:
            best_performance_metric = performance_metric
            best_kd = Kd
            best_kd_sys = sys
            best_kd_response = y_response
            best_kd_time = t_response

# Plot Stage 3 results - now including rise time
plt.subplot(3, 2, 5)
plt.plot(Kd_values, Kd_overshoots, 'mo-', linewidth=2, label='Overshoot (%)')
plt.plot(Kd_values, Kd_settling_times, 'co-', linewidth=2, label='Settling Time (s)')
plt.plot(Kd_values, Kd_rise_times, 'yo-', linewidth=2, label='Rise Time (s)')
plt.axhline(y=max_overshoot, color='m', linestyle='--', alpha=0.5, label=f'Max Overshoot ({max_overshoot}%)')
plt.axhline(y=max_settling_time, color='c', linestyle='--', alpha=0.5,
            label=f'Max Settling Time ({max_settling_time}s)')
plt.axhline(y=target_rise_time, color='y', linestyle='--', alpha=0.5, label=f'Target Rise Time ({target_rise_time}s)')
plt.axhline(y=target_rise_time + max_rise_time_deviation, color='y', linestyle=':', alpha=0.5)
plt.axhline(y=target_rise_time - max_rise_time_deviation, color='y', linestyle=':', alpha=0.5)

if best_kd is not None:
    plt.scatter([best_kd], [Kd_overshoots[np.where(Kd_values == best_kd)[0][0]]],
                color='red', s=100, zorder=5)
    plt.scatter([best_kd], [Kd_settling_times[np.where(Kd_values == best_kd)[0][0]]],
                color='red', s=100, zorder=5)
    plt.scatter([best_kd], [Kd_rise_times[np.where(Kd_values == best_kd)[0][0]]],
                color='red', s=100, zorder=5, label=f'Best Kd = {best_kd:.2f}')
plt.grid(True)
plt.xlabel('Kd Value')
plt.ylabel('Value')
plt.title(f'Stage 3: Performance vs. Kd (Kp={best_kp:.2f}, Ki={best_ki:.2f})')
plt.legend()

plt.subplot(3, 2, 6)
if best_kd is not None:
    metrics_stage3 = evaluate_response(best_kd_time, best_kd_response)
    plot_response(best_kd_time, best_kd_response, plt.gca(),
                  f'Stage 3: Response with Kp={best_kp:.2f}, Ki={best_ki:.2f}, Kd={best_kd:.2f}', metrics_stage3)

    print(f"Stage 3 Complete. Best Kd: {best_kd:.2f}")
    print(f"Rise Time: {metrics_stage3['rise_time']:.4f}s (Target: {target_rise_time}s)")
    print(f"Settling Time: {metrics_stage3['settling_time']:.4f}s (Max: {max_settling_time}s)")
    print(f"Overshoot: {metrics_stage3['overshoot']:.2f}% (Max: {max_overshoot}%)")
else:
    plt.text(0.5, 0.5, 'No valid Kd values found that meet all constraints',
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=14)
    print("Stage 3 Failed: No valid Kd values found that meet all constraints.")
    print("Consider relaxing constraints or expanding parameter search space.")

plt.tight_layout()

# Print final results
print("\nFinal PID Parameters:")
print(f"Kp = {best_kp:.2f}")
print(f"Ki = {best_ki:.2f}")
if best_kd is not None:
    print(f"Kd = {best_kd:.2f}")
    print(f"\nFinal Performance Metrics:")
    print(f"Rise Time: {metrics_stage3['rise_time']:.4f}s (Target: {target_rise_time}s)")
    print(f"Settling Time: {metrics_stage3['settling_time']:.4f}s (Max: {max_settling_time}s)")
    print(f"Percent Overshoot: {metrics_stage3['overshoot']:.2f}% (Max: {max_overshoot}%)")
    print(f"Steady-State Error: {metrics_stage3['ss_error']:.6f}")
else:
    print("\nNo valid Kd found. Final results use Kp and Ki only.")
    print(f"\nPartial Performance Metrics (without Kd):")
    print(f"Rise Time: {metrics_stage2['rise_time']:.4f}s (Target: {target_rise_time}s)")
    print(f"Settling Time: {metrics_stage2['settling_time']:.4f}s (Max: {max_settling_time}s)")
    print(f"Percent Overshoot: {metrics_stage2['overshoot']:.2f}% (Max: {max_overshoot}%)")
    print(f"Steady-State Error: {metrics_stage2['ss_error']:.6f}")

plt.show()