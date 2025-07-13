import numpy as np

try:
    stats = np.load("normal_speed_stats.npy", allow_pickle=True).item()
    mean_speed_routine = stats['mean']
    std_speed_routine = stats['std']
    print(f"Loaded from normal_speed_stats.npy:")
    print(f"  Mean Speed (Routine): {mean_speed_routine}")
    print(f"  Std Dev Speed (Routine): {std_speed_routine}")

    # Calculate the threshold being used
    speed_threshold_value = mean_speed_routine + 2 * std_speed_routine
    print(f"  Calculated Speed Anomaly Threshold: > {speed_threshold_value}")

except FileNotFoundError:
    print("Error: normal_speed_stats.npy not found!")
except KeyError:
    print(
        "Error: 'mean' or 'std' key not found in normal_speed_stats.npy. File might be corrupted or incorrectly generated.")