import os
import numpy as np

# ==================================================
# Observation via EXPONENTIAL MOVING AVERAGE (EMA)
# ==================================================

def ema_observe(y, alpha):
    y = np.asarray(y)
    y_hat = np.zeros_like(y, dtype=float)
    y_hat[0] = y[0]

    for t in range(1, len(y)):
        y_hat[t] = alpha * y[t] + (1 - alpha) * y_hat[t - 1]

    return y_hat


OBS_SCALES = {
    "fast": 0.5,
    "normal": 0.1,
    "slow": 0.02,
}

# ==================================================
# Import generators
# ==================================================

from synthetic_generators import (
    gen_smooth_sinusoidal,
    gen_sawtooth_series,
    gen_blocky_series,
    gen_sawtooth_plus_sinusoid,
    gen_multi_blocky_series,
    add_point_outliers,
    add_block_outliers,
    add_pattern_break,
)

# ==================================================
# Configuration
# ==================================================

sample_sizes = [2000]
seed = 1

PERIODS = {
    "slow": 75,
    "medium": 25,
    "fast": 7,
}

OUTLIER_FRACTIONS = [0.05, 0.25]
OUTLIER_STD_FACTOR = 4.0

all_series = {}

# ==================================================
# Generate LATENT series
# ==================================================

for n in sample_sizes:
    for speed_name, period in PERIODS.items():

        # 1. Smooth sinusoidal
        key = f"sinusoidal_{speed_name}_n{n}_seed{seed}"
        all_series[key] = gen_smooth_sinusoidal(
            n=n, seed=seed, period=period
        )

        # 2. Sawtooth
        key = f"sawtooth_{speed_name}_n{n}_seed{seed}"
        all_series[key] = gen_sawtooth_series(
            n=n, seed=seed, period=period, noise_std=0.5
        )

        
        # 3. Blocky
        block_duration = max(1, period // 4)

        # FIX: ensure divisibility
        while period % block_duration != 0:
            block_duration -= 1

        key = f"blocky_{speed_name}_n{n}_seed{seed}"
        all_series[key] = gen_blocky_series(
            n=n,
            seed=seed,
            period=period,
            block_duration=block_duration,
            amplitude=3.0,
            noise_std=0.1,
        )
    

        # 4. Sawtooth + sinusoid
        for sin_amp in [0.5, 2.0, 6.0]:
            key = f"saw_sin_{speed_name}_sinamp{sin_amp}_n{n}_seed{seed}"

            all_series[key] = gen_sawtooth_plus_sinusoid(
                n=n,
                seed=seed,
                saw_period=period * 4,
                sin_period=period,
                saw_amp=4.0,
                sin_amp=sin_amp,
                noise_std=0.1,
            )

        # 5. Multi-blocky
        key = f"multiblock_{speed_name}_n{n}_seed{seed}"

        all_series[key] = gen_multi_blocky_series(
            n=n,
            seed=seed,
            slow_block_duration=period * 2,
            fast_block_duration=max(1, period // 3),
            slow_amp=4.0,
            fast_amp=2.0,
            noise_std=0.1,
        )

# ==================================================
# Add disturbances
# ==================================================

all_series_with_disturbances = {}

for key, y in all_series.items():

    for frac in OUTLIER_FRACTIONS:
        new_key = f"{key}_point_{int(frac*100)}pct_std4"

        all_series_with_disturbances[new_key] = add_point_outliers(
            y,
            seed=seed,
            outlier_fraction=frac,
            outlier_std_factor=OUTLIER_STD_FACTOR,
        )

    new_key = f"{key}_block_25pct_std4"
    all_series_with_disturbances[new_key] = add_block_outliers(
        y,
        seed=seed,
        outlier_fraction=0.25,
        block_length=20,
        outlier_std_factor=OUTLIER_STD_FACTOR,
    )

    new_key = f"{key}_pattern_break"
    all_series_with_disturbances[new_key] = add_pattern_break(
        y,
        break_point=len(y) // 2,
        amplitude_shift=2.0,
        level_shift=10.0,
    )

# Merge
all_series.update(all_series_with_disturbances)

# ==================================================
# Build OBSERVED datasets (EMA)
# ==================================================

all_series_observed = {}

for key, y in all_series.items():
    for scale_name, alpha in OBS_SCALES.items():
        new_key = f"{key}_obs_{scale_name}"
        all_series_observed[new_key] = ema_observe(y, alpha)

all_series.update(all_series_observed)

# ==================================================
# Save dataset
# ==================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(SCRIPT_DIR, "synthetic_time_series.npz")

np.savez(out_path, **all_series)

print("✅ Dataset saved at:", out_path)
print("✅ Number of series:", len(all_series))