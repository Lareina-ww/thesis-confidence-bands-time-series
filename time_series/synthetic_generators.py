import numpy as np

# ==================================================
# Base generators
# ==================================================

def gen_smooth_sinusoidal(n=250, seed=1, period=12):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = 3.0 * np.sin(2 * np.pi * t / period)
    noise = rng.normal(0, 0.001, size=n)
    return 50 + seasonal + noise


def gen_sawtooth_series(n=250, seed=1, period=12, amplitude=3.0, noise_std=0.001):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    saw = (2 * (t % period) / period - 1) * amplitude
    noise = rng.normal(0, noise_std, size=n)
    return 50 + saw + noise


def gen_blocky_series(
    n=250,
    seed=1,
    period=12,
    block_duration=3,
    amplitude=3.0,
    noise_std=0.0,
    base_level=50.0,
):
    if period % block_duration != 0:
        raise ValueError("period must be divisible by block_duration")

    rng = np.random.default_rng(seed)
    t = np.arange(n)

    n_blocks = period // block_duration
    block_levels = rng.uniform(-amplitude, amplitude, size=n_blocks)

    block_index = (t % period) // block_duration
    y = base_level + block_levels[block_index]

    if noise_std > 0:
        y += rng.normal(0, noise_std, size=n)

    return y


def gen_sawtooth_plus_sinusoid(
    n=1000,
    seed=1,
    saw_period=200,
    sin_period=12,
    saw_amp=4.0,
    sin_amp=1.0,
    noise_std=0.1,
    base_level=50.0,
):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    saw = (2 * (t % saw_period) / saw_period - 1) * saw_amp
    sin = sin_amp * np.sin(2 * np.pi * t / sin_period)
    noise = rng.normal(0, noise_std, size=n)

    return base_level + saw + sin + noise


def gen_multi_blocky_series(
    n=1000,
    seed=1,
    slow_block_duration=50,
    fast_block_duration=10,
    slow_amp=4.0,
    fast_amp=1.5,
    noise_std=0.1,
    base_level=50.0,
):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    slow_idx = (t // slow_block_duration) % 5
    fast_idx = (t // fast_block_duration) % 5

    slow_levels = rng.uniform(-slow_amp, slow_amp, size=5)
    fast_levels = rng.uniform(-fast_amp, fast_amp, size=5)

    y = base_level + slow_levels[slow_idx] + fast_levels[fast_idx]

    if noise_std > 0:
        y += rng.normal(0, noise_std, size=n)

    return y


# ==================================================
# Outlier utilities
# ==================================================

def add_point_outliers(
    y,
    seed=1,
    outlier_fraction=None,
    n_outliers=None,
    outlier_std_factor=4.0,
):
    rng = np.random.default_rng(seed)
    y_out = y.copy()

    if outlier_fraction is not None:
        n_outliers = int(len(y) * outlier_fraction)

    if n_outliers is None or n_outliers == 0:
        return y_out

    idx = rng.choice(len(y), size=n_outliers, replace=False)
    scale = outlier_std_factor * np.std(y)
    signs = rng.choice([-1, 1], size=n_outliers)

    y_out[idx] += signs * scale
    return y_out


def add_block_outliers(
    y,
    seed=1,
    outlier_fraction=0.25,
    block_length=20,
    outlier_std_factor=4.0,
):
    rng = np.random.default_rng(seed)
    y_out = y.copy()
    n = len(y)

    total_points = int(n * outlier_fraction)
    n_blocks = max(1, total_points // block_length)

    scale = outlier_std_factor * np.std(y)

    for _ in range(n_blocks):
        start = rng.integers(0, n - block_length)
        y_out[start:start + block_length] += scale

    return y_out


def add_pattern_break(
    y,
    break_point=500,
    amplitude_shift=2.0,
    level_shift=10.0,
):
    y_out = y.copy()
    mean_before = np.mean(y_out[:break_point])

    y_out[break_point:] = (
        level_shift
        + amplitude_shift * (y_out[break_point:] - mean_before)
    )

    return y_out
