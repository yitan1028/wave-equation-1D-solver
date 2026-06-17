"""
one_d_solver.py

Teaching-oriented 1D acoustic forward solver.

V3 changes:
- Default velocity is 1D but variable, not flat.
- Output keeps enough intermediate values to show exact finite-difference updates.
- The numerical example selects one receiver/grid point and three time steps near wave arrival.

PDE:
    u_tt = v(x)^2 u_xx + source

Finite-difference update:
    u_next[i] = 2*u_curr[i] - u_prev[i]
                + (v[i]*dt/dx)^2 * (u_curr[i+1] - 2*u_curr[i] + u_curr[i-1])
                + source_term
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class SolverConfig:
    # Spatial grid
    nx: int = 601
    dx: float = 5.0

    # Time grid
    nt: int = 700
    dt: float = 0.0008

    # Velocity model
    # Options: "smooth_variable", "two_layer", "constant"
    velocity_model: str = "smooth_variable"
    velocity_value: float = 2000.0

    # Source
    source_x: float = 1500.0
    source_frequency: float = 18.0
    source_strength: float = 0.8

    # Receivers
    # If None, use many receivers for a better seismic-record image.
    receiver_xs: Optional[Sequence[float]] = None
    n_receivers: int = 61

    # Boundary damping
    damping_cells: int = 80
    damping_strength: float = 0.040

    # Numerical update explanation
    calculation_receiver_x: float = 2100.0
    calculation_steps: Optional[Sequence[int]] = None


@dataclass
class SolverResult:
    x: np.ndarray
    t: np.ndarray
    velocity: np.ndarray
    source_signal: np.ndarray
    source_index: int
    receiver_indices: np.ndarray
    receiver_xs: np.ndarray
    wavefield_history: np.ndarray
    receiver_data: np.ndarray
    damping_profile: np.ndarray
    cfl: float
    calculation_index: int
    calculation_examples: list[dict]


def ricker_wavelet(f0: float, t: np.ndarray, t0: Optional[float] = None) -> np.ndarray:
    """Normalized Ricker wavelet source-time function."""
    if t0 is None:
        t0 = 1.5 / f0
    a = np.pi * f0 * (t - t0)
    w = (1.0 - 2.0 * a**2) * np.exp(-a**2)
    m = np.max(np.abs(w))
    if m > 0:
        w = w / m
    return w


def make_velocity_model(config: SolverConfig) -> np.ndarray:
    """
    Create a 1D velocity profile v(x).

    smooth_variable is the default for the teaching demo:
    it is clearly not flat, but it avoids a sharp boundary so the first demo
    does not immediately become a reflection example.
    """
    x = np.arange(config.nx, dtype=float) * config.dx
    L = x[-1] - x[0]

    if config.velocity_model == "constant":
        return np.full(config.nx, config.velocity_value, dtype=float)

    if config.velocity_model == "smooth_variable":
        # Clear 1D velocity variation, roughly 1650--2550 m/s.
        # Trend + smooth low/high zones. No sharp layer boundary here.
        trend = 1650.0 + 620.0 * (x / L)
        low_zone = -170.0 * np.exp(-((x - 0.28 * L) / (0.13 * L)) ** 2)
        high_zone = 260.0 * np.exp(-((x - 0.68 * L) / (0.16 * L)) ** 2)
        gentle_wave = 55.0 * np.sin(2.0 * np.pi * x / L)
        v = trend + low_zone + high_zone + gentle_wave
        return v.astype(float)

    if config.velocity_model == "two_layer":
        # Kept for the next lesson: reflection/transmission.
        v = np.full(config.nx, 1800.0, dtype=float)
        v[x >= 0.58 * L] = 2600.0
        return v

    raise ValueError(
        f"Unknown velocity_model={config.velocity_model!r}. Use 'smooth_variable', 'two_layer', or 'constant'."
    )


def make_damping_profile(nx: int, damping_cells: int, damping_strength: float) -> np.ndarray:
    """Simple multiplicative absorbing/damping layer near both boundaries."""
    profile = np.ones(nx, dtype=float)
    if damping_cells <= 0:
        return profile

    damping_cells = min(damping_cells, nx // 2)
    for i in range(damping_cells):
        # strongest damping at the outer edge, smoothly decays to 1.0 inside
        r = (damping_cells - i) / damping_cells
        factor = np.exp(-damping_strength * r**2)
        profile[i] *= factor
        profile[-i - 1] *= factor
    return profile


def choose_default_receivers(x: np.ndarray, n_receivers: int) -> np.ndarray:
    """Many receivers across the interior of the domain."""
    margin = 0.10 * (x[-1] - x[0])
    return np.linspace(x[0] + margin, x[-1] - margin, n_receivers)


def positions_to_indices(x: np.ndarray, positions: Sequence[float]) -> np.ndarray:
    return np.array([int(np.argmin(np.abs(x - p))) for p in positions], dtype=int)


def check_cfl(velocity: np.ndarray, dt: float, dx: float) -> float:
    cfl = float(np.max(velocity) * dt / dx)
    if cfl > 1.0:
        raise ValueError(
            f"Unstable CFL={cfl:.3f}. Reduce dt, increase dx, or lower velocity."
        )
    return cfl


def choose_calculation_steps(
    config: SolverConfig,
    x: np.ndarray,
    velocity: np.ndarray,
    source_index: int,
    calculation_index: int,
) -> list[int]:
    """Choose three steps near arrival at the selected calculation point."""
    if config.calculation_steps is not None:
        return [int(s) for s in config.calculation_steps]

    distance = abs(x[calculation_index] - x[source_index])
    mean_v = 0.5 * (velocity[calculation_index] + velocity[source_index])
    arrival_time = distance / mean_v
    arrival_step = int(arrival_time / config.dt)

    steps = [arrival_step - 18, arrival_step, arrival_step + 18]
    return [int(np.clip(s, 5, config.nt - 5)) for s in steps]


def run_1d_forward(config: Optional[SolverConfig] = None) -> SolverResult:
    if config is None:
        config = SolverConfig()

    x = np.arange(config.nx, dtype=float) * config.dx
    t = np.arange(config.nt, dtype=float) * config.dt

    velocity = make_velocity_model(config)
    cfl = check_cfl(velocity, config.dt, config.dx)

    source_index = int(np.argmin(np.abs(x - config.source_x)))

    if config.receiver_xs is None:
        receiver_xs = choose_default_receivers(x, config.n_receivers)
    else:
        receiver_xs = np.asarray(config.receiver_xs, dtype=float)
    receiver_indices = positions_to_indices(x, receiver_xs)
    receiver_xs = x[receiver_indices]

    calculation_index = int(np.argmin(np.abs(x - config.calculation_receiver_x)))
    calculation_steps = choose_calculation_steps(
        config, x, velocity, source_index, calculation_index
    )
    calculation_step_set = set(calculation_steps)

    source_signal = config.source_strength * ricker_wavelet(
        config.source_frequency, t
    )

    damping_profile = make_damping_profile(
        config.nx, config.damping_cells, config.damping_strength
    )

    # Three time levels used in the leapfrog finite-difference update.
    u_prev = np.zeros(config.nx, dtype=float)  # u at n-1
    u_curr = np.zeros(config.nx, dtype=float)  # u at n
    u_next = np.zeros(config.nx, dtype=float)  # u at n+1

    wavefield_history = np.zeros((config.nt, config.nx), dtype=float)
    receiver_data = np.zeros((config.nt, len(receiver_indices)), dtype=float)

    alpha = (velocity * config.dt / config.dx) ** 2
    calculation_examples: list[dict] = []

    for n in range(config.nt):
        curvature = np.zeros(config.nx, dtype=float)
        curvature[1:-1] = u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2]

        u_next.fill(0.0)
        u_next[1:-1] = 2.0 * u_curr[1:-1] - u_prev[1:-1] + alpha[1:-1] * curvature[1:-1]

        # Source injection at the source grid point.
        source_term_at_calculation_point = source_signal[n] if calculation_index == source_index else 0.0
        u_next_before_source = u_next.copy()
        u_next[source_index] += source_signal[n]
        u_next_after_source = u_next.copy()

        # Damping near boundaries.
        u_next *= damping_profile

        if n in calculation_step_set:
            i = calculation_index
            raw_next = 2.0 * u_curr[i] - u_prev[i] + alpha[i] * curvature[i]
            after_source = raw_next + source_term_at_calculation_point
            after_damping = after_source * damping_profile[i]

            calculation_examples.append({
                "step": int(n),
                "time": float(t[n]),
                "index": int(i),
                "x": float(x[i]),
                "v": float(velocity[i]),
                "dt": float(config.dt),
                "dx": float(config.dx),
                "alpha": float(alpha[i]),
                "u_prev_i": float(u_prev[i]),
                "u_curr_im1": float(u_curr[i - 1]),
                "u_curr_i": float(u_curr[i]),
                "u_curr_ip1": float(u_curr[i + 1]),
                "curvature": float(curvature[i]),
                "alpha_times_curvature": float(alpha[i] * curvature[i]),
                "source_term": float(source_term_at_calculation_point),
                "damping_factor": float(damping_profile[i]),
                "u_next_before_source": float(raw_next),
                "u_next_after_source": float(after_source),
                "u_next_after_damping": float(after_damping),
                "u_prev_array": u_prev.copy(),
                "u_curr_array": u_curr.copy(),
                "u_next_before_source_array": u_next_before_source.copy(),
                "u_next_after_source_array": u_next_after_source.copy(),
                "u_next_array": u_next.copy(),
            })

        wavefield_history[n, :] = u_next
        receiver_data[n, :] = u_next[receiver_indices]

        u_prev, u_curr = u_curr, u_next.copy()

    calculation_examples.sort(key=lambda e: e["step"])

    return SolverResult(
        x=x,
        t=t,
        velocity=velocity,
        source_signal=source_signal,
        source_index=source_index,
        receiver_indices=receiver_indices,
        receiver_xs=receiver_xs,
        wavefield_history=wavefield_history,
        receiver_data=receiver_data,
        damping_profile=damping_profile,
        cfl=cfl,
        calculation_index=calculation_index,
        calculation_examples=calculation_examples,
    )


if __name__ == "__main__":
    res = run_1d_forward()
    print("V3 1D forward simulation complete.")
    print(f"Velocity range: {res.velocity.min():.1f} to {res.velocity.max():.1f} m/s")
    print(f"CFL: {res.cfl:.3f}")
    print(f"Wavefield history: {res.wavefield_history.shape}")
    print(f"Receiver data: {res.receiver_data.shape}")
    print(f"Calculation examples: {[e['step'] for e in res.calculation_examples]}")
