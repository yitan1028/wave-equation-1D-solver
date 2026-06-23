"""
Diagnostic experiment: verify that spatially varying velocity changes wave speed.

This script intentionally does not modify the teaching demo files. It runs a
small two-layer 1D finite-difference experiment and writes all outputs under
outputs/velocity_diagnostic/.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from one_d_solver import check_cfl, make_damping_profile


OUTPUT_DIR = Path("outputs") / "velocity_diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "matplotlib_config"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / "cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def first_arrival_time(trace: np.ndarray, t: np.ndarray, threshold_fraction: float = 0.05) -> float:
    """Return the first time where abs(trace) exceeds a fraction of its peak."""
    peak = float(np.max(np.abs(trace)))
    if peak == 0.0:
        return float("nan")

    threshold = threshold_fraction * peak
    hits = np.flatnonzero(np.abs(trace) >= threshold)
    if len(hits) == 0:
        return float("nan")
    return float(t[hits[0]])


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    nx = 401
    dx = 1.0
    dt = 0.25
    nt = 620

    x = np.arange(nx, dtype=float) * dx
    t = np.arange(nt, dtype=float) * dt

    source_index = nx // 2
    receiver_left = source_index - 100
    receiver_right = source_index + 100

    velocity = np.ones(nx, dtype=float)
    velocity[source_index:] = 2.0

    cfl = check_cfl(velocity, dt, dx)
    damping_profile = make_damping_profile(nx, damping_cells=30, damping_strength=0.025)

    u_prev = np.zeros(nx, dtype=float)
    u_curr = np.zeros(nx, dtype=float)
    u_next = np.zeros(nx, dtype=float)
    wavefield_history = np.zeros((nt, nx), dtype=float)
    receiver_data = np.zeros((nt, 2), dtype=float)

    alpha = (velocity * dt / dx) ** 2

    # Compact pulse centered near t=2.5 s. Its peak time is subtracted from
    # measured arrivals below so the reported times are propagation times.
    source_peak_time = 2.5
    source_width = 0.6
    source_signal = np.exp(-((t - source_peak_time) / source_width) ** 2)
    source_peak_index = int(np.argmax(source_signal))
    source_peak_time = float(t[source_peak_index])

    for n in range(nt):
        curvature = np.zeros(nx, dtype=float)
        curvature[1:-1] = u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2]

        u_next.fill(0.0)
        u_next[1:-1] = (
            2.0 * u_curr[1:-1]
            - u_prev[1:-1]
            + alpha[1:-1] * curvature[1:-1]
        )
        u_next[source_index] += source_signal[n]
        u_next *= damping_profile

        wavefield_history[n, :] = u_next
        receiver_data[n, 0] = u_next[receiver_left]
        receiver_data[n, 1] = u_next[receiver_right]

        u_prev, u_curr = u_curr, u_next.copy()

    left_trace = receiver_data[:, 0]
    right_trace = receiver_data[:, 1]

    measured_left_absolute = first_arrival_time(left_trace, t)
    measured_right_absolute = first_arrival_time(right_trace, t)
    measured_left = measured_left_absolute - source_peak_time
    measured_right = measured_right_absolute - source_peak_time

    expected_left = abs(source_index - receiver_left) * dx / 1.0
    expected_right = abs(receiver_right - source_index) * dx / 2.0

    np.save(output_dir / "x_positions.npy", x)
    np.save(output_dir / "time_seconds.npy", t)
    np.save(output_dir / "velocity.npy", velocity)
    np.save(output_dir / "wavefield_history_t_by_x.npy", wavefield_history)
    np.save(output_dir / "receiver_data_t_by_receiver.npy", receiver_data)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(x, velocity, linewidth=2.2)
    ax.axvline(x[source_index], color="black", linestyle="--", label="Source")
    ax.axvline(x[receiver_left], color="tab:blue", linestyle=":", label="Left receiver")
    ax.axvline(x[receiver_right], color="tab:orange", linestyle=":", label="Right receiver")
    ax.set_title("Velocity diagnostic two-layer model")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Velocity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_model.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t - source_peak_time, left_trace, label=f"left receiver x={x[receiver_left]:.0f}")
    ax.plot(t - source_peak_time, right_trace, label=f"right receiver x={x[receiver_right]:.0f}")
    ax.axvline(expected_left, color="tab:blue", linestyle="--", alpha=0.7, label="left expected")
    ax.axvline(expected_right, color="tab:orange", linestyle="--", alpha=0.7, label="right expected")
    ax.axvline(measured_left, color="tab:blue", linestyle="-.", alpha=0.7, label="left measured")
    ax.axvline(measured_right, color="tab:orange", linestyle="-.", alpha=0.7, label="right measured")
    ax.set_xlim(0, 130)
    ax.set_title("Receiver traces from simulated wavefield")
    ax.set_xlabel("Time after source peak")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "receiver_traces.png", dpi=180)
    plt.close(fig)

    vmax = np.percentile(np.abs(wavefield_history), 99.5)
    if vmax == 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(9, 5.5))
    extent = [x[0], x[-1], t[-1] - source_peak_time, t[0] - source_peak_time]
    im = ax.imshow(
        wavefield_history,
        aspect="auto",
        extent=extent,
        vmin=-vmax,
        vmax=vmax,
        cmap="seismic",
    )
    ax.axvline(x[source_index], color="black", linestyle="--", label="Source")
    ax.axvline(x[receiver_left], color="tab:blue", linestyle=":", label="Left receiver")
    ax.axvline(x[receiver_right], color="tab:orange", linestyle=":", label="Right receiver")
    ax.set_ylim(130, 0)
    ax.set_title("Simulated x-t wavefield")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Time after source peak")
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Wave amplitude")
    fig.tight_layout()
    fig.savefig(output_dir / "wavefield_xt.png", dpi=180)
    plt.close(fig)

    print("Velocity diagnostic complete")
    print(f"Output directory: {output_dir}")
    print(f"nx={nx}, dx={dx}, dt={dt}, nt={nt}, CFL={cfl:.3f}")
    print(f"source_index={source_index}, source_x={x[source_index]:.1f}")
    print(f"receiver_left_index={receiver_left}, receiver_left_x={x[receiver_left]:.1f}, velocity=1.0")
    print(f"receiver_right_index={receiver_right}, receiver_right_x={x[receiver_right]:.1f}, velocity=2.0")
    print(f"source_peak_time={source_peak_time:.3f}")
    print(f"expected_left_arrival={expected_left:.3f}")
    print(f"measured_left_arrival={measured_left:.3f}")
    print(f"expected_right_arrival={expected_right:.3f}")
    print(f"measured_right_arrival={measured_right:.3f}")


if __name__ == "__main__":
    main()
