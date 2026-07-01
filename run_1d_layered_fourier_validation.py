"""
Validate the layered finite-difference solver in a Fourier basis.

This is a full-operator Fourier-basis validation:
- the physical finite-difference update is implemented directly in this file;
- dense physical update matrices B and G are built for the exact same update;
- the dense operators are transformed as B_F = Phi_inv @ B @ Phi and
  G_F = Phi_inv @ G @ Phi;
- the Fourier-basis state is advanced and transformed back at each time step.

This is not a periodic toy FFT solver and not a constant-velocity diagonal
Fourier solver. The layered velocity and absorbing padding make B_F and G_F
dense matrices with mode coupling.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path("outputs") / "layered_fourier_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(tempfile.gettempdir()) / "wave_equation_1d_fourier_validation_matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHYSICAL_NX = 601
DX = 5.0
DT = 0.0008
NT = 2400
DAMPING_CELLS = 80
DAMPING_STRENGTH = 0.040
TOTAL_NX = PHYSICAL_NX + 2 * DAMPING_CELLS
PHYSICAL_DOMAIN_START = 0.0
PHYSICAL_DOMAIN_END = (PHYSICAL_NX - 1) * DX
SOURCE_X = 450.0
SOURCE_FREQUENCY = 18.0
SOURCE_STRENGTH = 0.8
SOURCE_DIRECTION = "right"
RECEIVER_XS_REQUESTED = np.linspace(450.0, 2550.0, 121)
INTERFACE_POSITIONS_M = (1050.0, 1950.0)
LAYER_VELOCITIES_M_PER_S = (3000.0, 2200.0, 1500.0)
ERROR_TOLERANCE = 1.0e-8


def ricker_wavelet(f0: float, t: np.ndarray, t0: float | None = None) -> np.ndarray:
    """Same normalized Ricker source-time function as one_d_solver.py."""
    if t0 is None:
        t0 = 1.5 / f0
    a = np.pi * f0 * (t - t0)
    w = (1.0 - 2.0 * a**2) * np.exp(-a**2)
    m = np.max(np.abs(w))
    if m > 0.0:
        w = w / m
    return w


def make_physical_velocity() -> np.ndarray:
    """Same three_layer_reflection velocity model as one_d_solver.py."""
    x_physical = np.arange(PHYSICAL_NX, dtype=float) * DX
    velocity = np.full(PHYSICAL_NX, LAYER_VELOCITIES_M_PER_S[0], dtype=float)
    velocity[x_physical >= INTERFACE_POSITIONS_M[0]] = LAYER_VELOCITIES_M_PER_S[1]
    velocity[x_physical >= INTERFACE_POSITIONS_M[1]] = LAYER_VELOCITIES_M_PER_S[2]
    return velocity


def make_absorbing_sigma(
    total_nx: int,
    nbc: int,
    dx: float,
    velmin: float,
) -> tuple[np.ndarray, float]:
    """Same external-padding sigma profile as one_d_solver.py."""
    sigma = np.zeros(total_nx, dtype=float)
    if nbc <= 0:
        return sigma, 0.0
    if nbc == 1:
        padding_width = dx
    else:
        padding_width = (nbc - 1) * dx

    sigma_max = 3.0 * velmin * np.log(1.0e7) / (2.0 * padding_width)
    ramp = np.linspace(1.0, 0.0, nbc) ** 2
    sigma[:nbc] = sigma_max * ramp
    sigma[-nbc:] = sigma_max * ramp[::-1]
    return sigma, float(sigma_max)


def positions_to_indices(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.array([int(np.argmin(np.abs(x - p))) for p in positions], dtype=int)


def build_grid_and_model() -> dict:
    physical_x = np.arange(PHYSICAL_NX, dtype=float) * DX
    x = (np.arange(TOTAL_NX, dtype=float) - DAMPING_CELLS) * DX
    t = np.arange(NT, dtype=float) * DT

    physical_velocity = make_physical_velocity()
    velocity = np.pad(physical_velocity, (DAMPING_CELLS, DAMPING_CELLS), mode="edge")
    sigma, sigma_max = make_absorbing_sigma(
        TOTAL_NX, DAMPING_CELLS, DX, float(np.min(velocity))
    )
    alpha = (velocity * DT / DX) ** 2
    kappa = sigma * DT
    cfl = float(np.max(velocity) * DT / DX)
    if cfl > 1.0:
        raise ValueError(f"Unstable CFL={cfl:.3f}")

    source_index = int(np.argmin(np.abs(x - SOURCE_X)))
    receiver_indices = positions_to_indices(x, RECEIVER_XS_REQUESTED)
    receiver_xs = x[receiver_indices]
    source_signal = SOURCE_STRENGTH * ricker_wavelet(SOURCE_FREQUENCY, t)

    return {
        "physical_x": physical_x,
        "x": x,
        "t": t,
        "velocity": velocity,
        "sigma": sigma,
        "sigma_max": sigma_max,
        "alpha": alpha,
        "kappa": kappa,
        "cfl": cfl,
        "source_index": source_index,
        "receiver_indices": receiver_indices,
        "receiver_xs": receiver_xs,
        "source_signal": source_signal,
    }


def initial_state(x: np.ndarray, velocity: np.ndarray, source_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Same right-going initial packet branch as one_d_solver.py."""
    u_prev = np.zeros(TOTAL_NX, dtype=float)
    u_curr = np.zeros(TOTAL_NX, dtype=float)

    if SOURCE_DIRECTION == "right":
        v_source = velocity[source_index]
        t0 = 1.5 / SOURCE_FREQUENCY
        packet_time = t0 - (x - SOURCE_X) / v_source
        packet = ricker_wavelet(
            SOURCE_FREQUENCY,
            np.concatenate((packet_time, packet_time - DT)),
        )
        u_curr = SOURCE_STRENGTH * packet[:TOTAL_NX]
        u_prev = SOURCE_STRENGTH * packet[TOTAL_NX:]
    elif SOURCE_DIRECTION != "both":
        raise ValueError(f"Unknown SOURCE_DIRECTION={SOURCE_DIRECTION!r}")

    return u_prev, u_curr


def source_vector_for_step(source_signal: np.ndarray, source_index: int, step: int) -> np.ndarray:
    """Match one_d_solver.py source handling exactly."""
    source = np.zeros(TOTAL_NX, dtype=float)
    if SOURCE_DIRECTION == "both":
        source[source_index] = source_signal[step]
    return source


def physical_step(
    u_prev: np.ndarray,
    u_curr: np.ndarray,
    velocity: np.ndarray,
    alpha: np.ndarray,
    kappa: np.ndarray,
    source_signal: np.ndarray,
    source_index: int,
    step: int,
) -> np.ndarray:
    """One time-domain update, matching one_d_solver.run_1d_forward."""
    curvature = np.zeros(TOTAL_NX, dtype=float)
    curvature[1:-1] = u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2]

    u_next = np.zeros(TOTAL_NX, dtype=float)
    u_next[1:-1] = (
        2.0 * u_curr[1:-1]
        - (1.0 - kappa[1:-1]) * u_prev[1:-1]
        + alpha[1:-1] * curvature[1:-1]
    ) / (1.0 + kappa[1:-1])

    c_left = velocity[0] * DT / DX
    c_right = velocity[-1] * DT / DX
    u_next[0] = u_curr[1] + ((c_left - 1.0) / (c_left + 1.0)) * (
        u_next[1] - u_curr[0]
    )
    u_next[-1] = u_curr[-2] + ((c_right - 1.0) / (c_right + 1.0)) * (
        u_next[-2] - u_curr[-1]
    )

    if SOURCE_DIRECTION == "both":
        u_next[source_index] += source_signal[step]

    return u_next


def build_physical_update_matrices(
    velocity: np.ndarray,
    alpha: np.ndarray,
    kappa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build dense B and G such that u_next = B @ u_curr + G @ u_prev + source.

    The rows include the same interior variable-alpha/kappa update and the
    same left/right boundary formulas as one_d_solver.py.
    """
    n = TOTAL_NX
    b = np.zeros((n, n), dtype=float)
    g = np.zeros((n, n), dtype=float)

    denom = 1.0 + kappa[1:-1]
    a = alpha[1:-1]
    interior = np.arange(1, n - 1)

    b[interior, interior - 1] = a / denom
    b[interior, interior] = (2.0 - 2.0 * a) / denom
    b[interior, interior + 1] = a / denom
    g[interior, interior] = -(1.0 - kappa[1:-1]) / denom

    c_left = velocity[0] * DT / DX
    q_left = (c_left - 1.0) / (c_left + 1.0)
    b[0, :] = q_left * b[1, :]
    g[0, :] = q_left * g[1, :]
    b[0, 1] += 1.0
    b[0, 0] += -q_left

    c_right = velocity[-1] * DT / DX
    q_right = (c_right - 1.0) / (c_right + 1.0)
    b[-1, :] = q_right * b[-2, :]
    g[-1, :] = q_right * g[-2, :]
    b[-1, -2] += 1.0
    b[-1, -1] += -q_right

    return b, g


def build_unitary_dft(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Build Phi with u = Phi @ a and a = Phi_inv @ u."""
    j = np.arange(n, dtype=float)[:, None]
    k = np.arange(n, dtype=float)[None, :]
    phi = np.exp(2.0j * np.pi * j * k / n) / np.sqrt(float(n))
    phi_inv = phi.conj().T
    return phi, phi_inv


def checked_matmul(left: np.ndarray, right: np.ndarray, label: str) -> np.ndarray:
    """
    Matrix product with an explicit finite check.

    Some macOS Accelerate/NumPy complex matmul calls can emit spurious floating
    warnings while returning finite results. The validation should fail on
    actual non-finite values, not on backend warning noise.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = left @ right
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f"Non-finite values produced during {label}")
    return result


def run_validation(model: dict) -> dict:
    x = model["x"]
    velocity = model["velocity"]
    alpha = model["alpha"]
    kappa = model["kappa"]
    source_index = model["source_index"]
    source_signal = model["source_signal"]
    receiver_indices = model["receiver_indices"]

    b, g = build_physical_update_matrices(velocity, alpha, kappa)
    phi, phi_inv = build_unitary_dft(TOTAL_NX)
    b_f = checked_matmul(phi_inv, checked_matmul(b, phi, "B @ Phi"), "Phi_inv @ B @ Phi")
    g_f = checked_matmul(phi_inv, checked_matmul(g, phi, "G @ Phi"), "Phi_inv @ G @ Phi")

    u_prev_time, u_curr_time = initial_state(x, velocity, source_index)
    a_prev = checked_matmul(phi_inv, u_prev_time, "Phi_inv @ u_prev")
    a_curr = checked_matmul(phi_inv, u_curr_time, "Phi_inv @ u_curr")

    wavefield_time = np.zeros((NT, TOTAL_NX), dtype=float)
    wavefield_fourier = np.zeros((NT, TOTAL_NX), dtype=float)
    receiver_time = np.zeros((NT, len(receiver_indices)), dtype=float)
    receiver_fourier = np.zeros((NT, len(receiver_indices)), dtype=float)
    error_over_time = np.zeros(NT, dtype=float)

    first_exceed_step: int | None = None

    for step in range(NT):
        u_next_time = physical_step(
            u_prev_time,
            u_curr_time,
            velocity,
            alpha,
            kappa,
            source_signal,
            source_index,
            step,
        )

        source = source_vector_for_step(source_signal, source_index, step)
        a_next = (
            checked_matmul(b_f, a_curr, "B_F @ a_curr")
            + checked_matmul(g_f, a_prev, "G_F @ a_prev")
            + checked_matmul(phi_inv, source, "Phi_inv @ source")
        )
        u_next_fourier_complex = checked_matmul(phi, a_next, "Phi @ a_next")
        u_next_fourier = np.real(u_next_fourier_complex)

        max_error = float(np.max(np.abs(u_next_time - u_next_fourier)))
        error_over_time[step] = max_error
        if first_exceed_step is None and max_error > ERROR_TOLERANCE:
            first_exceed_step = step

        wavefield_time[step, :] = u_next_time
        wavefield_fourier[step, :] = u_next_fourier
        receiver_time[step, :] = u_next_time[receiver_indices]
        receiver_fourier[step, :] = u_next_fourier[receiver_indices]

        u_prev_time, u_curr_time = u_curr_time, u_next_time
        a_prev, a_curr = a_curr, a_next

    return {
        "B": b,
        "G": g,
        "B_F": b_f,
        "G_F": g_f,
        "wavefield_time": wavefield_time,
        "wavefield_fourier": wavefield_fourier,
        "receiver_time": receiver_time,
        "receiver_fourier": receiver_fourier,
        "error_over_time": error_over_time,
        "first_exceed_step": first_exceed_step,
    }


def mark_geometry(ax, x: np.ndarray, source_index: int, receiver_xs: np.ndarray) -> None:
    ax.axvspan(x[0], PHYSICAL_DOMAIN_START, color="0.90", label="External padding")
    ax.axvspan(PHYSICAL_DOMAIN_END, x[-1], color="0.90")
    ax.axvspan(PHYSICAL_DOMAIN_START, PHYSICAL_DOMAIN_END, color="white", alpha=0.0, label="Physical region")
    ax.axvline(x[source_index], color="black", linestyle="--", linewidth=1.5, label="Source")
    for i, xpos in enumerate(INTERFACE_POSITIONS_M):
        ax.axvline(
            xpos,
            color="darkgreen",
            linestyle="-.",
            linewidth=1.3,
            label="Layer interfaces" if i == 0 else None,
        )
    subset = np.linspace(0, len(receiver_xs) - 1, 13, dtype=int)
    for i, r in enumerate(subset):
        ax.axvline(
            receiver_xs[r],
            color="0.35",
            linestyle=":",
            alpha=0.25,
            label="Receivers" if i == 0 else None,
        )


def save_velocity_plot(model: dict) -> Path:
    x = model["x"]
    path = OUTPUT_DIR / "velocity_model_with_padding.png"
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.step(x, model["velocity"], where="post", linewidth=2.2, label="Velocity")
    mark_geometry(ax, x, model["source_index"], model["receiver_xs"])
    ax.set_title("Layered velocity model with external padding")
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_xlim(x[0], x[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_source_signal_plot(model: dict) -> Path:
    path = OUTPUT_DIR / "source_signal.png"
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.plot(model["t"], model["source_signal"], linewidth=2.0)
    ax.set_title("Ricker source signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_error_plot(model: dict, validation: dict) -> Path:
    path = OUTPUT_DIR / "error_over_time.png"
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.semilogy(model["t"], validation["error_over_time"], linewidth=2.0)
    ax.axhline(ERROR_TOLERANCE, color="tab:red", linestyle="--", label="1e-8 tolerance")
    ax.set_title("Time-domain vs Fourier-basis max error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("max(abs(u_time - u_fourier))")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def selected_snapshot_steps(t: np.ndarray) -> list[int]:
    snapshot_times = [0.18, 0.36, 0.58, 0.82, 1.08]
    return [int(np.argmin(np.abs(t - ts))) for ts in snapshot_times]


def save_snapshot_comparison(model: dict, validation: dict) -> Path:
    path = OUTPUT_DIR / "snapshots_time_vs_fourier.png"
    x = model["x"]
    t = model["t"]
    steps = selected_snapshot_steps(t)
    wave_time = validation["wavefield_time"]
    wave_fourier = validation["wavefield_fourier"]
    ymax = float(np.percentile(np.abs(wave_time), 99.8))
    if ymax == 0.0:
        ymax = 1.0

    fig, axes = plt.subplots(len(steps), 1, figsize=(11, 9.5), sharex=True)
    for ax, step in zip(axes, steps):
        ax.plot(x, wave_time[step], linewidth=1.8, label="time-domain")
        ax.plot(x, wave_fourier[step], "--", linewidth=1.4, label="Fourier-basis")
        ax.set_ylim(-1.15 * ymax, 1.15 * ymax)
        ax.set_title(f"Wavefield comparison at t = {t[step]:.3f} s")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Position x (m)")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_snapshot_difference(model: dict, validation: dict) -> Path:
    path = OUTPUT_DIR / "snapshots_difference.png"
    x = model["x"]
    t = model["t"]
    steps = selected_snapshot_steps(t)
    diff = validation["wavefield_time"] - validation["wavefield_fourier"]
    ymax = float(np.max(np.abs(diff[np.array(steps)])))
    if ymax == 0.0:
        ymax = 1.0e-16

    fig, axes = plt.subplots(len(steps), 1, figsize=(11, 9.5), sharex=True)
    for ax, step in zip(axes, steps):
        ax.plot(x, diff[step], linewidth=1.7)
        ax.set_ylim(-1.15 * ymax, 1.15 * ymax)
        ax.set_title(f"Difference u_time - u_fourier at t = {t[step]:.3f} s")
        ax.set_ylabel("Difference")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Position x (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_receiver_trace_comparison(model: dict, validation: dict) -> Path:
    path = OUTPUT_DIR / "receiver_trace_comparison.png"
    t = model["t"]
    receiver_xs = model["receiver_xs"]
    receiver_time = validation["receiver_time"]
    receiver_fourier = validation["receiver_fourier"]
    selected_xs = [450.0, 750.0, 1050.0, 1500.0, 2100.0, 2550.0]
    selected = [int(np.argmin(np.abs(receiver_xs - xpos))) for xpos in selected_xs]
    selected = sorted(set(selected))

    data = receiver_time[:, selected]
    max_abs = float(np.max(np.abs(data)))
    if max_abs == 0.0:
        max_abs = 1.0
    offset = 1.8 * max_abs

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for row, r in enumerate(selected):
        base = row * offset
        ax.plot(
            t,
            receiver_time[:, r] + base,
            linewidth=1.4,
            label=f"time x={receiver_xs[r]:.0f} m",
        )
        ax.plot(
            t,
            receiver_fourier[:, r] + base,
            "--",
            linewidth=1.1,
            label=f"Fourier x={receiver_xs[r]:.0f} m",
        )
    ax.set_title("Receiver trace comparison")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude with vertical offset")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_xt_image(model: dict, wavefield: np.ndarray, filename: str, title: str) -> Path:
    path = OUTPUT_DIR / filename
    x = model["x"]
    t = model["t"]
    vmax = float(np.percentile(np.abs(wavefield), 99.5))
    if vmax == 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(11, 6.6))
    im = ax.imshow(
        wavefield,
        aspect="auto",
        extent=[x[0], x[-1], t[-1], t[0]],
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.axvline(PHYSICAL_DOMAIN_START, color="black", linewidth=1.2)
    ax.axvline(PHYSICAL_DOMAIN_END, color="black", linewidth=1.2, label="Physical domain")
    ax.axvline(model["x"][model["source_index"]], color="black", linestyle="--", linewidth=1.3, label="Source")
    for i, xpos in enumerate(INTERFACE_POSITIONS_M):
        ax.axvline(
            xpos,
            color="darkgreen",
            linestyle="-.",
            linewidth=1.2,
            label="Layer interfaces" if i == 0 else None,
        )
    ax.set_title(title)
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Amplitude")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_outputs(model: dict, validation: dict) -> list[Path]:
    paths = [
        save_velocity_plot(model),
        save_source_signal_plot(model),
        save_error_plot(model, validation),
        save_snapshot_comparison(model, validation),
        save_snapshot_difference(model, validation),
        save_receiver_trace_comparison(model, validation),
        save_xt_image(
            model,
            validation["wavefield_time"],
            "wavefield_xt_time.png",
            "Time-domain wavefield",
        ),
        save_xt_image(
            model,
            validation["wavefield_fourier"],
            "wavefield_xt_fourier.png",
            "Fourier-basis full-operator wavefield",
        ),
        save_xt_image(
            model,
            validation["wavefield_time"] - validation["wavefield_fourier"],
            "wavefield_xt_difference.png",
            "Difference: time-domain minus Fourier-basis",
        ),
    ]

    np.save(OUTPUT_DIR / "wavefield_time.npy", validation["wavefield_time"])
    np.save(OUTPUT_DIR / "wavefield_fourier.npy", validation["wavefield_fourier"])
    np.save(OUTPUT_DIR / "receiver_time.npy", validation["receiver_time"])
    np.save(OUTPUT_DIR / "receiver_fourier.npy", validation["receiver_fourier"])
    np.save(OUTPUT_DIR / "error_over_time.npy", validation["error_over_time"])

    max_error = float(np.max(validation["error_over_time"]))
    final_error = float(validation["error_over_time"][-1])
    first_exceed = validation["first_exceed_step"]
    metadata = {
        "validation_description": "Existing layered finite-difference update represented in a Fourier basis using full dense operators.",
        "uses_existing_external_padding_design": True,
        "fourier_basis_full_operator_representation": True,
        "not_a_diagonal_fourier_solver": True,
        "not_a_periodic_toy_fft_solver": True,
        "mode_coupling_statement": "Layered velocity, damping, and boundary rows make B_F and G_F dense/full matrices with Fourier-mode coupling.",
        "physical_nx": PHYSICAL_NX,
        "total_nx": TOTAL_NX,
        "dx_m": DX,
        "dt_s": DT,
        "nt": NT,
        "cfl": float(model["cfl"]),
        "physical_domain_range_m": [PHYSICAL_DOMAIN_START, PHYSICAL_DOMAIN_END],
        "computational_domain_range_m": [float(model["x"][0]), float(model["x"][-1])],
        "damping_cells": DAMPING_CELLS,
        "damping_strength": DAMPING_STRENGTH,
        "sigma_max_per_s": float(model["sigma_max"]),
        "source_position_m": SOURCE_X,
        "source_index": int(model["source_index"]),
        "source_frequency_hz": SOURCE_FREQUENCY,
        "source_strength": SOURCE_STRENGTH,
        "source_direction": SOURCE_DIRECTION,
        "source_handling": "For source_direction='right', the existing solver uses a right-going initial packet and no additive source during stepping.",
        "interface_positions_m": [float(v) for v in INTERFACE_POSITIONS_M],
        "layer_velocities_m_per_s": [float(v) for v in LAYER_VELOCITIES_M_PER_S],
        "receiver_range_m": [float(model["receiver_xs"][0]), float(model["receiver_xs"][-1])],
        "receiver_count": int(len(model["receiver_xs"])),
        "B_shape": list(validation["B"].shape),
        "G_shape": list(validation["G"].shape),
        "B_F_shape": list(validation["B_F"].shape),
        "G_F_shape": list(validation["G_F"].shape),
        "B_F_density_note": "Dense matrix produced by Phi_inv @ B @ Phi.",
        "G_F_density_note": "Dense matrix produced by Phi_inv @ G @ Phi.",
        "max_error_over_all_time_steps": max_error,
        "final_time_step_error": final_error,
        "error_tolerance": ERROR_TOLERANCE,
        "first_step_exceeding_tolerance": None if first_exceed is None else int(first_exceed),
        "output_files": [str(path) for path in paths],
        "saved_arrays": [
            str(OUTPUT_DIR / "wavefield_time.npy"),
            str(OUTPUT_DIR / "wavefield_fourier.npy"),
            str(OUTPUT_DIR / "receiver_time.npy"),
            str(OUTPUT_DIR / "receiver_fourier.npy"),
            str(OUTPUT_DIR / "error_over_time.npy"),
        ],
    }
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return paths


def print_summary(model: dict, validation: dict) -> None:
    max_error = float(np.max(validation["error_over_time"]))
    final_error = float(validation["error_over_time"][-1])
    first_exceed = validation["first_exceed_step"]

    print("Layered Fourier-basis full-operator validation complete")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"total_nx: {TOTAL_NX}")
    print(f"physical nx: {PHYSICAL_NX}")
    print(f"dx: {DX:.3f} m")
    print(f"dt: {DT:.6f} s")
    print(f"nt: {NT}")
    print(f"CFL: {model['cfl']:.3f}")
    print(f"Physical domain range: {PHYSICAL_DOMAIN_START:.1f}-{PHYSICAL_DOMAIN_END:.1f} m")
    print(f"Computational domain range: {model['x'][0]:.1f}-{model['x'][-1]:.1f} m")
    print(f"Padding cell count: {DAMPING_CELLS} per side")
    print(f"Source position and index: x={SOURCE_X:.1f} m, index={model['source_index']}")
    print(
        "Interface positions: "
        + ", ".join(f"{xpos:.1f} m" for xpos in INTERFACE_POSITIONS_M)
    )
    print(
        "Layer velocities: "
        f"left={LAYER_VELOCITIES_M_PER_S[0]:.1f} m/s, "
        f"middle={LAYER_VELOCITIES_M_PER_S[1]:.1f} m/s, "
        f"right={LAYER_VELOCITIES_M_PER_S[2]:.1f} m/s"
    )
    print(f"Max error over all time steps: {max_error:.12e}")
    print(f"Final time step error: {final_error:.12e}")
    if first_exceed is not None:
        print(
            "WARNING: error exceeded "
            f"{ERROR_TOLERANCE:.1e} first at step {first_exceed} "
            f"(t={model['t'][first_exceed]:.6f} s)."
        )


def main() -> None:
    model = build_grid_and_model()
    validation = run_validation(model)
    paths = save_outputs(model, validation)
    print_summary(model, validation)
    print("Generated outputs:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
