"""
Run a 1D layered-velocity reflection/transmission demo.

The solver PDE and finite-difference update are unchanged:
    u_tt = v(x)^2 u_xx + source

All outputs are written under outputs/layered_reflection_demo/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

OUTPUT_DIR = Path("outputs") / "layered_reflection_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "matplotlib_config"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / "cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from one_d_solver import (
    SolverConfig,
    SolverResult,
    run_1d_forward,
)


LAYER_VELOCITIES = (3000.0, 2200.0, 1500.0)
INTERFACE_POSITIONS_M = (1050.0, 1950.0)
PHYSICAL_MODEL_END_M = 3000.0


def interface_positions(result: SolverResult) -> list[float]:
    return list(INTERFACE_POSITIONS_M)


def mark_geometry(ax, result: SolverResult, interfaces: list[float], receivers: bool = True) -> None:
    ax.axvline(result.x[result.source_index], color="black", linestyle="--", linewidth=1.6, label="Source")
    for k, xpos in enumerate(interfaces):
        ax.axvline(xpos, color="darkgreen", linestyle="-.", linewidth=1.4, label="Layer interfaces" if k == 0 else None)
    if receivers:
        subset = np.linspace(0, len(result.receiver_xs) - 1, 13, dtype=int)
        for k, r in enumerate(subset):
            ax.axvline(result.receiver_xs[r], color="0.25", linestyle=":", alpha=0.30, label="Receivers" if k == 0 else None)


def plot_velocity_model(result: SolverResult, output_dir: Path, interfaces: list[float]) -> Path:
    path = output_dir / "01_velocity_model_layers.png"
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.step(result.x, result.velocity, where="post", linewidth=2.5, label="Velocity v(x)")
    mark_geometry(ax, result, interfaces, receivers=True)
    ax.set_title("Three-layer velocity model for reflection testing")
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Velocity v(x) (m/s)")
    ax.set_xlim(0.0, PHYSICAL_MODEL_END_M)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_source_signal(result: SolverResult, output_dir: Path) -> Path:
    path = output_dir / "02_source_signal.png"
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.plot(result.t, result.source_signal, linewidth=2.0)
    ax.set_title("Source time signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Source amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_wave_snapshots(result: SolverResult, output_dir: Path, interfaces: list[float]) -> Path:
    path = output_dir / "03_wavefield_snapshots.png"
    snapshot_times = [0.18, 0.36, 0.58, 0.82, 1.08]
    snapshot_steps = [int(np.argmin(np.abs(result.t - ts))) for ts in snapshot_times]
    ymax = np.percentile(np.abs(result.wavefield_history), 99.8)
    if ymax == 0:
        ymax = 1.0

    fig, axes = plt.subplots(len(snapshot_steps), 1, figsize=(11, 9.5), sharex=True)
    for ax, step in zip(axes, snapshot_steps):
        ax.plot(result.x, result.wavefield_history[step], linewidth=1.6)
        mark_geometry(ax, result, interfaces, receivers=False)
        ax.set_ylim(-1.15 * ymax, 1.15 * ymax)
        ax.set_xlim(0.0, PHYSICAL_MODEL_END_M)
        ax.set_title(f"Wavefield at t = {result.t[step]:.3f} s")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Position x (m)")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_receiver_traces(result: SolverResult, output_dir: Path, interfaces: list[float]) -> Path:
    path = output_dir / "04_receiver_traces.png"
    selected_xs = [
        0.08 * PHYSICAL_MODEL_END_M,
        0.20 * PHYSICAL_MODEL_END_M,
        0.32 * PHYSICAL_MODEL_END_M,
        0.50 * PHYSICAL_MODEL_END_M,
        0.78 * PHYSICAL_MODEL_END_M,
        0.92 * PHYSICAL_MODEL_END_M,
    ]
    selected = [int(np.argmin(np.abs(result.receiver_xs - xpos))) for xpos in selected_xs]
    selected = sorted(set(selected))
    data = result.receiver_data[:, selected]
    max_abs = np.max(np.abs(data))
    if max_abs == 0:
        max_abs = 1.0
    offset = 1.9 * max_abs

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for k, r in enumerate(selected):
        ax.plot(result.t, result.receiver_data[:, r] + k * offset, linewidth=1.35, label=f"x={result.receiver_xs[r]:.0f} m")
    ax.set_title("Receiver traces across layered model")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude with vertical offset")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", title="Receiver")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_seismic_record(result: SolverResult, output_dir: Path, interfaces: list[float]) -> Path:
    path = output_dir / "05_seismic_record_time_receiver.png"
    vmax = np.percentile(np.abs(result.receiver_data), 99.5)
    if vmax == 0:
        vmax = 1.0
    extent = [result.receiver_xs[0], result.receiver_xs[-1], result.t[-1], result.t[0]]

    fig, ax = plt.subplots(figsize=(10, 6.4))
    im = ax.imshow(result.receiver_data, aspect="auto", extent=extent, vmin=-vmax, vmax=vmax, cmap="seismic")
    mark_geometry(ax, result, interfaces, receivers=False)
    subset = np.linspace(0, len(result.receiver_xs) - 1, 13, dtype=int)
    for k, r in enumerate(subset):
        ax.axvline(
            result.receiver_xs[r],
            color="0.25",
            linestyle=":",
            alpha=0.30,
            label="Receivers" if k == 0 else None,
        )
    ax.set_title("Seismic record: time by receiver position")
    ax.set_xlabel("Receiver position x (m)")
    ax.set_ylabel("Time (s)")
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Recorded amplitude")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_wavefield_image(result: SolverResult, output_dir: Path, interfaces: list[float]) -> Path:
    path = output_dir / "06_full_wavefield_time_position.png"
    vmax = np.percentile(np.abs(result.wavefield_history), 99.5)
    if vmax == 0:
        vmax = 1.0
    extent = [result.x[0], result.x[-1], result.t[-1], result.t[0]]

    fig, ax = plt.subplots(figsize=(11, 6.8))
    im = ax.imshow(result.wavefield_history, aspect="auto", extent=extent, vmin=-vmax, vmax=vmax, cmap="seismic")
    mark_geometry(ax, result, interfaces, receivers=True)
    ax.set_xlim(0.0, PHYSICAL_MODEL_END_M)
    ax.set_title("Full simulated wavefield x-t image with interfaces")
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Time (s)")
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Wave amplitude")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def animate_wavefield(result: SolverResult, output_dir: Path, interfaces: list[float]) -> Path:
    path = output_dir / "07_wave_propagation_animation.gif"
    ymax = np.percentile(np.abs(result.wavefield_history), 99.8)
    if ymax == 0:
        ymax = 1.0

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    line, = ax.plot([], [], linewidth=1.7)
    time_text = ax.text(0.02, 0.90, "", transform=ax.transAxes)
    mark_geometry(ax, result, interfaces, receivers=False)
    ax.set_xlim(0.0, PHYSICAL_MODEL_END_M)
    ax.set_ylim(-1.15 * ymax, 1.15 * ymax)
    ax.set_title("Wave propagation in the three-layer model")
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Wave amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    frames = list(range(0, len(result.t), 7))

    def init():
        line.set_data([], [])
        time_text.set_text("")
        return line, time_text

    def update(frame):
        line.set_data(result.x, result.wavefield_history[frame])
        time_text.set_text(f"t = {result.t[frame]:.3f} s")
        return line, time_text

    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    anim.save(path, writer=PillowWriter(fps=25))
    plt.close(fig)
    return path


def main() -> None:
    nx = 601
    dx = 5.0
    nt = 2400
    dt = 0.0008
    domain_length = (nx - 1) * dx
    # The first 400 m is the left damping region, so place the source just beyond it.
    source_x = 450.0
    receiver_xs = np.linspace(450.0, 2550.0, 121)

    config = SolverConfig(
        nx=nx,
        dx=dx,
        nt=nt,
        dt=dt,
        velocity_model="three_layer_reflection",
        source_x=source_x,
        source_frequency=18.0,
        source_strength=0.8,
        source_direction="right",
        receiver_xs=receiver_xs,
        damping_cells=80,
        damping_strength=0.040,
        calculation_receiver_x=0.50 * PHYSICAL_MODEL_END_M,
    )
    result = run_1d_forward(config)
    interfaces = interface_positions(result)
    left_padding_range = [float(result.x[0]), 0.0]
    right_padding_range = [domain_length, float(result.x[-1])]
    physical_mask = (result.x >= 0.0) & (result.x <= domain_length)

    np.save(OUTPUT_DIR / "wavefield_history_t_by_x.npy", result.wavefield_history)
    np.save(OUTPUT_DIR / "receiver_data_t_by_receiver.npy", result.receiver_data)
    np.save(OUTPUT_DIR / "x_positions_m.npy", result.x)
    np.save(OUTPUT_DIR / "time_seconds.npy", result.t)
    np.save(OUTPUT_DIR / "velocity_m_per_s.npy", result.velocity)
    np.save(OUTPUT_DIR / "source_signal.npy", result.source_signal)

    paths = [
        plot_velocity_model(result, OUTPUT_DIR, interfaces),
        plot_source_signal(result, OUTPUT_DIR),
        plot_wave_snapshots(result, OUTPUT_DIR, interfaces),
        plot_receiver_traces(result, OUTPUT_DIR, interfaces),
        plot_seismic_record(result, OUTPUT_DIR, interfaces),
        plot_wavefield_image(result, OUTPUT_DIR, interfaces),
        animate_wavefield(result, OUTPUT_DIR, interfaces),
    ]

    metadata = {
        "velocity_model": config.velocity_model,
        "pde": "u_tt + sigma(x)*u_t = v(x)^2 u_xx + source",
        "finite_difference_update": "u_next[i] = (2*u_curr[i] - (1-kappa[i])*u_prev[i] + (v[i]*dt/dx)^2*(u_curr[i+1]-2*u_curr[i]+u_curr[i-1]))/(1+kappa[i]) + source_term, kappa=sigma*dt",
        "layer_velocities_m_per_s": list(LAYER_VELOCITIES),
        "interface_positions_m": [float(xpos) for xpos in interfaces],
        "physical_nx": int(config.nx),
        "total_nx": int(len(result.x)),
        "physical_x_range_m": [0.0, domain_length],
        "computational_x_range_m": [float(result.x[0]), float(result.x[-1])],
        "source_position_m": float(result.x[result.source_index]),
        "source_index": int(result.source_index),
        "source_direction": config.source_direction,
        "nbc": int(config.damping_cells),
        "damping_strength": float(config.damping_strength),
        "left_padding_range_m": left_padding_range,
        "right_padding_range_m": right_padding_range,
        "left_padding_velocity_m_per_s": float(result.velocity[0]),
        "right_padding_velocity_m_per_s": float(result.velocity[-1]),
        "receiver_range_m": [float(receiver_xs[0]), float(receiver_xs[-1])],
        "receiver_index_range": [
            int(result.receiver_indices[0]),
            int(result.receiver_indices[-1]),
        ],
        "sigma_max_per_s": float(np.max(result.sigma)),
        "sigma_max_in_physical_region_per_s": float(
            np.max(result.sigma[physical_mask])
        ),
        "dx_m": float(config.dx),
        "dt_s": float(config.dt),
        "cfl": float(result.cfl),
        "output_paths": [str(path) for path in paths],
    }
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Layered reflection demo complete")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Source position: {result.x[result.source_index]:.1f} m")
    print(f"Interface positions: {interfaces[0]:.1f} m, {interfaces[1]:.1f} m")
    print(
        "Layer velocities: "
        f"left={LAYER_VELOCITIES[0]:.1f} m/s, "
        f"middle={LAYER_VELOCITIES[1]:.1f} m/s, "
        f"right={LAYER_VELOCITIES[2]:.1f} m/s"
    )
    print(f"dx: {config.dx:.3f} m")
    print(f"dt: {config.dt:.6f} s")
    print(f"CFL number: {result.cfl:.3f}")
    print(f"Physical x range: 0.0-{domain_length:.1f} m")
    print(f"Computational x range: {result.x[0]:.1f}-{result.x[-1]:.1f} m")
    print(f"Padding cells per side: {config.damping_cells}")
    print(f"Damping strength: {config.damping_strength:.3f}")
    print(f"Left padding range: {left_padding_range[0]:.1f}-{left_padding_range[1]:.1f} m")
    print(f"Right padding range: {right_padding_range[0]:.1f}-{right_padding_range[1]:.1f} m")
    print(f"Left replicated padding velocity: {result.velocity[0]:.1f} m/s")
    print(f"Right replicated padding velocity: {result.velocity[-1]:.1f} m/s")
    print(f"Source computational index: {result.source_index}")
    print(
        "Receiver computational index range: "
        f"{result.receiver_indices[0]}-{result.receiver_indices[-1]}"
    )
    print(f"Receiver range: {receiver_xs[0]:.1f}-{receiver_xs[-1]:.1f} m")
    print(f"Sigma max: {np.max(result.sigma):.12f} 1/s")
    print("Generated outputs:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
