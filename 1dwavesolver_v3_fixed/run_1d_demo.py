"""
run_1d_demo.py

Run the V3 1D wave-solver teaching demo.

This version intentionally removes the old outputs folder before writing new figures,
so stale plots from earlier versions cannot be mistaken for the new result.
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np

from one_d_solver import SolverConfig, run_1d_forward
from one_d_visualize import make_all_visualizations


VERSION = "1D wave solver V3 - variable velocity + numeric update examples"


def main() -> None:
    output_dir = Path("outputs")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SolverConfig(
        nx=601,
        dx=5.0,
        nt=700,
        dt=0.0008,
        velocity_model="smooth_variable",
        source_x=1500.0,
        source_frequency=18.0,
        source_strength=0.8,
        receiver_xs=None,
        n_receivers=61,
        damping_cells=80,
        damping_strength=0.040,
        calculation_receiver_x=2100.0,
        calculation_steps=None,
    )

    result = run_1d_forward(config)

    np.save(output_dir / "wavefield_history_t_by_x.npy", result.wavefield_history)
    np.save(output_dir / "receiver_data_t_by_receiver.npy", result.receiver_data)
    np.save(output_dir / "x_positions_m.npy", result.x)
    np.save(output_dir / "time_seconds.npy", result.t)
    np.save(output_dir / "velocity_m_per_s.npy", result.velocity)
    np.save(output_dir / "source_signal.npy", result.source_signal)

    examples_for_json = []
    for ex in result.calculation_examples:
        examples_for_json.append({k: v for k, v in ex.items() if not k.endswith("array")})

    metadata = {
        "version": VERSION,
        "pde": "u_tt = v(x)^2 u_xx + source",
        "finite_difference_update": "u_next[i] = 2*u_curr[i] - u_prev[i] + (v[i]*dt/dx)^2*(u_curr[i+1]-2*u_curr[i]+u_curr[i-1]) + source_term",
        "velocity_model": config.velocity_model,
        "velocity_range_m_per_s": [float(result.velocity.min()), float(result.velocity.max())],
        "cfl": float(result.cfl),
        "source_x_m": float(result.x[result.source_index]),
        "receiver_count": int(len(result.receiver_xs)),
        "calculation_x_m": float(result.x[result.calculation_index]),
        "calculation_steps": [int(ex["step"]) for ex in result.calculation_examples],
        "calculation_examples": examples_for_json,
        "wavefield_history_shape": list(result.wavefield_history.shape),
        "receiver_data_shape": list(result.receiver_data.shape),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "VERSION.txt", "w", encoding="utf-8") as f:
        f.write(VERSION + "\n")
        f.write(f"velocity range: {result.velocity.min():.1f} to {result.velocity.max():.1f} m/s\n")
        f.write(f"receiver count: {len(result.receiver_xs)}\n")
        f.write(f"calculation steps: {[ex['step'] for ex in result.calculation_examples]}\n")

    paths = make_all_visualizations(result, output_dir)

    print("\n" + VERSION)
    print(f"Velocity range: {result.velocity.min():.1f} to {result.velocity.max():.1f} m/s")
    print(f"CFL number: {result.cfl:.3f}")
    print(f"Receiver count: {len(result.receiver_xs)}")
    print(f"Calculation point: x = {result.x[result.calculation_index]:.1f} m")
    print(f"Calculation steps: {[ex['step'] for ex in result.calculation_examples]}")
    print("\nGenerated figures:")
    for p in paths:
        print(f"  {p}")
    print("\nCheck outputs/VERSION.txt to confirm this is the new version.")


if __name__ == "__main__":
    main()
