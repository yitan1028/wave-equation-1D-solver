"""Compare boundary reflections with/without layers and sponge damping."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

OUTPUT_DIR = Path("outputs") / "boundary_diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "matplotlib_config"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / "cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from one_d_solver import SolverConfig, run_1d_forward


NX = 601
DX = 5.0
NT = 2400
DT = 0.0008
DOMAIN_END = (NX - 1) * DX
RECEIVER_XS = np.linspace(450.0, 2550.0, 121)
LATE_TIME = 0.9
REFLECTED_TIME = 1.0


def case_config(
    velocity_model: str,
    damping_cells: int,
    damping_strength: float,
) -> SolverConfig:
    return SolverConfig(
        nx=NX,
        dx=DX,
        nt=NT,
        dt=DT,
        velocity_model=velocity_model,
        velocity_value=1500.0,
        source_x=450.0,
        source_frequency=18.0,
        source_strength=0.8,
        source_direction="right",
        receiver_xs=RECEIVER_XS,
        damping_cells=damping_cells,
        damping_strength=damping_strength,
    )


def interface_positions(config: SolverConfig) -> list[float]:
    if config.velocity_model == "three_layer_reflection":
        return [0.35 * DOMAIN_END, 0.65 * DOMAIN_END]
    return []


def estimate_late_event_origin(result, config: SolverConfig) -> dict:
    """Estimate a boundary-reflection origin from the left-going characteristic."""
    du_dt = np.gradient(result.wavefield_history, config.dt, axis=0)
    du_dx = np.gradient(result.wavefield_history, config.dx, axis=1)
    left_going = np.abs(du_dt + result.velocity[np.newaxis, :] * du_dx)

    late = result.t > LATE_TIME
    search_start = 2600.0 if config.damping_cells > 0 else 2900.0
    boundary_zone = result.x >= search_start
    diagnostic = left_going[np.ix_(late, boundary_zone)]
    peak = float(np.max(diagnostic))

    if peak == 0.0:
        return {
            "estimated_origin_m": None,
            "estimated_origin_time_s": None,
            "origin_classification": "no detectable late event",
            "left_going_indicator_peak": 0.0,
        }

    # Find the earliest significant left-going signal, then its strongest x
    # location at that time. This distinguishes creation near the entrance from
    # a reflection that first appears at the physical boundary.
    threshold = 0.10 * peak
    significant_by_time = np.max(diagnostic, axis=1) >= threshold
    first_late_index = int(np.flatnonzero(significant_by_time)[0])
    origin_local_index = int(np.argmax(diagnostic[first_late_index]))
    late_indices = np.flatnonzero(late)
    boundary_indices = np.flatnonzero(boundary_zone)
    time_index = int(late_indices[first_late_index])
    x_index = int(boundary_indices[origin_local_index])
    origin_x = float(result.x[x_index])
    origin_t = float(result.t[time_index])
    if config.damping_cells == 0:
        classification = "near x=3000 physical boundary"
    else:
        entrance_distance = abs(origin_x - 2600.0)
        boundary_distance = abs(origin_x - DOMAIN_END)
        classification = (
            "near x=2600 sponge entrance"
            if entrance_distance <= boundary_distance
            else "near x=3000 physical boundary"
        )
    return {
        "estimated_origin_m": origin_x,
        "estimated_origin_time_s": origin_t,
        "origin_classification": classification,
        "left_going_indicator_peak": peak,
        "left_going_indicator_threshold": threshold,
    }


def compute_metrics(result, config: SolverConfig) -> dict:
    amplitude = np.abs(result.wavefield_history)
    late = result.t > LATE_TIME
    late_amplitude = amplitude[late]
    late_flat_index = int(np.argmax(late_amplitude))
    late_time_local, late_x_index = np.unravel_index(
        late_flat_index, late_amplitude.shape
    )
    late_time_indices = np.flatnonzero(late)
    late_time_index = int(late_time_indices[late_time_local])

    right_sponge = result.x >= 2600.0
    reflected_interior = (result.x >= 2000.0) & (result.x <= 2600.0)
    after_reflected_time = result.t > REFLECTED_TIME

    metrics = {
        "max_abs_amplitude_after_0_9_s": float(amplitude[late_time_index, late_x_index]),
        "max_abs_amplitude_x_m": float(result.x[late_x_index]),
        "max_abs_amplitude_time_s": float(result.t[late_time_index]),
        "max_abs_amplitude_right_sponge_x_ge_2600_m": float(
            np.max(amplitude[:, right_sponge])
        ),
        "max_abs_amplitude_interior_2000_to_2600_m_after_1_0_s": float(
            np.max(amplitude[np.ix_(after_reflected_time, reflected_interior)])
        ),
    }
    metrics.update(estimate_late_event_origin(result, config))
    return metrics


def draw_markers(ax, config: SolverConfig) -> None:
    ax.axvline(0.0, color="black", linewidth=1.4, label="Outer boundaries")
    ax.axvline(DOMAIN_END, color="black", linewidth=1.4)
    ax.axvline(config.source_x, color="lime", linestyle="--", label="Source")
    for index, xpos in enumerate(interface_positions(config)):
        ax.axvline(
            xpos,
            color="cyan",
            linestyle="--",
            label="Layer interfaces" if index == 0 else None,
        )
    if config.damping_cells > 0:
        sponge_width = config.damping_cells * config.dx
        ax.axvline(
            sponge_width,
            color="yellow",
            linestyle=":",
            linewidth=1.6,
            label="Sponge entrances",
        )
        ax.axvline(
            DOMAIN_END - sponge_width,
            color="yellow",
            linestyle=":",
            linewidth=1.6,
        )


def save_xt_image(case_id: str, title: str, result, config: SolverConfig) -> Path:
    path = OUTPUT_DIR / f"{case_id}_wavefield_xt.png"
    fig, ax = plt.subplots(figsize=(12, 6.5))
    scale = float(np.percentile(np.abs(result.wavefield_history), 99.5))
    if scale == 0.0:
        scale = 1.0
    image = ax.imshow(
        result.wavefield_history,
        aspect="auto",
        extent=[result.x[0], result.x[-1], result.t[-1], result.t[0]],
        cmap="seismic",
        vmin=-scale,
        vmax=scale,
    )
    draw_markers(ax, config)
    ax.set(title=f"{title}: full wavefield", xlabel="Position x (m)", ylabel="Time (s)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(image, ax=ax, label="Amplitude")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def save_velocity_image(case_id: str, title: str, result, config: SolverConfig) -> Path:
    path = OUTPUT_DIR / f"{case_id}_velocity_model.png"
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(result.x, result.velocity, color="tab:blue", linewidth=2.0)
    draw_markers(ax, config)
    ax.set(
        title=f"{title}: velocity model",
        xlabel="Position x (m)",
        ylabel="Velocity (m/s)",
        xlim=(0.0, DOMAIN_END),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def print_case_metrics(case_id: str, title: str, metrics: dict) -> None:
    print(f"\nCase {case_id}: {title}")
    print(
        "  Max |amplitude| after t > 0.9 s: "
        f"{metrics['max_abs_amplitude_after_0_9_s']:.12g}"
    )
    print(
        "  Location of late max: "
        f"x={metrics['max_abs_amplitude_x_m']:.1f} m, "
        f"t={metrics['max_abs_amplitude_time_s']:.6f} s"
    )
    print(
        "  Max |amplitude| in right sponge (x >= 2600 m): "
        f"{metrics['max_abs_amplitude_right_sponge_x_ge_2600_m']:.12g}"
    )
    print(
        "  Max |amplitude| in 2000-2600 m after t > 1.0 s: "
        f"{metrics['max_abs_amplitude_interior_2000_to_2600_m_after_1_0_s']:.12g}"
    )
    print(
        "  Estimated strongest late-event origin: "
        f"{metrics['origin_classification']} "
        f"(x={metrics['estimated_origin_m']}, "
        f"t={metrics['estimated_origin_time_s']})"
    )


def main() -> None:
    defaults = SolverConfig()
    current_damping_cells = defaults.damping_cells
    current_damping_strength = defaults.damping_strength
    cases = [
        (
            "A",
            "three layers, current damping",
            case_config(
                "three_layer_reflection",
                current_damping_cells,
                current_damping_strength,
            ),
        ),
        (
            "B",
            "constant 1500 m/s, current damping",
            case_config("constant", current_damping_cells, current_damping_strength),
        ),
        ("C", "constant 1500 m/s, no damping", case_config("constant", 0, current_damping_strength)),
        (
            "D",
            "three layers, no damping",
            case_config("three_layer_reflection", 0, current_damping_strength),
        ),
    ]

    print("Current diagnostic parameters")
    print(f"  damping_cells: {current_damping_cells}")
    print(f"  damping_strength: {current_damping_strength}")
    print(f"  total simulation time: {(NT - 1) * DT:.6f} s")
    print(
        "  right sponge entrance: "
        f"{DOMAIN_END - current_damping_cells * DX:.1f} m"
    )
    print(f"  receiver range: {RECEIVER_XS[0]:.1f}-{RECEIVER_XS[-1]:.1f} m")

    summaries = {}
    for case_id, title, config in cases:
        result = run_1d_forward(config)
        metrics = compute_metrics(result, config)
        xt_path = save_xt_image(case_id, title, result, config)
        velocity_path = save_velocity_image(case_id, title, result, config)
        sampled_velocity = {
            str(int(xpos)): float(result.velocity[int(round(xpos / config.dx))])
            for xpos in (2550.0, 2600.0, 2800.0, 3000.0)
        }
        config_metadata = asdict(config)
        config_metadata["receiver_xs"] = [float(xpos) for xpos in config.receiver_xs]
        metadata = {
            "case": case_id,
            "title": title,
            "config": config_metadata,
            "layer_interfaces_m": interface_positions(config),
            "left_sponge_entrance_m": (
                config.damping_cells * config.dx if config.damping_cells > 0 else None
            ),
            "right_sponge_entrance_m": (
                DOMAIN_END - config.damping_cells * config.dx
                if config.damping_cells > 0
                else None
            ),
            "receiver_range_m": [float(RECEIVER_XS[0]), float(RECEIVER_XS[-1])],
            "sampled_velocity_m_per_s": sampled_velocity,
            "metrics": metrics,
            "output_files": [str(xt_path), str(velocity_path)],
        }
        metadata_path = OUTPUT_DIR / f"{case_id}_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        summaries[case_id] = metadata
        print_case_metrics(case_id, title, metrics)
        print(
            "  Velocity at x=2550, 2600, 2800, 3000 m: "
            + ", ".join(
                f"{xpos} m/s" for xpos in sampled_velocity.values()
            )
        )

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nDiagnostic output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
