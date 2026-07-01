# 1D Wave Equation Solver

This repository contains a 1D acoustic wave-equation solver for a layered
velocity model. The main simulation uses a finite-difference time-domain
update with external absorbing padding. The repository also includes a
Fourier-basis validation experiment that represents the same layered
finite-difference update as a full dense operator in Fourier coordinates and
compares it step by step against the time-domain solver.

The Fourier validation is not a periodic toy FFT solver and not a diagonal
constant-velocity Fourier solver. It transforms the full layered update
operators, so the layered velocity, damping, and boundary rows produce mode
coupling in dense Fourier-space matrices.

## Setup

```bash
cd "/Users/tansheng/research/quantum coding/1dwavesolver"
bash setup_local.sh
```

`setup_local.sh` creates `.venv`, installs `requirements.txt`, removes the old
`outputs/` folder, and runs the main finite-difference layered reflection demo.

## Main Finite-Difference Demo

Run the main layered reflection/transmission simulation directly with:

```bash
python run_1d_layered_reflection_demo.py
```

Output:

```text
outputs/layered_reflection_demo/
```

This demo uses:

```text
physical nx = 601
dx = 5.0 m
dt = 0.0008 s
nt = 2400
velocity layers = 3000, 2200, 1500 m/s
external padding = 80 cells per side
```

Key result files include:

```text
outputs/layered_reflection_demo/01_velocity_model_layers.png
outputs/layered_reflection_demo/05_seismic_record_time_receiver.png
outputs/layered_reflection_demo/06_full_wavefield_time_position.png
outputs/layered_reflection_demo/07_wave_propagation_animation.gif
outputs/layered_reflection_demo/metadata.json
```

## Fourier-Basis Validation

Run the full-operator Fourier-basis validation with:

```bash
python run_1d_layered_fourier_validation.py
```

Output:

```text
outputs/layered_fourier_validation/
```

This script builds two independent solvers inside the validation file:

- a physical/time-domain reference update matching the layered finite-difference scheme;
- a Fourier-basis solver using dense transformed operators `B_F = Phi_inv @ B @ Phi` and `G_F = Phi_inv @ G @ Phi`.

It writes comparison plots, arrays, and metadata:

```text
outputs/layered_fourier_validation/error_over_time.png
outputs/layered_fourier_validation/snapshots_time_vs_fourier.png
outputs/layered_fourier_validation/wavefield_xt_difference.png
outputs/layered_fourier_validation/error_over_time.npy
outputs/layered_fourier_validation/metadata.json
```

Check the validation result with:

```bash
cat outputs/layered_fourier_validation/metadata.json
```

The expected max error is near floating-point precision and should remain below
`1e-8`.
