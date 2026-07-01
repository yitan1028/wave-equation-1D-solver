# 1D Layered Reflection Wave Solver

This project runs the three-layer 1D acoustic wave reflection/transmission demo.
The maintained run target writes only the layered reflection result set:

```text
outputs/layered_reflection_demo/
```

## Run

```bash
cd "/Users/tansheng/research/quantum coding/1dwavesolver"
bash setup_local.sh
```

The setup script deletes the old `outputs/` folder before making new figures, then runs:

```bash
python run_1d_layered_reflection_demo.py
```

## Confirm the result

After running, check:

```bash
cat outputs/layered_reflection_demo/metadata.json
```

The primary velocity figure is:

```text
outputs/layered_reflection_demo/01_velocity_model_layers.png
```
