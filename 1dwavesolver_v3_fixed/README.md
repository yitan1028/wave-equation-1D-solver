# 1D Wave Solver Teaching Demo — V3

This version fixes the problem from the first demo:

- velocity is **not flat** anymore;
- the 1D velocity model is a smooth variable profile;
- the seismic record uses many receivers;
- `07_numerical_update_examples.png` shows real finite-difference numeric substitutions at one selected receiver/grid point.

## Run

```bash
cd "/Users/tansheng/research/quantum coding/1dwavesolver"
unzip -o ~/Downloads/1dwavesolver_v3_fixed.zip
bash setup_local.sh
```

The script deletes the old `outputs/` folder before making new figures, so you will not accidentally look at stale plots.

## Confirm the new version

After running, check:

```bash
cat outputs/VERSION.txt
```

You should see:

```text
1D wave solver V3 - variable velocity + numeric update examples
```

The velocity figure should be titled:

```text
1D variable velocity model
```

not the old `1D velocity model` flat-line figure.
