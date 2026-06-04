# gaussian_dag — runnable examples

Five standalone scripts that reproduce the experiments of the accompanying
paper. Each script:

- reads no external configuration (constants at the top of the file);
- writes a result file `results/<name>.npz` and a figure
  `figures/<name>.pdf` next to itself;
- runs end-to-end on CPU in well under a minute.

Run any of them with:

```bash
uv run python examples/<script>.py
```

## Scripts

| Script | Paper reference | What it demonstrates |
| --- | --- | --- |
| `single_link_mimo.py` | Fig. 4 (a) | MIMO precoder optimisation `Y = (H F) X + Z` under `‖F‖_F² <= P`. PGA matches the classical water-filling optimum to ~6 digits. |
| `diamond_dag.py` | Fig. 4 (b) | Branch-precoder optimisation on the 4-node diamond DAG with two controllable branches and a fixed merge. Tracks the parent cross-covariance `K_{2,1}` and its contribution to the merging-node block. |
| `af_relay.py` | Fig. 4 (c) | Two-hop amplify-and-forward relay `V_0 → V_1 → V_2`. The relay gain `R` enters the edge matrix `A_{2,1} = H_2 R` and amplifies the relay noise; PGA balances the signal–noise trade-off without any per-hop derivation. |
| `input_covariance.py` | Fig. 4 (d) | Input-covariance shaping via a virtual edge `X = Q X̃` with `X̃ ~ CN(0, I)`. Recovers the water-filling optimum by treating `Q` as an ordinary controllable edge factor. |
| `multilayer_network.py` | Fig. 5 | Multi-layer Gaussian network with 11 nodes, 5 layers, 17 edges, `d = 4`. Nine relays carry shared controllable matrices `F_i` (broadcast across each relay's outgoing edges) jointly optimised under a shared total-power budget `P = 36`. End-to-end MI rises from 4.56 to 9.28 nats. |

## Output convention

Each run writes two files:

- `results/<script_name>.npz` — all numeric outputs (MI history, final
  parameters, topology metadata, budget settings). Loaded by the plotting
  helper inside the script and nothing else.
- `figures/<script_name>.pdf` — a publication-quality figure regenerated
  strictly from the `.npz` file (no hard-coded numbers).

The two output directories are created on first run and are listed in the
top-level `.gitignore`.

## Reproducibility

Every script seeds PyTorch and (where applicable) NumPy at the top. The
seeds are chosen to match the paper figures; changing them will change the
specific instance but not the qualitative conclusion.
