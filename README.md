# OpenPI Federated Learning (FL) Guide

(For pi0's original repo, see [pi0_ori.md](pi0_ori.md))

This guide explains the Flower-based, disk-exchange federated learning setup in this repo: how to run training (local simulation or remote), switch IID vs Non-IID, resume/snapshot, where artifacts/logs are written, and how to run inference from FL snapshots using the same centralized inference API.

Key files:
- Server (disk exchange): `scripts/federated_disk/server_flwr_disk.py`
- Client (IID, disk exchange): `scripts/federated_disk/client_flwr_disk.py`
- Client (Non-IID, disk exchange): `scripts/federated_disk/client_flwr_disk_noniid.py`
- Shared utils (save/load .npz, IID split, etc.): `scripts/tools_fl_fed.py`
- Launch scripts: `flwr.sh` (basic/auto-restart), `flwr_noniid.sh`
- Flower app config: `[tool.flwr.*]` sections in `pyproject.toml`


## 1. Overview

- Parameter exchange: no large tensors sent over the wire. The server aggregates and writes NPZ to disk. Clients receive only a file path and load locally, then apply to the model.
- Current global weights on server: `./cache/federated_disk/global/current/params.npz`
- Per-client per-round outputs: `./cache/federated_disk/client_<id>/round_<round>/params.npz` plus `meta.json` (contains examples, etc.).
- Optional periodic snapshots: every N rounds, the server saves an Orbax-style snapshot to `./checkpoints/<config-name>/<snapshot-exp>/<round>/` for inference/restore.
- Logs: under `./logs/<snapshot-exp>/` (contains `flwr.log`, `server.jsonl`, and `client_*.jsonl`).


## 2. Environment

- Python ≥ 3.11
- Dependencies listed in `pyproject.toml` (notably `flwr[simulation]`, `jax[cuda12]==0.5.3`, `flax==0.10.2`, `orbax-checkpoint==0.11.13`, etc.)
- Recommended to use uv (repo includes `uv.lock`):
  - Install: `uv sync`
  - Run: `uv run <your command>`

GPU/memory tips:
- Clients set `XLA_PYTHON_CLIENT_ALLOCATOR=platform` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` to reduce VRAM spikes.
- For multi-process on one machine, watch `CUDA_VISIBLE_DEVICES` and `XLA_PYTHON_CLIENT_MEM_FRACTION`.


## 2.1 Choose IID vs Non-IID (quick)

Switch `clientapp` under `[tool.flwr.app.components]` in `pyproject.toml`:

- IID (default):
  - `clientapp = "scripts.federated_disk.client_flwr_disk:app"`
- Non-IID:
  - `clientapp = "scripts.federated_disk.client_flwr_disk_noniid:app"`

Optional (only for Non-IID): configure `[tool.flwr.app.noniid.<config-name>].repo_id_by_client = [ ... ]` to give each client its own data source.


## 3. Running

### 3.1 Local simulation (recommended)

The repo already wires Flower app components in `pyproject.toml`:
- `[tool.flwr.app.components]`:
  - server: `scripts.federated_disk.server_flwr_disk:app`
  - client (IID default): `scripts.federated_disk.client_flwr_disk:app`

Quick start (logs to `./logs/<snapshot-exp>/flwr.log`):
- `./flwr.sh`
  - Reads `[tool.flwr.app.config]` from `pyproject.toml` and runs `uv run flwr run . local-simulation --stream`
  - Auto-restart flavor: `./flwr_noniid.sh` (name mentions noniid but it works for any config; includes failure restart and per-attempt logs)

Manual (dependencies installed):
- `uv run flwr run . local-simulation --stream`

### 3.2 Remote/local deployment

`[tool.flwr.federations]` in `pyproject.toml` provides multiple federations:
- `local-simulation` (default)
- `local-deployment` (local port)
- `remote-federation` (via SuperLink)

Pick one:
- `uv run flwr run . <federation-name> --stream`
- Or adapt the shell scripts accordingly.


## 4. App configuration (`pyproject.toml`)

Key options under `[tool.flwr.app.config]`:
- Rounds: `num-server-rounds`
- Client sampling: `min-fit-clients`, `min-available-clients`, `fraction-fit`
- Training: `config-name`, `total-clients`, `virtual-clients`, `local-steps`, `batch-size`, `num-workers`, `fsdp-devices`
- Snapshots: `snapshot-interval`, `snapshot-dir`, `snapshot-exp`
- Resume: `resume` (optionally `resume-from-round` – supported by the server strategy)
- Precision: `store-precision`, `agg-precision`
- FedOpt extension: `fed-opt = true/false` (when enabled, clients/servers include `opt_state`/`ema_params` in NPZ and aggregate them)

IID split seed:
- `[tool.flwr.app.iid.<config-name>].split_seed` (e.g., `split_seed = 42` for `pi0_libero_0813_fl`)

Non-IID data mapping (only with Non-IID client):
- `[tool.flwr.app.noniid.<config-name>].repo_id_by_client = [ ... ]`
  - `client_flwr_disk_noniid.py` replaces `TrainConfig.data.repo_id` with this per-client value.

Switch to Non-IID client:
- Set `[tool.flwr.app.components].clientapp = "scripts.federated_disk.client_flwr_disk_noniid:app"`
- Or override via a local branch/script.


## 5. Artifacts and logs

- Current global: `./cache/federated_disk/global/current/params.npz`
- Client per-round output: `./cache/federated_disk/client_<id>/round_<round>/params.npz` plus `meta.json`
- Server snapshots (if enabled): `./checkpoints/<config>/<snapshot-exp>/<round>/`
  - `params/` is an Orbax item that can be used for inference/restore
- Logs: `./logs/<snapshot-exp>/`
  - `flwr.log`: main Flower logs
  - `server.jsonl`: per-round sampling/aggregation/snapshot events
  - `client_*.jsonl`: client per-step loss and per-round summaries


## 6. Inference (aligned with centralized API)

Federated snapshots work with the same centralized inference API. Use one of the following:

### A) Build a Policy in Python and infer

Snapshot folder layout: `./checkpoints/<config-name>/<snapshot-exp>/<round>/` containing `params/` and `assets/`.

```python
from openpi.training import config as _config
from openpi.policies import policy_config

config_name = "pi0_libero_0813_fl"  # same as in FL training
checkpoint_dir = "./checkpoints/pi0_libero_0813_fl/flwr_iid_0820/10"  # pick a round to infer

policy = policy_config.create_trained_policy(
    _config.get_config(config_name),
    checkpoint_dir,
)

example = {
    "observation/exterior_image_1_left": ...,  # HWC or CHW depending on transforms
    "observation/wrist_image_left": ...,
    # ... other required inputs
    "prompt": "pick up the fork",
}

out = policy.infer(example)
action_chunk = out["actions"]
```

Notes:
- Same API as in README centralized inference; only `checkpoint_dir` points to the federated snapshot round.
- `assets/` holds normalization stats; `params/` holds model params, both produced by the server after aggregation.

### B) Run a Policy Server and infer locally/remotely

Use the unified script:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_libero_0813_fl \
  --policy.dir=./checkpoints/pi0_libero_0813_fl/flwr_iid_0820/10
```

Then follow the centralized docs (README) for remote inference or evaluation scripts (e.g., Libero).

### Compatibility and caveats

- Ensure `config-name` for inference matches the model config used in FL training; changes in model structure/dimensions can cause shape mismatches.
- If you must use the "current global" NPZ (`./cache/federated_disk/global/current/params.npz`): it contains params only and no `assets/`, so it doesn't directly fit `create_trained_policy`. Prefer snapshot directories. If you must use NPZ, you can manually assemble the model with a shape-intersection loader similar to the client's `_apply_model_params` (not recommended).


## 7. Switches and tips

- FedOpt (aggregate `opt_state`/`ema_params`): set `[tool.flwr.app.config].fed-opt = true`
- Per-round snapshots: `snapshot-interval = 1` (or larger)
- Resume: `resume = true` (server attempts to restore from the latest snapshot)
- Logs: `./logs/<snapshot-exp>/`; client step/avg_loss tracked in `client_*.jsonl`
- Memory: client code prefers CPU-side dtype/casting/host transfers to lower VRAM. If OOM persists, lower `batch-size`, `fsdp-devices`, or increase `virtual-clients` and aggregate more frequently.


## 8. File map (relevant to this guide)

- `scripts/federated_disk/server_flwr_disk.py`: server strategy (disk IO, snapshots, aggregation)
- `scripts/federated_disk/client_flwr_disk.py`: IID client (local train loop; avg loss = arithmetic mean over valid steps of the round)
- `scripts/federated_disk/client_flwr_disk_noniid.py`: Non-IID client (per-client dataset via TOML `repo_id_by_client`)
- `scripts/tools_fl_fed.py`: NPZ save/load, IID split, JSONL logging, etc.
- `flwr.sh` / `flwr_noniid.sh`: one-command Flower launch (local simulation), with log scoping and auto-restart
- `pyproject.toml`: Flower entry points, run config, federations, IID/Non-IID toggles
