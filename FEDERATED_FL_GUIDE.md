# OpenPI 联邦学习（FL）使用指南

（对于pi0自己的原生仓库 查看[pi0_ori.md](pi0_ori.md)）

本指南介绍当前仓库中基于 Flower 的“磁盘参数交换”联邦学习框架：如何运行训练（本地模拟或远程）、如何切换 IID/Non-IID、断点恢复/快照、训练产物与日志位置，以及如何使用快照做推理。

参考关键文件：
- 服务器端（磁盘交换）：`scripts/federated_disk/server_flwr_disk.py`
- 客户端（IID 磁盘交换）：`scripts/federated_disk/client_flwr_disk.py`
- 客户端（Non-IID 磁盘交换）：`scripts/federated_disk/client_flwr_disk_noniid.py`
- 共享工具方法（保存/加载 .npz、IID 切分等）：`scripts/tools_fl_fed.py`
- 运行脚本：`flwr.sh`（基础/自动重启版）、`flwr_noniid.sh`
- Flower 应用配置：`pyproject.toml` 的 `[tool.flwr.*]` 段


## 1. 框架概览

- 参数交换方式：不通过网络发送大张量，服务器聚合结果写入磁盘为 NPZ；客户端接收的“参数”也是磁盘路径，通过本地读取应用到模型。
- 服务器当前全局权重：`./cache/federated_disk/global/current/params.npz`
- 客户端每轮输出：`./cache/federated_disk/client_<id>/round_<round>/params.npz`，并写入 `meta.json`（包含 examples 等）。
- 周期性快照（可选）：服务器每隔 N 轮把“可复用推理快照”写入 `./checkpoints/<config-name>/<snapshot-exp>/<round>/`（Orbax 结构）。
- 日志：按实验名（snapshot-exp）归档在 `./logs/<snapshot-exp>/`（含 `flwr.log`、`server.jsonl`、`client_*.jsonl`）。


## 2. 环境准备

- Python ≥ 3.11
- 依赖见 `pyproject.toml`（包含 `flwr[simulation]`, `jax[cuda12]==0.5.3`, `flax==0.10.2`, `orbax-checkpoint==0.11.13` 等）
- 建议使用 uv 管理依赖（仓库自带 `uv.lock`）：
  - 首次安装：`uv sync`
  - 运行命令：`uv run <your command>`

GPU/内存建议（可选）：
- 客户端已在代码中设置 `XLA_PYTHON_CLIENT_ALLOCATOR=platform` 与 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 以减少显存峰值。
- 如使用单机多进程，请观察 `CUDA_VISIBLE_DEVICES` 与 `XLA_PYTHON_CLIENT_MEM_FRACTION`。

## 2.1 选择 IID / Non-IID（快速指引）

只需在 `pyproject.toml` 的 `[tool.flwr.app.components]` 下切换 `clientapp` 指向：

- IID（默认）：
  - `clientapp = "scripts.federated_disk.client_flwr_disk:app"`
- Non-IID：
  - `clientapp = "scripts.federated_disk.client_flwr_disk_noniid:app"`

可选（仅 Non-IID 生效）：在 `[tool.flwr.app.noniid.<config-name>]` 下配置 `repo_id_by_client = [ ... ]`，为每个客户端指定独立数据源。


## 3. 运行方式

### 3.1 本地模拟（推荐）

仓库已配置 Flower App 入口：
- `pyproject.toml` 中 `[tool.flwr.app.components]`：
  - server：`scripts.federated_disk.server_flwr_disk:app`
  - client（默认 IID）：`scripts.federated_disk.client_flwr_disk:app`

快速启动（会把日志写入 `./logs/<snapshot-exp>/flwr.log`）：
- 直接运行脚本：`./flwr.sh`
  - 自动读取 `pyproject.toml` 的 `[tool.flwr.app.config]`，使用 `uv run flwr run . local-simulation --stream` 启动
  - 支持自动重启版本：`./flwr_noniid.sh`（同名但可用于任意配置，含失败重启逻辑与分段日志）

也可手动运行（需已安装依赖）：
- `uv run flwr run . local-simulation --stream`

### 3.2 远程/本地部署

`pyproject.toml` 中 `[tool.flwr.federations]` 提供多种 Federation：
- `local-simulation`（默认）
- `local-deployment`（本地端口）
- `remote-federation`（通过 SuperLink）

选择方式：
- `uv run flwr run . <federation-name> --stream`
- 或在脚本中替换为对应 federation。


## 4. 配置说明（`pyproject.toml`）

位于 `[tool.flwr.app.config]` 的核心项：
- 训练轮数：`num-server-rounds`
- 客户端采样：`min-fit-clients`, `min-available-clients`, `fraction-fit`
- 训练参数：`config-name`, `total-clients`, `virtual-clients`, `local-steps`, `batch-size`, `num-workers`, `fsdp-devices`
- 快照：`snapshot-interval`, `snapshot-dir`, `snapshot-exp`
- 恢复：`resume`，可配合 `resume-from-round`（服务器端策略支持）
- 精度：`store-precision`, `agg-precision`
- 联邦优化器扩展：`fed-opt = true/false`（启用后客户端/服务器会在 NPZ 里带上 `opt_state`/`ema_params` 并做聚合）

IID 切分种子：
- `[tool.flwr.app.iid.<config-name>].split_seed`（例如 `pi0_libero_0813_fl` 下设置 `split_seed = 42`）

Non-IID 数据源映射（仅在使用 Non-IID 客户端时生效）：
- `[tool.flwr.app.noniid.<config-name>].repo_id_by_client = [ ... ]`
  - `client_flwr_disk_noniid.py` 会把每个 client 的 `TrainConfig.data.repo_id` 替换为这里的值。

切换 Non-IID 客户端：
- 把 `[tool.flwr.app.components].clientapp` 改成 `scripts.federated_disk.client_flwr_disk_noniid:app`
- 或在本地分支/脚本中临时覆盖。


## 5. 训练产物与日志

- 当前全局：`./cache/federated_disk/global/current/params.npz`
- 客户端每轮输出：`./cache/federated_disk/client_<id>/round_<round>/params.npz` 与 `meta.json`
- 服务器快照（若启用）：`./checkpoints/<config>/<snapshot-exp>/<round>/`
  - 其中 `params/` 为 Orbax 保存的参数目录，可用于推理/恢复
- 日志：`./logs/<snapshot-exp>/`
  - `flwr.log`：Flower 主流程日志
  - `server.jsonl`：每轮采样/聚合/快照事件
  - `client_*.jsonl`：客户端每步 `loss`、每轮汇总等


## 6. 推理流程（对齐集中式推理接口）

联邦训练得到的“服务器快照”可直接复用集中式推理接口。推荐使用下述两种方式之一：

### 方式 A：在 Python 中直接创建 Policy 并推理

快照目录形如：`./checkpoints/<config-name>/<snapshot-exp>/<round>/`（目录下包含 `params/` 与 `assets/`）。

```python
from openpi.training import config as _config
from openpi.policies import policy_config

config_name = "pi0_libero_0813_fl"            # 与联邦训练时一致
checkpoint_dir = "./checkpoints/pi0_libero_0813_fl/flwr_iid_0820/10"  # 选择要推理的轮次

# 创建已训练策略（与 README 集中式示例一致）
policy = policy_config.create_trained_policy(
    _config.get_config(config_name),
    checkpoint_dir,
)

# 构造一条与训练同分布的输入（键名需符合对应数据/模型 transforms）
example = {
    "observation/exterior_image_1_left": ...,  # HWC 或 CHW，取决于 transforms
    "observation/wrist_image_left": ...,
    # ... 其他必需观测
    "prompt": "pick up the fork",
}

out = policy.infer(example)
action_chunk = out["actions"]
```

说明：
- 该接口与 README“集中式”推理完全一致；唯一差别是 `checkpoint_dir` 指向联邦服务器保存的快照轮次目录。
- 快照中 `assets/` 存放了归一化统计，`params/` 存放模型参数，均由联邦服务器在聚合后写入。

### 方式 B：启动 Policy Server 再远程/本地推理

与集中式一致，使用统一的服务脚本：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_libero_0813_fl \
  --policy.dir=./checkpoints/pi0_libero_0813_fl/flwr_iid_0820/10
```

随后按 README 中的远程推理或评测脚本方式进行调用（例如 Libero 评测脚本，或 docs/remote_inference.md 的示例客户端）。

### 兼容性与注意事项

- 请确保用于推理的 `config-name` 与联邦训练时创建模型的配置一致；若中途改过模型结构或维度，可能出现形状不匹配。
- 如确需从“当前全局”NPZ（`./cache/federated_disk/global/current/params.npz`）推理：该文件仅含参数、不含 `assets/`，不直接适配集中式 `create_trained_policy`。建议优先使用“服务器快照”目录；若必须用 NPZ，可参考客户端 `_apply_model_params` 的按形状交集加载策略自行组装模型（不推荐）。


## 7. 常见开关与排错

- 启用 FedOpt（聚合 `opt_state`/`ema_params`）：`[tool.flwr.app.config].fed-opt = true`
- 每轮快照：`snapshot-interval = 1`（或更大间隔）
- 断点恢复：`resume = true`（服务器会尝试从最近快照恢复）
- 日志位置：`./logs/<snapshot-exp>/`；客户端 step/avg_loss 会滚动记录在 `client_*.jsonl`
- 内存：客户端代码已尽量在 CPU 上做 dtype/cast/host 化，仍 OOM 可减少 `batch-size`、`fsdp-devices`、或提高 `virtual-clients` 做小步聚合


## 8. 目录速览（与本指南相关）

- `scripts/federated_disk/server_flwr_disk.py`：服务器策略（磁盘读写、快照、聚合）
- `scripts/federated_disk/client_flwr_disk.py`：IID 客户端（本地训练循环、avg loss=当轮有效 step 的算术平均）
- `scripts/federated_disk/client_flwr_disk_noniid.py`：Non-IID 客户端（按 TOML 的 `repo_id_by_client` 映射）
- `scripts/tools_fl_fed.py`：NPZ save/load、IID 切分、日志追加等工具
- `flwr.sh` / `flwr_noniid.sh`：一键运行 Flower（本地模拟），含日志归档与自动重启
- `pyproject.toml`：Flower 入口、运行参数、federations、IID/Non-IID 配置


