#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# rm -rf ./cache/federated_disk

# 自动从 pyproject.toml 读取 snapshot-exp 的值
get_snapshot_exp() {
    if command -v python3 &> /dev/null; then
        python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
snap_exp = data.get('tool', {}).get('flwr', {}).get('app', {}).get('config', {}).get('snapshot-exp', 'flwr')
print(snap_exp)
"
    else
        # 备用方案：使用 grep 和 sed 解析
        grep -E "^snapshot-exp\s*=" pyproject.toml | sed -E 's/^snapshot-exp\s*=\s*"([^"]+)".*/\1/' | head -1 || echo "flwr"
    fi
}

SNAPSHOT_EXP=$(get_snapshot_exp)
LOG_DIR="./logs/${SNAPSHOT_EXP}"

echo "Using snapshot-exp: ${SNAPSHOT_EXP}"
echo "Log directory: ${LOG_DIR}"

# 确保日志目录存在
mkdir -p "${LOG_DIR}"

# 运行 Flower 并追加到正确的日志文件
uv run flwr run . local-simulation --stream 2>&1 | tee -a "${LOG_DIR}/flwr.log"
