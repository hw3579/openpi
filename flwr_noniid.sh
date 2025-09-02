#!/bin/bash

set -o pipefail  # 使 pipeline 返回非 0 时可检测到失败

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

# ================= 自动重启配置 =================
# 可选：最大重启次数 (0 或不设表示无限重启)
: "${RESTART_MAX:=0}"   # 导出 RESTART_MAX 环境变量可覆盖，例：RESTART_MAX=10 ./flwr.sh
# 可选：两次重启之间的延迟秒数（可为 0 代表立刻重启）
: "${RESTART_DELAY:=1}" # 覆盖方式：RESTART_DELAY=0 ./flwr.sh
# 可选：若存在这个文件则停止继续重启
STOP_FILE="${LOG_DIR}/stop"

echo "RESTART_MAX=${RESTART_MAX} (0=无限)" | tee -a "${LOG_DIR}/flwr.log"
echo "RESTART_DELAY=${RESTART_DELAY}" | tee -a "${LOG_DIR}/flwr.log"
echo "创建 ${STOP_FILE} 或 Ctrl-C 可停止循环" | tee -a "${LOG_DIR}/flwr.log"

ATTEMPT=1

trap 'echo "收到中断信号, 退出重启循环" | tee -a "${LOG_DIR}/flwr.log"; exit 130' INT TERM

while :; do
    echo "================ Attempt ${ATTEMPT} @ $(date '+%F %T') ================" | tee -a "${LOG_DIR}/flwr.log"
    ATTEMPT_LOG="${LOG_DIR}/attempt_${ATTEMPT}.log"
    # 单次运行日志同时写主日志和尝试日志
    # 使用子 shell 防止变量泄漏
            # 终端实时输出 + 追加主日志 + 写入本次 attempt 独立日志
            # 不使用子 shell 以便正确获取 PIPESTATUS[0]
            numactl --interleave=all uv run flwr run . local-simulation --stream 2>&1 \
                | tee >(tee "${ATTEMPT_LOG}" >/dev/null) -a "${LOG_DIR}/flwr.log"
            EXIT_CODE=${PIPESTATUS[0]}  # flower-simulation 的真实退出码

    # 模式匹配判定异常（即使 EXIT_CODE=0）
        if grep -qiE 'RESOURCE_EXHAUSTED|Out of memory|Traceback|RuntimeError|returned non-zero exit status|Failed to connect to GCS' "${ATTEMPT_LOG}"; then
        HAS_ERROR_LOG=1
    else
        HAS_ERROR_LOG=0
    fi

    if [ -f "${STOP_FILE}" ]; then
        echo "检测到停止文件 ${STOP_FILE}，不再重启 (exit code=${EXIT_CODE})" | tee -a "${LOG_DIR}/flwr.log"
        break
    fi

    # 判断是否认为成功
    if [ ${EXIT_CODE} -eq 0 ] && [ ${HAS_ERROR_LOG} -eq 0 ]; then
        echo "进程正常退出 (code=0)，未检测到异常关键字，停止重启" | tee -a "${LOG_DIR}/flwr.log"
        break
    fi

    if [ ${RESTART_MAX} -ne 0 ] && [ ${ATTEMPT} -ge ${RESTART_MAX} ]; then
        echo "已达到最大重启次数 ${RESTART_MAX}，停止 (最后退出码=${EXIT_CODE})" | tee -a "${LOG_DIR}/flwr.log"
        break
    fi

    echo "进程异常退出或检测到错误日志 (exit=${EXIT_CODE} errFlag=${HAS_ERROR_LOG})，准备重启 (下一次 Attempt $((ATTEMPT+1)))" | tee -a "${LOG_DIR}/flwr.log"
    ATTEMPT=$((ATTEMPT+1))
    if [ "${RESTART_DELAY}" != "0" ]; then
        sleep "${RESTART_DELAY}"
    fi
done

echo "退出 flwr.sh (总运行次数=${ATTEMPT})" | tee -a "${LOG_DIR}/flwr.log"
