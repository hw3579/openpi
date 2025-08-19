#!/bin/bash

# 低显存模式启动 Flower 客户端
# 用法: ./start_client_low_mem.sh [GPU_ID] [CLIENT_ID]

GPU_ID=${1:-0}
CLIENT_ID=${2:-0}

echo "启动客户端 $CLIENT_ID 在 GPU $GPU_ID (低显存模式)"

# 环境变量：严格控制 JAX 显存使用
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4  # 只使用 40% 显存
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_FORCE_GPU_ALLOW_GROWTH=true

# 强制单虚拟客户端，避免多模型副本
uv run python scripts/federated/client_flwr.py \
    --server-address 127.0.0.1:8080 \
    --client-id $CLIENT_ID \
    --total-clients 2 \
    --virtual-clients 1 \
    --fsdp-devices 1 \
    --batch-size 8 \
    --local-steps 10 \
    --offload-between-rounds \
    --grpc-max-mb 320
