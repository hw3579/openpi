# export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# rm -rf ./cache/federated_disk
uv run flwr run . local-simulation --stream 2>&1 | tee -a ./logs/flwr.log