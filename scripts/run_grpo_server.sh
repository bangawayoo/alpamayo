#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GRPO Post-Training with vLLM Server Mode
#
# Launches a vLLM server on one GPU and runs GRPO training on another.
# The server handles rollout generation; the trainer handles optimization.
#
# Usage:
#   ./scripts/run_grpo_server.sh                          # server on GPU 0, train on GPU 1
#   ./scripts/run_grpo_server.sh --smoke                  # quick smoke test
#   ./scripts/run_grpo_server.sh --server-gpu 2 --train-gpu 3
#   ./scripts/run_grpo_server.sh --model nvidia/Alpamayo-R1-10B --port 8000
#   ./scripts/run_grpo_server.sh --no-lora                # full-parameter training (FSDP enabled)
#   ./scripts/run_grpo_server.sh --gpu-mem-util 0.5        # limit vLLM GPU memory (default: 0.85)
#   ./scripts/run_grpo_server.sh --smoke --max-samples 5  # extra overrides forwarded to run_grpo.sh
#
# Prerequisites:
#   pip install vllm trl
#   At least 2 GPUs available

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a; source "$REPO_ROOT/.env" 2>/dev/null || true; set +a

# ---------------------------------------------------------------
# Auto-detect available GPUs
# ---------------------------------------------------------------
# Use CUDA_VISIBLE_DEVICES if set, otherwise query nvidia-smi.
# nvidia-smi -L lists both physical GPUs and MIG slices (indented with
# leading spaces). Count only non-indented "GPU N:" lines to avoid
# double-counting MIG instances as separate GPUs.  When MIG devices
# exist, count those instead since they are the actual compute units.
MIG_COUNT=$(nvidia-smi -L 2>/dev/null | grep -c "^  MIG" || true)
if [[ "$MIG_COUNT" -gt 0 ]]; then
    NUM_GPUS="$MIG_COUNT"
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true)
fi
if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "Error: no GPUs detected by nvidia-smi"
    exit 1
fi

# ---------------------------------------------------------------
# Defaults (adapt to available GPUs)
# ---------------------------------------------------------------
if [[ "$NUM_GPUS" -ge 2 ]]; then
    SERVER_GPU=0
    TRAIN_GPU=1
    TENSOR_PARALLEL_SIZE=1
else
    # Single GPU / MIG — share the device (server + training take turns)
    SERVER_GPU=0
    TRAIN_GPU=0
    TENSOR_PARALLEL_SIZE=1
fi
# Full AlpamayoR1 model — the VLM backbone is extracted automatically
# before launching the vLLM server (vLLM can't load the full model directly).
MODEL="nvidia/Alpamayo-R1-10B"
MODEL_IMPL="auto"
PORT=8000
NO_LORA=0
GPU_MEM_UTIL=0.9    # fraction of GPU memory vLLM may use
HEALTH_TIMEOUT=300   # seconds to wait for server readiness
HEALTH_INTERVAL=5    # seconds between health checks
EXTRA_ARGS=()        # forwarded to run_grpo.sh

# ---------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-gpu)
            SERVER_GPU="$2"
            shift 2
            ;;
        --train-gpu)
            TRAIN_GPU="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --model-impl)
            MODEL_IMPL="$2"
            shift 2
            ;;
        --tensor-parallel-size|--tp)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --no-lora)
            NO_LORA=1
            shift
            ;;
        --gpu-memory-utilization|--gpu-mem-util)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        --health-timeout)
            HEALTH_TIMEOUT="$2"
            shift 2
            ;;
        *)
            # Everything else forwarded to run_grpo.sh
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="/home/jovyan/conda/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Error: venv python not found at $VENV_PYTHON"
    echo "Run 'uv sync' first to create the virtual environment."
    exit 1
fi

# ---------------------------------------------------------------
# Validate GPUs
# ---------------------------------------------------------------
if [[ "$SERVER_GPU" != "$TRAIN_GPU" ]]; then
    :  # separate GPUs — ideal
elif [[ "$NUM_GPUS" -le 1 ]]; then
    echo "Warning: only $NUM_GPUS GPU(s) detected — server and trainer will share GPU $SERVER_GPU"
else
    echo "Error: --server-gpu and --train-gpu must be different (got $SERVER_GPU for both)"
    exit 1
fi

echo "================================================================"
echo "  Alpamayo-R1 GRPO — vLLM Server Mode"
echo "================================================================"
echo "  Available GPUs: $NUM_GPUS"
echo "  Server GPU:  $SERVER_GPU  (TP=$TENSOR_PARALLEL_SIZE)"
echo "  Train GPU:   $TRAIN_GPU"
echo "  Model:       $MODEL"
echo "  Port:        $PORT"
echo "  GPU mem util: $GPU_MEM_UTIL"
echo "  LoRA:        $(if [[ "$NO_LORA" -eq 1 ]]; then echo "disabled (full-parameter)"; else echo "enabled"; fi)"
echo "  Extra args:  ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"
echo "================================================================"
echo ""

# ---------------------------------------------------------------
# Cleanup: kill server on exit
# ---------------------------------------------------------------
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "Stopping vLLM server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "Server stopped."
    fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------
# 1. Extract VLM backbone from full model
# ---------------------------------------------------------------
# vLLM cannot load the full AlpamayoR1 architecture (Expert/Diffusion
# heads are unknown). We extract just the VLM (Qwen3VLForConditionalGeneration)
# to a cache directory so vLLM can serve it.
VLM_CACHE_DIR="$PROJECT_ROOT/.cache/vlm_extracted"

if [[ -f "$VLM_CACHE_DIR/config.json" ]]; then
    echo "Using cached VLM at $VLM_CACHE_DIR"
else
    echo "Extracting VLM backbone from $MODEL..."
    PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}" "$VENV_PYTHON" \
        "$SCRIPT_DIR/extract_vlm.py" \
        --model "$MODEL" \
        --output "$VLM_CACHE_DIR"
    echo "VLM extracted to $VLM_CACHE_DIR"
fi
echo ""

# ---------------------------------------------------------------
# 2. Start vLLM server
# ---------------------------------------------------------------
echo "Starting vLLM server on GPU $SERVER_GPU (port $PORT)..."

CUDA_VISIBLE_DEVICES="$SERVER_GPU" "$VENV_PYTHON" \
    "$SCRIPT_DIR/vllm/vllm_serve_patched.py" \
    --model "$VLM_CACHE_DIR" \
    --vllm_model_impl "$MODEL_IMPL" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --port "$PORT" \
    --max_model_len 8192 \
    --enforce_eager &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"

# ---------------------------------------------------------------
# 3. Wait for server to be ready
# ---------------------------------------------------------------
echo "Waiting for server health check (timeout: ${HEALTH_TIMEOUT}s)..."

ELAPSED=0
SERVER_URL="http://localhost:${PORT}"

while [[ $ELAPSED -lt $HEALTH_TIMEOUT ]]; do
    # Check server process is still alive
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: vLLM server process died unexpectedly."
        wait "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi

    # Try the health endpoint
    if curl -sf "${SERVER_URL}/health" >/dev/null 2>&1; then
        echo "Server is ready! (took ${ELAPSED}s)"
        break
    fi

    sleep "$HEALTH_INTERVAL"
    ELAPSED=$((ELAPSED + HEALTH_INTERVAL))
done

if [[ $ELAPSED -ge $HEALTH_TIMEOUT ]]; then
    echo "Error: server did not become ready within ${HEALTH_TIMEOUT}s"
    exit 1
fi

# ---------------------------------------------------------------
# 4. Run training on the other GPU
# ---------------------------------------------------------------
echo ""
echo "Starting GRPO training on GPU $TRAIN_GPU..."
echo "================================================================"

# With LoRA, FSDP is incompatible with vLLM server weight sync (LoRA merge
# fails on sharded weights) — use --no-fsdp so each GPU holds a full copy.
# Without LoRA (full-parameter), FSDP works fine with server sync since
# summon_full_params gathers weights before each sync.
TRAIN_EXTRA_ARGS=("--vllm-server" "vllm.server_port=$PORT")
if [[ "$NO_LORA" -eq 1 ]]; then
    TRAIN_EXTRA_ARGS+=("--no-lora" "--fsdp")
else
    TRAIN_EXTRA_ARGS+=("--no-fsdp")
fi

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$SCRIPT_DIR/run_grpo.sh" \
    "${TRAIN_EXTRA_ARGS[@]}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo ""
echo "================================================================"
echo "  Training complete! Shutting down server..."
echo "================================================================"
