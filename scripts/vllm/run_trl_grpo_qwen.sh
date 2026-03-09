#!/usr/bin/env bash
# TRL GRPO training for a standard Qwen model with vLLM server mode.
#
# This script:
#   1. Starts a vLLM server on dedicated GPU(s)
#   2. Waits for the server to become healthy
#   3. Runs GRPO training on separate GPU(s) using TRL's GRPOTrainer
#   4. Cleans up the server on exit
#
# Usage:
#   ./scripts/vllm/run_trl_grpo_qwen.sh                              # defaults: server GPU 0, train GPU 1
#   ./scripts/vllm/run_trl_grpo_qwen.sh --smoke                      # quick smoke test (20 samples, short gen)
#   ./scripts/vllm/run_trl_grpo_qwen.sh --model Qwen/Qwen2.5-7B-Instruct --server-gpus 0,1 --train-gpus 2,3
#   ./scripts/vllm/run_trl_grpo_qwen.sh --tp 2 --dp 1                # tensor parallel=2 on server
#
# Prerequisites:
#   pip install "trl[vllm]" datasets

set -euo pipefail

# ---------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------
MODEL="Qwen/Qwen3-0.6B"
SERVER_GPUS="0"
TRAIN_GPUS="1[]"
TP_SIZE=1           # tensor parallel size for vLLM server
DP_SIZE=1           # data parallel size for vLLM server
PORT=8000
HEALTH_TIMEOUT=300  # seconds to wait for server readiness
HEALTH_INTERVAL=5
SMOKE=0
EXTRA_TRAIN_ARGS=()

# ---------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2";       shift 2 ;;
        --server-gpus)    SERVER_GPUS="$2"; shift 2 ;;
        --train-gpus)     TRAIN_GPUS="$2";  shift 2 ;;
        --tp)             TP_SIZE="$2";     shift 2 ;;
        --dp)             DP_SIZE="$2";     shift 2 ;;
        --port)           PORT="$2";        shift 2 ;;
        --health-timeout) HEALTH_TIMEOUT="$2"; shift 2 ;;
        --smoke)          SMOKE=1;          shift ;;
        *)                EXTRA_TRAIN_ARGS+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Try project venv, then ar1_venv, then system python
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif [[ -x "$PROJECT_ROOT/ar1_venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/ar1_venv/bin/python"
else
    PYTHON="$(which python3)"
fi

echo "================================================================"
echo "  TRL GRPO — Qwen + vLLM Server Mode"
echo "================================================================"
echo "  Model:        $MODEL"
echo "  Server GPUs:  $SERVER_GPUS  (TP=$TP_SIZE, DP=$DP_SIZE)"
echo "  Train GPUs:   $TRAIN_GPUS"
echo "  Port:         $PORT"
echo "  Python:       $PYTHON"
if [[ "$SMOKE" -eq 1 ]]; then
echo "  Mode:         SMOKE TEST"
fi
echo "================================================================"
echo ""

# ---------------------------------------------------------------
# Validate: server and train GPUs must not overlap
# ---------------------------------------------------------------
IFS=',' read -ra SG <<< "$SERVER_GPUS"
IFS=',' read -ra TG <<< "$TRAIN_GPUS"
for sg in "${SG[@]}"; do
    for tg in "${TG[@]}"; do
        if [[ "$sg" == "$tg" ]]; then
            echo "Error: GPU $sg is in both --server-gpus and --train-gpus."
            echo "The vLLM server and trainer must run on separate CUDA devices."
            exit 1
        fi
    done
done

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
# 1. Start vLLM server
# ---------------------------------------------------------------
echo "Starting vLLM server on GPU(s) $SERVER_GPUS..."

CUDA_VISIBLE_DEVICES="$SERVER_GPUS" "$PYTHON" -c "
import sys; sys.argv = ['trl'] + sys.argv[1:]
from trl.cli import main; main()
" vllm-serve \
    --model "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --data-parallel-size "$DP_SIZE" \
    --port "$PORT" \
    --trust-remote-code \
    &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# ---------------------------------------------------------------
# 2. Wait for server to be healthy
# ---------------------------------------------------------------
echo "Waiting for server health check (timeout: ${HEALTH_TIMEOUT}s)..."

ELAPSED=0
SERVER_URL="http://localhost:${PORT}"

while [[ $ELAPSED -lt $HEALTH_TIMEOUT ]]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: vLLM server process died unexpectedly."
        wait "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi

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
echo ""

# ---------------------------------------------------------------
# 3. Build training arguments
# ---------------------------------------------------------------
TRAIN_ARGS=()

if [[ "$SMOKE" -eq 1 ]]; then
    TRAIN_ARGS+=(
        --max-samples 20
        --num-generations 2
        --max-completion-length 128
        --batch-size 2
        --gradient-accumulation-steps 1
        --num-epochs 1
        --save-steps 999999
        --report-to none
        --output-dir outputs/trl_grpo_qwen_smoke
    )
else
    TRAIN_ARGS+=(
        --output-dir outputs/trl_grpo_qwen
    )
fi

TRAIN_ARGS+=(
    --model "$MODEL"
    --vllm-server-url "$SERVER_URL"
    "${EXTRA_TRAIN_ARGS[@]+"${EXTRA_TRAIN_ARGS[@]}"}"
)

# ---------------------------------------------------------------
# 4. Run training on separate GPUs
# ---------------------------------------------------------------
echo "Starting GRPO training on GPU(s) $TRAIN_GPUS..."
echo "================================================================"

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$PYTHON" -m accelerate.commands.launch \
    "$SCRIPT_DIR/trl_grpo_qwen.py" \
    "${TRAIN_ARGS[@]}"

echo ""
echo "================================================================"
echo "  Training complete!"
echo "================================================================"
