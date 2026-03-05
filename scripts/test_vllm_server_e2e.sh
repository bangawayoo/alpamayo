#!/usr/bin/env bash
# Launch the vLLM server (same config as run_grpo_server.sh) and run
# the diagnostic test script against it.
#
# Usage:
#   ./scripts/test_vllm_server_e2e.sh                      # text-only tests
#   ./scripts/test_vllm_server_e2e.sh --with-images         # include multi-image tests
#   ./scripts/test_vllm_server_e2e.sh --server-gpu 0,1 --tp 2 --with-images

set -euo pipefail
export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN env var}

# ---------------------------------------------------------------
# Defaults (mirror run_grpo_server.sh)
# ---------------------------------------------------------------
SERVER_GPU=0
MODEL_IMPL="auto"
TENSOR_PARALLEL_SIZE=1
PORT=8000
HEALTH_TIMEOUT=300
HEALTH_INTERVAL=5
WITH_IMAGES=""

# ---------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-gpu)       SERVER_GPU="$2";            shift 2 ;;
        --model-impl)       MODEL_IMPL="$2";            shift 2 ;;
        --tp|--tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --port)             PORT="$2";                  shift 2 ;;
        --health-timeout)   HEALTH_TIMEOUT="$2";        shift 2 ;;
        --with-images)      WITH_IMAGES="--with-images"; shift ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Error: venv python not found at $VENV_PYTHON"
    exit 1
fi

echo "================================================================"
echo "  vLLM Server Diagnostic Test"
echo "================================================================"
echo "  Server GPU:  $SERVER_GPU"
echo "  TP size:     $TENSOR_PARALLEL_SIZE"
echo "  Port:        $PORT"
echo "  With images: ${WITH_IMAGES:-no}"
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
# 1. Extract VLM backbone (reuse cache)
# ---------------------------------------------------------------
MODEL="nvidia/Alpamayo-R1-10B"
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
SERVER_LOG="$PROJECT_ROOT/.cache/vllm_server.log"
echo "Starting vLLM server on GPU $SERVER_GPU (port $PORT)..."
echo "  Server log: $SERVER_LOG"

VENV_TRL="$PROJECT_ROOT/.venv/bin/trl"
CUDA_VISIBLE_DEVICES="$SERVER_GPU" "$VENV_TRL" vllm-serve \
    --model "$VLM_CACHE_DIR" \
    --vllm_model_impl "$MODEL_IMPL" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --port "$PORT" \
    --max_model_len 8192 \
    --enforce_eager \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"

# ---------------------------------------------------------------
# 3. Wait for server to be ready
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

    if curl -sf "${SERVER_URL}/health/" >/dev/null 2>&1; then
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
# 4. Run diagnostic tests
# ---------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Running diagnostic tests..."
echo "================================================================"

TEST_RC=0
PYTHONUNBUFFERED=1 PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}" \
    "$VENV_PYTHON" "$SCRIPT_DIR/test_vllm_server.py" \
    --port "$PORT" \
    --model "$VLM_CACHE_DIR" \
    --full-model "$MODEL" \
    ${WITH_IMAGES} || TEST_RC=$?

if [[ $TEST_RC -ne 0 ]]; then
    echo ""
    echo "================================================================"
    echo "  Tests failed (exit $TEST_RC). Last 50 lines of server log:"
    echo "================================================================"
    tail -n 50 "$SERVER_LOG"
fi

echo ""
echo "================================================================"
echo "  Done. Shutting down server..."
echo "================================================================"
echo "  Full server log: $SERVER_LOG"
exit $TEST_RC
