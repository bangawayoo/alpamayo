import os
import torch

print("Python:", os.sys.version.split()[0])
print("Torch:", torch.__version__)
print("Torch built for CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# --- vLLM import + basic metadata ---
import vllm
from vllm import LLM, SamplingParams

print("vLLM:", getattr(vllm, "__version__", "unknown"))

# --- minimal smoke test: load a tiny model and generate 1 short completion ---
# Set VLLM_TEST_MODEL to a local path if your environment blocks downloads.
model_id = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

print("Loading model:", model_id)
llm = LLM(
    model=model_id,
    tensor_parallel_size=int(os.environ.get("TP", "1")),
    trust_remote_code=True,
)

params = SamplingParams(max_tokens=16, temperature=0.0)
out = llm.generate(["Say 'vllm ok' and nothing else."], params)

print("Generation OK.")
print("Output:", out[0].outputs[0].text.strip())
