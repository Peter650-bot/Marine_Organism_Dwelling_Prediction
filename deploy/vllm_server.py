"""Serve a strong open model as an OpenAI-compatible vLLM endpoint on Modal.

This replaces the Colab-T4 + cloudflared setup with a Modal GPU: no per-call
rate limit (unlike hosted free APIs), a stable HTTPS URL, and enough VRAM for a
72B model that a free T4 can't hold. Runs against Modal's free monthly compute
credits.

The TabLLM pipeline is backend-agnostic (OpenAI-compatible), so nothing in the
analysis code changes — you just point it at this server's URL.

--------------------------------------------------------------------------
ONE-TIME SETUP (on the machine running the pipeline)
    pip install modal
    modal token new                      # opens a browser to authenticate

DEPLOY THE SERVER (gives a stable https URL)
    modal deploy deploy/vllm_server.py
    # -> prints:  https://<user>--vllm-tabllm-serve.modal.run

POINT THE PIPELINE AT IT
    export VLLM_BASE_URL="https://<user>--vllm-tabllm-serve.modal.run/v1"
    python tabllm_pipeline.py \
        --species "Caretta caretta" \
        --velocity Datasets/Oceanic_data.nc \
        --species-csv Datasets/caretta_data.csv \
        --base-url "$VLLM_BASE_URL" \
        --model "Qwen/Qwen2.5-72B-Instruct-AWQ" \
        --n-test 300 --shots 8,16,32,64

STOP BILLING WHEN DONE (containers also auto-scale-down after idle)
    modal app stop vllm-tabllm
--------------------------------------------------------------------------
"""
import modal

# ---- knobs -----------------------------------------------------------------
# 4x A10G (24 GB each = 96 GB) is the strongest CARD-FREE tier on Modal's free
# credit (A100/H100 require a payment method). Tensor-parallel across the 4 GPUs
# holds the 72B AWQ model (~40 GB weights) with ample KV-cache headroom.
MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"   # ~40 GB in 4-bit AWQ
GPU = "A10G:4"                             # 4x24 GB, card-free
TENSOR_PARALLEL = 4                        # must match the GPU count
VLLM_PORT = 8000
MAX_MODEL_LEN = 8192                       # plenty for k<=32 prompts
MINUTES = 60

# ---- image: CUDA + vLLM, with fast HF downloads ----------------------------
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Persist model weights + vLLM compile cache across cold starts (avoids
# re-downloading ~40 GB every time the container scales to zero).
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("vllm-tabllm")


@app.function(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=15 * MINUTES,   # keep warm 15 min after the last request
    timeout=60 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=64)     # let vLLM batch many in-flight requests
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve():
    import subprocess

    cmd = (
        f"vllm serve {MODEL} --served-model-name {MODEL} "
        f"--quantization awq --tensor-parallel-size {TENSOR_PARALLEL} "
        f"--max-model-len {MAX_MODEL_LEN} "
        f"--gpu-memory-utilization 0.92 --port {VLLM_PORT}"
    )
    # Popen (not run): return immediately so Modal can proxy the port while
    # vLLM keeps serving in the background.
    subprocess.Popen(cmd, shell=True)
