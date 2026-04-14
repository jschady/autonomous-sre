FROM vllm/vllm-openai:latest

# Model and inference configuration (overridable via env at runtime)
ENV MODEL=meta-llama/Llama-3.1-8B-Instruct
ENV GPU_UTIL=0.9
ENV MAX_MODEL_LEN=32768
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# HuggingFace token required for gated models (mount as secret or env var)
# ENV HF_TOKEN=<your_token>

ENTRYPOINT ["sh", "-c", "exec vllm serve \"${MODEL}\" --gpu-memory-utilization \"${GPU_UTIL}\" --max-model-len \"${MAX_MODEL_LEN}\" --host \"${HOST}\" --port \"${PORT}\""]
CMD []
