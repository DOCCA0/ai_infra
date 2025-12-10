#!/bin/bash

# Define variables
MODEL_PATH="/data/gpt2"

# Set QUANTIZATION to an empty string to disable quantization, or set to
# a quantization name (for example: "awq_marlin") and mount the
# required config file into the container at /data/quant_config.json.
# Example to enable AWQ Marlin (after placing a config at
# ai_infra_week2/quant_config.json):
#   QUANTIZATION="awq_marlin"
QUANTIZATION=""

# Batch sizes to test (simulated using max-num-seqs)
BATCH_SIZES=(1 2)

echo "Running benchmarks..."
echo "------------------------------------------------"

for bs in "${BATCH_SIZES[@]}"; do
    echo "Testing Batch Size (max-num-seqs): $bs"

    # Build optional quantization arg
    QUANT_ARG=""
    if [ -n "$QUANTIZATION" ]; then
      QUANT_ARG="--quantization $QUANTIZATION"
    fi

    docker run --gpus all --rm \
      -v $(pwd)/ai_infra_week2:/data \
      --entrypoint /bin/bash \
      vllm/vllm-openai:latest \
      -c "vllm bench throughput \
        --model $MODEL_PATH \
        $QUANT_ARG \
        --num-prompts 100 \
        --input-len 64 \
        --output-len 32 \
        --dtype float16 \
        --gpu-memory-utilization 0.7 \
        --max-num-seqs $bs"
done