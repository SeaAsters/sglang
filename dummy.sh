source /usr/local/Ascend/cann-9.0.0/opp/vendors/customize/bin/set_env.bash


MODEL_PATH=/home/weights/GLM-5-Next-0808/hf

export PYTHONPATH=/home/rjw/sglang-glmx-0813/python:$PYTHONPATH


# Ascend NPU：禁止进入 CUDA/MUSA MHC kernel
export SGLANG_ENABLE_JIT_DEEPGEMM=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export HCCL_BUFFSIZE=1536
export SGLANG_DSA_NPU_TOPK_NATIVE=False
export SGLANG_DSA_NPU_TOPK_TRITON=False

python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 16 \
        --nnodes 1 \
        --chunked-prefill-size -1 \
        --max-prefill-tokens 512 \
        --trust-remote-code \
        --mem-fraction-static 0.78 \
	--page-size 128 \
        --served-model-name GLM-NEXT \
        --moe-a2a-backend deepep --deepep-mode auto \
        --max-running-requests 16 \
        --load-balance-method round_robin \
        --disable-radix-cache \
        --disable-fast-image-processor \
        --watchdog-timeout 9000 \
        --device npu --host 61.47.19.70 --port 8899

# --disable-cuda-graph \
# --skip-server-warmup \
# --moe-a2a-backend deepep --deepep-mode auto \
# --disable-fast-image-processor
	# --skip-tokenizer-init \
        # --enable-dp-attention --dp-size 2 \
# curl --location 'http://61.47.19.70:8810/generate' --header 'Content-Type: application/json' --data '{
#         "input_ids": [1,2,3],
#         "sampling_params": {
#                 "temperature": 0,
#                 "max_new_tokens": 20
#         }
# }'


# curl --location 'http://61.47.19.70:8810/generate' --header 'Content-Type: application/json' --data '{
#         "text": "The capital of China is",
#         "sampling_params": {
#         "temperature": 0,
#         "max_new_tokens": 20
#         }
# }'