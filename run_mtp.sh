source /usr/local/Ascend/cann-9.0.0/opp/vendors/customize/bin/set_env.bash

export PYTHONPATH=/home/rjw/sglang-glmx-0813/python:$PYTHONPATH


# Ascend NPU：禁止进入 CUDA/MUSA MHC kernel
export SGLANG_ENABLE_JIT_DEEPGEMM=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export HCCL_BUFFSIZE=1536
export SGLANG_DSA_NPU_TOPK_NATIVE=False
export SGLANG_DSA_NPU_TOPK_TRITON=False

# mtp
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_DEBUG_NPU_MTP_ACCURACY=1


MODEL_PATH=/home/weights/GLM-5-Next-0808/hf/

python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 16 \
        --nnodes 1 \
	--disable-fast-image-processor \
        --chunked-prefill-size -1 \
        --max-prefill-tokens 8192 \
        --trust-remote-code \
        --mem-fraction-static 0.82 \
	--page-size 128 \
        --served-model-name GLM-NEXT \
        --speculative-algorithm NEXTN \
        --speculative-num-steps 3 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 4 \
        --cuda-graph-bs 16 \
        --max-running-requests 16 \
        --load-balance-method round_robin \
        --disable-radix-cache \
        --watchdog-timeout 9000 \
        --skip-server-warmup \
        --device npu --host 61.47.19.70 --port 8899

        # --disable-cuda-graph \
        #--speculative-algorithm NEXTN \
        #--speculative-num-steps 1 \
        #--speculative-eagle-topk 1 \
        #--speculative-num-draft-tokens 2 \
	# --skip-tokenizer-init \
        # --enable-dp-attention --dp-size 2 \
# curl --location 'http://61.47.19.70:8810/generate' --header 'Content-Type: application/json' --data '{
#         "input_ids": [1,2,3],
#         "sampling_params": {
#                 "temperature": 0,
#                 "max_new_tokens": 20
#         }
# }'
