export PYTHONPATH=`pwd`/python:$PYTHONPATH

export SGLANG_ENABLE_JIT_DEEPGEMM=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_NPU_PROFILING=0
export ASCEND_LAUNCH_BLOCKING=0
export HCCL_BUFFSIZE=256

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh

tp=$1
MODEL_PATH="/mnt/share/w00936111/weights/GLM-5.3-Flash-BF16"

CUDA_GRAPH_BS="1 8 64"
MEM_FRACTION="0.87"
EXTRA_ARGS="--enable-dp-attention --dp-size 2 --load-balance-method round_robin"

python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size ${tp} \
        --nnodes 1 \
        --chunked-prefill-size 8192 \
        --max-prefill-tokens 8192 \
        --trust-remote-code \
        --mem-fraction-static ${MEM_FRACTION} \
        --page-size 64 \
        --served-model-name GLM-NEXT \
        --load-format auto \
        --max-running-requests 16 \
        --pre-warm-nccl \
        --watchdog-timeout 1200 \
        --cuda-graph-bs ${CUDA_GRAPH_BS} \
        --host 127.0.0.1 \
        --port 8810

        #--moe-a2a-backend deepep \
        #${EXTRA_ARGS} \

