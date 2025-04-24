


set -xe

export VLLM_USE_V1="1"
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')


# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


# Prefilling instance
CUDA_VISIBLE_DEVICES=0 vllm serve \
    unsloth/Llama-3.2-1B \
    --host 0.0.0.0 \
    --port 20001 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --served-model-name Llama \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 10 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e9","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.0.11.188","proxy_port":"30001","http_port":"20001"}}'


# Decoding instance
CUDA_VISIBLE_DEVICES=1 vllm serve \
    unsloth/Llama-3.2-1B \
    --host 0.0.0.0 \
    --port 20002 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --served-model-name Llama \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 10 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"1e9","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.0.11.188","proxy_port":"30001","http_port":"20002"}}'


# wait_for_server 20001
# wait_for_server 20002


# Proxy server
python disagg_prefill_proxy_xpyd.py



# Client
curl -X POST -s http://localhost:9001/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "Llama",
"prompt": "San Francisco is a",
"max_tokens": 100,
"temperature": 0
}'
