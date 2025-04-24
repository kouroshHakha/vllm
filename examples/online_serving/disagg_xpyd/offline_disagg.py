# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of disaggregated prefilling
We will launch 2 vllm instances (GPU 0 for prefill and GPU 1 for decode),
and then transfer the KV cache between them.
"""
import os
import time
import ray

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

context = "hi " * 1000
# context = ""


@ray.remote
def run_prefill(prefill_done):
    # We use GPU 0 for prefill node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # The prefill node receives two requests, while the decode node receives
    # three requests. So the decode node will only receive the KV Cache for
    # requests 1 and 3. The decode node will use the KV Cache of requests 1
    # and 3 and do prefilling on request 2.
    prompts = [
        context,
        # "Hi, your name is",
        # # The decode node will actually "prefill" this request.
        # "Tell me a very long story",
    ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer",'
        '"kv_buffer_size":"1e9","kv_port":"21001","kv_connector_extra_config":{}}'
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    # llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    llm = LLM(model="unsloth/Llama-3.2-1B",
              kv_transfer_config=ktc,
              max_model_len=2000,
              enforce_eager=True,
              gpu_memory_utilization=0.8)

    llm.generate(prompts, sampling_params)
    print("Prefill node is finished.")
    prefill_done.set.remote()
    print("Prefill done is set.")

    # To keep the prefill node running in case the decode node is not done;
    # otherwise, the script might exit prematurely, causing incomplete decoding.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script stopped by user.")


@ray.remote
def run_decode(prefill_done):
    # We use GPU 1 for decode node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    prompts = [
        context,
        # context + "Hello, my name is" * 10,
        # "Hi, your name is",
        # "Tell me a very long story",
    ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95)


    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer",'
        '"kv_buffer_size":"1e9","kv_port":"22001","kv_connector_extra_config":{}}'

    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    # llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    llm = LLM(model="unsloth/Llama-3.2-1B",
              kv_transfer_config=ktc,
              max_model_len=2000,
              enforce_eager=True,
              gpu_memory_utilization=0.8)

    # Wait for the producer to start the pipe
    print("Waiting for prefill node to finish...")
    ray.get(prefill_done.wait.remote())
    print("KV cache is ready.")

    # At this point when the prefill_done is set, the kv-cache should have been
    # transferred to this decode node, so we can start decoding.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


@ray.remote
class EventActor:
    def __init__(self):
        self._is_set = False
    
    def set(self):
        self._is_set = True
        
    def wait(self):
        while not self._is_set:
            time.sleep(0.1)  # Small sleep to avoid busy waiting
        return True


def main():
    # Initialize Ray
    ray.init(
        runtime_env={
            "env_vars": {
                "VLLM_USE_V1": "1",
            }
        }
    )
    
    # Create an event actor to coordinate between processes
    prefill_done = EventActor.remote()
    
    # Start prefill node
    prefill_task = run_prefill.remote(prefill_done)
    
    # Start decode node
    decode_task = run_decode.remote(prefill_done)

    # Wait for decode to finish
    ray.get(decode_task)
    
    # Terminate the prefill task
    ray.cancel(prefill_task, force=True)
    


if __name__ == "__main__":
    main()
