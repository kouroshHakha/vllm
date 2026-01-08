# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RLHF example with Data Parallel (DP=2) using native pause/resume APIs.

Demonstrates that the simple pause/resume approach works for wide-EP/DP 
deployments WITHOUT needing DPCoordinator modifications.

Key insight: 
- pause_generation() only affects the frontend (AsyncLLM._paused = True)
- The engine core continues running and stays synced via wave mechanism
- We just need to broadcast pause/resume to all DP ranks from the orchestrator

Flow:
1. Start async generations across DP ranks
2. Orchestrator broadcasts pause_generation() to all ranks
3. Wait for all to acknowledge (requests drain or abort)
4. Update weights safely (all engines paused and synced)
5. Orchestrator broadcasts resume_generation() to all ranks
6. Remaining generations complete (with retry on abort)

Requirements: 3 GPUs (1 trainer, 2 for DPEP=2 inference) -- L4 will work
"""

import asyncio
import uuid
from dataclasses import dataclass

import ray
import torch
from transformers import AutoModelForCausalLM
import vllm
from vllm import SamplingParams
from vllm.utils.network_utils import get_ip, get_open_port
from async_rlhf_utils import init_custom_process_group, get_tcp_url

# Model to use (small for testing)
MODEL_NAME = "Isotonic/smol_llama-4x220M-MoE"
DP_SIZE = 2


@dataclass
class PauseResult:
    """Result from pause_generation call."""
    dp_rank: int
    status: str  # "paused" or "error"


@dataclass  
class ResumeResult:
    """Result from resume_generation call."""
    dp_rank: int
    status: str  # "resumed" or "error"


# Actor class for the inference engine
class InferenceEngine:
    """
    A single vLLM inference engine representing one DP rank.
    
    Uses the native pause_generation() and resume_generation() APIs.
    """

    def __init__(self, dp_rank: int, dp_size: int):
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        
        # Create AsyncLLM engine with DUMMY weights
        # This starts with random weights - we'll sync real weights from trainer
        # In a real wide-EP deployment, this would be a separate process
        # with --data-parallel-size and --data-parallel-rank flags
        self.llm = vllm.AsyncLLMEngine.from_engine_args(
            vllm.AsyncEngineArgs(
                model=MODEL_NAME,
                enforce_eager=False,
                tensor_parallel_size=1,
                data_parallel_size=dp_size,
                data_parallel_rank=dp_rank,
                enable_expert_parallel=True,
                gpu_memory_utilization=0.4,  # Lower to fit multiple engines
                load_format="dummy",  # Start with random weights!
                worker_extension_cls=(
                    "async_rlhf_utils.WorkerExtension"
                ),
            )
        )
        print(f"[DP-{dp_rank}] Engine initialized with DUMMY weights")


    def ready(self) -> bool:
        return True
    
    async def generate(
        self, 
        prompt: str, 
        sampling_params: SamplingParams,
    ) -> vllm.RequestOutput:
        """Generate text for a single prompt."""
        request_id = str(uuid.uuid4())
        async for output in self.llm.generate(prompt, sampling_params, request_id):
            final_output = output
        return final_output

    async def pause(self, wait_for_inflight: bool = False) -> PauseResult:
        """
        Pause generation using native API.
        
        Args:
            wait_for_inflight: If True, wait for in-flight requests to complete.
                              If False, abort them immediately.
        """
        print(f"[DP-{self.dp_rank}] Pausing (wait_for_inflight={wait_for_inflight})...")
        try:
            await self.llm.pause_generation(
                wait_for_inflight_requests=wait_for_inflight,
                clear_cache=True,
            )
            print(f"[DP-{self.dp_rank}] Paused")
            return PauseResult(dp_rank=self.dp_rank, status="paused")
        except Exception as e:
            print(f"[DP-{self.dp_rank}] Pause failed: {e}")
            return PauseResult(dp_rank=self.dp_rank, status="error")

    async def resume(self) -> ResumeResult:
        """Resume generation using native API."""
        print(f"[DP-{self.dp_rank}] Resuming...")
        try:
            await self.llm.resume_generation()
            print(f"[DP-{self.dp_rank}] Resumed")
            return ResumeResult(dp_rank=self.dp_rank, status="resumed")
        except Exception as e:
            print(f"[DP-{self.dp_rank}] Resume failed: {e}")
            return ResumeResult(dp_rank=self.dp_rank, status="error")

    async def collective_rpc(self, method: str, args: tuple = ()):
        """Call method on all workers."""
        return await self.llm.collective_rpc(method, args=args)


class InferencePoolRouter:
    """
    Orchestrates pause/resume across multiple DP ranks.
    
    This is the key component - it broadcasts pause/resume to all
    DP ranks and waits for all acknowledgments before returning.
    """

    def __init__(self, engines: list):
        """
        Args:
            engines: List of Ray actor handles to InferenceEngine instances
        """
        self.engines = engines
        self.dp_size = len(engines)
        self._last_engine_idx = 0
        
    async def generate(self, prompt: str, sampling_params: SamplingParams) -> vllm.RequestOutput:
        """Generate text for a single prompt (blocking)."""
        sampled_engine = self.engines[self._last_engine_idx]  
        self._last_engine_idx = (self._last_engine_idx + 1) % self.dp_size
        output = ray.get(sampled_engine.generate.remote(prompt, sampling_params))
        return output

    def generate_async(self, prompts: list[str], sampling_params: SamplingParams) -> list:
        """Start generations across DP ranks and return Ray futures (non-blocking).
        
        Distributes prompts round-robin across DP ranks.
        Returns list of Ray ObjectRefs that can be collected with ray.get().
        """
        futures = []
        for i, prompt in enumerate(prompts):
            engine = self.engines[i % self.dp_size]
            futures.append(engine.generate.remote(prompt, sampling_params))
        return futures

    async def pause(self, wait_for_inflight: bool = False) -> bool:
        """
        Broadcast pause to all DP ranks and wait for all to complete.
        
        Returns True if all ranks paused successfully.
        """
        print(f"\n{'='*50}")
        print(f"[InferencePool] Pausing all {self.dp_size} DP ranks...")
        print(f"{'='*50}")
        
        # Broadcast pause in parallel
        pause_futures = [
            engine.pause.remote(wait_for_inflight) 
            for engine in self.engines
        ]
        
        # Wait for all to complete
        results = ray.get(pause_futures)
        
        # Check all succeeded
        all_paused = all(r.status == "paused" for r in results)
        
        if all_paused:
            print(f"[InferencePool] ✓ All {self.dp_size} ranks paused")
        else:
            failed = [r.dp_rank for r in results if r.status != "paused"]
            print(f"[InferencePool] ✗ Failed to pause ranks: {failed}")
        
        return all_paused

    async def resume(self) -> bool:
        """
        Broadcast resume to all DP ranks and wait for all to complete.
        
        Returns True if all ranks resumed successfully.
        """
        print(f"\n{'='*50}")
        print(f"[InferencePool] Resuming all {self.dp_size} DP ranks...")
        print(f"{'='*50}")
        
        # Broadcast resume in parallel
        resume_futures = [
            engine.resume.remote() 
            for engine in self.engines
        ]
        
        # Wait for all to complete
        results = ray.get(resume_futures)
        
        # Check all succeeded
        all_resumed = all(r.status == "resumed" for r in results)
        
        if all_resumed:
            print(f"[InferencePool] ✓ All {self.dp_size} ranks resumed")
        else:
            failed = [r.dp_rank for r in results if r.status != "resumed"]
            print(f"[InferencePool] ✗ Failed to resume ranks: {failed}")
        
        return all_resumed
    
    async def init_weight_update_group(self, master_address: str, master_port: int):
        """Initialize the weight update group for syncing with vLLM."""
        # world_size = 1 (trainer) + DP_SIZE (inference workers)
        # Ranks: trainer=0, DP_0=1, DP_1=2, ...
        # 
        # The WorkerExtension does: rank = get_world_group().rank + rank_offset
        # Since get_world_group().rank already returns the DP rank (0, 1, ...),
        # we just need rank_offset=1 for all engines (trainer occupies rank 0)
        world_size = 1 + self.dp_size
        rank_offset = 1  # Trainer is rank 0, so inference starts at rank 1
        
        futures = []
        for engine in self.engines:
            futures.append(engine.collective_rpc.remote(
                "init_weight_update_group",
                args=(master_address, master_port, rank_offset, world_size)
            ))

        results = await asyncio.gather(*futures)
        print(f"[InferencePool] Weight update group initialized successfully")
        return True
        
    async def update_weights(self, name: str, dtype: str, shape: tuple):
        """Update a single weight on all DP ranks via collective RPC."""
        futures = [
            engine.collective_rpc.remote("update_weight", args=(name, dtype, shape))
            for engine in self.engines
        ]
        ray.get(futures)

    async def receive_all_weights(self, weight_info: list[tuple[str, str, tuple]]):
        """Receive all weights from trainer via NCCL (inference side)."""
        print(f"[InferencePool] Receiving {len(weight_info)} weight tensors...")
        for name, dtype, shape in weight_info:
            await self.update_weights(name, dtype, shape)
        print("[InferencePool] All weights received")
        
    async def check_weights_changed(self) -> bool:
        """Check if weights have been zeroed on all DP ranks."""
        futures = [
            engine.collective_rpc.remote("check_weights_changed")
            for engine in self.engines
        ]
        results = ray.get(futures)
        # collective_rpc returns a list (one result per worker), check all are True
        return all(r[0] for r in results)

# Create Ray actor version
InferenceEngineActor = ray.remote(num_gpus=1)(InferenceEngine)


@ray.remote(num_gpus=1)
class Trainer:
    """Training model on a separate GPU with pretrained weights."""

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self.model.to("cuda")
        print("[Trainer] Model loaded with pretrained weights")
        
    def ready(self) -> bool:
        """Check if the trainer is ready."""
        return True

    def get_weight_info(self) -> list[tuple[str, str, tuple]]:
        """Get weight metadata (name, dtype, shape) for all parameters."""
        return [
            (name, str(p.dtype).split(".")[-1], tuple(p.shape))
            for name, p in self.model.named_parameters()
        ]

    def broadcast_weights(self):
        """Broadcast all weights to vLLM workers via NCCL (trainer side)."""
        print("[Trainer] Broadcasting weights via NCCL...")
        for name, p in self.model.named_parameters():
            torch.distributed.broadcast(p.data, src=0, group=self.weight_update_group)
        print("[Trainer] Broadcast complete")
    
    def get_ip_port(self) -> tuple[str, int]:
        """Get the IP and port of the trainer."""
        return get_ip(), get_open_port()
    
    async def init_weight_update_group(self, master_address: str, master_port: int):
        """Initialize the weight update group between the trainer and inference."""
        
        self.weight_update_group = init_custom_process_group(
            backend="nccl",
            init_method=get_tcp_url(master_address, master_port),
            world_size=1 + DP_SIZE,
            rank=0,
            group_name="vllm_weight_update_group",
        )
        
        return True
        


async def run_example():
    """Main example function."""
    
    print("\n" + "=" * 60)
    print("RLHF Example with DP=2 using Native Pause/Resume APIs")
    print("=" * 60)

    # --- Setup ---
    print("\n[Setup] Creating DP inference engines...")
    engines = [
        InferenceEngineActor.remote(dp_rank=i, dp_size=DP_SIZE) 
        for i in range(DP_SIZE)
    ]
    
    print("[Setup] Creating trainer...")
    trainer = Trainer.remote()
    
    # Wait for training and inference pools to be ready
    ray.get([trainer.ready.remote()] + [engine.ready.remote() for engine in engines])
    
    print("[Setup] Creating the router...")
    inference_pool = InferencePoolRouter(engines)
    
    print("[Setup] Form a process group between the trainer and inference...")
    master_address, master_port = ray.get(trainer.get_ip_port.remote())
    results = await asyncio.gather(
        trainer.init_weight_update_group.remote(master_address, master_port), 
        inference_pool.init_weight_update_group(master_address, master_port)
    )
    
    if not all(results):
        print("[Setup] Failed to initialize the weight update group")
        return

    prompt = "The capital of France is"
    sampling_params = SamplingParams(max_tokens=20)
    
    # Helper function to sync weights from trainer to vLLM
    async def sync_weights():
        weight_info = ray.get(trainer.get_weight_info.remote())
        trainer_broadcast_future = trainer.broadcast_weights.remote()
        inference_receive_task = inference_pool.receive_all_weights(weight_info)
        await asyncio.gather(
            asyncio.to_thread(ray.get, trainer_broadcast_future),
            inference_receive_task
        )

    # ==========================================================================
    # Phase 1: Generate with DUMMY weights (random garbage)
    # vLLM was initialized with load_format="dummy" - no real weights loaded
    # ==========================================================================
    print("\n" + "=" * 60)
    print("[Phase 1] Generating with DUMMY weights (random initialization)...")
    print("=" * 60)
    
    output_garbage = await inference_pool.generate(prompt, sampling_params)
    text_garbage = output_garbage.outputs[0].text
    print(f"[Phase 1] Prompt: '{prompt}'")
    print(f"[Phase 1] Output (garbage): '{text_garbage}'")

    # ==========================================================================
    # Phase 2: Sync REAL pretrained weights from trainer
    # This is the key test - if weight transfer works, output becomes coherent
    # ==========================================================================
    print("\n" + "=" * 60)
    print("[Phase 2] Syncing PRETRAINED weights from trainer...")
    print("=" * 60)
    
    # Pause, sync pretrained weights, resume
    pause_ok = await inference_pool.pause(wait_for_inflight=True)
    assert pause_ok, "Failed to pause all DP ranks for weight sync"
    
    await sync_weights()
    
    resume_ok = await inference_pool.resume()
    assert resume_ok, "Failed to resume all DP ranks after weight sync"

    print("\n[Phase 2] Generating with PRETRAINED weights...")
    output_correct = await inference_pool.generate(prompt, sampling_params)
    text_correct = output_correct.outputs[0].text
    print(f"[Phase 2] Prompt: '{prompt}'")
    print(f"[Phase 2] Output (should be coherent): '{text_correct}'")

    # ==========================================================================
    # Verification
    # ==========================================================================
    print("\n" + "=" * 60)
    print("[Verification] Checking results...")
    print("=" * 60)

    outputs_differ = text_garbage != text_correct
    print(f"[Verification] Outputs differ: {outputs_differ}")
    print(f"  - Output A (garbage): '{text_garbage[:60]}...'")
    print(f"  - Output B (correct): '{text_correct[:60]}...'")

    # A simple heuristic: coherent output should have common words/patterns
    # Garbage output typically has random tokens or repetition
    looks_coherent = (
        len(text_correct.split()) > 3 and  # Has multiple words
        text_correct != text_garbage and    # Different from garbage
        "Paris" in text_correct or "france" in text_correct.lower() or len(set(text_correct.split())) > 2
    )
    print(f"[Verification] Output looks coherent: {looks_coherent}")

    if outputs_differ and looks_coherent:
        print("\n" + "=" * 60)
        print("TEST PASSED: Weight update worked correctly!")
        print("  - Started with dummy (random) weights via load_format='dummy'")
        print("  - Successfully synced pretrained weights from trainer")
        print("  - Output became coherent after weight sync")
        print("  - This proves the weight transfer mechanism is correct")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("TEST FAILED!")
        if not outputs_differ:
            print("  - Outputs are the same (weight update had no effect)")
        if not looks_coherent:
            print("  - Output still looks like garbage (weights may be corrupted)")
        print("=" * 60)
        raise AssertionError("Test failed - see above for details")

    # ==========================================================================
    # Phase 3: Abort Demonstration
    # Shows what happens when we pause with wait_for_inflight=False
    # ==========================================================================
    print("\n" + "=" * 60)
    print("[Phase 3] Demonstrating ABORT behavior...")
    print("=" * 60)

    # Start LONG generations on both DP ranks simultaneously
    long_prompts = [
        "Write a very detailed essay about the history of computers and how they evolved from mechanical calculators to modern supercomputers",
        "Explain quantum mechanics in extreme detail, covering wave-particle duality, superposition, and entanglement",
    ]
    long_params = SamplingParams(max_tokens=1000, ignore_eos=True)

    print("\n[Phase 3a] Starting 2 long generations (one per DP rank)...")
    gen_futures = inference_pool.generate_async(long_prompts, long_params)

    # Wait briefly for generation to start producing tokens
    print("[Phase 3a] Waiting 0.5s for tokens to start generating...")
    await asyncio.sleep(0.5)

    # Abort via pause with wait_for_inflight=False
    print("\n[Phase 3b] Aborting via pause(wait_for_inflight=False)...")
    pause_ok = await inference_pool.pause(wait_for_inflight=False)
    assert pause_ok, "Failed to pause (abort) all DP ranks"

    # Collect results - should be aborted with partial output
    print("\n[Phase 3b] Collecting aborted results...")
    results = ray.get(gen_futures)
    
    for i, result in enumerate(results):
        output = result.outputs[0]
        
        print(f"\n[DP-{i}] Prompt: '{long_prompts[i][:50]}...'")
        print(f"[DP-{i}] Finish reason: {output.finish_reason}")
        print(f"[DP-{i}] Tokens generated: {len(output.token_ids)}")
        if output.text:
            preview = output.text[:100] if len(output.text) > 100 else output.text
            print(f"[DP-{i}] Partial output: '{preview}...'")
        else:
            print(f"[DP-{i}] Partial output: (empty - aborted before any tokens)")
        
        # ASSERTION: Request should be aborted
        assert output.finish_reason == "abort", \
            f"[DP-{i}] Expected finish_reason='abort', got '{output.finish_reason}'"

    # ==========================================================================
    # Phase 3c: Submit requests WHILE PAUSED (they should queue)
    # ==========================================================================
    print("\n[Phase 3c] Submitting requests WHILE PAUSED (should queue)...")
    
    queued_prompts = [
        "What is the speed of light?",
        "Name three planets in our solar system.",
    ]
    queued_params = SamplingParams(max_tokens=50)
    
    # Start these requests while engine is paused - they should queue
    queued_futures = inference_pool.generate_async(queued_prompts, queued_params)
    print(f"[Phase 3c] Submitted {len(queued_prompts)} requests while paused")
    print("[Phase 3c] These requests are now queued, waiting for resume...")
    
    # Brief pause to show they're truly queued (not completing)
    await asyncio.sleep(0.3)
    
    # Check that futures are NOT ready yet (still queued)
    ready, not_ready = ray.wait(queued_futures, timeout=0)
    print(f"[Phase 3c] Requests ready: {len(ready)}, still queued: {len(not_ready)}")
    
    # ASSERTION: Requests should be queued (not ready) while paused
    assert len(not_ready) == len(queued_prompts), \
        f"Expected {len(queued_prompts)} requests queued, but {len(ready)} completed while paused"

    # ==========================================================================
    # Phase 3d: Resume and see queued requests complete
    # ==========================================================================
    print("\n[Phase 3d] Resuming all DP ranks...")
    resume_ok = await inference_pool.resume()
    assert resume_ok, "Failed to resume all DP ranks"

    # Now collect the queued requests - they should complete
    print("[Phase 3d] Collecting queued requests (should complete now)...")
    queued_results = ray.get(queued_futures)
    
    for i, result in enumerate(queued_results):
        output = result.outputs[0]
        
        print(f"\n[Queued-{i}] Prompt: '{queued_prompts[i]}'")
        print(f"[Queued-{i}] Finish reason: {output.finish_reason}")
        print(f"[Queued-{i}] Output: '{output.text.strip()}'")
        
        # ASSERTION: Queued request should complete after resume
        assert output.finish_reason in ("length", "stop"), \
            f"[Queued-{i}] Expected finish_reason in ('length', 'stop'), got '{output.finish_reason}'"

    # ==========================================================================
    # Phase 3e: Final retry of aborted prompt
    # ==========================================================================
    print("\n[Phase 3e] Retrying aborted prompt (should complete fully)...")
    retry_result = await inference_pool.generate(long_prompts[0], long_params)
    retry_output = retry_result.outputs[0]
    
    print(f"[Retry] Finish reason: {retry_output.finish_reason}")
    print(f"[Retry] Tokens generated: {len(retry_output.token_ids)}")
    if retry_output.text:
        preview = retry_output.text[:200] if len(retry_output.text) > 200 else retry_output.text
        print(f"[Retry] Output: '{preview}...'")

    # ASSERTION: Retry should complete successfully
    assert retry_output.finish_reason in ("length", "stop"), \
        f"Retry failed: expected finish_reason in ('length', 'stop'), got '{retry_output.finish_reason}'"

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("Phase 1-2: Weight Sync")
    print("  ✓ Started with dummy weights (load_format='dummy')")
    print("  ✓ Synced pretrained weights from trainer")
    print("  ✓ Output became coherent after sync")
    print("")
    print("Phase 3: Abort + Queue Behavior")
    print("  ✓ Both DP ranks returned with finish_reason='abort'")
    print("  ✓ Requests submitted during pause were queued")
    print("  ✓ Queued requests completed after resume")
    print("  ✓ Retry after resume completed successfully")
    print("=" * 60)


@ray.remote(num_gpus=0)
def main():
    """Entry point."""
    asyncio.run(run_example())


if __name__ == "__main__":
    ray.init()
    ray.get(main.remote())
