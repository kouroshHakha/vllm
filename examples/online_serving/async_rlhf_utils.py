# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import ipaddress
import uuid
from typing import Any

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from transformers import AutoModelForCausalLM

import vllm
from vllm import SamplingParams
from vllm.utils.network_utils import get_ip, get_open_port


def get_tcp_url(host: str, port: int) -> str:
    """
    Formats the TCP URL for the given host and port,
    handling IPv6 addresses correctly.
    Args:
        host (str): The hostname or IP address.
        port (int): The port number.
    Returns:
        str: The formatted TCP URL.
    """
    try:
        if isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address):
            return f"tcp://[{host}]:{port}"
    except ValueError:
        # not a literal IP, probably a hostname
        pass
    return f"tcp://{host}:{port}"


def init_custom_process_group(
    backend: str | Backend = None,
    init_method: str | None = None,
    timeout: Any | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str = None,
    pg_options: Any | None = None,
):
    print(f"DEBUG: init_custom_process_group rank={rank}/{world_size}")
    assert (store is None) or (init_method is None), (
        "Cannot specify both init_method and store."
    )

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = (
        "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    By defining an extension class, the code can work no matter what is
    the underlying worker class.

    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size
    ):
        from vllm.distributed.parallel_state import get_world_group

        rank = get_world_group().rank + rank_offset
        self.model_update_group = init_custom_process_group(
            backend="nccl",
            init_method=get_tcp_url(master_address, master_port),
            world_size=world_size,
            rank=rank,
            group_name="vllm_weight_update_group",
        )

    def update_weight(self, name, dtype_name, shape):
        dtype = getattr(torch, dtype_name)
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, src=0, group=self.model_update_group)
        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight  

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated
