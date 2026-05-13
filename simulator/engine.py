"""Discrete time-tick simulation engine.

Simulates LLM inference across edge devices with:
- Layer occupation flags (busy/free per layer per device)
- KV Cache growth tracking
- Memory usage over time
- Request lifecycle: arrive -> prefill -> decode -> complete
"""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from simulator.model import ModelConfig
from simulator.device import DeviceCluster
from simulator.request import Request
from simulator.cost import compute_request_delay


class RequestPhase(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    COMPLETE = "complete"


@dataclass
class RequestState:
    request: Request
    z_q: np.ndarray            # [U, K] routing for this request
    phase: RequestPhase = RequestPhase.WAITING
    current_layer: int = 0     # which layer is being processed
    tokens_generated: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    prefill_end_time: float = 0.0

    # Time remaining for current layer's computation
    layer_compute_remaining: float = 0.0
    # Time remaining for transfer to next layer's device
    transfer_remaining: float = 0.0
    is_transferring: bool = False


@dataclass
class SimSnapshot:
    """A snapshot of the simulation state at a point in time."""
    time: float
    device_memory_used: list[float]     # per device: total memory used
    device_kv_cache: list[float]        # per device: KV cache only
    active_requests: int
    completed_requests: int


class SimulationEngine:
    def __init__(self, model: ModelConfig, cluster: DeviceCluster,
                 tick_s: float = 0.01):
        self.model = model
        self.cluster = cluster
        self.tick_s = tick_s

    def run(self, requests: list[Request], z: np.ndarray,
            x: np.ndarray) -> dict:
        """Run simulation with given routing z and placement x.

        Args:
            requests: list of user requests
            z: [Q, U, K] routing matrix
            x: [U, K] placement matrix

        Returns:
            dict with snapshots, per-request results, etc.
        """
        U = self.model.num_layers
        K = self.cluster.num_devices

        # Initialize device static memory from placement
        device_layer_mem = np.zeros(K)
        for k in range(K):
            for u in range(U):
                if x[u, k] > 0.5:
                    device_layer_mem[k] += self.model.layer_size_mb(u)

        # KV cache per device (dynamic)
        device_kv = np.zeros(K)
        # Track which layers on which devices are occupied (per request)
        # layer_busy[u][k] = set of request indices using this layer
        layer_busy: dict[tuple[int, int], set] = {}
        for u in range(U):
            for k in range(K):
                layer_busy[(u, k)] = set()

        # Initialize request states
        req_states: list[RequestState] = []
        for q, req in enumerate(requests):
            rs = RequestState(request=req, z_q=z[q])
            req_states.append(rs)

        snapshots: list[SimSnapshot] = []
        current_time = 0.0
        max_time = 1000.0  # safety limit

        while current_time < max_time:
            all_done = all(rs.phase == RequestPhase.COMPLETE for rs in req_states)
            if all_done:
                break

            # Record snapshot
            mem_used = [
                device_layer_mem[k] + device_kv[k]
                for k in range(K)
            ]
            snapshots.append(SimSnapshot(
                time=round(current_time, 4),
                device_memory_used=[round(m, 2) for m in mem_used],
                device_kv_cache=[round(device_kv[k], 2) for k in range(K)],
                active_requests=sum(
                    1 for rs in req_states
                    if rs.phase in (RequestPhase.PREFILL, RequestPhase.DECODE)
                ),
                completed_requests=sum(
                    1 for rs in req_states if rs.phase == RequestPhase.COMPLETE
                ),
            ))

            # Process each request
            for q, rs in enumerate(req_states):
                if rs.phase == RequestPhase.COMPLETE:
                    continue

                # Check if request has arrived
                if rs.phase == RequestPhase.WAITING:
                    if current_time >= rs.request.arrival_time:
                        rs.phase = RequestPhase.PREFILL
                        rs.start_time = current_time
                        rs.current_layer = 0
                        dev_k = int(np.argmax(rs.z_q[0]))
                        c_k = self.cluster.devices[dev_k].tokens_per_second_per_layer
                        rs.layer_compute_remaining = rs.request.prompt_length / c_k
                        rs.is_transferring = False
                        layer_busy[(0, dev_k)].add(q)
                    continue

                # Handle transfer between layers
                if rs.is_transferring:
                    rs.transfer_remaining -= self.tick_s
                    if rs.transfer_remaining <= 0:
                        rs.is_transferring = False
                        # Start computing on next layer
                        u = rs.current_layer
                        dev_k = int(np.argmax(rs.z_q[u]))
                        c_k = self.cluster.devices[dev_k].tokens_per_second_per_layer
                        if rs.phase == RequestPhase.PREFILL:
                            rs.layer_compute_remaining = rs.request.prompt_length / c_k
                        else:
                            rs.layer_compute_remaining = 1.0 / c_k
                        layer_busy[(u, dev_k)].add(q)
                    continue

                # Compute on current layer
                rs.layer_compute_remaining -= self.tick_s
                if rs.layer_compute_remaining <= 0:
                    u = rs.current_layer
                    dev_k = int(np.argmax(rs.z_q[u]))
                    layer_busy[(u, dev_k)].discard(q)

                    # Add KV cache for this layer on this device (1 token for decode, prompt_len for prefill)
                    if rs.phase == RequestPhase.PREFILL:
                        kv_bytes = (self.model.kv_cache_bytes_per_layer_per_token
                                    * rs.request.prompt_length)
                    else:
                        kv_bytes = self.model.kv_cache_bytes_per_layer_per_token
                    device_kv[dev_k] += kv_bytes / (1024 * 1024)

                    # Move to next layer
                    if u + 1 < U:
                        rs.current_layer = u + 1
                        next_dev = int(np.argmax(rs.z_q[u + 1]))
                        if next_dev != dev_k:
                            # Need to transfer
                            rs.is_transferring = True
                            rs.transfer_remaining = self.cluster.transfer_time_s(
                                dev_k, next_dev, self.model.activation_size_bytes
                            )
                        else:
                            # Same device, start next layer immediately
                            c_next = self.cluster.devices[next_dev].tokens_per_second_per_layer
                            if rs.phase == RequestPhase.PREFILL:
                                rs.layer_compute_remaining = rs.request.prompt_length / c_next
                            else:
                                rs.layer_compute_remaining = 1.0 / c_next
                            layer_busy[(u + 1, next_dev)].add(q)
                    else:
                        # Finished all layers for this pass
                        if rs.phase == RequestPhase.PREFILL:
                            rs.prefill_end_time = current_time
                            rs.phase = RequestPhase.DECODE
                            rs.tokens_generated = 1
                            # Start decode: go back to layer 1 (skip embedding)
                            rs.current_layer = 1
                            dev_1 = int(np.argmax(rs.z_q[1]))
                            c_1 = self.cluster.devices[dev_1].tokens_per_second_per_layer
                            rs.layer_compute_remaining = 1.0 / c_1
                            layer_busy[(1, dev_1)].add(q)
                        else:
                            rs.tokens_generated += 1
                            if rs.tokens_generated >= rs.request.output_length:
                                rs.phase = RequestPhase.COMPLETE
                                rs.end_time = current_time
                                # Release KV cache for this request
                                for uu in range(U):
                                    dk = int(np.argmax(rs.z_q[uu]))
                                    total_tokens = rs.request.prompt_length + rs.tokens_generated
                                    kv_release = (self.model.kv_cache_bytes_per_layer_per_token
                                                  * total_tokens) / (1024 * 1024)
                                    device_kv[dk] = max(0, device_kv[dk] - kv_release)
                            else:
                                # Next decode step: back to layer 1
                                rs.current_layer = 1
                                dev_1 = int(np.argmax(rs.z_q[1]))
                                c_1 = self.cluster.devices[dev_1].tokens_per_second_per_layer
                                rs.layer_compute_remaining = 1.0 / c_1
                                layer_busy[(1, dev_1)].add(q)

            current_time += self.tick_s

        # Collect results
        request_results = []
        for q, rs in enumerate(req_states):
            # Also compute analytical delay for comparison
            analytical = compute_request_delay(
                rs.z_q, self.model, self.cluster, rs.request
            )
            request_results.append({
                "request_id": rs.request.id,
                "prompt_length": rs.request.prompt_length,
                "output_length": rs.request.output_length,
                "simulated_total_s": round(rs.end_time - rs.start_time, 4),
                "simulated_prefill_s": round(rs.prefill_end_time - rs.start_time, 4),
                "simulated_decode_s": round(rs.end_time - rs.prefill_end_time, 4),
                "analytical_total_s": round(analytical["t_total"], 4),
                "tokens_generated": rs.tokens_generated,
            })

        return {
            "snapshots": snapshots,
            "request_results": request_results,
            "total_simulated_time": round(current_time, 4),
        }
