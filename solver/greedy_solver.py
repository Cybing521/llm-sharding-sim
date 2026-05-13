"""Greedy solver: assign consecutive layers to devices, largest-first."""

import time
import numpy as np
from solver.base_solver import BaseSolver, SolverResult
from simulator.model import ModelConfig
from simulator.device import DeviceCluster
from simulator.request import Request


class GreedySolver(BaseSolver):
    """Assign contiguous blocks of layers to devices.

    Strategy:
    1. Sort devices by memory capacity (descending).
    2. Greedily pack consecutive layers onto each device until it's full.
    3. Prefer placing layers on faster devices when possible.
    4. Route each request through the placed layers (only valid option per layer).
    """

    def solve(self, requests: list[Request]) -> SolverResult:
        t0 = time.time()
        U = self.model.num_layers
        K = self.cluster.num_devices

        x = np.zeros((U, K), dtype=int)

        # Sort devices: prefer faster (higher tokens/s), break ties by memory
        device_order = sorted(
            range(K),
            key=lambda k: (
                self.cluster.devices[k].tokens_per_second_per_layer,
                self.cluster.devices[k].memory_mb,
            ),
            reverse=True,
        )

        remaining_cap = [self.cluster.devices[k].memory_mb for k in range(K)]
        layer_idx = 0

        for k in device_order:
            if layer_idx >= U:
                break
            while layer_idx < U:
                layer_size = self.model.layer_size_mb(layer_idx)
                if remaining_cap[k] >= layer_size:
                    x[layer_idx, k] = 1
                    remaining_cap[k] -= layer_size
                    layer_idx += 1
                else:
                    break

        # If not all layers placed, do a fallback pass
        if layer_idx < U:
            for u in range(layer_idx, U):
                layer_size = self.model.layer_size_mb(u)
                for k in device_order:
                    if remaining_cap[k] >= layer_size:
                        x[u, k] = 1
                        remaining_cap[k] -= layer_size
                        break
                else:
                    # Infeasible -- not enough total memory
                    return SolverResult(x, np.zeros((len(requests), U, K)),
                                        "greedy", "infeasible", time.time() - t0)

        # Routing: each request uses the only device where each layer is placed
        Q = len(requests)
        z = np.zeros((Q, U, K), dtype=int)
        for q in range(Q):
            for u in range(U):
                # Pick the device that has this layer
                devices_with_layer = np.where(x[u] > 0.5)[0]
                if len(devices_with_layer) == 1:
                    z[q, u, devices_with_layer[0]] = 1
                else:
                    # Multiple copies: pick the one closest to arrival device
                    arr_dev = requests[q].arrival_device
                    best_k = min(devices_with_layer,
                                 key=lambda k: self.cluster.latency_ms[arr_dev, k])
                    z[q, u, best_k] = 1

        solve_time = time.time() - t0
        return SolverResult(x, z, "greedy", "feasible", solve_time)
