"""贪心求解器：按设备容量从大到小，依次将连续层分配给设备。"""

import time
import numpy as np
from solver.base_solver import BaseSolver, SolverResult
from simulator.model import ModelConfig
from simulator.device import DeviceCluster
from simulator.request import Request


class GreedySolver(BaseSolver):
    """将连续的层块分配给各设备。

    策略：
    1. 按设备内存容量降序排列。
    2. 贪心地将连续层填充到每个设备，直到装满。
    3. 优先将层放置在计算速度更快的设备上。
    4. 每个请求通过已放置的层进行路由（每层只有唯一有效选项）。
    """

    def solve(self, requests: list[Request]) -> SolverResult:
        t0 = time.time()
        U = self.model.num_layers
        K = self.cluster.num_devices

        x = np.zeros((U, K), dtype=int)

        # 设备排序：优先选择更快的（更高的 tokens/s），内存容量作为次要排序依据
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

        # 如果未能放置所有层，进行回退分配
        if layer_idx < U:
            for u in range(layer_idx, U):
                layer_size = self.model.layer_size_mb(u)
                for k in device_order:
                    if remaining_cap[k] >= layer_size:
                        x[u, k] = 1
                        remaining_cap[k] -= layer_size
                        break
                else:
                    # 不可行——总内存不足以放置所有层
                    return SolverResult(x, np.zeros((len(requests), U, K)),
                                        "greedy", "infeasible", time.time() - t0)

        # 路由：每个请求使用各层所在的唯一设备
        Q = len(requests)
        z = np.zeros((Q, U, K), dtype=int)
        for q in range(Q):
            for u in range(U):
                # 选择拥有该层的设备
                devices_with_layer = np.where(x[u] > 0.5)[0]
                if len(devices_with_layer) == 1:
                    z[q, u, devices_with_layer[0]] = 1
                else:
                    # 存在多个副本时：选择离请求到达设备最近的
                    arr_dev = requests[q].arrival_device
                    best_k = min(devices_with_layer,
                                 key=lambda k: self.cluster.latency_ms[arr_dev, k])
                    z[q, u, best_k] = 1

        solve_time = time.time() - t0
        return SolverResult(x, z, "greedy", "feasible", solve_time)
