"""层放置求解器的基类接口。"""

from abc import ABC, abstractmethod
import numpy as np
from simulator.model import ModelConfig
from simulator.device import DeviceCluster
from simulator.request import Request


class SolverResult:
    def __init__(self, x: np.ndarray, z: np.ndarray, solver_name: str,
                 status: str = "optimal", solve_time: float = 0.0):
        """
        参数：
            x: [U, K] 二值层放置矩阵
            z: [Q, U, K] 二值路由矩阵
            solver_name: 使用的求解器名称
            status: 求解状态，"optimal"、"feasible"、"infeasible"
            solve_time: 求解耗时（秒）
        """
        self.x = x
        self.z = z
        self.solver_name = solver_name
        self.status = status
        self.solve_time = solve_time

    def layer_assignment(self) -> dict[int, list[int]]:
        """返回 {设备索引: [层索引列表]} 的映射。"""
        U, K = self.x.shape
        assignment = {k: [] for k in range(K)}
        for u in range(U):
            for k in range(K):
                if self.x[u, k] > 0.5:
                    assignment[k].append(u)
        return assignment

    def print_assignment(self, cluster: DeviceCluster, model: ModelConfig):
        assignment = self.layer_assignment()
        print(f"\n{'='*60}")
        print(f"Solver: {self.solver_name} | Status: {self.status} | Time: {self.solve_time:.3f}s")
        print(f"{'='*60}")
        for k, layers in assignment.items():
            dev = cluster.devices[k]
            mem_used = sum(model.layer_size_mb(u) for u in layers)
            layer_str = ", ".join(str(u) for u in layers) if layers else "(none)"
            print(f"  {dev.name} [{dev.memory_mb}MB]: "
                  f"layers [{layer_str}] = {mem_used:.0f}MB / {dev.memory_mb}MB")
        print()


class BaseSolver(ABC):
    def __init__(self, model: ModelConfig, cluster: DeviceCluster):
        self.model = model
        self.cluster = cluster

    @abstractmethod
    def solve(self, requests: list[Request]) -> SolverResult:
        pass
