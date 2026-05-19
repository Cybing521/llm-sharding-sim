"""离散时间步仿真引擎。

在多边缘设备上仿真 LLM 推理过程，包括：
- 层占用标记（每层每设备的忙闲状态）
- KV Cache 增长追踪
- 内存使用随时间的变化
- 请求生命周期：到达 -> 预填充(prefill) -> 解码(decode) -> 完成
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
    z_q: np.ndarray            # [U, K] 该请求的路由矩阵
    phase: RequestPhase = RequestPhase.WAITING
    current_layer: int = 0     # 当前正在处理的层索引
    tokens_generated: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    prefill_end_time: float = 0.0

    # 当前层剩余计算时间
    layer_compute_remaining: float = 0.0
    # 传输到下一层所在设备的剩余时间
    transfer_remaining: float = 0.0
    is_transferring: bool = False


@dataclass
class SimSnapshot:
    """某一时刻的仿真状态快照。"""
    time: float
    device_memory_used: list[float]     # 每个设备的总内存使用量
    device_kv_cache: list[float]        # 每个设备的 KV Cache 使用量
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
        """使用给定的路由矩阵 z 和放置矩阵 x 运行仿真。

        参数：
            requests: 用户请求列表
            z: [Q, U, K] 路由矩阵
            x: [U, K] 层放置矩阵

        返回：
            包含快照、各请求结果等信息的字典。
        """
        U = self.model.num_layers
        K = self.cluster.num_devices

        # 根据层放置方案初始化设备静态内存占用
        device_layer_mem = np.zeros(K)
        for k in range(K):
            for u in range(U):
                if x[u, k] > 0.5:
                    device_layer_mem[k] += self.model.layer_size_mb(u)

        # 每设备的 KV Cache（动态增长）
        device_kv = np.zeros(K)
        # 追踪各层在各设备上的占用情况（按请求）
        # layer_busy[u][k] = 正在使用该层的请求索引集合
        layer_busy: dict[tuple[int, int], set] = {}
        for u in range(U):
            for k in range(K):
                layer_busy[(u, k)] = set()

        # 初始化各请求的状态
        req_states: list[RequestState] = []
        for q, req in enumerate(requests):
            rs = RequestState(request=req, z_q=z[q])
            req_states.append(rs)

        snapshots: list[SimSnapshot] = []
        current_time = 0.0
        max_time = 1000.0  # 安全上限，防止无限循环

        while current_time < max_time:
            all_done = all(rs.phase == RequestPhase.COMPLETE for rs in req_states)
            if all_done:
                break

            # 记录当前时刻的快照
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

            # 处理每个请求
            for q, rs in enumerate(req_states):
                if rs.phase == RequestPhase.COMPLETE:
                    continue

                # 检查请求是否已到达
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

                # 处理层间传输
                if rs.is_transferring:
                    rs.transfer_remaining -= self.tick_s
                    if rs.transfer_remaining <= 0:
                        rs.is_transferring = False
                        # 开始在下一层上计算
                        u = rs.current_layer
                        dev_k = int(np.argmax(rs.z_q[u]))
                        c_k = self.cluster.devices[dev_k].tokens_per_second_per_layer
                        if rs.phase == RequestPhase.PREFILL:
                            rs.layer_compute_remaining = rs.request.prompt_length / c_k
                        else:
                            rs.layer_compute_remaining = 1.0 / c_k
                        layer_busy[(u, dev_k)].add(q)
                    continue

                # 在当前层上进行计算
                rs.layer_compute_remaining -= self.tick_s
                if rs.layer_compute_remaining <= 0:
                    u = rs.current_layer
                    dev_k = int(np.argmax(rs.z_q[u]))
                    layer_busy[(u, dev_k)].discard(q)

                    # 为该层在该设备上添加 KV Cache（decode 阶段为 1 token，prefill 阶段为 prompt_len 个 token）
                    if rs.phase == RequestPhase.PREFILL:
                        kv_bytes = (self.model.kv_cache_bytes_per_layer_per_token
                                    * rs.request.prompt_length)
                    else:
                        kv_bytes = self.model.kv_cache_bytes_per_layer_per_token
                    device_kv[dev_k] += kv_bytes / (1024 * 1024)

                    # 移动到下一层
                    if u + 1 < U:
                        rs.current_layer = u + 1
                        next_dev = int(np.argmax(rs.z_q[u + 1]))
                        if next_dev != dev_k:
                            # 需要跨设备传输
                            rs.is_transferring = True
                            rs.transfer_remaining = self.cluster.transfer_time_s(
                                dev_k, next_dev, self.model.activation_size_bytes
                            )
                        else:
                            # 同一设备，直接开始下一层计算
                            c_next = self.cluster.devices[next_dev].tokens_per_second_per_layer
                            if rs.phase == RequestPhase.PREFILL:
                                rs.layer_compute_remaining = rs.request.prompt_length / c_next
                            else:
                                rs.layer_compute_remaining = 1.0 / c_next
                            layer_busy[(u + 1, next_dev)].add(q)
                    else:
                        # 当前轮次所有层处理完毕
                        if rs.phase == RequestPhase.PREFILL:
                            rs.prefill_end_time = current_time
                            rs.phase = RequestPhase.DECODE
                            rs.tokens_generated = 1
                            # 开始解码阶段：回到第 1 层（跳过 embedding 层）
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
                                # 释放该请求占用的 KV Cache
                                for uu in range(U):
                                    dk = int(np.argmax(rs.z_q[uu]))
                                    total_tokens = rs.request.prompt_length + rs.tokens_generated
                                    kv_release = (self.model.kv_cache_bytes_per_layer_per_token
                                                  * total_tokens) / (1024 * 1024)
                                    device_kv[dk] = max(0, device_kv[dk] - kv_release)
                            else:
                                # 下一个解码步骤：回到第 1 层
                                rs.current_layer = 1
                                dev_1 = int(np.argmax(rs.z_q[1]))
                                c_1 = self.cluster.devices[dev_1].tokens_per_second_per_layer
                                rs.layer_compute_remaining = 1.0 / c_1
                                layer_busy[(1, dev_1)].add(q)

            current_time += self.tick_s

        # 收集仿真结果
        request_results = []
        for q, rs in enumerate(req_states):
            # 同时计算解析延迟，用于与仿真结果对比
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
