"""代价函数：计算给定层分配方案下的 t_prc（预填充）、t_dec（解码）、t_trans（传输）。

给定：
  x[u][k]    = 1 表示层 u 被放置在设备 k 上
  z[q][u][k] = 1 表示请求 q 在层 u 使用设备 k
  model, cluster, requests -- 配置对象

计算每个请求的延迟及总代价。
"""

import numpy as np
from simulator.model import ModelConfig
from simulator.device import DeviceCluster
from simulator.request import Request


def compute_request_delay(
    z_q: np.ndarray,           # [U, K] 单请求的二值分配矩阵
    model: ModelConfig,
    cluster: DeviceCluster,
    request: Request,
) -> dict:
    """计算单个请求的 t_prc（预填充时间）、t_dec（解码时间）、t_trans（传输时间）。

    参数：
        z_q: 形状 [num_layers, num_devices]，z_q[u][k] = 1 表示请求在层 u 使用设备 k。
    返回：
        包含 t_prc、t_dec、t_trans、t_total 的字典（单位：秒）。
    """
    U, K = z_q.shape
    r_q = request.prompt_length
    g_q = request.output_length

    # t_prc：预填充阶段——所有层处理 |r_q| 个 token
    t_prc = 0.0
    for u in range(U):
        for k in range(K):
            if z_q[u, k] > 0.5:
                c_k = cluster.devices[k].tokens_per_second_per_layer
                t_prc += r_q / c_k

    # t_dec：解码阶段——(g_q - 1) 步，每层每步处理 1 个 token
    # 解码时跳过 embedding 层（u=0），因为它只是查表操作
    t_dec = 0.0
    if g_q > 1:
        for u in range(1, U):
            for k in range(K):
                if z_q[u, k] > 0.5:
                    c_k = cluster.devices[k].tokens_per_second_per_layer
                    t_dec += (g_q - 1) * (1.0 / c_k)

    # t_trans：相邻层在不同设备上时产生的传输时间
    a = model.activation_size_bytes
    t_trans = 0.0
    for u in range(U - 1):
        dev_u = int(np.argmax(z_q[u]))
        dev_u1 = int(np.argmax(z_q[u + 1]))
        if dev_u != dev_u1:
            per_hop = cluster.transfer_time_s(dev_u, dev_u1, a)
            # 预填充：一次性传输 |r_q| 个 token 的隐藏状态
            # 解码：每步传输 1 个 token 的隐藏状态，共 (g_q - 1) 次
            # 简化处理：共 g_q 次跳转（预填充视为 1 次批量传输 + 解码步骤）
            t_trans += g_q * per_hop

    return {
        "t_prc": t_prc,
        "t_dec": t_dec,
        "t_trans": t_trans,
        "t_total": t_prc + t_dec + t_trans,
    }


def compute_total_cost(
    x: np.ndarray,             # [U, K] 层放置矩阵
    z: np.ndarray,             # [Q, U, K] 路由矩阵
    model: ModelConfig,
    cluster: DeviceCluster,
    requests: list[Request],
) -> dict:
    """计算所有请求的总代价。

    返回：
        包含各请求延迟列表和汇总总延迟的字典。
    """
    per_request = []
    total_delay = 0.0
    for q, req in enumerate(requests):
        delays = compute_request_delay(z[q], model, cluster, req)
        delays["request_id"] = req.id
        per_request.append(delays)
        total_delay += delays["t_total"]

    return {
        "per_request": per_request,
        "total_delay": total_delay,
    }


def check_memory_feasibility(
    x: np.ndarray,
    model: ModelConfig,
    cluster: DeviceCluster,
) -> list[dict]:
    """检查层放置方案 x 是否满足内存约束（仅静态权重部分）。"""
    U, K = x.shape
    results = []
    for k in range(K):
        used = sum(x[u, k] * model.layer_size_mb(u) for u in range(U))
        cap = cluster.devices[k].memory_mb
        results.append({
            "device": cluster.devices[k].name,
            "used_mb": used,
            "capacity_mb": cap,
            "feasible": used <= cap,
        })
    return results
