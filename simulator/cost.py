"""Cost functions: t_prc, t_dec, t_trans for a given layer assignment.

Given:
  x[u][k]    = 1 if layer u is placed on device k
  z[q][u][k] = 1 if request q uses device k for layer u
  model, cluster, requests  -- config objects

Compute total delay per request and aggregate cost.
"""

import numpy as np
from simulator.model import ModelConfig
from simulator.device import DeviceCluster
from simulator.request import Request


def compute_request_delay(
    z_q: np.ndarray,           # [U, K] binary assignment for one request
    model: ModelConfig,
    cluster: DeviceCluster,
    request: Request,
) -> dict:
    """Compute t_prc, t_dec, t_trans for a single request.

    Args:
        z_q: shape [num_layers, num_devices], z_q[u][k] = 1 if request uses
             device k for layer u.
    Returns:
        dict with t_prc, t_dec, t_trans, t_total (all in seconds).
    """
    U, K = z_q.shape
    r_q = request.prompt_length
    g_q = request.output_length

    # t_prc: prefill -- all layers process |r_q| tokens
    t_prc = 0.0
    for u in range(U):
        for k in range(K):
            if z_q[u, k] > 0.5:
                c_k = cluster.devices[k].tokens_per_second_per_layer
                t_prc += r_q / c_k

    # t_dec: decode -- (g_q - 1) steps, each layer processes 1 token
    # Skip embedding layer (u=0) in decode since it's just a lookup
    t_dec = 0.0
    if g_q > 1:
        for u in range(1, U):
            for k in range(K):
                if z_q[u, k] > 0.5:
                    c_k = cluster.devices[k].tokens_per_second_per_layer
                    t_dec += (g_q - 1) * (1.0 / c_k)

    # t_trans: transmission when consecutive layers are on different devices
    a = model.activation_size_bytes
    t_trans = 0.0
    for u in range(U - 1):
        dev_u = int(np.argmax(z_q[u]))
        dev_u1 = int(np.argmax(z_q[u + 1]))
        if dev_u != dev_u1:
            per_hop = cluster.transfer_time_s(dev_u, dev_u1, a)
            # Prefill: transfer hidden states of |r_q| tokens once
            # Decode: transfer 1 token's hidden state (g_q - 1) times
            # Simplified: g_q total hops (prefill counted as 1 batch transfer + decode steps)
            t_trans += g_q * per_hop

    return {
        "t_prc": t_prc,
        "t_dec": t_dec,
        "t_trans": t_trans,
        "t_total": t_prc + t_dec + t_trans,
    }


def compute_total_cost(
    x: np.ndarray,             # [U, K] placement
    z: np.ndarray,             # [Q, U, K] routing
    model: ModelConfig,
    cluster: DeviceCluster,
    requests: list[Request],
) -> dict:
    """Compute total cost across all requests.

    Returns:
        dict with per_request list and aggregate total_delay.
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
    """Check if placement x satisfies memory constraints (static only)."""
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
