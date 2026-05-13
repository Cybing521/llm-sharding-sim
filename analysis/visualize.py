"""Visualization: memory timelines, delay comparison, layer heatmap."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from simulator.model import ModelConfig
from simulator.device import DeviceCluster


def plot_memory_timeline(snapshots: list, cluster: DeviceCluster,
                         title: str = "Device Memory Usage Over Time",
                         save_path: str = "results/memory_timeline.png"):
    """Plot memory usage per device over simulation time."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    times = [s.time for s in snapshots]
    K = cluster.num_devices

    fig, axes = plt.subplots(K, 1, figsize=(12, 3 * K), sharex=True)
    if K == 1:
        axes = [axes]

    for k in range(K):
        ax = axes[k]
        mem = [s.device_memory_used[k] for s in snapshots]
        kv = [s.device_kv_cache[k] for s in snapshots]
        static = [m - kvc for m, kvc in zip(mem, kv)]

        ax.fill_between(times, 0, static, alpha=0.6, label="Layer Weights")
        ax.fill_between(times, static, mem, alpha=0.6, label="KV Cache")
        ax.axhline(y=cluster.devices[k].memory_mb, color="red",
                    linestyle="--", linewidth=1.5, label="Capacity")
        ax.set_ylabel("MB")
        ax.set_title(f"{cluster.devices[k].name} ({cluster.devices[k].memory_mb} MB)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, cluster.devices[k].memory_mb * 1.1)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved memory timeline -> {save_path}")


def plot_delay_comparison(results_by_solver: dict,
                          save_path: str = "results/delay_comparison.png"):
    """Bar chart comparing total delay across solvers.

    Args:
        results_by_solver: {solver_name: [{"request_id": ..., "t_total": ...}, ...]}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    solver_names = list(results_by_solver.keys())
    request_ids = [r["request_id"] for r in results_by_solver[solver_names[0]]]
    n_requests = len(request_ids)
    n_solvers = len(solver_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.8 / n_solvers
    x = np.arange(n_requests)

    for i, name in enumerate(solver_names):
        delays = [r["simulated_total_s"] for r in results_by_solver[name]]
        bars = ax.bar(x + i * bar_width, delays, bar_width, label=name)
        for bar, val in zip(bars, delays):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Request")
    ax.set_ylabel("Total Delay (s)")
    ax.set_title("Delay Comparison: ILP vs Greedy")
    ax.set_xticks(x + bar_width * (n_solvers - 1) / 2)
    ax.set_xticklabels(request_ids)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved delay comparison -> {save_path}")


def plot_layer_assignment_heatmap(x: np.ndarray, model: ModelConfig,
                                  cluster: DeviceCluster,
                                  solver_name: str = "",
                                  save_path: str = "results/layer_heatmap.png"):
    """Heatmap showing which layers are on which devices."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    U, K = x.shape

    fig, ax = plt.subplots(figsize=(max(8, K * 1.5), max(6, U * 0.3)))
    cmap = ListedColormap(["#f0f0f0", "#4CAF50"])
    im = ax.imshow(x, cmap=cmap, aspect="auto", interpolation="nearest")

    layer_labels = []
    for u in range(U):
        lt = model.layers[u].layer_type
        if lt == "embedding":
            layer_labels.append(f"L{u} (emb)")
        elif lt == "lm_head":
            layer_labels.append(f"L{u} (head)")
        else:
            layer_labels.append(f"L{u}")

    device_labels = [d.name.replace("jetson_", "").replace("_", " ") for d in cluster.devices]

    ax.set_xticks(range(K))
    ax.set_xticklabels(device_labels, rotation=45, ha="right")
    ax.set_yticks(range(U))
    ax.set_yticklabels(layer_labels, fontsize=7)
    ax.set_xlabel("Device")
    ax.set_ylabel("Layer")
    ax.set_title(f"Layer Assignment Heatmap ({solver_name})")

    for u in range(U):
        for k in range(K):
            if x[u, k] > 0.5:
                ax.text(k, u, f"{model.layer_size_mb(u):.0f}",
                        ha="center", va="center", fontsize=6, color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved layer heatmap -> {save_path}")


def plot_prompt_length_sweep(results: dict,
                             save_path: str = "results/prompt_sweep.png"):
    """Line plot: delay vs prompt length for each solver."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for solver_name, data in results.items():
        lengths = [d["prompt_length"] for d in data]
        delays = [d["total_delay"] for d in data]
        ax.plot(lengths, delays, "o-", label=solver_name, linewidth=2, markersize=8)

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Total Delay (s)")
    ax.set_title("Delay vs Prompt Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved prompt sweep -> {save_path}")
