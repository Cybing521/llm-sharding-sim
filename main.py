"""Main entry: load config -> solve -> simulate -> visualize.

Usage:
    python main.py                          # run all experiments
    python main.py --experiment 1           # run specific experiment
    python main.py --solver greedy          # single solver
    python main.py --solver ilp --time-limit 60
"""

import argparse
import os
import sys
import time
import copy

import numpy as np

from simulator.model import ModelConfig
from simulator.device import DeviceCluster
from simulator.request import Request, load_requests
from simulator.cost import compute_total_cost, check_memory_feasibility
from simulator.engine import SimulationEngine
from solver.greedy_solver import GreedySolver
from solver.ilp_solver import ILPSolver
from solver.base_solver import SolverResult
from analysis.compare import compare_solvers
from analysis.visualize import (
    plot_memory_timeline,
    plot_delay_comparison,
    plot_layer_assignment_heatmap,
    plot_prompt_length_sweep,
)


def run_experiment_1(model, cluster, requests, results_dir):
    """Experiment 1: Single request, ILP vs Greedy."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Single request -- ILP vs Greedy")
    print("=" * 60)

    single_req = [requests[0]]
    exp_dir = os.path.join(results_dir, "exp1_single_request")
    os.makedirs(exp_dir, exist_ok=True)

    greedy = GreedySolver(model, cluster)
    greedy_result = greedy.solve(single_req)
    greedy_result.print_assignment(cluster, model)

    ilp = ILPSolver(model, cluster, time_limit=120)
    ilp_result = ilp.solve(single_req)
    ilp_result.print_assignment(cluster, model)

    report = compare_solvers([greedy_result, ilp_result], model, cluster, single_req)
    print(report)
    with open(os.path.join(exp_dir, "report.txt"), "w") as f:
        f.write(report)

    # Simulate both
    engine = SimulationEngine(model, cluster, tick_s=0.005)
    sim_results = {}
    for label, result in [("Greedy", greedy_result), ("ILP", ilp_result)]:
        if result.status in ("infeasible", "not_solved"):
            print(f"  Skipping simulation for {label}: {result.status}")
            continue
        sim = engine.run(single_req, result.z, result.x)
        sim_results[label] = sim["request_results"]
        plot_memory_timeline(
            sim["snapshots"], cluster,
            title=f"Exp1: Memory Timeline ({label})",
            save_path=os.path.join(exp_dir, f"memory_{label.lower()}.png"),
        )
        plot_layer_assignment_heatmap(
            result.x, model, cluster, solver_name=label,
            save_path=os.path.join(exp_dir, f"heatmap_{label.lower()}.png"),
        )

    if sim_results:
        plot_delay_comparison(
            sim_results,
            save_path=os.path.join(exp_dir, "delay_comparison.png"),
        )


def run_experiment_2(model, cluster, requests, results_dir):
    """Experiment 2: Multi-request concurrent, observe KV Cache pressure."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Multi-request concurrent -- KV Cache pressure")
    print("=" * 60)

    exp_dir = os.path.join(results_dir, "exp2_multi_request")
    os.makedirs(exp_dir, exist_ok=True)

    greedy = GreedySolver(model, cluster)
    greedy_result = greedy.solve(requests)
    greedy_result.print_assignment(cluster, model)

    report = compare_solvers([greedy_result], model, cluster, requests)
    print(report)
    with open(os.path.join(exp_dir, "report.txt"), "w") as f:
        f.write(report)

    engine = SimulationEngine(model, cluster, tick_s=0.005)
    sim = engine.run(requests, greedy_result.z, greedy_result.x)

    plot_memory_timeline(
        sim["snapshots"], cluster,
        title="Exp2: Multi-Request Memory Timeline (Greedy)",
        save_path=os.path.join(exp_dir, "memory_timeline.png"),
    )
    plot_layer_assignment_heatmap(
        greedy_result.x, model, cluster, solver_name="Greedy",
        save_path=os.path.join(exp_dir, "heatmap.png"),
    )

    print("\nSimulated request results:")
    for r in sim["request_results"]:
        print(f"  {r['request_id']}: simulated={r['simulated_total_s']:.3f}s "
              f"analytical={r['analytical_total_s']:.3f}s "
              f"tokens={r['tokens_generated']}")


def run_experiment_3(model, cluster, results_dir):
    """Experiment 3: Sweep prompt lengths, compare delay."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Prompt length sweep")
    print("=" * 60)

    exp_dir = os.path.join(results_dir, "exp3_prompt_sweep")
    os.makedirs(exp_dir, exist_ok=True)

    prompt_lengths = [32, 64, 128, 256]
    output_len = 64

    sweep_results = {"Greedy": [], "ILP": []}

    for pl in prompt_lengths:
        req = [Request(id=f"r_pl{pl}", prompt_length=pl,
                       output_length=output_len, arrival_device=0,
                       arrival_time=0.0)]

        greedy = GreedySolver(model, cluster)
        gr = greedy.solve(req)
        cost_g = compute_total_cost(gr.x, gr.z, model, cluster, req)
        sweep_results["Greedy"].append({
            "prompt_length": pl,
            "total_delay": cost_g["total_delay"],
        })

        ilp = ILPSolver(model, cluster, time_limit=60)
        ir = ilp.solve(req)
        if ir.status == "optimal":
            cost_i = compute_total_cost(ir.x, ir.z, model, cluster, req)
            sweep_results["ILP"].append({
                "prompt_length": pl,
                "total_delay": cost_i["total_delay"],
            })
        else:
            sweep_results["ILP"].append({
                "prompt_length": pl,
                "total_delay": cost_g["total_delay"],
            })

        print(f"  prompt_length={pl}: Greedy={cost_g['total_delay']:.4f}s", end="")
        if ir.status == "optimal":
            print(f"  ILP={cost_i['total_delay']:.4f}s")
        else:
            print(f"  ILP={ir.status}")

    plot_prompt_length_sweep(
        sweep_results,
        save_path=os.path.join(exp_dir, "prompt_sweep.png"),
    )

    with open(os.path.join(exp_dir, "sweep_data.txt"), "w") as f:
        for solver_name, data in sweep_results.items():
            f.write(f"--- {solver_name} ---\n")
            for d in data:
                f.write(f"  prompt_length={d['prompt_length']}: "
                        f"total_delay={d['total_delay']:.6f}s\n")


def run_experiment_4(model, cluster, requests, results_dir):
    """Experiment 4: Bandwidth sweep, analyze transmission bottleneck."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Bandwidth sweep")
    print("=" * 60)

    exp_dir = os.path.join(results_dir, "exp4_bandwidth_sweep")
    os.makedirs(exp_dir, exist_ok=True)

    single_req = [requests[0]]
    bw_factors = [0.5, 1.0, 2.0, 5.0, 10.0]
    base_bw = cluster.bandwidth_mbps.copy()

    sweep_data = {"Greedy": [], "ILP": []}

    for factor in bw_factors:
        # Scale bandwidth
        test_cluster = copy.deepcopy(cluster)
        test_cluster.bandwidth_mbps = base_bw * factor
        # Keep diagonal at 0
        np.fill_diagonal(test_cluster.bandwidth_mbps, 0)

        bw_label = f"{factor}x"

        greedy = GreedySolver(model, test_cluster)
        gr = greedy.solve(single_req)
        cost_g = compute_total_cost(gr.x, gr.z, model, test_cluster, single_req)
        sweep_data["Greedy"].append({
            "bw_factor": factor,
            "total_delay": cost_g["total_delay"],
        })

        ilp = ILPSolver(model, test_cluster, time_limit=60)
        ir = ilp.solve(single_req)
        if ir.status == "optimal":
            cost_i = compute_total_cost(ir.x, ir.z, model, test_cluster, single_req)
            sweep_data["ILP"].append({
                "bw_factor": factor,
                "total_delay": cost_i["total_delay"],
            })
        else:
            sweep_data["ILP"].append({
                "bw_factor": factor,
                "total_delay": cost_g["total_delay"],
            })

        print(f"  bandwidth={bw_label}: Greedy={cost_g['total_delay']:.4f}s", end="")
        if ir.status == "optimal":
            print(f"  ILP={cost_i['total_delay']:.4f}s")
        else:
            print(f"  ILP={ir.status}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for solver_name, data in sweep_data.items():
        factors = [d["bw_factor"] for d in data]
        delays = [d["total_delay"] for d in data]
        ax.plot(factors, delays, "o-", label=solver_name, linewidth=2, markersize=8)
    ax.set_xlabel("Bandwidth Factor (x base)")
    ax.set_ylabel("Total Delay (s)")
    ax.set_title("Delay vs Network Bandwidth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    save_path = os.path.join(exp_dir, "bandwidth_sweep.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved bandwidth sweep -> {save_path}")

    with open(os.path.join(exp_dir, "sweep_data.txt"), "w") as f:
        for solver_name, data in sweep_data.items():
            f.write(f"--- {solver_name} ---\n")
            for d in data:
                f.write(f"  bw_factor={d['bw_factor']}: "
                        f"total_delay={d['total_delay']:.6f}s\n")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Make plt available for exp4
    globals()["plt"] = plt

    parser = argparse.ArgumentParser(description="LLM Sharding Simulator")
    parser.add_argument("--experiment", type=int, default=0,
                        help="Run specific experiment (1-4), 0 for all")
    parser.add_argument("--config-dir", type=str, default="config",
                        help="Config directory path")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Results output directory")
    args = parser.parse_args()

    config_dir = args.config_dir
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    print("Loading configurations...")
    model = ModelConfig.from_yaml(os.path.join(config_dir, "model.yaml"))
    cluster = DeviceCluster.from_yaml(os.path.join(config_dir, "devices.yaml"))
    requests = load_requests(os.path.join(config_dir, "requests.yaml"))

    print(f"  Model: {model.name} | {model.num_layers} layers | "
          f"{model.total_size_mb:.0f} MB total")
    print(f"  Devices: {cluster.num_devices}")
    for d in cluster.devices:
        print(f"    {d.name}: {d.memory_mb} MB, "
              f"{d.tokens_per_second_per_layer} tok/s/layer")
    print(f"  Requests: {len(requests)}")
    for r in requests:
        print(f"    {r.id}: prompt={r.prompt_length}, output={r.output_length}, "
              f"device={r.arrival_device}")

    exp = args.experiment
    if exp == 0 or exp == 1:
        run_experiment_1(model, cluster, requests, results_dir)
    if exp == 0 or exp == 2:
        run_experiment_2(model, cluster, requests, results_dir)
    if exp == 0 or exp == 3:
        run_experiment_3(model, cluster, results_dir)
    if exp == 0 or exp == 4:
        run_experiment_4(model, cluster, requests, results_dir)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {os.path.abspath(results_dir)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
