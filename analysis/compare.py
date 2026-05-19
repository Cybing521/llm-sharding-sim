"""策略对比工具函数。"""

from simulator.model import ModelConfig
from simulator.device import DeviceCluster
from simulator.request import Request
from simulator.cost import compute_total_cost, check_memory_feasibility
from solver.base_solver import SolverResult


def compare_solvers(results: list[SolverResult],
                    model: ModelConfig,
                    cluster: DeviceCluster,
                    requests: list[Request]) -> str:
    """生成多个求解器结果的文本对比报告。"""
    lines = []
    lines.append("=" * 70)
    lines.append("SOLVER COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append(f"Model: {model.name} | Layers: {model.num_layers} | "
                 f"Total size: {model.total_size_mb:.0f} MB")
    lines.append(f"Devices: {cluster.num_devices} | "
                 f"Requests: {len(requests)}")
    lines.append("")

    for result in results:
        lines.append(f"--- {result.solver_name} ---")
        lines.append(f"  Status: {result.status} | Solve time: {result.solve_time:.3f}s")

        # Memory check
        mem_check = check_memory_feasibility(result.x, model, cluster)
        for mc in mem_check:
            flag = "OK" if mc["feasible"] else "OVER"
            lines.append(f"  {mc['device']}: {mc['used_mb']:.0f}/{mc['capacity_mb']:.0f} MB [{flag}]")

        # Cost
        cost = compute_total_cost(result.x, result.z, model, cluster, requests)
        lines.append(f"  Total delay: {cost['total_delay']:.4f} s")
        for pr in cost["per_request"]:
            lines.append(
                f"    {pr['request_id']}: "
                f"prc={pr['t_prc']:.4f}s  dec={pr['t_dec']:.4f}s  "
                f"trans={pr['t_trans']:.4f}s  total={pr['t_total']:.4f}s"
            )

        # Layer assignment
        assignment = result.layer_assignment()
        lines.append("  Layer assignment:")
        for k, layers in assignment.items():
            if layers:
                lines.append(f"    Device {k} ({cluster.devices[k].name}): "
                             f"layers {layers}")
        lines.append("")

    return "\n".join(lines)
