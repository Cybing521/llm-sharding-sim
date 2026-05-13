"""ILP solver using PuLP, matching the Notion formulation exactly.

Decision variables:
    x[u,k]      -- binary: layer u placed on device k
    z[q,u,k]    -- binary: request q uses device k for layer u
    y[q,u,k,k'] -- auxiliary: linearizes the product z[q,u,k] * z[q,u+1,k']

Objective:
    min  sum_q  t_delay(r_q)
    t_delay = t_prc + t_dec + t_trans

Constraints:
    1. Memory:   sum_u  x[u,k] * size(l_u) <= h_k          for all k
    2. Unique:   sum_k  z[q,u,k] = 1                        for all q, u
    3. Placement: z[q,u,k] <= x[u,k]                        for all q, u, k
    4. Linearization of y = z[q,u,k] * z[q,u+1,k']:
       y >= z[q,u,k] + z[q,u+1,k'] - 1
       y <= z[q,u,k]
       y <= z[q,u+1,k']
"""

import time
import numpy as np
import pulp

from solver.base_solver import BaseSolver, SolverResult
from simulator.model import ModelConfig
from simulator.device import DeviceCluster
from simulator.request import Request


class ILPSolver(BaseSolver):
    def __init__(self, model: ModelConfig, cluster: DeviceCluster,
                 time_limit: int = 300, solver_name: str = "PULP_CBC_CMD"):
        super().__init__(model, cluster)
        self.time_limit = time_limit
        self.pulp_solver_name = solver_name

    def solve(self, requests: list[Request]) -> SolverResult:
        t0 = time.time()
        U = self.model.num_layers
        K = self.cluster.num_devices
        Q = len(requests)

        prob = pulp.LpProblem("LLM_Sharding", pulp.LpMinimize)

        # --- Decision variables ---
        x = {}
        for u in range(U):
            for k in range(K):
                x[u, k] = pulp.LpVariable(f"x_{u}_{k}", cat="Binary")

        z = {}
        for q in range(Q):
            for u in range(U):
                for k in range(K):
                    z[q, u, k] = pulp.LpVariable(f"z_{q}_{u}_{k}", cat="Binary")

        # Auxiliary variables for linearizing z[q,u,k] * z[q,u+1,k']
        y = {}
        for q in range(Q):
            for u in range(U - 1):
                for k in range(K):
                    for kp in range(K):
                        if k == kp:
                            continue
                        y[q, u, k, kp] = pulp.LpVariable(
                            f"y_{q}_{u}_{k}_{kp}", lowBound=0, upBound=1, cat="Binary"
                        )

        # --- Objective function ---
        obj = pulp.LpAffineExpression()

        for q in range(Q):
            r_q = requests[q].prompt_length
            g_q = requests[q].output_length
            a = self.model.activation_size_bytes

            # t_prc: prefill
            for u in range(U):
                for k in range(K):
                    c_k = self.cluster.devices[k].tokens_per_second_per_layer
                    obj += z[q, u, k] * (r_q / c_k)

            # t_dec: decode (skip embedding u=0)
            if g_q > 1:
                for u in range(1, U):
                    for k in range(K):
                        c_k = self.cluster.devices[k].tokens_per_second_per_layer
                        obj += z[q, u, k] * ((g_q - 1) / c_k)

            # t_trans: transmission between different devices
            for u in range(U - 1):
                for k in range(K):
                    for kp in range(K):
                        if k == kp:
                            continue
                        a_mb = a / (1024 * 1024)
                        bw = self.cluster.bandwidth_mbps[k, kp]
                        lat = self.cluster.latency_ms[k, kp] / 1000.0
                        if bw > 0:
                            transfer_cost = a_mb / bw + lat
                        else:
                            transfer_cost = 1e6
                        obj += y[q, u, k, kp] * (g_q * transfer_cost)

        prob += obj, "total_delay"

        # --- Constraints ---

        # 1. Memory constraint
        for k in range(K):
            prob += (
                pulp.lpSum(x[u, k] * self.model.layer_size_mb(u) for u in range(U))
                <= self.cluster.devices[k].memory_mb,
                f"memory_{k}"
            )

        # 2. Each layer must be placed on at least one device
        for u in range(U):
            prob += (
                pulp.lpSum(x[u, k] for k in range(K)) >= 1,
                f"layer_placed_{u}"
            )

        # 3. Unique routing: each request uses exactly one device per layer
        for q in range(Q):
            for u in range(U):
                prob += (
                    pulp.lpSum(z[q, u, k] for k in range(K)) == 1,
                    f"unique_{q}_{u}"
                )

        # 4. Routing respects placement
        for q in range(Q):
            for u in range(U):
                for k in range(K):
                    prob += (
                        z[q, u, k] <= x[u, k],
                        f"route_place_{q}_{u}_{k}"
                    )

        # 5. Linearization: y[q,u,k,k'] = z[q,u,k] * z[q,u+1,k']
        for q in range(Q):
            for u in range(U - 1):
                for k in range(K):
                    for kp in range(K):
                        if k == kp:
                            continue
                        prob += (
                            y[q, u, k, kp] >= z[q, u, k] + z[q, u + 1, kp] - 1,
                            f"lin_lb_{q}_{u}_{k}_{kp}"
                        )
                        prob += (
                            y[q, u, k, kp] <= z[q, u, k],
                            f"lin_ub1_{q}_{u}_{k}_{kp}"
                        )
                        prob += (
                            y[q, u, k, kp] <= z[q, u + 1, kp],
                            f"lin_ub2_{q}_{u}_{k}_{kp}"
                        )

        # --- Solve ---
        solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=self.time_limit)
        prob.solve(solver)

        status_map = {
            pulp.constants.LpStatusOptimal: "optimal",
            pulp.constants.LpStatusNotSolved: "not_solved",
            pulp.constants.LpStatusInfeasible: "infeasible",
            pulp.constants.LpStatusUnbounded: "unbounded",
        }
        status = status_map.get(prob.status, "unknown")

        # Extract solution
        x_arr = np.zeros((U, K), dtype=int)
        z_arr = np.zeros((Q, U, K), dtype=int)

        if status in ("optimal",):
            for u in range(U):
                for k in range(K):
                    x_arr[u, k] = int(round(x[u, k].varValue or 0))
            for q in range(Q):
                for u in range(U):
                    for k in range(K):
                        z_arr[q, u, k] = int(round(z[q, u, k].varValue or 0))

        solve_time = time.time() - t0
        return SolverResult(x_arr, z_arr, "ILP (PuLP/CBC)", status, solve_time)
