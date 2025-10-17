import pulp
import numpy as np
import wandb
from itertools import product
from src.utility.utility_function import *

def run_cvrp_mip():
    run = wandb.init(
        project='Capacitated-Vehicle-Routing',
        job_type='MIP_CVRP',
        name='MIP_CVRP_Run'
    )

    # --- Load preprocessed data ---
    artifact = run.use_artifact('scsthilakarathne-nibm/Capacitated-Vehicle-Routing/CVRP_Dataset:v0', type='dataset')
    artifact_dir = artifact.download()
    dist_matrix, demands, coords = load_preprocessed_data(artifact_dir)
    vehicle_capacity = int(demands[:, 1].max() * 5)
    n_nodes = len(demands)
    nodes = list(range(n_nodes))
    customers = list(range(1, n_nodes))  # exclude depot 0

    # --- Define PuLP problem ---
    prob = pulp.LpProblem("CVRP", pulp.LpMinimize)

    # Decision variables: x[i,j] = 1 if route goes from i to j
    x = pulp.LpVariable.dicts("x", (nodes, nodes), cat="Binary")

    # MTZ variables for sub-tour elimination
    u = pulp.LpVariable.dicts("u", customers, lowBound=0, upBound=vehicle_capacity, cat="Continuous")

    # --- Objective: minimize total distance ---
    prob += pulp.lpSum(dist_matrix[i, j] * x[i][j] for i in nodes for j in nodes if i != j)

    # --- Constraints ---

    # 1. Each customer visited exactly once
    for j in customers:
        prob += pulp.lpSum(x[i][j] for i in nodes if i != j) == 1
        prob += pulp.lpSum(x[j][i] for i in nodes if i != j) == 1

    # 2. Depot constraints: vehicles leave and return
    prob += pulp.lpSum(x[0][j] for j in customers) <= 4  # max number of vehicles
    prob += pulp.lpSum(x[j][0] for j in customers) <= 4

    # 3. MTZ sub-tour elimination constraints
    for i, j in product(customers, repeat=2):
        if i != j:
            prob += u[i] - u[j] + vehicle_capacity * x[i][j] <= vehicle_capacity - demands[j,1]

    # --- Solve ---
    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=300)
    prob.solve(solver)

    # --- Extract solution ---
    solution_edges = [(i, j) for i in nodes for j in nodes if i != j and pulp.value(x[i][j]) > 0.5]

    # --- Reconstruct routes ---
    def reconstruct_routes(edges):
        routes = []
        unused = set(customers)
        while unused:
            route = [0]
            current = 0
            while True:
                next_nodes = [j for i,j in edges if i == current and j in unused]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                route.append(next_node)
                unused.remove(next_node)
                current = next_node
            route.append(0)
            routes.append(route)
        return routes

    best_routes = reconstruct_routes(solution_edges)
    total_distance = pulp.value(prob.objective)
    best_fitness = 1 / total_distance
    num_routes = len(best_routes)
    avg_customers_per_route = np.mean([len(r) - 2 for r in best_routes])  # exclude depot

    # --- Metrics ---
    metrics = {
        "avg_customers_per_route": avg_customers_per_route,
        "best_fitness": best_fitness,
        "final_best_fitness": best_fitness,
        "final_total_distance": total_distance,
        "generation": 0,  # not applicable for MIP
        "num_routes": num_routes
    }

    # --- Log metrics to W&B ---
    wandb.log(metrics)

    # Save routes and objective as artifact
    np.save("mip_best_routes.npy", np.array(best_routes))
    artifact_out = wandb.Artifact("MIP_CVRP_Solution", type="model", description="MIP solution for CVRP")
    artifact_out.add_file("mip_best_routes.npy")
    wandb.log_artifact(artifact_out)
    run.finish()

    print(f"✅ MIP Total Distance: {total_distance:.2f}")
    print(f"Routes ({num_routes} vehicles): {best_routes}")
    print(f"Metrics: {metrics}")

