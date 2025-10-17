import wandb
import numpy as np
import matplotlib.pyplot as plt
import random

# Load preprocessed data (coordinates)
def load_coords_from_dataset(artifact_name, type="dataset"):
    artifact = wandb.use_artifact(artifact_name, type=type)
    artifact_dir = artifact.download()
    coords = np.load(f"{artifact_dir}/coords.npy")
    return coords

# --- Function to visualize GA routes ---
def plot_ga_routes(artifact_name='scsthilakarathne-nibm/Capacitated-Vehicle-Routing/GA_Model:v1',
                   dataset_artifact='scsthilakarathne-nibm/Capacitated-Vehicle-Routing/CVRP_Dataset:v0'):
    # Init W&B run for logging
    run = wandb.init(project="Capacitated-Vehicle-Routing", job_type="plot_ga_routes")

    # Load GA solution
    ga_artifact = run.use_artifact(artifact_name, type='model')
    ga_dir = ga_artifact.download()
    best_solution = np.load(f"{ga_dir}/best_solution.npy")

    # Load dataset coordinates
    coords = load_coords_from_dataset(dataset_artifact)

    # Load preprocessed demands
    demands = np.load(f"{wandb.use_artifact(dataset_artifact, type='dataset').download()}/demands.npy")
    vehicle_capacity = int(demands[:,1].max() * 5)

    # Split GA solution into routes
    def split_routes(chromosome, demands, vehicle_capacity):
        routes = []
        current_route = [0]  # depot
        current_load = 0
        for customer in chromosome:
            demand = demands[customer, 1]
            if current_load + demand <= vehicle_capacity:
                current_route.append(customer)
                current_load += demand
            else:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0, customer]
                current_load = demand
        current_route.append(0)
        routes.append(current_route)
        return routes

    routes = split_routes(best_solution, demands, vehicle_capacity)

    # Plot
    plt.figure(figsize=(8,6))
    colors = plt.cm.get_cmap('tab20', len(routes))
    for idx, route in enumerate(routes):
        xs = [coords[i,1] for i in route]
        ys = [coords[i,2] for i in route]
        plt.plot(xs, ys, color=colors(idx), marker='o', label=f'Vehicle {idx+1}')
    plt.scatter(coords[0,1], coords[0,2], c='red', s=100, marker='s', label='Depot')
    plt.title("GA CVRP Routes")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Log figure as artifact
    plt.savefig("ga_routes.png")
    run.log({"GA Routes": wandb.Image("ga_routes.png")})
    run.finish()


# --- Function to visualize MIP routes ---
def plot_mip_routes(artifact_name='scsthilakarathne-nibm/Capacitated-Vehicle-Routing/MIP_CVRP_Solution:v0',
                    dataset_artifact='scsthilakarathne-nibm/Capacitated-Vehicle-Routing/CVRP_Dataset:v0'):
    run = wandb.init(project="Capacitated-Vehicle-Routing", job_type="plot_mip_routes")

    # Load MIP solution
    mip_artifact = run.use_artifact(artifact_name, type='model')
    mip_dir = mip_artifact.download()
    best_routes = np.load(f"{mip_dir}/mip_best_routes.npy", allow_pickle=True)

    # Load dataset coordinates
    coords = load_coords_from_dataset(dataset_artifact)

    plt.figure(figsize=(8,6))
    colors = plt.cm.get_cmap('tab20', len(best_routes))
    for idx, route in enumerate(best_routes):
        xs = [coords[i,1] for i in route]
        ys = [coords[i,2] for i in route]
        plt.plot(xs, ys, color=colors(idx), marker='o', label=f'Vehicle {idx+1}')
    plt.scatter(coords[0,1], coords[0,2], c='red', s=100, marker='s', label='Depot')
    plt.title("MIP CVRP Routes")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Log figure as artifact
    plt.savefig("mip_routes.png")
    run.log({"MIP Routes": wandb.Image("mip_routes.png")})
    run.finish()
