import numpy as np
import matplotlib.pyplot as plt
import re
import os
import wandb

from config import RAW_DATASET, DEMANDS_PATH, COORDS_PATH, DIST_MATRIX_PATH



def parse_vrp_file(filepath):
    """Parse CVRPLIB .vrp file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    name, capacity, coords, demands, depot = None, None, [], [], []
    section = None
    for line in lines:
        line = line.strip()
        if line.startswith("NAME"):
            name = line.split(":")[1].strip()
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1])
        elif "NODE_COORD_SECTION" in line:
            section = "coords"
            continue
        elif "DEMAND_SECTION" in line:
            section = "demands"
            continue
        elif "DEPOT_SECTION" in line:
            section = "depot"
            continue
        elif "EOF" in line:
            break

        # Parse data
        if section == "coords" and line:
            parts = re.split(r"\s+", line)
            if len(parts) >= 3:
                coords.append((int(parts[0]), float(parts[1]), float(parts[2])))
        elif section == "demands" and line:
            parts = re.split(r"\s+", line)
            if len(parts) >= 2:
                demands.append((int(parts[0]), int(parts[1])))
        elif section == "depot" and line and line != "-1":
            depot.append(int(line))

    coords = np.array(coords)
    demands = np.array(demands)
    depot = depot[0] if depot else 1
    return name, capacity, coords, demands, depot


def compute_distance_matrix(coords):
    n = coords.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.sqrt((coords[i,1]-coords[j,1])**2 + (coords[i,2]-coords[j,2])**2)
    return dist


def data_preprocessing():

    wandb.init(project='Capacitated-Vehicle-Routing', job_type="data_preprocessing", name= "data_preprocessing")

    name, capacity, coords, demands, depot = parse_vrp_file(RAW_DATASET)
    print(f"Dataset: {name}")
    print(f"Vehicle Capacity: {capacity}")
    print(f"Total customers: {len(demands)-1}")
    print(f"Depot node: {depot}")

    # Sort by node index
    coords = coords[np.argsort(coords[:,0])]
    demands = demands[np.argsort(demands[:,0])]

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(coords)

    # Basic stats
    print(f"Total demand: {demands[1:,1].sum()}")
    print(f"Max demand: {demands[1:,1].max()}")
    print(f"Average demand: {demands[1:,1].mean():.2f}")

    # # Plot coordinates
    # plt.figure(figsize=(8,6))
    # plt.scatter(coords[1:,1], coords[1:,2], c='blue', label='Customers')
    # plt.scatter(coords[0,1], coords[0,2], c='red', marker='s', s=100, label='Depot')
    # for i in range(1, len(coords)):
    #     plt.text(coords[i,1]+0.5, coords[i,2]+0.5, str(int(coords[i,0])), fontsize=8)
    # plt.title(f"{name} - Customer Distribution")
    # plt.xlabel("X Coordinate")
    # plt.ylabel("Y Coordinate")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # Save distance matrix for later use
    np.save(DIST_MATRIX_PATH, dist_matrix)
    np.save(DEMANDS_PATH, demands)
    np.save(COORDS_PATH, coords)

    artifact = wandb.Artifact(
        name=f"CVRP_Dataset",
        type="dataset",
        description="Preprocessed datasets"
    )

    # Add files
    artifact.add_file(RAW_DATASET)
    artifact.add_file(DIST_MATRIX_PATH)
    artifact.add_file(DEMANDS_PATH)
    artifact.add_file(COORDS_PATH)


    wandb.log_artifact(artifact)
    wandb.finish()
    

    print("\n✅ Data exploration completed and saved.")

