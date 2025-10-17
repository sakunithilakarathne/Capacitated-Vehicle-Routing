import numpy as np
import random
import wandb
import matplotlib.pyplot as plt
import re
import os
from tqdm import tqdm


def load_preprocessed_data(artifact_dir):
    """Load preprocessed CVRP data from artifact directory"""
    dist_matrix = np.load(os.path.join(artifact_dir, 'distance_matrix.npy'))
    demands = np.load(os.path.join(artifact_dir, 'demands.npy'))
    coords = np.load(os.path.join(artifact_dir, 'coords.npy'))
    return dist_matrix, demands, coords


def calculate_route_cost(route, dist_matrix):
    """Compute total distance of a single route"""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += dist_matrix[route[i], route[i+1]]
    return total_distance


def split_routes(chromosome, demands, vehicle_capacity):
    """Split chromosome into feasible vehicle routes based on capacity"""
    routes = []
    current_route = [0]  # start from depot
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


def fitness_function(chromosome, dist_matrix, demands, vehicle_capacity):
    """Fitness = inverse of total distance of all routes"""
    routes = split_routes(chromosome, demands, vehicle_capacity)
    total_cost = sum(calculate_route_cost(route, dist_matrix) for route in routes)
    return 1 / total_cost  # minimize distance