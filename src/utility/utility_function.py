import numpy as np
import random
import wandb
import matplotlib.pyplot as plt
import re
import os
from tqdm import tqdm


def load_preprocessed_data(artifact_dir):
    """Load preprocessed CVRP data from artifact directory"""
    dist_matrix = np.load(f"{artifact_dir}/dist_matrix.npy")
    demands = np.load(f"{artifact_dir}/demands.npy")
    coords = np.load(f"{artifact_dir}/coords.npy")
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



def selection(population, fitnesses):
    """
    Tournament selection: randomly pick two and return the fitter one.
    """
    i, j = random.sample(range(len(population)), 2)
    return population[i] if fitnesses[i] > fitnesses[j] else population[j], \
           population[j] if fitnesses[i] > fitnesses[j] else population[i]


def crossover(parent1, parent2):
    """
    Ordered crossover (OX): preserves order and position of many genes.
    """
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]

    # fill remaining positions with order from parent2
    ptr = 0
    for gene in parent2:
        if gene not in child:
            while child[ptr] is not None:
                ptr += 1
            child[ptr] = gene
    return child


def mutate(chromosome, mutation_rate):
    """
    Swap mutation: randomly swaps two positions with given mutation rate.
    """
    chrom = chromosome[:]
    for i in range(len(chrom)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(chrom) - 1)
            chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom
