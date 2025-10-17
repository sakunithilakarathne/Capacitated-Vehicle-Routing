import numpy as np
import random
import wandb
import matplotlib.pyplot as plt
import re
import os
from tqdm import tqdm

from src.utility.utility_function import *

num_generations=200
pop_size=100
crossover_rate=0.9
mutation_rate=0.02

def selection(population, fitnesses):
    """Tournament selection"""
    selected = random.choices(population, weights=fitnesses, k=2)
    return selected[0], selected[1]


def crossover(parent1, parent2):
    """Ordered Crossover (OX)"""
    a, b = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[a:b] = parent1[a:b]

    fill_values = [x for x in parent2 if x not in child]
    pointer = 0
    for i in range(len(child)):
        if child[i] is None:
            child[i] = fill_values[pointer]
            pointer += 1
    return child


def mutate(chromosome, mutation_rate=0.02):
    """Swap mutation"""
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(chromosome) - 1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome



def run_genetic_algorithm():

    run = wandb.init(project='Capacitated-Vehicle-Routing', job_type='genetic_algorithm', name='GA_CVRP_Run')
    
    raw_artifact = run.use_artifact('scsthilakarathne-nibm/Capacitated-Vehicle-Routing/CVRP_Dataset:v0', type='dataset')
    raw_artifact_dir = raw_artifact.download()


    dist_matrix, demands, coords = load_preprocessed_data(raw_artifact_dir)
    vehicle_capacity = int(demands[:, 1].max() * 5)  # Approximation
    n_customers = len(demands) - 1
    customers = list(range(1, n_customers + 1))

    # Initialize population
    population = [random.sample(customers, len(customers)) for _ in range(pop_size)]

    best_fitness = 0
    best_solution = None
    fitness_progress = []

    for gen in tqdm(range(num_generations), desc="Running GA"):
        fitnesses = [fitness_function(ch, dist_matrix, demands, vehicle_capacity) for ch in population]
        new_population = []

        for _ in range(pop_size // 2):
            p1, p2 = selection(population, fitnesses)
            if random.random() < crossover_rate:
                c1 = crossover(p1, p2)
                c2 = crossover(p2, p1)
            else:
                c1, c2 = p1[:], p2[:]

            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)
            new_population.extend([c1, c2])

        population = new_population

        gen_best_fitness = max(fitnesses)
        fitness_progress.append(gen_best_fitness)

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_solution = population[np.argmax(fitnesses)]

        run.log({"generation": gen, "best_fitness": best_fitness})

    best_routes = split_routes(best_solution, demands, vehicle_capacity)
    best_distance = 1 / best_fitness

    print(f"\n Best routes:", best_routes)
    # Optionally, log routes to wandb
    run.log({"best_routes": best_routes})

    print(f"\n✅ Best total distance: {best_distance:.2f}")

    # Plot convergence
    plt.figure(figsize=(8, 5))
    plt.plot(fitness_progress)
    plt.title("GA Convergence - Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (1/Distance)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ga_convergence.png")
    wandb.log({"GA Convergence": wandb.Image("ga_convergence.png")})

    # Save results as artifact
    np.save("best_solution.npy", np.array(best_solution))
    np.save("fitness_progress.npy", np.array(fitness_progress))

    artifact = run.Artifact("GA_Model", type="model", description="Best GA solution for CVRP")
    artifact.add_file("best_solution.npy")
    artifact.add_file("fitness_progress.npy")
    artifact.add_file("ga_convergence.png")
    run.log_artifact(artifact)

    wandb.finish()

    return best_solution, best_distance, fitness_progress