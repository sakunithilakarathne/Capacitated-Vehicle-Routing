import wandb
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utility.utility_function import *



def run_ga_with_config(config=None):

    with wandb.init(config=config, project='Capacitated-Vehicle-Routing', job_type='GA_Tuning') as run:
        config = wandb.config

        # Load dataset artifact
        raw_artifact = run.use_artifact('scsthilakarathne-nibm/Capacitated-Vehicle-Routing/CVRP_Dataset:v0', type='dataset')
        raw_artifact_dir = raw_artifact.download()

        dist_matrix, demands, coords = load_preprocessed_data(raw_artifact_dir)
        vehicle_capacity = int(demands[:, 1].max() * 5)
        n_customers = len(demands) - 1
        customers = list(range(1, n_customers + 1))

        # Initialize population
        population = [random.sample(customers, len(customers)) for _ in range(config.pop_size)]

        best_fitness = 0
        best_solution = None
        fitness_progress = []

        for gen in tqdm(range(config.num_generations), desc="Running GA"):
            fitnesses = [fitness_function(ch, dist_matrix, demands, vehicle_capacity) for ch in population]
            new_population = []

            for _ in range(config.pop_size // 2):
                p1, p2 = selection(population, fitnesses)
                if random.random() < config.crossover_rate:
                    c1 = crossover(p1, p2)
                    c2 = crossover(p2, p1)
                else:
                    c1, c2 = p1[:], p2[:]

                c1 = mutate(c1, config.mutation_rate)
                c2 = mutate(c2, config.mutation_rate)
                new_population.extend([c1, c2])

            population = new_population

            gen_best_fitness = max(fitnesses)
            fitness_progress.append(gen_best_fitness)

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_solution = population[np.argmax(fitnesses)]

            wandb.log({"generation": gen, "best_fitness": best_fitness})

        # Calculate final metrics
        best_routes = split_routes(best_solution, demands, vehicle_capacity)
        best_distance = 1 / best_fitness
        num_routes = len(best_routes)
        avg_customers_per_route = np.mean([len(r) for r in best_routes])

        # Log final results
        wandb.log({
            "final_best_fitness": best_fitness,
            "final_total_distance": best_distance,
            "num_routes": num_routes,
            "avg_customers_per_route": avg_customers_per_route
        })

        print({
            "final_best_fitness": best_fitness,
            "final_total_distance": best_distance,
            "num_routes": num_routes,
            "avg_customers_per_route": avg_customers_per_route
        })

        # Save fitness convergence plot
        plt.figure(figsize=(8, 5))
        plt.plot(fitness_progress)
        plt.title("GA Convergence - Fitness over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (1/Distance)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("ga_convergence.png")
        wandb.log({"GA Convergence": wandb.Image("ga_convergence.png")})

        # Save best solution
        np.save("best_solution.npy", np.array(best_solution))
        np.save("fitness_progress.npy", np.array(fitness_progress))

        # Upload as artifact
        artifact = wandb.Artifact(f"GA_Model_{wandb.run.id}", type="model", description="Tuned GA solution for CVRP")
        artifact.add_file("best_solution.npy")
        artifact.add_file("fitness_progress.npy")
        artifact.add_file("ga_convergence.png")
        wandb.log_artifact(artifact)


# =========================
# W&B Sweep Configuration
# =========================
sweep_config = {
    "method": "bayes",  # or "grid" / "random"
    "metric": {"name": "final_total_distance", "goal": "minimize"},
    "parameters": {
        "pop_size": {"values": [50, 100, 150]},
        "num_generations": {"values": [100, 200, 300]},
        "mutation_rate": {"values": [0.01, 0.05, 0.1]},
        "crossover_rate": {"values": [0.6, 0.8, 0.9]},
    },
}

# Create and run the sweep
sweep_id = wandb.sweep(sweep_config, project="Capacitated-Vehicle-Routing")
wandb.agent(sweep_id, function=run_ga_with_config, count=10)
