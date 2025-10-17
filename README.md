# Capacitated Vehicle Routing Problem (CVRP) – GA & MIP Approaches

This project implements and compares **Genetic Algorithm (GA)** and **Mixed-Integer Programming (MIP)** approaches to solve the Capacitated Vehicle Routing Problem (CVRP). The project uses the **A-n32-k5** instance from **CVRPLIB** as a benchmark and includes data preprocessing, model implementations, visualization, and experiment tracking using **Weights & Biases (W&B)**.

---

## Table of Contents

- [Problem Description](#problem-description)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)

---

## Problem Description

The **Capacitated Vehicle Routing Problem (CVRP)** is defined as:

- **Objective:** Minimize the total travel distance of a fleet of vehicles serving all customers.
- **Decision Variables:** Route assignments for each vehicle.
- **Constraints:**
  - Each customer is visited exactly once.
  - Vehicle capacities must not be exceeded.
  - All routes start and end at the depot.

The project compares **heuristic (GA)** and **exact (MIP)** solutions in terms of solution quality, route feasibility, and computational performance.

---

## Dataset

- Source: [CVRPLIB](http://vrp.atd-lab.inf.puc-rio.br/index.php/en/)
- Instance: **A-n32-k5**
- Preprocessing:
  - Distance matrix computation
  - Sorting by node index
  - Demand extraction
  - Saved as W&B artifact (`CVRP_Dataset:v0`)

---

## Project Structure

CVRP-Project/
│
├─ data_preprocessing.py # Parsing and preprocessing CVRPLIB files
├─ ga_cvrp.py # Genetic Algorithm implementation
├─ mip_cvrp.py # MIP implementation with PuLP
├─ utils.py # Utility functions (fitness, split_routes, etc.)
├─ visualization.py # Route visualization for GA and MIP
├─ README.md # Project documentation
└─ notebooks/ # Optional Colab notebooks for experiments


---

## Environment Setup

Use **conda** to set up the environment:

```bash
conda create -n cvrp python=3.10 -y
conda activate cvrp

# Required packages
pip install numpy matplotlib pulp wandb tqdm
