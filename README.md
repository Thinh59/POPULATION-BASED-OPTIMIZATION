# AI Fundamental Lab 03 – Population Methods

Implementation and Analysis of Population-based Optimization Algorithms (Chapter 9)  
Evolutionary Algorithms · Swarm Intelligence · Hybrid Optimization

---

## Overview
This project focuses on population-based optimization methods from Chapter 9 of
*Algorithms for Optimization*. It implements and compares evolutionary algorithms,
swarm intelligence algorithms, and a hybrid approach on continuous optimization problems.

Key goals:
- Compare convergence behavior and solution quality
- Visualize population evolution in 2D
- Perform statistical benchmarking over multiple runs

---

## Algorithms Implemented
- Genetic Algorithm (GA)
- Differential Evolution (DE)
- Particle Swarm Optimization (PSO)
- Firefly Algorithm (FA)
- Cuckoo Search (CS)
- Hybrid GA + PSO

---

## Benchmark Functions
- Ackley Function
- Sphere Function
- Rastrigin Function

(Default: 10 dimensions for benchmarking, 2 dimensions for visualization)

---

## Project Structure
Source/
├── configs/            # YAML configuration files
├── menu/               # Interactive menu and helpers
├── src/
│   ├── optimizers/     # Optimization algorithms
│   ├── problems/       # Benchmark problems
│   ├── visualize/      # Plotting and visualization
│   └── utils/          # Metrics and utilities
├── results/            # CSV and JSON result files
├── visualizations/     # Generated figures
└── main.py             # Program entry point

---

## Installation
Require Python 3.8+

pip install numpy pandas matplotlib pyyaml seaborn

---

## Usage
Run the program from the root directory:

python Source/main.py

Modes:
1. Demo Mode: visualize convergence and 2D population evolution
2. Benchmark Mode: run all algorithms with 30 runs and statistical analysis

All outputs are saved automatically to the visualizations and results folders.

---

## Outputs
- Convergence curves
- 2D evolution plots (book-style)
- Radar charts and bar charts
- CSV metrics and JSON convergence histories

---

## Authors
Bang My Linh – 23122009  
Lai Nguyen Hong Thanh – 23122018  
Phan Huynh Chau Thinh – 23122019  
Nguyen Trong Hoa – 23122029  

University of Science – VNUHCM  
AI Fundamentals Course
