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
├── main.py                 # Program entry point
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
│
├── configs/                # YAML configuration files
│   ├── algorithms/         # Algorithm parameter settings (CS, PSO, GA, DE, ...)
│   └── problems/           # Benchmark problem definitions
│
├── menu/                   # Execution modes and helper functions
│   ├── continuous_mode.py
│   └── helper.py
│
├── src/                    # Core source code
│   ├── optimizers/
│   │   └── continuous/     # Continuous optimization algorithms
│   │
│   ├── problems/
│   │   └── continuous/     # Continuous benchmark functions
│   │
│   ├── utils/              # Configuration loading and metric comparison
│   │
│   └── visualize/
│       └── continuous/     # Visualization scripts
│
├── results/                # Experimental results
│   └── continuous/
│       └── performance/    # CSV and JSON performance metrics
│
└── visualizations/         # Generated figures for analysis and reporting
    ├── continuous/
    └── theory/

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
