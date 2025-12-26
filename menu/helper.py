import os
import sys

from src.utils.config_loader import load_algorithm_config
            

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("\n" + "="*70)
    print("   AI LAB 03 - POPULATION METHODS (CHAPTER 9)")
    print("="*70)
    print("   Algorithms:")
    print("     1. Genetic Algorithm (GA)")
    print("     2. Firefly Algorithm (FA)")
    print("     3. Cuckoo Search (CS)")
    print("     4. Particle Swarm Optimization (PSO)")
    print("     5. Differential Evolution (DE)")
    print("     6. Hybrid Method (GA + PSO)")
    print("\n   Problems: Sphere, Ackley, Rastrigin, Branin, Michalewicz...")
    print("="*70 + "\n")

def get_algorithm_params(use_yaml=True):
    default_params = {
        'GA': {'population_size': 50, 'generations': 100, 'crossover_rate': 0.8, 'mutation_rate': 0.1, 'elite_size': 2, 'tournament_size': 3},
        'FA': {'population_size': 50, 'generations': 100, 'alpha': 0.5, 'beta0': 1.0, 'gamma': 1.0},
        'CS': {'n_nests': 50, 'max_iter': 100, 'pa': 0.25, 'beta': 1.5, 'alpha': 0.01},
        'PSO': {'population_size': 50, 'generations': 100, 'w': 0.7, 'c1': 1.5, 'c2': 1.5},
        'DE': {'population_size': 50, 'generations': 100, 'w': 0.5, 'p': 0.9},
        'HYBRID': {'population_size': 50, 'generations': 100, 'w': 0.5, 'c1': 1.5, 'c2': 1.5, 'crossover_rate': 0.8, 'mutation_rate': 0.1}
    }
    
    if use_yaml:
        try:
            loaded_params = {}
            for alg in default_params.keys():
                yaml_conf = load_algorithm_config(alg)
                if yaml_conf:
                    loaded_params[alg] = yaml_conf
                else:
                    loaded_params[alg] = default_params[alg]
            return loaded_params
        except ImportError:
            pass
        except Exception as e:
            print(f"Error loading config: {e}")
            
    return default_params
