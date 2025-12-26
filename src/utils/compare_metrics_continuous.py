import numpy as np
import pandas as pd
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimizers.continuous.ga_optimizer import GeneticAlgorithm
from optimizers.continuous.fa_optimizer import FireflyAlgorithm
from optimizers.continuous.cs_optimizer import CuckooSearch

from optimizers.continuous.pso_optimizer import ParticleSwarmOptimization
from optimizers.continuous.de_optimizer import DifferentialEvolution      
from optimizers.continuous.hybrid_optimizer import HybridGAPSO            

from problems.continuous.sphere_function import SphereFunction
from problems.continuous.ackley_function import AckleyFunction
from problems.continuous.rastrigin_function import RastriginFunction

def run_algorithm_multiple_times(algorithm_name, problem, n_runs=10, **alg_params):
    optimizers = {
        'ga': GeneticAlgorithm,
        'fa': FireflyAlgorithm,
        'cs': CuckooSearch,
        'pso': ParticleSwarmOptimization, 
        'de': DifferentialEvolution,
        'hybrid': HybridGAPSO
    }

    algo_class = optimizers.get(algorithm_name.lower())
    if not algo_class:
        print(f"Error: Unknown algorithm '{algorithm_name}'")
        return None

    results = {
        'best_fitness': [],
        'convergence_curves': [],
        'execution_times': []
    }
    
    bounds = problem.get_bounds()
    
    for run in range(n_runs):
        start_time = time.time()

        try:
            optimizer = algo_class(
                objective_function=problem.evaluate,
                bounds=bounds,
                random_seed=42 + run,
                **alg_params
            )
        except TypeError:
            optimizer = algo_class(
                seed=42 + run,
                **alg_params
            )

        try:
            _, best_fit = optimizer.optimize(verbose=False)

            try:
                hist = optimizer.get_history()['best_fitness']
            except:
                hist = []

        except Exception:
            res = optimizer.optimize(
                objective_func=problem.evaluate,
                dim=problem.dimensions,
                bounds=bounds[0],
                verbose=False
            )
            if isinstance(res, dict):
                best_fit = res.get('best_fitness', float('inf'))
                hist = res.get('history', [])
            elif isinstance(res, tuple):
                _, best_fit = res
                hist = []
            else:
                best_fit = float('inf')
                hist = []

        end_time = time.time()

        results['best_fitness'].append(best_fit)
        results['execution_times'].append(end_time - start_time)
        results['convergence_curves'].append(hist)
    
    return results

def compute_convergence_speed(convergence_curves, threshold=0.99):
    if not convergence_curves: return 0
    convergence_iterations = []
    for curve in convergence_curves:
        curve = np.array(curve)
        if len(curve) == 0: continue
        start_val = curve[0]
        end_val = curve[-1]
        if start_val == end_val: 
            convergence_iterations.append(0)
            continue
            
        target_val = start_val - threshold * (start_val - end_val)
        converged_at = len(curve)
        for i, val in enumerate(curve):
            if val <= target_val:
                converged_at = i
                break
        convergence_iterations.append(converged_at)
    return np.mean(convergence_iterations) if convergence_iterations else 0

def compute_robustness(fitness_values):
    std = np.std(fitness_values)
    if std == 0: return 1.0
    return 1.0 / (1.0 + std)

def benchmark_algorithms(algorithms_config, problem, n_runs=30, output_dir='results/continuous/performance'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    problem_name = problem.__class__.__name__.replace('Function', '')
    
    print(f"\n{'='*70}")
    print(f"BENCHMARKING ALGORITHMS ON {problem_name.upper()} FUNCTION")
    print(f"Runs per algorithm: {n_runs}")
    print(f"{'='*70}")
    
    results_rows = []
    detailed_results = {}
    
    for alg_name, alg_params in algorithms_config.items():
        print(f"Processing {alg_name}...", end=" ", flush=True)
        
        results = run_algorithm_multiple_times(
            alg_name, problem, n_runs=n_runs, **alg_params
        )
        
        if not results:
            print("Failed.")
            continue

        best_fitness = np.min(results['best_fitness'])
        mean_fitness = np.mean(results['best_fitness'])
        std_fitness = np.std(results['best_fitness'])
        mean_time = np.mean(results['execution_times'])
        std_time = np.std(results['execution_times'])
        conv_speed = compute_convergence_speed(results['convergence_curves'])
        robustness = compute_robustness(results['best_fitness'])

        best_run_idx = np.argmin(results['best_fitness'])
        
        detailed_results[alg_name] = {
            'best_fitness_values': results['best_fitness'],
            'convergence': results['convergence_curves'][best_run_idx], # Quan trọng cho visualize
            'execution_times': results['execution_times']
        }
        
        results_rows.append({
            'Algorithm': alg_name,
            'Best Fitness': best_fitness,
            'Mean Fitness': mean_fitness,
            'Std Fitness': std_fitness,
            'Time (s)': mean_time,
            'Robustness': robustness,
            'Convergence Speed': conv_speed
        })
        
        print(f"Done. (Mean Fit: {mean_fitness:.6f})")

    df = pd.DataFrame(results_rows)
    csv_file = output_path / f'{problem_name.lower()}_metrics.csv'
    df.to_csv(csv_file, index=False)
    print(f" Saved metrics to {csv_file}")

    json_file = output_path / f'{problem_name.lower()}_detailed.json'
    with open(json_file, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    return df, detailed_results

def benchmark_scalability(algorithms_config, problem_class, dimensions_list=[5, 10, 20], n_runs=5, output_dir='results/continuous/performance'):  
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    problem_name = problem_class.__name__.replace('Function', '')
    
    print(f"\n>>> SCALABILITY ANALYSIS ({problem_name})")
    
    scalability_data = {alg: {'dimensions': [], 'times': []} for alg in algorithms_config}
    
    for dim in dimensions_list:
        print(f" Dimension: {dim}")
        problem = problem_class(dimensions=dim)
        
        for alg_name, alg_params in algorithms_config.items():
            res = run_algorithm_multiple_times(alg_name, problem, n_runs=n_runs, **alg_params)
            if res:
                avg_time = np.mean(res['execution_times'])
                scalability_data[alg_name]['dimensions'].append(dim)
                scalability_data[alg_name]['times'].append(avg_time)

    results_rows = []
    for alg, data in scalability_data.items():
        if len(data['times']) > 1:
            slope = np.polyfit(data['dimensions'], data['times'], 1)[0]
            results_rows.append({'Algorithm': alg, 'Scalability Coefficient': slope})
            
    df = pd.DataFrame(results_rows)
    df.to_csv(output_path / f'{problem_name.lower()}_scalability.csv', index=False)
    
    with open(output_path / f'{problem_name.lower()}_scalability_data.json', 'w') as f:
        json.dump(scalability_data, f, indent=2)
        
    print(f" Saved scalability analysis.")
    return df