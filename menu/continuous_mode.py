import sys
import os
import json
import numpy as np
import pandas as pd

from menu.helper import clear_screen, print_header, get_algorithm_params

from src.visualize.continuous.visualize_evolution import visualize_2d_evolution
from src.visualize.continuous.visualize_initialization import visualize_population_distribution
from src.visualize.continuous.visualize_convergence import visualize_all_convergence
from src.visualize.continuous.visualize_swarm_classic_comparison import visualize_all_paradigm_comparison
from src.utils.compare_metrics_continuous import run_algorithm_multiple_times

from src.problems.continuous.sphere_function import SphereFunction
from src.problems.continuous.ackley_function import AckleyFunction
from src.problems.continuous.rastrigin_function import RastriginFunction
from src.problems.continuous.wheelers_ridge import WheelersRidge
from src.problems.continuous.branin import BraninFunction
from src.problems.continuous.michalewicz import MichalewiczFunction

from src.optimizers.continuous.ga_optimizer import GeneticAlgorithm
from src.optimizers.continuous.fa_optimizer import FireflyAlgorithm
from src.optimizers.continuous.cs_optimizer import CuckooSearch
from src.optimizers.continuous.pso_optimizer import ParticleSwarmOptimization
from src.optimizers.continuous.de_optimizer import DifferentialEvolution
from src.optimizers.continuous.hybrid_optimizer import HybridGAPSO

def get_problem_instance(choice, dim=10):
    if choice == 'ackley': return AckleyFunction(dim)
    if choice == 'sphere': return SphereFunction(dim)
    if choice == 'rastrigin': return RastriginFunction(dim)
    if choice == 'wheeler': return WheelersRidge(dim)
    if choice == 'branin': return BraninFunction(dim=2)         # Luôn là 2D
    if choice == 'michalewicz': return MichalewiczFunction(dim)
    return None

def get_optimizer_class(name):
    map_class = {
        'GA': GeneticAlgorithm, 'FA': FireflyAlgorithm, 'CS': CuckooSearch, 
        'PSO': ParticleSwarmOptimization, 'DE': DifferentialEvolution, 'Hybrid': HybridGAPSO
    }
    return map_class.get(name)

def run_visual_demo_logic(algo_name, problem_key):
    dim = 2 
    problem = get_problem_instance(problem_key, dim)
    if not problem: 
        print("Problem not found!")
        return

    all_params = get_algorithm_params(use_yaml=True)
    cfg = all_params.get(algo_name.upper(), {}).copy()

    cfg['population_size'] = 25 
    cfg['generations'] = 50

    if algo_name == 'FA':
        cfg['alpha'] = 0.5
        cfg['beta0'] = 1.0
        cfg['gamma'] = 0.1
    
    OptimizerClass = get_optimizer_class(algo_name)
    print(f"   >>> Visualizing {algo_name} on {problem.get_name()}...", end=" ", flush=True)

    try:
        opt = OptimizerClass(problem.evaluate, problem.get_bounds(), **cfg)
        is_new_style = True
    except:
        opt = OptimizerClass(**cfg)
        is_new_style = False

    try:
        if is_new_style:
            opt.optimize(verbose=False)
        else:
            opt.optimize(problem.evaluate, dim, problem.get_bounds(), verbose=False)

        visualize_2d_evolution(opt, problem, algo_name)
        
    except Exception as e:
        print(f"Error details: {e}")

def run_visual_demo_wizard():
    while True:
        clear_screen()
        print_header()
        print(" --- VISUAL DEMO MODE (2D ANIMATION STYLE) ---")
        
        print("\n [STEP 1] SELECT ALGORITHM:")
        print("  1. DE (Differential Evolution)")
        print("  2. PSO (Particle Swarm)")
        print("  3. GA    4. CS    5. Hybrid")
        print("  6. FA (Firefly Algorithm)")
        print("  0. Return")
        algo_choice = input("\n  Choose Algorithm (0-6): ").strip()
        
        if algo_choice == '0': return
        
        algo_map = {'1':'DE', '2':'PSO', '3':'GA', '4':'CS', '5':'Hybrid', '6':'FA'}
        selected_algo = algo_map.get(algo_choice)
        if not selected_algo: continue

        print("\n [STEP 2] SELECT PROBLEM:")
        print("  1. Ackley Function")
        print("  2. Wheeler's Ridge")
        print("  3. Sphere Function")
        print("  4. Rastrigin Function")
        print("  5. Branin Function (Chuẩn cho CS/FA)")      
        print("  6. Michalewicz Function (Chuẩn cho GA)")    
        print("  0. Return")
        prob_choice = input("\n  Choose Problem (0-6): ").strip()
        
        if prob_choice == '0': continue
        
        prob_key = 'ackley' 
        if prob_choice == '2': prob_key = 'wheeler'
        if prob_choice == '3': prob_key = 'sphere'
        if prob_choice == '4': prob_key = 'rastrigin'
        if prob_choice == '5': prob_key = 'branin'
        if prob_choice == '6': prob_key = 'michalewicz'
            
        print("\n" + "="*50)
        print(f" >>> GENERATING 2D VISUALIZATIONS...")
        print("="*50 + "\n")
        
        run_visual_demo_logic(selected_algo, prob_key)
                    
        input("\n  [DONE] Check 'visualizations/continuous/demo_visual_2d/'. Press Enter...")

def run_benchmark_suite():
    print("\n" + "="*60)
    print(" >>> RUNNING FULL STATISTICAL BENCHMARK (30 RUNS)")
    print("="*60)
    
    problems = [SphereFunction(10), AckleyFunction(10), RastriginFunction(10)]

    configs = get_algorithm_params(use_yaml=True)
    
    algos = ['GA', 'FA', 'CS', 'PSO', 'DE', 'Hybrid']
    
    output_dir = 'results/continuous/performance'
    os.makedirs(output_dir, exist_ok=True)
    
    for prob in problems:
        print(f"\n [Problem: {prob.get_name()}]")
        detailed = {}
        metrics = []
        
        for alg in algos:
            print(f"   Running {alg}...", end=" ", flush=True)
            alg_cfg = configs.get(alg.upper(), {})
            
            res = run_algorithm_multiple_times(alg, prob, n_runs=10, **alg_cfg)
            
            if res and len(res['best_fitness']) > 0:
                metrics.append({
                    'Algorithm': alg,
                    'Best Fitness': np.mean(res['best_fitness']),
                    'Time (s)': np.mean(res['execution_times'])
                })
                
                best_idx = np.argmin(res['best_fitness'])
                raw_curve = res['convergence_curves'][best_idx]
                
                safe_curve = []
                if isinstance(raw_curve, (int, float, np.number)): safe_curve = [float(raw_curve)] * 100 
                elif len(raw_curve) == 0: safe_curve = [float(res['best_fitness'][best_idx])] * 100
                else:
                    if isinstance(raw_curve, np.ndarray): safe_curve = raw_curve.tolist()
                    else: safe_curve = list(raw_curve)

                detailed[alg] = {'convergence_curves': safe_curve}
                print("Done.")
            else:
                print("Failed.")
        
        if metrics:
            p_name = prob.__class__.__name__.lower().replace("function","")
            pd.DataFrame(metrics).to_csv(f'{output_dir}/{p_name}_metrics.csv', index=False)
            with open(f'{output_dir}/{p_name}_detailed.json', 'w') as f:
                try: json.dump(detailed, f)
                except: json.dump(detailed, f, default=str)

    print("\n >>> GENERATING PLOTS...")
    try:
        visualize_all_convergence(
            problems=['sphere', 'ackley', 'rastrigin'], 
            results_dir=output_dir, 
            output_dir='visualizations/continuous/convergence'
        )
        visualize_all_paradigm_comparison(
            problems=['sphere', 'ackley', 'rastrigin'], 
            results_dir=output_dir, 
            output_dir='visualizations/continuous/comparison'
        )
        print(" [SUCCESS] Check folder 'visualizations/continuous/'")
    except Exception as e:
        print(f" Error plotting: {e}")
    
    input("\n Press Enter to return...")

def continuous_menu():
    while True:
        clear_screen()
        print_header()
        print(" 1. Run Visual Demo (2D Swarm Behavior)")
        print("    (Ackley, Wheeler, Branin, Michalewicz...)")
        print(" 2. Run Benchmark Suite (Performance Graphs)")
        print("    (Sphere, Ackley, Rastrigin - 10D)")
        print(" 3. Generate Theory Plots (Population Distribution)")
        print("    (Uniform vs Normal vs Cauchy - Figure 9.1)")
        print(" 0. Exit")
        
        c = input("\n Choose: ").strip()
        
        if c == '1': run_visual_demo_wizard()
        elif c == '2': run_benchmark_suite()
        elif c == '3': 
            visualize_population_distribution()
        elif c == '0': sys.exit()