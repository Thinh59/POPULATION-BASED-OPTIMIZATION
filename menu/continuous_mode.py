import sys
import os
import json
import numpy as np
import pandas as pd

# --- FIX LỖI POPUP WINDOW ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from menu.helper import clear_screen, print_header, get_algorithm_params

# Import Utils & Visualizers
from src.utils.compare_metrics_continuous import run_algorithm_multiple_times
from src.visualize.continuous.visualize_convergence import visualize_all_convergence
from src.visualize.continuous.visualize_swarm_classic_comparison import visualize_all_paradigm_comparison

# Import Problems
from src.problems.continuous.sphere_function import SphereFunction
from src.problems.continuous.ackley_function import AckleyFunction
from src.problems.continuous.rastrigin_function import RastriginFunction
# [MỚI] Thêm hàm Wheeler's Ridge
from src.problems.continuous.wheelers_ridge import WheelersRidge

# Import Optimizers
from src.optimizers.continuous.ga_optimizer import GeneticAlgorithm
from src.optimizers.continuous.fa_optimizer import FireflyAlgorithm
from src.optimizers.continuous.cs_optimizer import CuckooSearch
from src.optimizers.continuous.pso_optimizer import ParticleSwarmOptimization
from src.optimizers.continuous.de_optimizer import DifferentialEvolution
from src.optimizers.continuous.hybrid_optimizer import HybridGAPSO

# ===============================================================================
#                               HELPER FUNCTIONS
# ===============================================================================

def get_problem_instance(choice, dim=10):
    if choice == 'ackley': return AckleyFunction(dim)
    if choice == 'sphere': return SphereFunction(dim)
    if choice == 'rastrigin': return RastriginFunction(dim)
    # [MỚI] Thêm lựa chọn Wheeler
    if choice == 'wheeler': return WheelersRidge(dim) 
    return None

def get_optimizer_class(name):
    map_class = {
        'GA': GeneticAlgorithm, 'FA': FireflyAlgorithm, 'CS': CuckooSearch, 
        'PSO': ParticleSwarmOptimization, 'DE': DifferentialEvolution, 'Hybrid': HybridGAPSO
    }
    return map_class.get(name)

# --- VẼ HÌNH 2D (VISUAL DEMO) ---
def visualize_2d_evolution(optimizer, problem, name):
    try:
        history = optimizer.get_history()['population']
    except:
        return

    # 1. Tạo lưới điểm contour
    # Lấy bounds chuẩn
    bounds = problem.get_bounds()[0] 
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = problem.evaluate(np.array([X[i,j], Y[i,j]]))

    # 2. Vẽ 8 hình snapshots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes = axes.flatten()
    
    total_gen = len(history)
    if total_gen >= 40:
        snapshots = [0, 1, 2, 4, 8, 15, 25, 39]
    else:
        snapshots = np.linspace(0, total_gen-1, 8, dtype=int)
    
    for i, gen_idx in enumerate(snapshots):
        ax = axes[i]
        
        # [QUAN TRỌNG] Dùng contourf (filled) thay vì contour (lines) để hình mượt trở lại
        ax.contourf(X, Y, Z, levels=50, cmap='viridis_r', alpha=0.95)
        
        if gen_idx < total_gen:
            pop = history[gen_idx]
            # Vẽ chấm đen
            ax.scatter(pop[:, 0], pop[:, 1], c='black', s=25, edgecolors='white', linewidth=0.5)
            
        ax.set_title(f"Gen {gen_idx}", fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[0], bounds[1])

    plt.suptitle(f"{name} on {problem.get_name()} (Book Style Demo)", fontsize=16, fontweight='bold')
    
    save_dir = 'visualizations/continuous/demo_visual_2d'
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/{name}_{problem.__class__.__name__}_2D.png"
    plt.savefig(filename, dpi=100)
    plt.close('all')
    print(f"   [SAVED] 2D Visual saved to: {filename}")

# ===============================================================================
#                               CORE LOGIC
# ===============================================================================

def run_visual_demo_logic(algo_name, problem_key):
    dim = 2 
    problem = get_problem_instance(problem_key, dim)
    if not problem: 
        print("Problem not found!")
        return

    cfg = get_algorithm_params()[algo_name.upper()].copy()
    cfg['population_size'] = 20
    cfg['generations'] = 40
    
    # Riêng cho Wheeler's Ridge cần chỉnh Bounds lại trong config nếu cần thiết
    # Nhưng class WheelersRidge đã hardcode bounds chuẩn rồi.
    
    OptimizerClass = get_optimizer_class(algo_name)
    print(f"   >>> Visualizing {algo_name} on {problem.get_name()}...", end=" ", flush=True)
    
    try:
        if algo_name in ['DE', 'PSO', 'Hybrid', 'GA', 'FA']:
             opt = OptimizerClass(problem.evaluate, problem.get_bounds(), **cfg)
             is_new_style = True
        else:
             opt = OptimizerClass(**cfg)
             is_new_style = False
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
        print(f"Error: {e}")

def run_visual_demo_wizard():
    while True:
        clear_screen()
        print_header()
        print(" --- VISUAL DEMO MODE (2D ANIMATION STYLE) ---")
        
        # 1. Chọn Thuật toán (Đã bổ sung FA)
        print("\n [STEP 1] SELECT ALGORITHM:")
        print("  1. DE (Differential Evolution)")
        print("  2. PSO (Particle Swarm)")
        print("  3. GA    4. CS    5. Hybrid")
        print("  6. FA (Firefly Algorithm)") # <--- Đã thêm lại
        print("  0. Return")
        algo_choice = input("\n  Choose Algorithm (0-6): ").strip()
        
        if algo_choice == '0': return
        
        # Cập nhật map đầy đủ 6 thuật toán
        algo_map = {
            '1':'DE', '2':'PSO', 
            '3':'GA', '4':'CS', '5':'Hybrid',
            '6':'FA' 
        }
        selected_algo = algo_map.get(algo_choice)
        if not selected_algo: continue

        # 2. Chọn Bài toán
        print("\n [STEP 2] SELECT PROBLEM:")
        print("  1. Ackley Function (Hố sâu)")
        print("  2. Wheeler's Ridge (Thung lũng cong - GIỐNG SÁCH)")
        print("  3. Sphere Function (Hình cầu)")
        print("  4. Rastrigin Function")
        print("  0. Return")
        prob_choice = input("\n  Choose Problem (0-4): ").strip()
        
        if prob_choice == '0': continue
        
        prob_key = 'ackley'
        if prob_choice == '2': prob_key = 'wheeler'
        if prob_choice == '3': prob_key = 'sphere'
        if prob_choice == '4': prob_key = 'rastrigin'
            
        print("\n" + "="*50)
        print(f" >>> GENERATING 2D VISUALIZATIONS...")
        print("="*50 + "\n")
        
        run_visual_demo_logic(selected_algo, prob_key)
                    
        input("\n  [DONE] Check 'visualizations/continuous/demo_visual_2d/'. Press Enter...")

def run_benchmark_suite():
    print("\n" + "="*60)
    print(" >>> RUNNING FULL STATISTICAL BENCHMARK (30 RUNS)")
    print("="*60)
    
    # Chỉ benchmark trên 3 hàm chuẩn, không cần Wheeler cho thống kê
    problems = [SphereFunction(10), AckleyFunction(10), RastriginFunction(10)]
    configs = get_algorithm_params()
    algos = ['GA', 'FA', 'CS', 'PSO', 'DE', 'Hybrid']
    
    output_dir = 'results/continuous/performance'
    os.makedirs(output_dir, exist_ok=True)
    
    for prob in problems:
        print(f"\n [Problem: {prob.get_name()}]")
        detailed = {}
        metrics = []
        
        for alg in algos:
            print(f"   Running {alg}...", end=" ", flush=True)
            res = run_algorithm_multiple_times(alg, prob, n_runs=10, **configs.get(alg.upper(), {}))
            
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

# ===============================================================================
#                               MAIN MENU
# ===============================================================================

def continuous_menu():
    while True:
        clear_screen()
        print_header()
        print(" 1. Run Visual Demo (2D Swarm Behavior)")
        print("    (Ackley, Wheeler's Ridge...)")
        print(" 2. Run Benchmark Suite (Performance Graphs)")
        print("    (Sphere, Ackley, Rastrigin - 30D)")
        print(" 0. Exit")
        
        c = input("\n Choose: ").strip()
        
        if c == '1': run_visual_demo_wizard()
        elif c == '2': run_benchmark_suite()
        elif c == '0': sys.exit()