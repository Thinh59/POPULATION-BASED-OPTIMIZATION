import os
import sys
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

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
    print("     5. Differential Evolution (DE) [NEW]")
    print("     6. Hybrid Method (GA + PSO)    [NEW]")
    print("\n   Problems: Sphere, Ackley, Rastrigin")
    print("="*70 + "\n")

def get_algorithm_params(use_yaml=True):
    # Cấu hình mặc định (Fallback nếu không load được YAML)
    default_params = {
        'GA': {
            'population_size': 50, 'generations': 100, 
            'crossover_rate': 0.8, 'mutation_rate': 0.1, 
            'elite_size': 2, 'tournament_size': 3
        },
        'FA': {
            'population_size': 50, 'generations': 100, 
            'alpha': 0.5, 'beta0': 1.0, 'gamma': 1.0
        },
        'CS': {
            'n_nests': 50, 'max_iter': 100, 
            'pa': 0.25, 'beta': 1.5, 'alpha': 0.01
        },
        'PSO': {
            'population_size': 50, 'generations': 100, 
            'w': 0.7, 'c1': 1.5, 'c2': 1.5
        },
        'DE': {
            'population_size': 50, 'generations': 100, 
            'w': 0.5, 'p': 0.9
        },
        'HYBRID': {
            'population_size': 50, 'generations': 100, 
            'w': 0.5, 'c1': 1.5, 'c2': 1.5,      # <-- Bổ sung
            'crossover_rate': 0.8, 'mutation_rate': 0.1 # <-- Bổ sung
        }
    }
    
    # Logic load YAML
    try:
        from src.utils.config_loader import load_algorithm_config
        if use_yaml:
            loaded = {}
            for alg in default_params.keys():
                conf = load_algorithm_config(alg)
                # Nếu load được config thì dùng, không thì dùng default
                loaded[alg] = conf if conf else default_params[alg]
            return loaded
    except Exception as e:
        # print(f"Config load error: {e}") # Uncomment để debug nếu cần
        pass
        
    return default_params

def visualize_2d_evolution(optimizer, problem, name):
    """
    Vẽ quá trình tiến hóa trên không gian 2D (Recreating Book Figures)
    Layout: 2x4 (8 snapshots)
    """
    try:
        history = optimizer.get_history()['population']
    except:
        print("\n [ERROR] Thuật toán này chưa hỗ trợ lưu 'population_history'.")
        return

    # Chỉ vẽ nếu là bài toán 2D
    if problem.dimensions != 2:
        print("\n [INFO] Bài toán không phải 2D. Đang chuyển về view 2D...")
    
    print(" >>> Rendering 2D Evolution Plot (Book Style 2x4)...")
    
    # 1. Tạo lưới điểm (Contour background)
    bounds = problem.get_bounds()[0] # Lấy bound của chiều đầu tiên
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Tính giá trị hàm tại từng điểm lưới
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = problem.evaluate(np.array([X[i,j], Y[i,j]]))

    # 2. Cấu hình vẽ hình lưới 2x4 = 8 hình
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes = axes.flatten()
    
    # Các mốc thời gian (Snapshots) giống hệt sách hoặc chia đều nếu số thế hệ khác
    total_gen = len(history)
    if total_gen >= 40:
        snapshots = [0, 1, 2, 4, 8, 15, 25, 39] # Chuẩn sách
    else:
        # Nếu chạy ít thế hệ hơn thì chia đều
        snapshots = np.linspace(0, total_gen-1, 8, dtype=int)
    
    for i, gen_idx in enumerate(snapshots):
        ax = axes[i]
        
        # Vẽ nền (Contour) - Dùng màu viridis_r đảo ngược giống sách (Tâm sáng)
        ax.contourf(X, Y, Z, levels=50, cmap='viridis_r', alpha=0.95)
        
        # Vẽ quần thể (Scatter)
        if gen_idx < total_gen:
            pop = history[gen_idx]
            # Vẽ điểm đen (s=25) giống code mẫu
            ax.scatter(pop[:, 0], pop[:, 1], c='black', s=25)
            
        ax.set_title(f"Gen {gen_idx}", fontweight='bold')
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[0], bounds[1])
        
        # Tắt trục số cho đẹp (giống sách)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f"{name} on {problem.get_name()} (Recreating Figure 9.10 Style)", fontsize=16, fontweight='bold')
    
    # Lưu file
    save_dir = 'visualizations/continuous/demo_book_style'
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/{name}_{problem.__class__.__name__}_2D_BookStyle.png"
    plt.savefig(filename, dpi=150)
    print(f"\n [SAVED] Image saved to: {filename}")
    plt.show()