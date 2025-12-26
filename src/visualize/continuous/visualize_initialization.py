import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_population_distribution():
    print("\n >>> GENERATING POPULATION DISTRIBUTION COMPARISON...")
    N = 1000
    np.random.seed(42) 

    uniform_data = np.random.uniform(-2, 2, (N, 2))

    normal_data = np.random.normal(0, 1, (N, 2))
    cauchy_data = np.random.standard_cauchy((N, 2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    limit = 4 
    distributions = [
        ("Uniform", uniform_data),
        ("Normal", normal_data),
        ("Cauchy", cauchy_data)
    ]

    for ax, (name, data) in zip(axes, distributions):
        ax.scatter(data[:, 0], data[:, 1], s=15, color='#56B4E9', alpha=0.7, edgecolors='none')
        
        ax.set_title(name, fontsize=16, fontfamily='serif')
        ax.set_xlabel('$x_1$', fontsize=14)
        ax.set_ylabel('$x_2$', fontsize=14)
        
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')

        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        ax.tick_params(direction='in', top=True, right=True)

    save_dir = 'visualizations/theory'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/population_initialization_comparison.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [DONE] Theory Image saved to: {save_path}")
    input("\n Press Enter to return...")