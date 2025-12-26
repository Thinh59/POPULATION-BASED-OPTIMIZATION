import sys
import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.problems.continuous.sphere_function import SphereFunction
from src.problems.continuous.ackley_function import AckleyFunction
from src.problems.continuous.rastrigin_function import RastriginFunction
from src.problems.continuous.wheelers_ridge import WheelersRidge
from src.problems.continuous.branin import BraninFunction
from src.problems.continuous.michalewicz import MichalewiczFunction

def visualize_2d_evolution(optimizer, problem, name):
    try:
        history = optimizer.get_history()['population']
    except:
        return
    bounds = problem.get_bounds()[0] 
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = problem.evaluate(np.array([X[i,j], Y[i,j]]))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes = axes.flatten()
    
    total_gen = len(history)
    if total_gen >= 40:
        snapshots = [0, 1, 2, 4, 8, 15, 25, 39]
    else:
        snapshots = np.linspace(0, total_gen-1, 8, dtype=int)
    
    for i, gen_idx in enumerate(snapshots):
        ax = axes[i]

        ax.contourf(X, Y, Z, levels=50, cmap='viridis_r', alpha=0.95)
        
        if gen_idx < total_gen:
            pop = history[gen_idx]
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