import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import numpy as np

def visualize_all_convergence(problems, results_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n==================VISUALIZING CONVERGENCE ABILITY=========================\n")

    for problem_name in problems:
        print(f"Visualizing {problem_name.title()} Function:")
        
        # 1. Đọc file JSON
        json_path = os.path.join(results_dir, f"{problem_name.lower()}_detailed.json")
        
        if not os.path.exists(json_path):
            print(f"File not found: {json_path}")
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Cannot read JSON: {e}")
            continue

        if not data:
            print("Empty data.")
            continue

        # 2. Vẽ biểu đồ
        try:
            plt.figure(figsize=(10, 6))

            for algo_name, content in data.items():

                curve = content.get('convergence_curves', content.get('convergence', []))
                
                if isinstance(curve, (int, float)):
                    curve = [curve] * 100
                elif isinstance(curve, list) and len(curve) == 0:
                    continue 

                plt.plot(curve, label=algo_name, linewidth=1.5)

            plt.title(f"Convergence Comparison - {problem_name.title()} Function")
            plt.xlabel("Iterations")
            plt.ylabel("Best Fitness (Log Scale)")
            plt.yscale("log") # Dùng Log scale để dễ nhìn sự khác biệt
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.legend()

            save_path = os.path.join(output_dir, f"{problem_name.lower()}_convergence.png")
            plt.savefig(save_path, dpi=300)
            plt.close() 
            
            print(f"{save_path}")

        except Exception as e:
            print(f"Plotting failed: {e}")
            import traceback
            traceback.print_exc()

    print("\nVisualization finished.")