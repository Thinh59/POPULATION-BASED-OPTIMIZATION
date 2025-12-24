import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

# Cấu hình màu sắc riêng biệt cho 6 thuật toán để dễ phân biệt
ALGORITHM_COLORS = {
    'GA': '#1f77b4',      # Xanh dương đậm
    'DE': '#9467bd',      # Tím
    'PSO': '#d62728',     # Đỏ
    'FA': '#ff7f0e',      # Cam
    'CS': '#2ca02c',      # Xanh lá
    'Hybrid': '#8c564b'   # Nâu
}

def load_metrics(problem_name, results_dir):
    csv_file = Path(results_dir) / f'{problem_name.lower()}_metrics.csv'
    if not csv_file.exists(): return None
    return pd.read_csv(csv_file)

def visualize_all_paradigm_comparison(problems, results_dir, output_dir):
    """
    Vẽ biểu đồ so sánh trực tiếp tất cả các thuật toán (Không chia nhóm)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    for problem_name in problems:
        df = load_metrics(problem_name, results_dir)
        if df is not None:
            df['Problem'] = problem_name.title()
            all_data.append(df)
    
    if not all_data: 
        print("  [WARN] No data found to visualize comparison.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)

    # Cấu hình Style
    sns.set_style("whitegrid")
    
    # --- VẼ BIỂU ĐỒ CỘT (BAR CHART) ---
    # So sánh các chỉ số quan trọng: Best Fitness, Time, Std Fitness
    metrics_to_plot = [col for col in combined_df.columns if col not in ['Algorithm', 'Problem']]
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        
        # Vẽ Barplot: X là Bài toán, Y là Chỉ số, Màu theo Thuật toán
        ax = sns.barplot(
            data=combined_df, 
            x='Problem', 
            y=metric, 
            hue='Algorithm', 
            palette=ALGORITHM_COLORS
        )
        
        plt.title(f'Comparison of {metric} (All Algorithms)', fontsize=14, fontweight='bold')
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Nếu là Fitness thì dùng Log Scale cho dễ nhìn (vì chênh lệch thường rất lớn)
        if 'Fitness' in metric:
            plt.yscale('log')
            plt.ylabel(f'{metric} (Log Scale)')
        
        plt.tight_layout()
        
        # Lưu file
        safe_metric_name = metric.replace(" ", "_").replace("(", "").replace(")", "").lower()
        save_file = output_path / f'compare_all_{safe_metric_name}.png'
        plt.savefig(save_file, dpi=300)
        plt.close('all')

    print(f"  [SAVED] All comparison charts saved to: {output_dir}")