#!/usr/bin/env python3
"""
Plot performance delta vs training steps for each model and metric.

Creates a faceted plot with:
- x-axis: training step
- y-axis: delta from base model (percentage points)
- columns: metrics (GSM8K, MATH, IFEval, MBPP)
- hue/color: model configuration
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Base model scores (Qwen3-1.7B-Base)
BASE_SCORES = {
    'gsm8k_4shot_flex': 0.6748,
    'math_qwen': 0.387,
    'ifeval_prompt': 0.22,
    'mbpp': 0.558,
}

# Metric display names
METRIC_NAMES = {
    'gsm8k_4shot_flex': 'GSM8K',
    'math_qwen': 'MATH',
    'ifeval_prompt': 'IFEval',
    'mbpp': 'MBPP',
}

def main():
    # Load data
    csv_path = Path("/efs/rlvr-experiments/experiments/all_results.csv")
    df = pd.read_csv(csv_path)

    # Create model identifier from checkpoint name (remove _stepN suffix)
    df['model'] = df['checkpoint'].str.replace(r'_step\d+$', '', regex=True)

    # Calculate deltas for each metric
    metrics = ['gsm8k_4shot_flex', 'math_qwen', 'ifeval_prompt', 'mbpp']

    # Melt the dataframe to long format for seaborn
    records = []
    for _, row in df.iterrows():
        for metric in metrics:
            if pd.notna(row.get(metric)):
                delta = (row[metric] - BASE_SCORES[metric]) * 100  # Convert to percentage points
                records.append({
                    'step': row['step'],
                    'model': row['model'],
                    'metric': METRIC_NAMES[metric],
                    'delta': delta,
                    'lr': row['lr'],
                    'order': row['order'],
                    'staleness': row['staleness'],
                })

    plot_df = pd.DataFrame(records)

    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)

    # Color palette for models
    models = plot_df['model'].unique()
    palette = sns.color_palette('husl', n_colors=len(models))
    color_map = dict(zip(models, palette))

    # Plot each metric
    for ax, metric_name in zip(axes, ['GSM8K', 'MATH', 'IFEval', 'MBPP']):
        metric_df = plot_df[plot_df['metric'] == metric_name]

        for model in models:
            model_df = metric_df[metric_df['model'] == model].sort_values('step')
            if len(model_df) > 0:
                ax.plot(model_df['step'], model_df['delta'],
                       marker='o', markersize=4, linewidth=1.5,
                       color=color_map[model], label=model, alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Delta (pp)' if ax == axes[0] else '')
        ax.set_title(metric_name)
        ax.set_xlim(50, 650)

    # Create legend outside the plot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
               title='Model', fontsize=8, title_fontsize=9)

    plt.suptitle('Performance Delta vs Base Model by Training Step', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save the plot
    output_path = Path("/efs/rlvr-experiments/experiments/delta_by_step.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}")

    # Also create a simplified version with fewer models (top performers only)
    top_models = [
        'Hadadv_Mixed_lr5e6',
        'Hadadv_Seq_lr5e6',
        'Mixed_lr1e6',
        'Seq_lr1e6',
        'Random_lr5e6',
        'Random_lr5e6_s1',
    ]

    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    palette2 = sns.color_palette('tab10', n_colors=len(top_models))
    color_map2 = dict(zip(top_models, palette2))

    for ax, metric_name in zip(axes2, ['GSM8K', 'MATH', 'IFEval', 'MBPP']):
        metric_df = plot_df[plot_df['metric'] == metric_name]

        for model in top_models:
            model_df = metric_df[metric_df['model'] == model].sort_values('step')
            if len(model_df) > 0:
                ax.plot(model_df['step'], model_df['delta'],
                       marker='o', markersize=5, linewidth=2,
                       color=color_map2[model], label=model, alpha=0.9)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Delta (pp)' if ax == axes2[0] else '')
        ax.set_title(metric_name)
        ax.set_xlim(50, 650)

    handles2, labels2 = axes2[0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc='center left', bbox_to_anchor=(1.0, 0.5),
                title='Model', fontsize=9, title_fontsize=10)

    plt.suptitle('Performance Delta vs Base Model (Main Configurations)', fontsize=14, y=1.02)
    plt.tight_layout()

    output_path2 = Path("/efs/rlvr-experiments/experiments/delta_by_step_main.png")
    plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved simplified plot to: {output_path2}")


if __name__ == "__main__":
    main()
