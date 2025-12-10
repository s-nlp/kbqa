"""
Plot Hit@1 results for different reranking approaches across multiple candidate sources.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure matplotlib for publication quality
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 10
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 8
rcParams['figure.titlesize'] = 11

# File paths
files = {
    'T5-Large-SSM': 'tabs/t5largessm_all_results_hit_at_n.tex',
    'T5-XL-SSM': 'tabs/t5xlssm_all_results_hit_at_n.tex',
    'Mixtral': 'tabs/mixtral_all_results_hit_at_n.tex',
    'Mistral': 'tabs/mistral_all_results_hit_at_n.tex'
}

def parse_value(value_str):
    """Parse values like '0.2213' or '$0.2442\\pm0.0004$'"""
    value_str = value_str.strip().replace('$', '')

    if '\\pm' in value_str:
        parts = value_str.split('\\pm')
        mean = float(parts[0])
        error = float(parts[1])
        return mean, error
    else:
        try:
            return float(value_str), None
        except:
            return None, None

def parse_latex_table(filename):
    """Parse LaTeX table and extract Hit@1 values with errors."""
    with open(filename, 'r') as f:
        content = f.read()

    results = {}
    lines = content.split('\n')
    current_model = None

    for line in lines:
        # Skip control lines
        if any(x in line for x in ['\\toprule', '\\midrule', '\\bottomrule', '\\cmidrule', 
                                     'textbf{Reranking Model}', '\\caption', '\\label',
                                     '\\begin{', '\\end{', '\\setlength', '\\fontsize']):
            continue

        # Check if it's a data line
        if '&' in line and '\\\\' in line:
            parts = [p.strip() for p in line.split('&')]

            if len(parts) >= 3:
                model = parts[0]
                features = parts[1] if len(parts) > 1 else ''
                hit1 = parts[2] if len(parts) > 2 else ''

                # Handle multirow
                if model and '\\multirow' not in model:
                    if model != '':
                        current_model = model

                if '\\multirow' in model:
                    match = re.search(r'\\multirow\{[^}]+\}\{[^}]+\}\{([^}]+)\}', model)
                    if match:
                        current_model = match.group(1)
                        model = ''

                if model == '' and current_model:
                    model = current_model

                # Parse Hit@1 value
                mean, error = parse_value(hit1)

                if mean is not None:
                    if features:
                        # Normalize "Det.Lin." to "Det. Lin." for consistency
                        features = features.replace('Det.Lin.', 'Det. Lin.')
                        key = f"{model} - {features}"
                    else:
                        key = model

                    results[key] = {'mean': mean, 'error': error}

    return results

# Parse all data
all_data = {}
for model_name, filename in files.items():
    all_data[model_name] = parse_latex_table(filename)

# Define comprehensive method selection - prioritize most important ones
# Note: Some methods may not exist for all models (will show as missing/0)
selected_approaches = [
    'Without reranking',
    'Majority Vote',
    # 'Semantic reranking - Text',
    'RankGPT (Mistral) - Text',
    'RankGPT (Mistral) - Text + G2T (Det. Lin.)',
    'RankGPT (Qwen2.5 7B) - Text',
    'RankGPT (Qwen2.5 7B) - Text + G2T (Det. Lin.)',
    'Linear Regression - Text',
    'Linear Regression - Text + Graph',
    'Linear Regression - Text + Graph + G2T (Det. Lin.)',
    'Linear Regression - Text + Graph + G2T (T5)',
    'Linear Regression - Text + Graph + G2T (GAP)',
    'Linear Regression - Text + G2T (Det. Lin.)',
    'Linear Regression - Text + G2T (T5)',
    'Linear Regression - Text + G2T (GAP)',
    'Logistic Regression - Text',
    'Logistic Regression - Text + Graph',
    'Logistic Regression - Text + Graph + G2T (Det. Lin.)',
    'Logistic Regression - Text + Graph + G2T (T5)',
    'Logistic Regression - Text + Graph + G2T (GAP)',
    'Logistic Regression - Text + G2T (Det. Lin.)',
    'Logistic Regression - Text + G2T (T5)',
    'Logistic Regression - Text + G2T (GAP)',
    'CatBoost - Text',
    'CatBoost - Text + Graph',
    'CatBoost - Text + Graph + G2T (Det. Lin.)',
    'CatBoost - Text + Graph + G2T (T5)',
    'CatBoost - Text + Graph + G2T (GAP)',
    'CatBoost - Text + G2T (Det. Lin.)',
    'CatBoost - Text + G2T (T5)',
    'CatBoost - Text + G2T (GAP)',
    'MPNet - Text',
    'MPNet - Text + G2T (Det. Lin.)',
    'MPNet - Text + G2T (T5)',
    'MPNet - Text + G2T (GAP)',
]

def shorten_method_name(name):
    """Shorten method names for better display."""
    replacements = {
        'Linear Regression': 'Lin. Reg.',
        'Logistic Regression': 'Log. Reg.',
        'RankGPT (Mistral)': 'RankGPT (Mistral)',
        'RankGPT (Qwen2.5 7B)': 'RankGPT (Qwen)',
    }
    result = name
    for full, short in replacements.items():
        result = result.replace(full, short)
    return result

def get_method_group(method_name):
    """Determine which group a method belongs to for spacing."""
    if method_name in ['Without reranking', 'Majority Vote', 'Semantic reranking - Text']:
        return 'baseline'
    elif method_name.startswith('RankGPT'):
        return 'rankgpt'
    elif method_name.startswith('Linear Regression'):
        return 'linear'
    elif method_name.startswith('Logistic Regression'):
        return 'logistic'
    elif method_name.startswith('CatBoost'):
        return 'catboost'
    elif method_name.startswith('MPNet'):
        return 'mpnet'
    else:
        return 'other'

# High contrast earthy colors palette
colors = {
    'Without reranking': '#6B5344',
    'Majority Vote': '#E74C3C',
    'Semantic reranking - Text': '#A0522D',
    'RankGPT (Mistral) - Text': '#8B4513',
    'RankGPT (Mistral) - Text + G2T (Det. Lin.)': '#7A3513',
    'RankGPT (Qwen2.5 7B) - Text': '#9B5523',
    'RankGPT (Qwen2.5 7B) - Text + G2T (Det. Lin.)': '#8A4523',
    'Linear Regression - Text': '#BC8F8F',
    'Linear Regression - Text + Graph': '#A0826D',
    'Linear Regression - Text + Graph + G2T (Det. Lin.)': '#8B7355',
    'Linear Regression - Text + Graph + G2T (T5)': '#7A6345',
    'Linear Regression - Text + Graph + G2T (GAP)': '#6B5535',
    'Linear Regression - Text + G2T (Det. Lin.)': '#9A8365',
    'Linear Regression - Text + G2T (T5)': '#897355',
    'Linear Regression - Text + G2T (GAP)': '#786345',
    'Logistic Regression - Text': '#DEB887',
    'Logistic Regression - Text + Graph': '#D2B48C',
    'Logistic Regression - Text + Graph + G2T (Det. Lin.)': '#C8A882',
    'Logistic Regression - Text + Graph + G2T (T5)': '#BE9C78',
    'Logistic Regression - Text + Graph + G2T (GAP)': '#B4906E',
    'Logistic Regression - Text + G2T (Det. Lin.)': '#D8B892',
    'Logistic Regression - Text + G2T (T5)': '#CEA882',
    'Logistic Regression - Text + G2T (GAP)': '#C49872',
    'CatBoost - Text': '#DAA520',
    'CatBoost - Text + Graph': '#B8860B',
    'CatBoost - Text + Graph + G2T (Det. Lin.)': '#9B7708',
    'CatBoost - Text + Graph + G2T (T5)': '#8B6806',
    'CatBoost - Text + Graph + G2T (GAP)': '#7B5905',
    'CatBoost - Text + G2T (Det. Lin.)': '#856A07',
    'CatBoost - Text + G2T (T5)': '#755A06',
    'CatBoost - Text + G2T (GAP)': '#654A05',
    'MPNet - Text': '#CD853F',
    'MPNet - Text + G2T (Det. Lin.)': '#A0522D',
    'MPNet - Text + G2T (T5)': '#8F451D',
    'MPNet - Text + G2T (GAP)': '#7E380D',
}

def plot_results(files_dict, selected_approaches_list, colors_dict, y_axis_limits_dict, 
                 title, output_prefix):
    """Plot Hit@1 results for given files and methods."""
    # Parse all data
    all_data = {}
    for model_name, filename in files_dict.items():
        all_data[model_name] = parse_latex_table(filename)
    
    # Extract data for plotting
    models = list(files_dict.keys())
    
    # Prepare data arrays for each model
    data_by_model = {}
    for j, model in enumerate(models):
        means = []
        errors = []
        labels = []
        colors_list = []
        
        for approach in selected_approaches_list:
            if approach in all_data[model]:
                means.append(all_data[model][approach]['mean'])
                err = all_data[model][approach]['error']
                errors.append(err if err is not None else 0.0)
                labels.append(shorten_method_name(approach))
                colors_list.append(colors_dict.get(approach, '#808080'))
        
        data_by_model[model] = {
            'means': np.array(means), 
            'errors': np.array(errors),
            'labels': labels,
            'colors': colors_list
        }
    
    # Create subplot figure - 1 column, n_models rows
    n_models = len(models)
    n_cols = 1
    n_rows = n_models
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows))
    if n_models == 1:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each model in a separate subplot
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        means = data_by_model[model]['means']
        errors = data_by_model[model]['errors']
        labels = data_by_model[model]['labels']
        colors_list = data_by_model[model]['colors']
        
        n_methods = len(means)
        
        # Group methods and calculate positions with spacing
        # First, collect which approaches are present and their groups
        present_approaches = []
        for approach in selected_approaches_list:
            if approach in all_data[model]:
                present_approaches.append(approach)
        
        # Group consecutive methods of the same category
        groups = []
        current_group = None
        group_start = 0
        
        for i, approach in enumerate(present_approaches):
            group = get_method_group(approach)
            if current_group is None:
                current_group = group
            elif group != current_group:
                groups.append((current_group, group_start, i - group_start))
                current_group = group
                group_start = i
        if current_group is not None:
            groups.append((current_group, group_start, len(present_approaches) - group_start))
        
        # Calculate x positions with spacing between groups
        x = []
        group_spacing = 0.8
        bar_spacing_within_group = 0.85  # 50% less than 1.0
        current_pos = 0
        
        for group_name, start_idx, group_size in groups:
            for i in range(group_size):
                x.append(current_pos)
                current_pos += bar_spacing_within_group
            current_pos += group_spacing
        
        x = np.array(x)
        width = 0.65
        
        # Plot bars
        bars = ax.bar(x, means, width, 
                       color=colors_list,
                       edgecolor='black',
                       linewidth=0.8,
                       alpha=0.9)
        
        # Add error bars
        ax.errorbar(x, means, yerr=errors,
                    fmt='none',
                    ecolor='black',
                    elinewidth=1.2,
                    capsize=3.5,
                    capthick=1.2,
                    alpha=0.7,
                    zorder=2)
        
        # Add labels below bars at 35 degree angle
        ax.set_xticks(x)
        ax.set_xticklabels(labels,
                           rotation=35, ha='right', fontsize=8)
        
        # Focused Y-axis scaling for each model
        if len(means) > 0:
            if model in y_axis_limits_dict and y_axis_limits_dict[model] is not None:
                y_min, y_max = y_axis_limits_dict[model]
            else:
                data_range = means.max() - means.min()
                y_min = means.min() - data_range * 0.05
                y_max = means.max() + data_range * 0.05
                y_min = max(0, y_min)
            ax.set_ylim(y_min, y_max)
        
        # Customize subplot
        ax.set_ylabel('Hit@1 Score', fontweight='bold', fontsize=9)
        ax.set_title(f'{model}', fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.7, color='gray')
        ax.set_axisbelow(True)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    # fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 1])
    
    # Save figure
    plt.savefig(f'{output_prefix}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}.png', format='png', dpi=300, bbox_inches='tight')
    print(f"✓ Figures saved: {output_prefix}.pdf and {output_prefix}.png")
    
    # Report what was plotted
    for model in models:
        print(f"  {model}: {len(data_by_model[model]['means'])} methods")
    
    return data_by_model

# Manual Y-axis limits for focused scaling (optional, None for auto)
y_axis_limits = {
    'T5-Large-SSM': None,
    'T5-XL-SSM': None,
    'Mixtral': None,
    'Mistral': None,
}

# Plot Mintaka dataset
data_by_model = plot_results(
    files, 
    selected_approaches, 
    colors, 
    y_axis_limits,
    'Hit@1 Performance: Reranking Approaches Across Candidate Sources on Mintaka Dataset',
    'hit_at_1_comparison'
)

for model in list(files.keys()):
    means = data_by_model[model]['means']
    errors = data_by_model[model]['errors']
    labels = data_by_model[model]['labels']
    print(f"Statistics for model: {model}")
    for method, mean, std in zip(labels, means, errors):
        print(f"  {method}: mean={mean:.4f}, std={std:.4f}")

# MKQA dataset files
mkqa_files = {
    'T5-Large-SSM': 'tabs/mkqa_t5largessm_all_results_hit_at_n.tex',
    'T5-XL-SSM': 'tabs/mkqa_t5xlssm_all_results_hit_at_n.tex',
}

# MKQA selected approaches (subset of methods available in MKQA)
mkqa_selected_approaches = [
    'Without reranking',
    'Majority Vote',
    # 'Semantic reranking - Text',
    'RankGPT (Mistral) - Text',
    'RankGPT (Mistral) - Text + G2T (Det. Lin.)',
    'RankGPT (Qwen2.5 7B) - Text',
    'RankGPT (Qwen2.5 7B) - Text + G2T (Det. Lin.)',
    'Linear Regression - Text',
    'Linear Regression - Graph',
    'Linear Regression - Text + Graph',
    'Linear Regression - G2T (Det. Lin.)',
    'Linear Regression - Text + G2T (Det. Lin.)',
    'Logistic Regression - Text',
    'Logistic Regression - Graph',
    'Logistic Regression - Text + Graph',
    'Logistic Regression - G2T (Det. Lin.)',
    'Logistic Regression - Text + G2T (Det. Lin.)',
    'CatBoost - Text',
    'CatBoost - Graph',
    'CatBoost - Text + Graph',
    'CatBoost - G2T (Det. Lin.)',
    'CatBoost - Text + G2T (Det. Lin.)',
    'MPNet - Text',
    'MPNet - Text + G2T (Det. Lin.)',
]

# Add colors for MKQA-specific methods
mkqa_colors = colors.copy()
mkqa_colors.update({
    'Linear Regression - Graph': '#B5A08D',
    'Logistic Regression - Graph': '#E8C8A2',
    'CatBoost - Graph': '#C9A00B',
})

# MKQA Y-axis limits
mkqa_y_axis_limits = {
    'T5-Large-SSM': None,
    'T5-XL-SSM': None,
}

# Plot MKQA dataset
print("\n" + "="*60)
print("Plotting MKQA dataset results...")
print("="*60)
mkqa_data_by_model = plot_results(
    mkqa_files,
    mkqa_selected_approaches,
    mkqa_colors,
    mkqa_y_axis_limits,
    'Hit@1 Performance: Reranking Approaches Across Candidate Sources on MKQA Dataset',
    'hit_at_1_comparison_mkqa'
)