import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.patches as patches

def load_assessment_results(file_path):
    """Load assessment results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_ias_scores(data):
    """Extract IAS scores for each model from the assessment data."""
    ias_data = []
    
    for entry in data:
        model = entry.get('model')
        agent_id = entry.get('agent_id')
        
        # Check if IAS assessment exists and was successful
        if 'assessments' in entry and 'IAS' in entry['assessments']:
            ias_assessment = entry['assessments']['IAS']
            
            if ias_assessment.get('status') == 'success' and 'scores' in ias_assessment:
                scores = ias_assessment['scores']
                
                # Extract octant scores
                octants = scores.get('octants', {})
                
                ias_data.append({
                    'model': model,
                    'agent_id': agent_id,
                    'PA': octants.get('PA', np.nan),  # Assured-Dominant
                    'BC': octants.get('BC', np.nan),  # Arrogant-Calculating
                    'DE': octants.get('DE', np.nan),  # Cold-Hearted
                    'FG': octants.get('FG', np.nan),  # Aloof-Introverted
                    'HI': octants.get('HI', np.nan),  # Unassured-Submissive
                    'JK': octants.get('JK', np.nan),  # Unassuming-Ingenuous
                    'LM': octants.get('LM', np.nan),  # Warm-Agreeable
                    'NO': octants.get('NO', np.nan)   # Gregarious-Extraverted
                })
    
    return pd.DataFrame(ias_data)

def extract_ias_scores_with_targets(data):
    """Extract IAS scores with target warmth/dominance for prompted assessments."""
    ias_data = []
    
    for entry in data:
        model = entry.get('model')
        agent_id = entry.get('agent_id')
        target_warmth = entry.get('target_warmth')
        target_dominance = entry.get('target_dominance')
        
        # Check if IAS assessment exists and was successful
        if 'assessments' in entry and 'IAS' in entry['assessments']:
            ias_assessment = entry['assessments']['IAS']
            
            if ias_assessment.get('status') == 'success' and 'scores' in ias_assessment:
                scores = ias_assessment['scores']
                
                # Extract warmth and dominance from IAS scores
                warmth = scores.get('warmth', np.nan)
                dominance = scores.get('dominance', np.nan)
                
                # Extract octant scores
                octants = scores.get('octants', {})
                
                ias_data.append({
                    'model': model,
                    'agent_id': agent_id,
                    'target_warmth': target_warmth,
                    'target_dominance': target_dominance,
                    'actual_warmth': warmth,
                    'actual_dominance': dominance,
                    'PA': octants.get('PA', np.nan),  # Assured-Dominant
                    'BC': octants.get('BC', np.nan),  # Arrogant-Calculating
                    'DE': octants.get('DE', np.nan),  # Cold-Hearted
                    'FG': octants.get('FG', np.nan),  # Aloof-Introverted
                    'HI': octants.get('HI', np.nan),  # Unassured-Submissive
                    'JK': octants.get('JK', np.nan),  # Unassuming-Ingenuous
                    'LM': octants.get('LM', np.nan),  # Warm-Agreeable
                    'NO': octants.get('NO', np.nan)   # Gregarious-Extraverted
                })
    
    return pd.DataFrame(ias_data)

def get_model_family(model_name):
    """Categorize models into families for color coding."""
    model_lower = model_name.lower()
    if 'gpt' in model_lower or any(x in model_lower for x in ['o1', 'o2', 'o3', 'o4']):
        return 'OpenAI'
    elif 'claude' in model_lower:
        return 'Claude'
    elif 'gemini' in model_lower:
        return 'Gemini'
    else:
        return 'Other'

def calculate_assessment_success_rates(data):
    """Calculate successful and unsuccessful assessment counts for each model."""
    success_counts = {}
    total_counts = {}
    
    for entry in data:
        model = entry.get('model')
        if model not in total_counts:
            total_counts[model] = 0
            success_counts[model] = 0
        
        total_counts[model] += 1
        
        # Check if IAS assessment exists and was successful
        if 'assessments' in entry and 'IAS' in entry['assessments']:
            ias_assessment = entry['assessments']['IAS']
            if ias_assessment.get('status') == 'success' and 'scores' in ias_assessment:
                success_counts[model] += 1
    
    # Create a list of dictionaries with success rates
    success_data = []
    for model in total_counts:
        successful = success_counts[model]
        unsuccessful = total_counts[model] - successful
        success_rate = (successful / total_counts[model]) * 100 if total_counts[model] > 0 else 0
        
        success_data.append({
            'model': model,
            'successful': successful,
            'unsuccessful': unsuccessful,
            'total': total_counts[model],
            'success_rate': success_rate
        })
    
    return success_data

def create_radar_charts_grid(df):
    """Create a grid of radar charts for each model."""
    
    # IAS octants in the correct circumplex order (clockwise from right)
    # LM (Warm-Agreeable) at 0°, NO (Gregarious-Extraverted) at 45°, PA (Assured-Dominant) at 90°, 
    # BC (Arrogant-Calculating) at 135°, DE (Cold-Hearted) at 180°, FG (Aloof-Introverted) at 225°, 
    # HI (Unassured-Submissive) at 270°, JK (Unassuming-Ingenuous) at 315°
    dimensions = ['LM', 'NO', 'PA', 'BC', 'DE', 'FG', 'HI', 'JK']
    dimension_names = ['Warm-Agreeable', 'Gregarious-Extraverted', 'Assured-Dominant', 'Arrogant-Calculating', 
                      'Cold-Hearted', 'Aloof-Introverted', 'Unassured-Submissive', 'Unassuming-Ingenuous']
    
    # Number of models and calculate grid dimensions
    n_models = len(df)
    n_cols = 6
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), subplot_kw=dict(projection='polar'))
    fig.suptitle('IAS Interpersonal Octant Profiles - Radar Charts Grid', fontsize=16, fontweight='bold', y=0.98)
    
    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Colors for different model families
    family_colors = {
        'OpenAI': '#1f77b4',
        'Claude': '#9467bd', 
        'Gemini': '#2ca02c',
        'Other': '#d62728'
    }
    
    for i, (_, row) in enumerate(df.iterrows()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get model data
        model_name = row['model']
        model_family = get_model_family(model_name)
        values = [row[dim] for dim in dimensions]
        
        # Create angles for polar plot - align with interpersonal circumplex
        # LM at 0°, NO at 45°, PA at 90°, BC at 135°, DE at 180°, FG at 225°, HI at 270°, JK at 315°
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        angles += angles[:1]  # Close the polygon
        values += values[:1]  # Close the polygon
        
        # Set up polar plot
        ax.set_theta_offset(0)  # Start at 0 degrees (right side)
        ax.set_theta_direction(-1)  # Clockwise direction
        
        # Plot radar chart
        ax.plot(angles, values, 'o-', linewidth=2, color=family_colors.get(model_family, '#666666'))
        ax.fill(angles, values, alpha=0.25, color=family_colors.get(model_family, '#666666'))
        
        # Set up the axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, color='black', size=8, fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        ax.set_rlabel_position(0)
        
        # Set y-axis limits based on IAS score range (typically 1-8)
        max_score = max(values[:-1]) if values[:-1] else 8
        min_score = min(values[:-1]) if values[:-1] else 1
        y_ticks = list(range(int(min_score), int(max_score) + 2))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(tick) for tick in y_ticks], color='black', size=8)
        ax.set_ylim(min_score - 0.5, max_score + 0.5)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set title
        ax.set_title(f'{model_name}\n({model_family})', fontsize=10, fontweight='bold', pad=10)
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_grouped_bar_chart(df):
    """Create a grouped bar chart comparing IAS octants across models."""
    
    # Prepare data for plotting
    dimensions = ['PA', 'BC', 'DE', 'FG', 'HI', 'JK', 'LM', 'NO']
    dimension_names = ['Assured-Dominant', 'Arrogant-Calculating', 'Cold-Hearted', 'Aloof-Introverted', 
                      'Unassured-Submissive', 'Unassuming-Ingenuous', 'Warm-Agreeable', 'Gregarious-Extraverted']
    
    # Add model family for color coding
    df['model_family'] = df['model'].apply(get_model_family)
    
    # Create figure with 2 rows and 4 columns (8 subplots total)
    fig, axes = plt.subplots(2, 4, figsize=(25, 12))
    fig.suptitle('IAS Interpersonal Octant Scores by Model', fontsize=16, fontweight='bold')
    
    # Colors for different model families
    family_colors = {
        'OpenAI': '#1f77b4',
        'Claude': '#9467bd', 
        'Gemini': '#2ca02c',
        'Other': '#d62728'
    }
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    for i, (dim, name) in enumerate(zip(dimensions, dimension_names)):
        ax = axes[i]
        
        # Sort by model family and then by model name
        df_sorted = df.sort_values(['model_family', 'model'])
        
        # Create bars
        x_pos = np.arange(len(df_sorted))
        bars = ax.bar(x_pos, df_sorted[dim], 
                     color=[family_colors.get(fam, '#666666') for fam in df_sorted['model_family']],
                     alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Customize plot
        ax.set_title(f'{name} ({dim})', fontweight='bold', fontsize=10)
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        
        # Set y-axis limits based on IAS score range
        max_score = df_sorted[dim].max()
        min_score = df_sorted[dim].min()
        ax.set_ylim(min_score - 0.5, max_score + 0.5)
        y_ticks = list(range(int(min_score), int(max_score) + 2))
        ax.set_yticks(y_ticks)
        
        # Set x-axis labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_sorted['model'], rotation=45, ha='right', fontsize=6)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, df_sorted[dim]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{value:.1f}', ha='center', va='bottom', fontsize=6)
    
    # Add legend at the bottom
    legend_elements = [patches.Patch(color=color, label=family) 
                      for family, color in family_colors.items()]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
              ncol=4, fontsize=12)
    
    plt.tight_layout()
    # Adjust layout to make room for legend
    plt.subplots_adjust(bottom=0.1)
    return fig

def create_heatmap_matrices(df):
    """Create one large heatmap with models on x-axis and IAS octants on y-axis."""
    
    # Prepare data for heatmap
    dimensions = ['PA', 'BC', 'DE', 'FG', 'HI', 'JK', 'LM', 'NO']
    dimension_names = ['Assured-Dominant', 'Arrogant-Calculating', 'Cold-Hearted', 'Aloof-Introverted', 
                      'Unassured-Submissive', 'Unassuming-Ingenuous', 'Warm-Agreeable', 'Gregarious-Extraverted']
    
    # Add model family and sort
    df['model_family'] = df['model'].apply(get_model_family)
    df_sorted = df.sort_values(['model_family', 'model'])
    
    # Create heatmap data (transpose to get dimensions as rows, models as columns)
    heatmap_data = df_sorted[dimensions].T.values  # Transpose the data
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
    
    # Set up axes
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['model'], fontsize=9, rotation=45, ha='right')
    ax.set_yticks(range(len(dimensions)))
    ax.set_yticklabels(dimension_names, fontsize=10, fontweight='bold')
    
    # Add value annotations
    for row in range(len(dimensions)):
        for col in range(len(df_sorted)):
            value = heatmap_data[row, col]
            # Determine text color based on value (assuming range 1-8)
            color = 'white' if value > 4.5 else 'black'
            ax.text(col, row, f'{value:.1f}', ha='center', va='center', 
                   color=color, fontsize=8, fontweight='bold')
    
    # Add model family separators
    current_family = None
    for i, family in enumerate(df_sorted['model_family']):
        if family != current_family:
            if current_family is not None:
                ax.axvline(x=i-0.5, color='black', linewidth=2)
            current_family = family
    
    # Add title
    ax.set_title('IAS Interpersonal Octant Scores Heatmap', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add color bar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_scatter_plots_grid(df):
    """Create a 5x6 grid of scatter plots for target vs actual warmth/dominance."""
    
    # Filter out rows with NaN or Inf values
    df_clean = df.dropna(subset=['target_warmth', 'target_dominance', 'actual_warmth', 'actual_dominance'])
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    
    if df_clean.empty:
        print("No valid data found after removing NaN/Inf values!")
        return None
    
    # Get unique models from clean data
    models = df_clean['model'].unique()
    n_models = len(models)
    
    if n_models == 0:
        print("No models with valid data found!")
        return None
    
    # Calculate grid dimensions (5 columns, 6 rows max)
    n_cols = 5
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots for each model (2 subplots per model: warmth and dominance)
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(30, 5 * n_rows))
    fig.suptitle('Target vs Actual Warmth/Dominance by Model', fontsize=16, fontweight='bold', y=0.98)
    
    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Colors for different model families
    family_colors = {
        'OpenAI': '#1f77b4',
        'Claude': '#9467bd', 
        'Gemini': '#2ca02c',
        'Other': '#d62728'
    }
    
    for i, model in enumerate(models):
        if i >= len(axes) // 2:  # Each model gets 2 subplots
            break
            
        # Get data for this model
        model_data = df_clean[df_clean['model'] == model]
        model_family = get_model_family(model)
        color = family_colors.get(model_family, '#d62728')
        
        # Left subplot: Target vs Actual Warmth
        ax_warmth = axes[i * 2]
        ax_warmth.set_title(f'{model} - Warmth\n({model_family})', fontsize=10, fontweight='bold')
        
        # Plot target vs actual warmth
        ax_warmth.scatter(model_data['target_warmth'], model_data['actual_warmth'], 
                         c=color, alpha=0.7, s=50)
        
        # Add perfect correlation diagonal
        min_warmth = min(model_data['target_warmth'].min(), model_data['actual_warmth'].min())
        max_warmth = max(model_data['target_warmth'].max(), model_data['actual_warmth'].max())
        
        # Ensure bounds are finite
        if np.isfinite(min_warmth) and np.isfinite(max_warmth):
            ax_warmth.plot([min_warmth, max_warmth], [min_warmth, max_warmth], 'k--', alpha=0.5, linewidth=1)
            
            # Set labels and limits
            ax_warmth.set_xlabel('Target Warmth')
            ax_warmth.set_ylabel('Actual Warmth')
            ax_warmth.set_xlim(min_warmth - 0.1, max_warmth + 0.1)
            ax_warmth.set_ylim(min_warmth - 0.1, max_warmth + 0.1)
        ax_warmth.grid(True, alpha=0.3)
        
        # Right subplot: Target vs Actual Dominance
        ax_dominance = axes[i * 2 + 1]
        ax_dominance.set_title(f'{model} - Dominance\n({model_family})', fontsize=10, fontweight='bold')
        
        # Plot target vs actual dominance
        ax_dominance.scatter(model_data['target_dominance'], model_data['actual_dominance'], 
                            c=color, alpha=0.7, s=50)
        
        # Add perfect correlation diagonal
        min_dom = min(model_data['target_dominance'].min(), model_data['actual_dominance'].min())
        max_dom = max(model_data['target_dominance'].max(), model_data['actual_dominance'].max())
        
        # Ensure bounds are finite
        if np.isfinite(min_dom) and np.isfinite(max_dom):
            ax_dominance.plot([min_dom, max_dom], [min_dom, max_dom], 'k--', alpha=0.5, linewidth=1)
            
            # Set labels and limits
            ax_dominance.set_xlabel('Target Dominance')
            ax_dominance.set_ylabel('Actual Dominance')
            ax_dominance.set_xlim(min_dom - 0.1, max_dom + 0.1)
            ax_dominance.set_ylim(min_dom - 0.1, max_dom + 0.1)
        ax_dominance.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_models * 2, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_correlation_heatmap(df):
    """Create correlation coefficient heatmap for warmth and dominance."""
    
    # Filter out rows with NaN or Inf values
    df_clean = df.dropna(subset=['target_warmth', 'target_dominance', 'actual_warmth', 'actual_dominance'])
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    
    if df_clean.empty:
        print("No valid data found for correlation heatmap!")
        return None
    
    # Calculate correlations for each model
    correlations = []
    
    for model in df_clean['model'].unique():
        model_data = df_clean[df_clean['model'] == model]
        
        # Only calculate correlations if we have at least 2 data points
        if len(model_data) >= 2:
            # Calculate warmth correlation
            warmth_corr = model_data['target_warmth'].corr(model_data['actual_warmth'])
            
            # Calculate dominance correlation
            dominance_corr = model_data['target_dominance'].corr(model_data['actual_dominance'])
            
            # Only include if correlations are not NaN
            if not np.isnan(warmth_corr) and not np.isnan(dominance_corr):
                correlations.append({
                    'model': model,
                    'warmth_correlation': warmth_corr,
                    'dominance_correlation': dominance_corr,
                    'avg_correlation': (warmth_corr + dominance_corr) / 2
                })
    
    if not correlations:
        print("No valid correlations found!")
        return None
    
    # Create DataFrame and sort by average correlation
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('avg_correlation', ascending=False)
    
    # Add model family for grouping
    corr_df['model_family'] = corr_df['model'].apply(get_model_family)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(corr_df) * 0.3)))
    
    # Prepare data for heatmap
    heatmap_data = corr_df[['warmth_correlation', 'dominance_correlation']].values
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # Set labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Warmth\nCorrelation', 'Dominance\nCorrelation'], fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df['model'], fontsize=10)
    
    # Add correlation values as text
    for i in range(len(corr_df)):
        for j in range(2):
            value = heatmap_data[i, j]
            if not np.isnan(value):
                color = 'white' if value < 0.5 else 'black'
                ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                       color=color, fontsize=9, fontweight='bold')
    
    # Add model family separators
    current_family = None
    for i, family in enumerate(corr_df['model_family']):
        if family != current_family:
            if current_family is not None:
                ax.axhline(y=i-0.5, color='black', linewidth=2)
            current_family = family
    
    # Add title
    ax.set_title('Model Responsiveness: Target vs Actual Correlation Coefficients', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add color bar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_model_summary_table(df):
    """Create a summary table of IAS scores by model."""
    summary = df.groupby('model')[['PA', 'BC', 'DE', 'FG', 'HI', 'JK', 'LM', 'NO']].agg(['mean', 'std']).round(3)
    return summary

def main():
    """Main function to run the analysis."""
    
    import sys
    
    # File path - can be modified to use different files
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        file_path = Path('assessment_results/assessment_models_claude_gemini_prompted.json')
    
    # Load data
    print("Loading assessment results...")
    data = load_assessment_results(file_path)
    
    # Check if this is prompted or unprompted data based on filename
    file_path_str = str(file_path)
    # Use word boundaries to avoid matching "prompted" within "unprompted"
    is_prompted = 'prompted' in file_path_str and 'unprompted' not in file_path_str
    is_unprompted = 'unprompted' in file_path_str
    
    # Also check if data has target_warmth/target_dominance as backup
    has_targets = any('target_warmth' in entry and entry.get('target_warmth') is not None for entry in data)
    
    # Determine the type of analysis to run
    if is_prompted or (not is_unprompted and has_targets):
        analysis_type = 'prompted'
    else:
        analysis_type = 'unprompted'
    
    if analysis_type == 'prompted':
        print("Detected prompted assessment data - creating target vs actual plots...")
        df = extract_ias_scores_with_targets(data)
        
        if df.empty:
            print("No IAS data found in the prompted assessment results!")
            return
        
        print(f"Found IAS data for {len(df)} entries across {df['model'].nunique()} models")
        
        # Create prompted assessment plots
        print("Creating scatter plots grid...")
        fig1 = create_scatter_plots_grid(df)
        
        print("Creating correlation heatmap...")
        fig2 = create_correlation_heatmap(df)
        
        # Save the plots only if they were created successfully
        Path('figures').mkdir(exist_ok=True)
        
        # Generate dynamic file names based on input file
        base_name = file_path.stem
        if fig1 is not None:
            fig1.savefig(f'figures/{base_name}_scatter_plots_grid.png', dpi=300, bbox_inches='tight')
            print(f"Scatter plots grid saved to figures/{base_name}_scatter_plots_grid.png")
        else:
            print("No scatter plots created - no valid data")
        
        if fig2 is not None:
            fig2.savefig(f'figures/{base_name}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"Correlation heatmap saved to figures/{base_name}_correlation_heatmap.png")
        else:
            print("No correlation heatmap created - no valid data")
        
        # Create and display correlation summary only if we have valid data
        if fig2 is not None:
            print("\nCreating correlation summary...")
            correlations = []
            df_clean = df.dropna(subset=['target_warmth', 'target_dominance', 'actual_warmth', 'actual_dominance'])
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
            
            for model in df_clean['model'].unique():
                model_data = df_clean[df_clean['model'] == model]
                if len(model_data) >= 2:
                    warmth_corr = model_data['target_warmth'].corr(model_data['actual_warmth'])
                    dominance_corr = model_data['target_dominance'].corr(model_data['actual_dominance'])
                    if not np.isnan(warmth_corr) and not np.isnan(dominance_corr):
                        avg_corr = (warmth_corr + dominance_corr) / 2
                        correlations.append({
                            'model': model,
                            'warmth_correlation': warmth_corr,
                            'dominance_correlation': dominance_corr,
                            'avg_correlation': avg_corr
                        })
            
            if correlations:
                corr_df = pd.DataFrame(correlations).sort_values('avg_correlation', ascending=False)
                
                # Calculate assessment success rates
                success_data = calculate_assessment_success_rates(data)
                success_df = pd.DataFrame(success_data)
                
                # Merge correlation data with success data
                merged_df = corr_df.merge(success_df[['model', 'successful', 'unsuccessful', 'total', 'success_rate']], 
                                        on='model', how='left')
                
                print("\nModel Responsiveness Rankings:")
                print("=" * 120)
                # Display with better formatting to show all columns
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                print(merged_df.round(3))
                
                # Save correlation summary to CSV
                corr_csv_path = f'assessment_results/{base_name}_correlation_summary.csv'
                merged_df.to_csv(corr_csv_path, index=False)
                print(f"\nCorrelation summary saved to {corr_csv_path}")
                
                # Also save with standard ias_summary naming convention
                if 'prompted' in base_name:
                    ias_corr_path = 'assessment_results/ias_correlation_summary.csv'
                else:
                    ias_corr_path = 'assessment_results/ias_correlation_summary_unprompted.csv'
                merged_df.to_csv(ias_corr_path, index=False)
                print(f"Correlation summary also saved to {ias_corr_path}")
            else:
                print("No valid correlations found for summary")
        else:
            print("No correlation summary created - no valid data")
        
    else:  # analysis_type == 'unprompted'
        print("Detected unprompted assessment data - creating octant plots...")
        df = extract_ias_scores(data)
        
        if df.empty:
            print("No IAS data found in the assessment results!")
            return
        
        print(f"Found IAS data for {len(df)} entries across {df['model'].nunique()} models")
        
        # Create the three original plots
        print("Creating radar charts grid...")
        fig1 = create_radar_charts_grid(df)
        
        print("Creating grouped bar chart...")
        fig2 = create_grouped_bar_chart(df)
        
        print("Creating heatmap matrices...")
        fig3 = create_heatmap_matrices(df)
        
        # Save the plots
        Path('figures').mkdir(exist_ok=True)
        
        # Generate dynamic file names based on input file
        base_name = file_path.stem
        
        fig1.savefig(f'figures/{base_name}_radar_charts_grid.png', dpi=300, bbox_inches='tight')
        print(f"Radar charts grid saved to figures/{base_name}_radar_charts_grid.png")
        
        fig2.savefig(f'figures/{base_name}_grouped_bar_chart.png', dpi=300, bbox_inches='tight')
        print(f"Grouped bar chart saved to figures/{base_name}_grouped_bar_chart.png")
        
        fig3.savefig(f'figures/{base_name}_heatmap_matrices.png', dpi=300, bbox_inches='tight')
        print(f"Heatmap matrices saved to figures/{base_name}_heatmap_matrices.png")
        
        # Create and display summary table
        print("\nCreating summary table...")
        summary = create_model_summary_table(df)
        
        # Calculate assessment success rates
        success_data = calculate_assessment_success_rates(data)
        success_df = pd.DataFrame(success_data)
        
        # Flatten the multi-level index and columns of the summary table
        summary_flat = summary.reset_index()
        summary_flat.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in summary_flat.columns]
        

        
        # Ensure the model column is properly named
        if 'model_' in summary_flat.columns:
            summary_flat = summary_flat.rename(columns={'model_': 'model'})
        elif 'level_0' in summary_flat.columns:
            summary_flat = summary_flat.rename(columns={'level_0': 'model'})
        elif 'model' not in summary_flat.columns:
            # If model is not in columns, it might be the index name
            summary_flat = summary_flat.reset_index()
            if 'index' in summary_flat.columns:
                summary_flat = summary_flat.rename(columns={'index': 'model'})
        
        # Ensure both DataFrames have the same data type for the model column
        summary_flat['model'] = summary_flat['model'].astype(str)
        success_df['model'] = success_df['model'].astype(str)
        
        # Merge summary data with success data
        merged_summary = summary_flat.merge(success_df[['model', 'successful', 'unsuccessful', 'total', 'success_rate']], 
                                          on='model', how='left')
        
        print("\nIAS Scores Summary by Model:")
        print("=" * 120)
        # Display with better formatting to show all columns
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(merged_summary)
        
        # Save summary to CSV
        summary_csv_path = f'assessment_results/{base_name}_summary.csv'
        merged_summary.to_csv(summary_csv_path)
        print(f"\nSummary table saved to {summary_csv_path}")
        
        # Also save with standard ias_summary naming convention
        if 'prompted' in base_name:
            ias_summary_path = 'assessment_results/ias_summary.csv'
        else:
            ias_summary_path = 'assessment_results/ias_summary_unprompted.csv'
        merged_summary.to_csv(ias_summary_path)
        print(f"Summary table also saved to {ias_summary_path}")
    
    # Display basic statistics
    print(f"\nBasic Statistics:")
    print(f"Total entries: {len(df)}")
    print(f"Number of models: {df['model'].nunique()}")
    print(f"Models: {', '.join(sorted(df['model'].unique()))}")
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    main() 