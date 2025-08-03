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

def extract_tki_scores(data):
    """Extract TKI scores for each model from the assessment data."""
    tki_data = []
    
    for entry in data:
        model = entry.get('model')
        agent_id = entry.get('agent_id')
        
        # Check if TKI assessment exists and was successful
        if 'assessments' in entry and 'TKI' in entry['assessments']:
            tki_assessment = entry['assessments']['TKI']
            
            if tki_assessment.get('status') == 'success' and 'scores' in tki_assessment:
                scores = tki_assessment['scores']
                
                tki_data.append({
                    'model': model,
                    'agent_id': agent_id,
                    'competing': scores.get('competing', np.nan),
                    'collaborating': scores.get('collaborating', np.nan),
                    'compromising': scores.get('compromising', np.nan),
                    'avoiding': scores.get('avoiding', np.nan),
                    'accommodating': scores.get('accommodating', np.nan)
                })
    
    return pd.DataFrame(tki_data)

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

def create_radar_charts_grid(df):
    """Create a grid of radar charts for each model."""
    
    # TKI dimensions and their full names
    dimensions = ['competing', 'collaborating', 'compromising', 'avoiding', 'accommodating']
    dimension_names = ['Competing', 'Collaborating', 'Compromising', 'Avoiding', 'Accommodating']
    
    # Number of models and calculate grid dimensions
    n_models = len(df)
    n_cols = 6
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), subplot_kw=dict(projection='polar'))
    fig.suptitle('TKI Conflict Mode Profiles - Radar Charts Grid', fontsize=16, fontweight='bold', y=0.98)
    
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
        
        # Create angles for polar plot
        N = len(dimensions)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the polygon
        values += values[:1]  # Close the polygon
        
        # Set up polar plot
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Plot radar chart
        ax.plot(angles, values, 'o-', linewidth=2, color=family_colors.get(model_family, '#666666'))
        ax.fill(angles, values, alpha=0.25, color=family_colors.get(model_family, '#666666'))
        
        # Set up the axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, color='black', size=8, fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        ax.set_rlabel_position(0)
        
        # Set y-axis limits based on TKI score range (typically 0-12)
        max_score = max(values[:-1]) if values[:-1] else 12
        y_ticks = list(range(0, max_score + 1, 2))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(tick) for tick in y_ticks], color='black', size=8)
        ax.set_ylim(0, max_score + 1)
        
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
    """Create a grouped bar chart comparing conflict modes across models."""
    
    # Prepare data for plotting
    dimensions = ['competing', 'collaborating', 'compromising', 'avoiding', 'accommodating']
    dimension_names = ['Competing', 'Collaborating', 'Compromising', 'Avoiding', 'Accommodating']
    
    # Add model family for color coding
    df['model_family'] = df['model'].apply(get_model_family)
    
    # Create figure with 2 rows and 3 columns (5 subplots total)
    fig, axes = plt.subplots(2, 3, figsize=(25, 12))
    fig.suptitle('TKI Conflict Mode Scores by Model', fontsize=16, fontweight='bold')
    
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
        ax.set_title(f'{name}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        
        # Set y-axis limits based on TKI score range
        max_score = df_sorted[dim].max()
        ax.set_ylim(0, max_score + 1)
        y_ticks = list(range(0, int(max_score) + 2, 2))
        ax.set_yticks(y_ticks)
        
        # Set x-axis labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_sorted['model'], rotation=45, ha='right', fontsize=8)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, df_sorted[dim]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{value:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Hide the unused subplot
    axes[5].set_visible(False)
    
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
    """Create one large heatmap with models on x-axis and TKI dimensions on y-axis."""
    
    # Prepare data for heatmap
    dimensions = ['competing', 'collaborating', 'compromising', 'avoiding', 'accommodating']
    dimension_names = ['Competing', 'Collaborating', 'Compromising', 'Avoiding', 'Accommodating']
    
    # Add model family and sort
    df['model_family'] = df['model'].apply(get_model_family)
    df_sorted = df.sort_values(['model_family', 'model'])
    
    # Create heatmap data (transpose to get dimensions as rows, models as columns)
    heatmap_data = df_sorted[dimensions].T.values  # Transpose the data
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
    
    # Set up axes
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['model'], fontsize=9, rotation=45, ha='right')
    ax.set_yticks(range(len(dimensions)))
    ax.set_yticklabels(dimension_names, fontsize=12, fontweight='bold')
    
    # Add value annotations
    for row in range(len(dimensions)):
        for col in range(len(df_sorted)):
            value = heatmap_data[row, col]
            # Determine text color based on value (assuming max around 12)
            color = 'white' if value > 6 else 'black'
            ax.text(col, row, f'{value:.0f}', ha='center', va='center', 
                   color=color, fontsize=8, fontweight='bold')
    
    # Add model family separators
    current_family = None
    for i, family in enumerate(df_sorted['model_family']):
        if family != current_family:
            if current_family is not None:
                ax.axvline(x=i-0.5, color='black', linewidth=2)
            current_family = family
    
    # Add title
    ax.set_title('TKI Conflict Mode Scores Heatmap', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add color bar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_model_summary_table(df):
    """Create a summary table of TKI scores by model."""
    summary = df.groupby('model')[['competing', 'collaborating', 'compromising', 'avoiding', 'accommodating']].agg(['mean', 'std']).round(3)
    return summary

def main():
    """Main function to run the analysis."""
    
    # File path
    file_path = Path('assessment_results/assessment_results.json')
    
    # Check if file exists
    if not file_path.exists():
        print(f"Error: File {file_path} not found!")
        return
    
    # Load data
    print("Loading assessment results...")
    data = load_assessment_results(file_path)
    
    # Extract TKI scores
    print("Extracting TKI scores...")
    df = extract_tki_scores(data)
    
    if df.empty:
        print("No TKI data found in the assessment results!")
        return
    
    print(f"Found TKI data for {len(df)} entries across {df['model'].nunique()} models")
    
    # Create the three plots
    print("Creating radar charts grid...")
    fig1 = create_radar_charts_grid(df)
    
    print("Creating grouped bar chart...")
    fig2 = create_grouped_bar_chart(df)
    
    print("Creating heatmap matrices...")
    fig3 = create_heatmap_matrices(df)
    
    # Save the plots
    Path('figures').mkdir(exist_ok=True)
    
    fig1.savefig('figures/tki_radar_charts_grid.png', dpi=300, bbox_inches='tight')
    print("Radar charts grid saved to figures/tki_radar_charts_grid.png")
    
    fig2.savefig('figures/tki_grouped_bar_chart.png', dpi=300, bbox_inches='tight')
    print("Grouped bar chart saved to figures/tki_grouped_bar_chart.png")
    
    fig3.savefig('figures/tki_heatmap_matrices.png', dpi=300, bbox_inches='tight')
    print("Heatmap matrices saved to figures/tki_heatmap_matrices.png")
    
    # Create and display summary table
    print("\nCreating summary table...")
    summary = create_model_summary_table(df)
    print("\nTKI Scores Summary by Model:")
    print("=" * 80)
    print(summary)
    
    # Save summary to CSV
    summary_csv_path = 'assessment_results/tki_summary.csv'
    summary.to_csv(summary_csv_path)
    print(f"\nSummary table saved to {summary_csv_path}")
    
    # Display basic statistics
    print(f"\nBasic Statistics:")
    print(f"Total entries: {len(df)}")
    print(f"Number of models: {df['model'].nunique()}")
    print(f"Models: {', '.join(sorted(df['model'].unique()))}")
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    main() 