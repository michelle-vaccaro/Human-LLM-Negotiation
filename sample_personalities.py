import pandas as pd
import os
from itertools import product

# ===========================
# PARAMETERS
# ===========================
models = ['o4-mini', 'o3', 'o3-mini', 'o1', 'gpt-4.1', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1-nano', 
          'gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4',
          'claude-3-haiku-20240307', 'claude-3-5-sonnet-20240620', 'claude-3-5-sonnet-20241022',
          'claude-3-5-haiku-20241022', 'claude-3-7-sonnet-20250219', 'claude-sonnet-4-20250514',
          'claude-opus-4-20250514',
          'gemini-1.5-pro', 'gemini-1.5-flash-8b', 'gemini-1.5-flash', 'gemini-2.0-flash-live-001',
          'gemini-2.0-flash-lite', 'gemini-2.0-flash', 'gemini-2.5-flash-lite', 'gemini-2.5-flash',
          'gemini-2.5-pro']
prompted = [True, False]
# prompted = [False]
warmth_scores = [-100, -50, 0, 50, 100]
dominance_scores = [-100, -50, 0, 50, 100]

# ===========================
# FILE PATHS
# ===========================
prompt_file = 'prompts/personality_template.txt'

# Determine filename based on prompted setting
if True in prompted:
    sample_file = f'samples/models_all_prompted.csv'
    # sample_file = f'samples/models_claude_gemini_prompted.csv'
else:
    # sample_file = f'samples/models_all_unprompted.csv'
    sample_file = f'samples/models_claude_gemini_unprompted.csv'

# ===========================
# MAIN SCRIPT
# ===========================
def create_personas(prompt_file, sample_file):
    # Load prompt template
    with open(prompt_file, 'r') as f:
        prompt_template = f.read()

    # Sample from the space of possible persona
    personas = []
    for value in prompted:
        if value:
            for model, warmth_score, dominance_score in product(models, warmth_scores, dominance_scores):
                prompt = prompt_template.format(warmth_score=warmth_score, dominance_score=dominance_score)
                personas.append({
                    'model': model,
                    'warmth_score': warmth_score,
                    'dominance_score': dominance_score,
                    'prompt': prompt
                })
        else:
            for model in models:
                personas.append({
                    'model': model,
                    'warmth_score': None,
                    'dominance_score': None,
                    'prompt': ''
                })

    # Convert to DataFrame
    personas_df = pd.DataFrame(personas)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(sample_file), exist_ok=True)

    # Save to CSV
    personas_df.to_csv(sample_file, index=False)
    print(f"Sample file created: {sample_file} ({len(personas_df)} rows)")

if __name__ == "__main__":
    create_personas(prompt_file, sample_file)
