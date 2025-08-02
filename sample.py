import pandas as pd
import os
from itertools import product

# ===========================
# PARAMETERS
# ===========================
models = ['o4-mini', 'o3', 'o3-mini', 'o1', 'gpt-4.1', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1-nano', 
          'gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4']
# prompted = [True, False]
prompted = [False]
warmth_scores = [-100, 50, 0, 50, 100]
dominance_scores = [-100, 50, 0, 50, 100]
n_iter = 1

# ===========================
# FILE PATHS
# ===========================
prompt_file = 'prompts/personality_template.txt'
sample_file = f'samples/samples_n{n_iter}.csv'

# ===========================
# MAIN SCRIPT
# ===========================
def create_pairings(prompt_file, sample_file):
    # Load prompt template
    with open(prompt_file, 'r') as f:
        prompt_template = f.read()

    # Merge to get every possible pairing
    pairings = []
    for _ in range(n_iter):
        for value in prompted:
            if value:
                for model, warmth_score, dominance_score in product(models, warmth_scores, dominance_scores):
                    prompt = prompt_template.format(warmth_score=warmth_score, dominance_score=dominance_score)
                    pairings.append({
                        'model': model,
                        'warmth_score': warmth_score,
                        'dominance_score': dominance_score,
                        'prompt': prompt
                    })
            else:
                for model in models:
                    pairings.append({
                        'model': model,
                        'warmth_score': None,
                        'dominance_score': None,
                        'prompt': ''
                    })

    # Convert to DataFrame
    pairings_df = pd.DataFrame(pairings)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(sample_file), exist_ok=True)

    # Save to CSV
    pairings_df.to_csv(sample_file, index=False)
    print(f"Sample file created: {sample_file} ({len(pairings_df)} rows)")

if __name__ == "__main__":
    create_pairings(prompt_file, sample_file)
