import pandas as pd
import openai
import json
import re
import os
from tqdm import tqdm  # For progress tracking

def setup_api():
    """Configure OpenAI API key from environment"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    openai.api_key = api_key

def load_data(filename):
    """Load and prepare the prompts data"""
    df = pd.read_csv(filename)
    return df.fillna('')

def create_query_template():
    """Define the scoring query template"""
    return """Return a json formatted string with the following values:

"dominance_score": How much the prompt emphasizes dominance (0 = not at all, 100 = very much)
"warmth_score": How much the prompt emphasizes warmth (0 = not at all, 100 = very much)

Please note that dominance and warmth are *orthogonal* constructs, meaning it's possible to be high on one without being low on the other. For example, being dominant does not mean you are cold, and being warm does not mean you are submissive. 

<prompt>
{prompt}
</prompt>

<dominance>
Dominance is related to acting assertive, firm, or forceful, and advocating for your own needs, interests, and positions. They can be about how to set anchors, how to leverage your BATNA (best alternative to a negotiated agreement), and how to respond to counteroffers in ways that benefit you.
</dominance>

<warmth>
Warmth is related to acting friendly, sympathetic, or sociable, and demonstrating empathy and nonjudgmental understanding of other people's needs, interests, and positions. They can be about how to maintain a positive rapport, how to enhance counterpart subjective value, and using language to show empathy and kindness. 
</warmth>
"""

def score_prompt(prompt, model, query_template):
    """Score a single prompt using the OpenAI API"""
    formatted_query = query_template.format(prompt=prompt)
    
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": formatted_query}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API error: {e}")
        return None

def parse_json_response(json_str):
    """Parse JSON from API response"""
    if not json_str:
        return {"dominance_score": None, "warmth_score": None}
        
    # Clean the response
    cleaned = json_str.replace('```', '').replace('json', '').strip()
    
    # Extract JSON object
    match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
    if not match:
        return {"dominance_score": None, "warmth_score": None}
    
    try:
        json_part = match.group(1)
        return json.loads(json_part)
    except json.JSONDecodeError:
        return {"dominance_score": None, "warmth_score": None}

def process_prompts(df, model, query_template, save_path, batch_size=10):
    """Process all prompts and save results periodically"""
    total_rows = len(df)
    
    for i in tqdm(range(0, total_rows, batch_size)):
        batch_end = min(i + batch_size, total_rows)
        batch = df.iloc[i:batch_end]
        
        for idx, row in batch.iterrows():
            if pd.isna(df.loc[idx, 'annotation']):  # Only process if not already done
                annotation = score_prompt(row['prompt'], model, query_template)
                df.loc[idx, 'annotation'] = annotation
                
                if annotation:
                    parsed = parse_json_response(annotation)
                    df.loc[idx, 'dominance_score'] = parsed.get('dominance_score')
                    df.loc[idx, 'warmth_score'] = parsed.get('warmth_score')
        
        # Save after each batch for recovery in case of failure
        df.to_csv(save_path, index=False)
        
    return df

def main():
    # Configuration
    round = '1'
    model = 'gpt-4.5-preview'

    input_file = f'prompts/round{round}_prompts.csv'
    output_file = f'outcomes/round{round}_scores_{model}.csv'
    
    # Setup
    setup_api()
    query_template = create_query_template()
    
    # Check if output file exists to resume processing
    if os.path.exists(output_file):
        print(f"Found existing output file. Resuming from {output_file}")
        df = load_data(output_file)
    else:
        print(f"Starting new processing from {input_file}")
        df = load_data(input_file)
        # Initialize new columns
        df['annotation'] = None
        df['dominance_score'] = None
        df['warmth_score'] = None
    
    # Process prompts
    df = process_prompts(df, model, query_template, output_file)
    
    # Final save
    df.to_csv(output_file, index=False)
    print(f"Processing complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()