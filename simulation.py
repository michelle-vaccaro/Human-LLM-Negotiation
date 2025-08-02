import os
import sys
import time
import random
import openai
import numpy as np
import pandas as pd
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

# ========== SETUP ==========

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Negotiation settings
temperature = 0
max_rounds = 50
chunk_size = 50
num_tries = 2
exercise = "table"

# Filepaths
sample_path = f'samples/samples_full.csv'
negotiation_path = f'negotiations/{round}_{exercise}_negotiations.csv'

# Load prompts
df_prompts = pd.read_csv(sample_path).fillna('')
with open('prompts/prelude.txt', 'r') as file:
    prompt_prelude = file.read()
with open(f'prompts/{exercise}_role1_instructions.txt', 'r') as file:
    role1_instructions = file.read()
with open(f'prompts/{exercise}_role2_instructions.txt', 'r') as file:
    role2_instructions = file.read()
with open(f'prompts/{exercise}_role1_svi_instructions.txt', 'r') as file:
    role1_svi_prompt = file.read()
with open(f'prompts/{exercise}_role2_svi_instructions.txt', 'r') as file:
    role2_svi_prompt = file.read()
with open('prompts/svi_survey.txt', 'r') as file:
    svi_survey = file.read()

# ========== SANITY CHECKS ==========
test_mode = False
if test_mode:
    df_prompts = df_prompts.head(10)

# ========== DEFINITIONS ==========

# Survey administration
def administer_svi(bot_history, model, temperature):
    bot_history.append({"role": "user", "content": svi_survey})
    response = openai.chat.completions.create(
        model=model,
        messages=bot_history,
        temperature=temperature,
    )
    answer = response.choices[0].message.content
    bot_history.append({"role": "assistant", "content": answer})
    return answer

# Reasoning schemas
class Step(BaseModel):
    name: str
    explanation: str

class Reasoning(BaseModel):
    steps: list[Step]
    final_response: str

# Core negotiation simulation
def simulate_negotiation(
    role1_instructions,
    role2_instructions,
    role1_competition_prompt,
    role2_competition_prompt,
    role1="Role 1",
    role2="Role 2",
    model="gpt-4",
    temperature=0.2,
    max_rounds=20,
):
    conversation_history = []

    role1_history = [{"role": "system", "content": f"{role1_instructions}\n\n{role1_competition_prompt}"}]
    role2_history = [{"role": "system", "content": f"{role2_instructions}\n\n{role2_competition_prompt}"}]

    current_speaker = random.choice([role1, role2])

    if current_speaker == role1:
        role1_response = openai.chat.completions.create(
            model=model, messages=role1_history, temperature=temperature
        ).choices[0].message.content.strip()
        role1_history.append({"role": "assistant", "content": role1_response})
        role2_history.append({"role": "user", "content": role1_response})
        conversation_history.append(role1_response)
        current_speaker = role2
    else:
        role2_response = openai.chat.completions.create(
            model=model, messages=role2_history, temperature=temperature
        ).choices[0].message.content.strip()
        role2_history.append({"role": "assistant", "content": role2_response})
        role1_history.append({"role": "user", "content": role2_response})
        conversation_history.append(role2_response)
        current_speaker = role1

    # Negotiation rounds
    for i in range(max_rounds):
        if current_speaker == role1:
            role1_response = openai.chat.completions.create(
                model=model, messages=role1_history, temperature=temperature
            ).choices[0].message.content.strip()
            if role1_response:
                role1_history.append({"role": "assistant", "content": role1_response})
                role2_history.append({"role": "user", "content": role1_response})
                conversation_history.append(role1_response)
            current_speaker = role2
        else:
            role2_response = openai.chat.completions.create(
                model=model, messages=role2_history, temperature=temperature
            ).choices[0].message.content.strip()
            if role2_response:
                role2_history.append({"role": "assistant", "content": role2_response})
                role1_history.append({"role": "user", "content": role2_response})
                conversation_history.append(role2_response)
            current_speaker = role1

        # Check for end conditions
        if "[DEAL REACHED]" in role1_response or "[DEAL REACHED]" in role2_response \
            or "[NO DEAL]" in role1_response or "[NO DEAL]" in role2_response \
            or i == max_rounds - 1:
            
            # Administer SVI before ending
            role1_svi = administer_svi(role1_history, model, temperature)
            role2_svi = administer_svi(role2_history, model, temperature)
            conversation_history.append(f"{role1}: {role1_svi}")
            conversation_history.append(f"{role2}: {role2_svi}")
            break

    return conversation_history

# Wrapper for one row
def simulate_negotiation_wrapper(row):
    if pd.notna(row['conversation']):
        return row['conversation']

    for attempt in range(num_tries):
        try:
            return simulate_negotiation(
                role1_instructions=role1_instructions,
                role2_instructions=role2_instructions,
                role1_competition_prompt=prompt_prelude + row['prompt_1'],
                role2_competition_prompt=prompt_prelude + row['prompt_2'],
                # role1="Buyer",
                # role2="Seller",
                model=model,
                temperature=temperature,
                max_rounds=max_rounds,
            )
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for row {row.name}: {e}")
            continue
    return None

# ========== MAIN LOOP ==========
# Mark rows where conversation is missing
def is_missing(x):
    return (x is None) or (pd.isna(x)) or (str(x).strip() == "")

prompt_cols = df_prompts.columns.tolist()        # columns that uniquely define a prompt
conv_col     = "conversation"

# Load or initialize negotiations
if os.path.exists(negotiation_path):
    df_negotiations = pd.read_csv(negotiation_path)
else:
    df_negotiations = df_prompts.copy()

# Make sure 'conversation' column exists
if 'conversation' not in df_negotiations.columns:
    df_negotiations['conversation'] = None
else:
    df_negotiations['conversation'] = df_negotiations['conversation'].astype('object')

# Only work on rows where conversation is missing
incomplete_rows = df_negotiations[df_negotiations['conversation'].apply(is_missing)]
rows = [row for _, row in incomplete_rows.iterrows()]
chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]

# Start timing
start = time.time()

# Process chunks
for chunk_index, chunk in enumerate(chunks):
    print(f"\rProcessing chunk {chunk_index + 1}/{len(chunks)}", end="\t")
    sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=chunk_size) as executor:
        responses = list(executor.map(simulate_negotiation_wrapper, chunk))

    for i, row in enumerate(chunk):
        df_negotiations.at[row.name, 'conversation'] = responses[i]
    
    df_negotiations.to_csv(negotiation_path, index=False)

# End timing
end = time.time()
print(f"\nTotal time: {(end - start) / 60:.2f} minutes")