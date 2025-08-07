import os
import sys
import time
import random
import anthropic
import google.generativeai as genai
import numpy as np
import pandas as pd
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from anthropic import Anthropic
from concurrent.futures import ThreadPoolExecutor

# ========== SETUP ==========
# Load environment variables from .env file
load_dotenv()

# Set API keys and clients
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
gemini_key = os.getenv("GOOGLE_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=openai_key) if openai_key else None
anthropic_client = Anthropic(api_key=anthropic_key) if anthropic_key else None
if gemini_key:
    genai.configure(api_key=gemini_key)

# Negotiation settings
temperature = 0
max_rounds = 50
chunk_size = 50
num_tries = 2
exercise = "table"

# Filepaths
sample_path = f'samples/dyads_all_unprompted_1.csv'

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

# ========== PROVIDER FUNCTIONS ==========

def get_provider(model_name):
    """Determine the provider for a given model name."""
    model_lower = model_name.lower()
    if 'gpt' in model_lower or any(x in model_lower for x in ['o1', 'o2', 'o3', 'o4']):
        return 'openai'
    elif 'claude' in model_lower:
        return 'anthropic'
    elif 'gemini' in model_lower:
        return 'google'
    else:
        return 'openai'  # default

def create_chat_completion(model, messages, temperature):
    """Create a chat completion using the appropriate provider."""
    provider = get_provider(model)
    
    try:
        if provider == 'openai':
            if not openai_client:
                print(f"Warning: OpenAI API key not set for model {model}")
                return None
            
            # Models that don't support temperature parameter
            models_without_temperature = ["o3", "o3-mini", "o1", "o4-mini", "o4"]
            
            if model in models_without_temperature:
                # For models that don't support temperature, omit the parameter
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            else:
                # For models that support temperature, include it
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
            return response.choices[0].message.content
        
        elif provider == 'anthropic':
            if not anthropic_client:
                print(f"Warning: Anthropic API key not set for model {model}")
                return None
            # Convert OpenAI format to Anthropic format
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                elif msg['role'] == 'user':
                    user_messages.append(msg['content'])
                elif msg['role'] == 'assistant':
                    user_messages.append(msg['content'])
            
            # Combine messages for Anthropic
            full_prompt = system_message + "\n\n" + "\n\n".join(user_messages)
            
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=temperature,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.content[0].text
        
        elif provider == 'google':
            if not gemini_key:
                print(f"Warning: Google API key not set for model {model}")
                return None
            # Convert to Google format
            model_obj = genai.GenerativeModel(model)
            
            # Combine messages for Google
            full_prompt = ""
            for msg in messages:
                if msg['role'] == 'system':
                    full_prompt += f"System: {msg['content']}\n\n"
                elif msg['role'] == 'user':
                    full_prompt += f"User: {msg['content']}\n\n"
                elif msg['role'] == 'assistant':
                    full_prompt += f"Assistant: {msg['content']}\n\n"
            
            response = model_obj.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=1000
                )
            )
            return response.text
        
    except Exception as e:
        print(f"Error with {provider} model {model}: {e}")
        return None

# ========== DEFINITIONS ==========

# Survey administration
def administer_svi(bot_history, model, temperature):
    bot_history.append({"role": "user", "content": svi_survey})
    answer = create_chat_completion(model, bot_history, temperature)
    if answer:
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
        role1_response = create_chat_completion(model, role1_history, temperature)
        if role1_response:
            role1_response = role1_response.strip()
            role1_history.append({"role": "assistant", "content": role1_response})
            role2_history.append({"role": "user", "content": role1_response})
            conversation_history.append(role1_response)
        current_speaker = role2
    else:
        role2_response = create_chat_completion(model, role2_history, temperature)
        if role2_response:
            role2_response = role2_response.strip()
            role2_history.append({"role": "assistant", "content": role2_response})
            role1_history.append({"role": "user", "content": role2_response})
            conversation_history.append(role2_response)
        current_speaker = role1

    for i in range(max_rounds):
        if current_speaker == role1:
            role1_response = create_chat_completion(model, role1_history, temperature)
            if role1_response:
                role1_response = role1_response.strip()
                role1_history.append({"role": "assistant", "content": role1_response})
                role2_history.append({"role": "user", "content": role1_response})
                conversation_history.append(role1_response)
            current_speaker = role2
        else:
            role2_response = create_chat_completion(model, role2_history, temperature)
            if role2_response:
                role2_response = role2_response.strip()
                role2_history.append({"role": "assistant", "content": role2_response})
                role1_history.append({"role": "user", "content": role2_response})
                conversation_history.append(role2_response)
            current_speaker = role1

        # Check for end conditions
        if (role1_response and ("[DEAL REACHED]" in role1_response or "[NO DEAL]" in role1_response)) \
            or (role2_response and ("[DEAL REACHED]" in role2_response or "[NO DEAL]" in role2_response)) \
            or i == max_rounds - 1:
            
            # Administer SVI before ending
            role1_svi = administer_svi(role1_history, model, temperature)
            role2_svi = administer_svi(role2_history, model, temperature)
            if role1_svi:
                conversation_history.append(f"{role1}: {role1_svi}")
            if role2_svi:
                conversation_history.append(f"{role2}: {role2_svi}")
            break

    return conversation_history

# Wrapper for one row
def simulate_negotiation_wrapper(row):
    if pd.notna(row['conversation']):
        return row['conversation']

    # Determine which model to use for this negotiation
    # For now, use model_1 as the primary model for the negotiation
    model = row['model_1'] if pd.notna(row['model_1']) else 'gpt-4'

    # Handle empty prompt values
    prompt_1 = str(row['prompt_1']) if pd.notna(row['prompt_1']) and str(row['prompt_1']).strip() != '' else ''
    prompt_2 = str(row['prompt_2']) if pd.notna(row['prompt_2']) and str(row['prompt_2']).strip() != '' else ''

    for attempt in range(num_tries):
        try:
            return simulate_negotiation(
                role1_instructions=role1_instructions,
                role2_instructions=role2_instructions,
                role1_competition_prompt=prompt_prelude + prompt_1,
                role2_competition_prompt=prompt_prelude + prompt_2,
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
def main():
    # Check if any API keys are available
    if not any([openai_client, anthropic_client, gemini_key]):
        print("Error: No API keys found. Please set at least one of:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY") 
        print("  - GOOGLE_API_KEY")
        return
    
    # Generate output path based on sample path
    # Replace "samples/" with "negotiations/" and "dyads" with "negotiations"
    negotiation_path = sample_path.replace('samples/', 'negotiations/').replace('dyads', 'negotiations')
    
    # Load prompts - handle empty values properly
    df_prompts = pd.read_csv(sample_path)
    # Fill NaN values with empty strings for string columns, but preserve numeric columns as NaN
    string_columns = ['model_1', 'model_2', 'prompt_1', 'prompt_2']
    for col in string_columns:
        if col in df_prompts.columns:
            df_prompts[col] = df_prompts[col].fillna('')
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
    
    print(f"Total rows in file: {len(df_negotiations)}")
    print(f"Rows with missing conversations: {len(incomplete_rows)}")
    print(f"Number of chunks to process: {len(chunks)}")
    
    if len(incomplete_rows) == 0:
        print("No conversations to process. All conversations are already complete.")
        return

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

if __name__ == "__main__":
    main()