import pandas as pd
import itertools
import argparse
from pathlib import Path
import random

def load_agents_from_csv(file_path):
    """Load agents from CSV file."""
    df = pd.read_csv(file_path)
    
    # Extract unique model names (agents)
    agents = df['model'].dropna().unique().tolist()
    
    print(f"Loaded {len(agents)} unique agents from {file_path}")
    print(f"Agents: {agents}")
    
    return df

def generate_all_pairings(agents, n_iter=1, include_self_pairs=False):
    """
    Generate all possible pairings of agents.
    
    Args:
        agents (list): List of agent names
        n_iter (int): Number of iterations for each pairing
        include_self_pairs (bool): Whether to include pairs of an agent with itself
    
    Returns:
        list: List of tuples containing (agent1, agent2, iteration_number)
    """
    pairings = []
    
    if include_self_pairs:
        # Include pairs of an agent with itself
        for agent in agents:
            for i in range(n_iter):
                pairings.append((agent, agent, i + 1))
    
    # Generate all possible pairs of different agents
    for agent1, agent2 in itertools.combinations(agents, 2):
        for i in range(n_iter):
            pairings.append((agent1, agent2, i + 1))
    
    return pairings

def generate_all_permutations(df, n_iter=1, include_self_pairs=True):
    """
    Generate all possible permutations of agents (including both directions).
    
    Args:
        df (DataFrame): DataFrame containing agent data
        n_iter (int): Number of iterations for each pairing
        include_self_pairs (bool): Whether to include pairs of an agent with itself
    
    Returns:
        list: List of dictionaries containing pairing data
    """
    agents = df['model'].dropna().unique().tolist()
    pairings = []
    
    if include_self_pairs:
        # Include pairs of an agent with itself
        for agent in agents:
            agent_data = df[df['model'] == agent].iloc[0]
            for i in range(n_iter):
                pairing = {
                    'model_1': agent,
                    'warmth_score_1': agent_data['warmth_score'],
                    'dominance_score_1': agent_data['dominance_score'],
                    'prompt_1': agent_data['prompt'],
                    'model_2': agent,
                    'warmth_score_2': agent_data['warmth_score'],
                    'dominance_score_2': agent_data['dominance_score'],
                    'prompt_2': agent_data['prompt'],
                    'iteration': i + 1
                }
                pairings.append(pairing)
    
    # Generate all possible permutations of different agents
    for agent1, agent2 in itertools.permutations(agents, 2):
        agent1_data = df[df['model'] == agent1].iloc[0]
        agent2_data = df[df['model'] == agent2].iloc[0]
        for i in range(n_iter):
            pairing = {
                'model_1': agent1,
                'warmth_score_1': agent1_data['warmth_score'],
                'dominance_score_1': agent1_data['dominance_score'],
                'prompt_1': agent1_data['prompt'],
                'model_2': agent2,
                'warmth_score_2': agent2_data['warmth_score'],
                'dominance_score_2': agent2_data['dominance_score'],
                'prompt_2': agent2_data['prompt'],
                'iteration': i + 1
            }
            pairings.append(pairing)
    
    return pairings

def save_pairings_to_csv(pairings, output_file):
    """Save pairings to CSV file."""
    df = pd.DataFrame(pairings)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(pairings)} pairings to {output_file}")

def print_pairing_summary(pairings, agents):
    """Print a summary of the generated pairings."""
    print(f"\nPairing Summary:")
    print(f"Total pairings: {len(pairings)}")
    print(f"Number of agents: {len(agents)}")
    
    # Count unique pairs (excluding iterations)
    unique_pairs = set()
    for pairing in pairings:
        agent1, agent2 = pairing['model_1'], pairing['model_2']
        if agent1 <= agent2:  # Sort to avoid counting (A,B) and (B,A) as different
            unique_pairs.add((agent1, agent2))
        else:
            unique_pairs.add((agent2, agent1))
    
    print(f"Unique pairs: {len(unique_pairs)}")
    
    # Show some examples
    print(f"\nExample pairings:")
    for i, pairing in enumerate(pairings[:10]):
        print(f"  {i+1}. {pairing['model_1']} vs {pairing['model_2']} (iteration {pairing['iteration']})")
    
    if len(pairings) > 10:
        print(f"  ... and {len(pairings) - 10} more pairings")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate all possible agent pairings')
    parser.add_argument('input_file', help='Path to CSV file containing agents')
    parser.add_argument('--n_iter', type=int, default=1, help='Number of iterations for each pairing (default: 1)')
    parser.add_argument('--output', help='Output CSV file path (default: samples/dyads_nX.csv)')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the final list of pairings')
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found!")
        return
    
    # Load agents
    df = load_agents_from_csv(input_path)
    agents = df['model'].dropna().unique().tolist()
    
    if len(agents) < 2:
        print("Error: Need at least 2 agents to generate pairings!")
        return
    
    # Generate pairings (only permutations with self-pairs included)
    pairings = generate_all_permutations(df, args.n_iter, include_self_pairs=True)
    
    # Shuffle if requested
    if args.shuffle:
        random.shuffle(pairings)
        print("Pairings have been shuffled")
    
    # Set output file path
    if args.output:
        output_file = args.output
    else:
        output_file = f"samples/dyads_n{args.n_iter}.csv"
    
    # Save pairings
    save_pairings_to_csv(pairings, output_file)
    
    # Print summary
    print_pairing_summary(pairings, agents)
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"  Pairing type: permutations")
    print(f"  Iterations per pair: {args.n_iter}")
    print(f"  Self-pairs included: True")
    
    expected_pairs = len(agents) * len(agents) * args.n_iter
    print(f"  Expected pairs: {expected_pairs}")

if __name__ == "__main__":
    main() 