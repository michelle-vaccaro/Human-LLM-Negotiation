import csv
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Optional imports for other providers
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

class PersonalityAssessment:
    """Class to administer personality assessments to AI agents"""
    
    def __init__(self, openai_api_key: Optional[str] = None, 
                 gemini_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None):
        # Initialize API clients
        self.openai_client = None
        self.gemini_client = None
        self.anthropic_client = None
        
        # Initialize OpenAI client
        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
        
        # Initialize Gemini client
        if GEMINI_AVAILABLE:
            gemini_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
            if gemini_key:
                genai.configure(api_key=gemini_key)
                self.gemini_client = genai
        
        # Initialize Anthropic client
        if ANTHROPIC_AVAILABLE:
            anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key:
                self.anthropic_client = Anthropic(api_key=anthropic_key)
        
        # Verify at least one API key is available
        if not any([self.openai_client, self.gemini_client, self.anthropic_client]):
            raise ValueError(
                "No API keys found. Please either:\n"
                "1. Create a .env file with at least one of:\n"
                "   - OPENAI_API_KEY=your-key-here\n"
                "   - GOOGLE_API_KEY=your-key-here\n"
                "   - ANTHROPIC_API_KEY=your-key-here\n"
                "2. Set the corresponding environment variables\n"
                "3. Pass the api_key parameters when initializing"
            )
        
        # BFI-44 items (abbreviated for space - full version would have all 44)
        self.bfi_items = {
            # Extraversion
            1: ("Is talkative", "E", False),
            6: ("Is reserved", "E", True),
            11: ("Is full of energy", "E", False),
            16: ("Generates a lot of enthusiasm", "E", False),
            21: ("Tends to be quiet", "E", True),
            26: ("Has an assertive personality", "E", False),
            31: ("Is sometimes shy, inhibited", "E", True),
            36: ("Is outgoing, sociable", "E", False),
            
            # Agreeableness
            2: ("Tends to find fault with others", "A", True),
            7: ("Is helpful and unselfish with others", "A", False),
            12: ("Starts quarrels with others", "A", True),
            17: ("Has a forgiving nature", "A", False),
            22: ("Is generally trusting", "A", False),
            27: ("Can be cold and aloof", "A", True),
            32: ("Is considerate and kind to almost everyone", "A", False),
            37: ("Is sometimes rude to others", "A", True),
            42: ("Likes to cooperate with others", "A", False),
            
            # Conscientiousness
            3: ("Does a thorough job", "C", False),
            8: ("Can be somewhat careless", "C", True),
            13: ("Is a reliable worker", "C", False),
            18: ("Tends to be disorganized", "C", True),
            23: ("Tends to be lazy", "C", True),
            28: ("Perseveres until the task is finished", "C", False),
            33: ("Does things efficiently", "C", False),
            38: ("Makes plans and follows through with them", "C", False),
            43: ("Is easily distracted", "C", True),
            
            # Neuroticism
            4: ("Is depressed, blue", "N", False),
            9: ("Is relaxed, handles stress well", "N", True),
            14: ("Can be tense", "N", False),
            19: ("Worries a lot", "N", False),
            24: ("Is emotionally stable, not easily upset", "N", True),
            29: ("Can be moody", "N", False),
            34: ("Remains calm in tense situations", "N", True),
            39: ("Gets nervous easily", "N", False),
            
            # Openness
            5: ("Is original, comes up with new ideas", "O", False),
            10: ("Is curious about many different things", "O", False),
            15: ("Is ingenious, a deep thinker", "O", False),
            20: ("Has an active imagination", "O", False),
            25: ("Is inventive", "O", False),
            30: ("Values artistic, aesthetic experiences", "O", False),
            35: ("Prefers work that is routine", "O", True),
            40: ("Likes to reflect, play with ideas", "O", False),
            41: ("Has few artistic interests", "O", True),
            44: ("Is sophisticated in art, music, or literature", "O", False)
        }
        
        # TKI items (simplified version)
        self.tki_items = [
            ("I try to find a compromise solution", "Compromising"),
            ("I attempt to deal with all of the other's and my concerns", "Collaborating"),
            ("I am usually firm in pursuing my goals", "Competing"),
            ("I might try to soothe the other's feelings and preserve our relationship", "Accommodating"),
            ("I try to avoid creating unpleasantness for myself", "Avoiding"),
            ("I try to win my position", "Competing"),
            ("I try to postpone the issue until I have had some time to think it over", "Avoiding"),
            ("I give up some points in exchange for others", "Compromising"),
            ("I am usually firm in pursuing my goals", "Competing"),
            ("I tell the other my ideas and ask for theirs", "Collaborating"),
            ("I try to show the logic and benefits of my position", "Competing"),
            ("I might try to soothe the other's feelings and preserve our relationship", "Accommodating"),
            ("I attempt to get all concerns and issues immediately out in the open", "Collaborating"),
            ("I try to do what is necessary to avoid tensions", "Avoiding"),
            ("I try not to hurt the other's feelings", "Accommodating"),
        ]
        
        # IAS octants
        self.ias_items = {
            "PA": ["Leading", "Dominant", "Assertive", "Self-assured"],
            "BC": ["Arrogant", "Calculating", "Cocky", "Boastful"],
            "DE": ["Cold-hearted", "Cruel", "Hostile", "Rebellious"],
            "FG": ["Aloof", "Introverted", "Reserved", "Withdrawn"],
            "HI": ["Unassured", "Submissive", "Timid", "Meek"],
            "JK": ["Agreeable", "Warm-hearted", "Tender", "Kind"],
            "LM": ["Sociable", "Outgoing", "Enthusiastic", "Cheerful"],
            "NO": ["Self-confident", "Independent", "Persistent", "Self-reliant"]
        }
    
    def create_assessment_prompt(self, assessment_type: str, persona_prompt: str) -> str:
        """Create a prompt for the agent to complete an assessment"""        
        if assessment_type == "BFI":
            return self._create_bfi_prompt()
        elif assessment_type == "TKI":
            return self._create_tki_prompt()
        elif assessment_type == "IAS":
            # return persona_prompt + self._create_ias_prompt()
            return self._create_ias_prompt()
    
    def _create_bfi_prompt(self) -> str:
        """Create BFI assessment prompt"""
        prompt = """Below are characteristics that may or may not apply to you. 
For each statement, indicate the extent to which you agree or disagree using this scale:
1 = Disagree strongly
2 = Disagree a little
3 = Neither agree nor disagree
4 = Agree a little
5 = Agree strongly

Please respond with ONLY the number (1-5) for each item, separated by commas.

I see myself as someone who...\n"""
        
        for item_id, (statement, _, _) in sorted(self.bfi_items.items()):
            prompt += f"{item_id}. {statement}\n"
            
        prompt += "\nYour response (e.g., 5,3,4,2,1,3...):"
        return prompt
    
    def _create_tki_prompt(self) -> str:
        """Create TKI assessment prompt"""
        prompt = """For each statement below, rate how likely you are to engage in this behavior 
when facing a conflict situation, using this scale:
1 = Never
2 = Rarely
3 = Sometimes
4 = Often
5 = Always

Please respond with ONLY the number (1-5) for each item, separated by commas.

"""
        for i, (statement, _) in enumerate(self.tki_items, 1):
            prompt += f"{i}. {statement}\n"
            
        prompt += "\nYour response (e.g., 3,4,2,5,1...):"
        return prompt
    
    def _create_ias_prompt(self) -> str:
        """Create IAS assessment prompt"""
        prompt = """Below are adjectives that may or may not describe you.
For each adjective, rate how accurately it describes you using this scale:
1 = Very inaccurate
2 = Moderately inaccurate
3 = Neither accurate nor inaccurate
4 = Moderately accurate
5 = Very accurate

Please respond with ONLY the number (1-5) for each item, separated by commas.

"""
        item_num = 1
        for octant, adjectives in self.ias_items.items():
            for adj in adjectives:
                prompt += f"{item_num}. {adj}\n"
                item_num += 1
                
        prompt += "\nYour response (e.g., 4,2,5,3,1...):"
        return prompt
    
    def get_agent_response(self, model: str, system_prompt: str, user_prompt: str, 
                          temperature: float = 0.0, max_retries: int = 3) -> Optional[str]:
        """Get response from the AI agent via API"""
        
        # Model provider mapping
        openai_models = ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", 
                        "gpt-4-turbo", "gpt-3.5-turbo"]
        anthropic_models = ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229", 
                           "claude-3-sonnet-20240229", "claude-3-haiku-20240229", "claude-2.1", "claude-2.0", 
                           "claude-instant-1.2"]
        gemini_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro", "gemini-1.0-pro-vision"]
        
        # Models that don't support temperature parameter
        models_without_temperature = ["o3", "o3-mini", "o1", "o4-mini", "o4"]
        
        for attempt in range(max_retries):
            try:
                # Determine provider and call appropriate API
                if model in openai_models or model in models_without_temperature:
                    if not self.openai_client:
                        raise ValueError(f"OpenAI client not initialized for model: {model}")
                    
                    if model in models_without_temperature:
                        response = self.openai_client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            n=1
                        )
                    else:
                        response = self.openai_client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=temperature,
                            n=1
                        )
                    return response.choices[0].message.content.strip()
                
                elif model in anthropic_models:
                    if not self.anthropic_client:
                        if not ANTHROPIC_AVAILABLE:
                            raise ValueError(f"Anthropic package not available for model: {model}. Install with: pip install anthropic")
                        else:
                            raise ValueError(f"Anthropic client not initialized for model: {model}. Add ANTHROPIC_API_KEY to your .env file")
                    
                    response = self.anthropic_client.messages.create(
                        model=model,
                        max_tokens=1000,
                        temperature=temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                    return response.content[0].text.strip()
                
                elif model in gemini_models:
                    if not self.gemini_client:
                        if not GEMINI_AVAILABLE:
                            raise ValueError(f"Google Generative AI package not available for model: {model}. Install with: pip install google-generativeai")
                        else:
                            raise ValueError(f"Gemini client not initialized for model: {model}. Add GOOGLE_API_KEY to your .env file")
                    
                    # Create model instance
                    model_instance = self.gemini_client.GenerativeModel(model)
                    
                    # Combine system and user prompts for Gemini
                    full_prompt = f"{system_prompt}\n\n{user_prompt}"
                    
                    response = model_instance.generate_content(
                        full_prompt,
                        generation_config=self.gemini_client.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=1000
                        )
                    )
                    return response.text.strip()
                
                else:
                    raise ValueError(f"Unknown model: {model}. Supported models: {openai_models + anthropic_models + gemini_models}")
                
            except Exception as e:
                print(f"  API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return None
    
    def parse_assessment_response(self, response_text: str, expected_items: int) -> Optional[List[int]]:
        """Parse the agent's response into a list of integers"""
        
        # Try to extract numbers from the response
        # First, try to find a comma-separated list
        import re
        
        # Look for comma-separated numbers
        numbers_match = re.findall(r'\b[1-5]\b', response_text)
        
        if len(numbers_match) >= expected_items:
            # Take only the expected number of items
            return [int(n) for n in numbers_match[:expected_items]]
        
        # If not enough numbers found, try to parse line by line
        lines = response_text.strip().split('\n')
        parsed_responses = []
        
        for line in lines:
            # Look for patterns like "1. 5" or "1: 5" or just "5"
            match = re.search(r'(?:^\d+[\.\:\)\s]+)?([1-5])\s*$', line.strip())
            if match:
                parsed_responses.append(int(match.group(1)))
        
        if len(parsed_responses) == expected_items:
            return parsed_responses
        
        print(f"  Warning: Expected {expected_items} responses but found {len(parsed_responses)}")
        return None
    
    def score_bfi(self, responses: List[int]) -> Dict[str, float]:
        """Score BFI responses"""
        scores = {"E": [], "A": [], "C": [], "N": [], "O": []}
        
        for (item_id, (_, factor, reverse)), response in zip(sorted(self.bfi_items.items()), responses):
            score = response if not reverse else (6 - response)
            scores[factor].append(score)
        
        return {factor: sum(values) / len(values) for factor, values in scores.items()}
    
    def score_tki(self, responses: List[int]) -> Dict[str, float]:
        """Score TKI responses"""
        mode_scores = {"Competing": 0, "Collaborating": 0, "Compromising": 0, 
                      "Avoiding": 0, "Accommodating": 0}
        
        for (_, mode), response in zip(self.tki_items, responses):
            mode_scores[mode] += response
            
        # Normalize scores
        total = sum(mode_scores.values())
        return {mode: score / total * 100 for mode, score in mode_scores.items()}

    def score_ias(self, responses: List[int]) -> Dict[str, float]:
        """Score IAS responses"""
        octant_scores = {}
        response_idx = 0
        
        for octant, adjectives in self.ias_items.items():
            octant_sum = 0
            for _ in adjectives:
                octant_sum += responses[response_idx]
                response_idx += 1
            octant_scores[octant] = octant_sum / len(adjectives)
        
        # IAS interpersonal circumplex scoring
        # Dominance (vertical axis): Dominant-Submissive
        # Warmth (horizontal axis): Friendly-Hostile
        
        # Calculate based on octant positions in the circumplex
        # PA (90°): Dominant
        # BC (45°): Dominant-Hostile  
        # DE (0°): Hostile
        # FG (-45°): Submissive-Hostile
        # HI (-90°): Submissive
        # JK (-135°): Submissive-Friendly
        # LM (180°): Friendly
        # NO (135°): Dominant-Friendly
        
        # Dominance calculation (vertical component)
        dom_raw = (
            octant_scores["PA"] * 1.0 +  # Full dominance
            octant_scores["NO"] * 0.707 +  # 45° angle
            octant_scores["BC"] * 0.707 +  # 45° angle
            octant_scores["LM"] * 0.0 +  # Neutral
            octant_scores["DE"] * 0.0 +  # Neutral
            octant_scores["JK"] * -0.707 +  # -45° angle
            octant_scores["FG"] * -0.707 +  # -45° angle
            octant_scores["HI"] * -1.0  # Full submission
        )
        
        # Warmth calculation (horizontal component)
        warm_raw = (
            octant_scores["LM"] * 1.0 +  # Full warmth
            octant_scores["NO"] * 0.707 +  # 45° angle
            octant_scores["JK"] * 0.707 +  # 45° angle
            octant_scores["PA"] * 0.0 +  # Neutral
            octant_scores["HI"] * 0.0 +  # Neutral
            octant_scores["BC"] * -0.707 +  # -45° angle
            octant_scores["FG"] * -0.707 +  # -45° angle
            octant_scores["DE"] * -1.0  # Full coldness
        )
        
        # Normalize to -100-100 scale
        dom_score = dom_raw
        warm_score = warm_raw

        # Map to 0-100 where 50 is neutral
        # dom_score = 50 + (dom_raw * 12.5)
        # warm_score = 50 + (warm_raw * 12.5)
        
        return {
            "dominance": dom_score,
            "warmth": warm_score,
            "octants": octant_scores,
            "raw_dominance": dom_raw,
            "raw_warmth": warm_raw
        }
    
    def administer_single_assessment(self, model: str, persona_prompt: str, 
                                   assessment_type: str) -> Dict:
        """Administer a single assessment to an agent and return results"""
        
        # Create assessment prompt
        assessment_prompt = self.create_assessment_prompt(assessment_type, persona_prompt)
        
        # Get expected number of items
        if assessment_type == "BFI":
            expected_items = len(self.bfi_items)
        elif assessment_type == "TKI":
            expected_items = len(self.tki_items)
        elif assessment_type == "IAS":
            expected_items = sum(len(adjs) for adjs in self.ias_items.values())
        
        print(f"    Getting {assessment_type} responses from API...")
        
        # Get response from agent
        response_text = self.get_agent_response(
            model=model,
            system_prompt=persona_prompt,
            user_prompt=assessment_prompt,
            temperature=0.0  # Lower temperature for more consistent responses
        )
        
        if not response_text:
            return {
                "status": "error",
                "error": "Failed to get response from API",
                "raw_response": None,
                "scores": None
            }
        
        # Parse response
        parsed_responses = self.parse_assessment_response(response_text, expected_items)
        
        if not parsed_responses:
            return {
                "status": "error",
                "error": "Failed to parse response",
                "raw_response": response_text,
                "scores": None
            }
        
        # Score the assessment
        try:
            if assessment_type == "BFI":
                scores = self.score_bfi(parsed_responses)
            elif assessment_type == "TKI":
                scores = self.score_tki(parsed_responses)
            elif assessment_type == "IAS":
                scores = self.score_ias(parsed_responses)
            
            return {
                "status": "success",
                "raw_response": response_text,
                "parsed_responses": parsed_responses,
                "scores": scores
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Scoring failed: {str(e)}",
                "raw_response": response_text,
                "parsed_responses": parsed_responses,
                "scores": None
            }

def create_env_template():
    """Create a .env template file if it doesn't exist"""
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("# API Configuration\n")
            f.write("# Add at least one of the following API keys:\n")
            f.write("\n# OpenAI API Configuration\n")
            f.write("OPENAI_API_KEY=your-openai-api-key-here\n")
            f.write("\n# Google Gemini API Configuration\n")
            f.write("GOOGLE_API_KEY=your-google-api-key-here\n")
            f.write("\n# Anthropic Claude API Configuration\n")
            f.write("ANTHROPIC_API_KEY=your-anthropic-api-key-here\n")
            f.write("\n# Optional: Override the default output directory\n")
            f.write("# OUTPUT_DIR=assessment_results\n")
        print("Created .env template file. Please add at least one API key.")
        return False
    return True

def administer_assessments(csv_file: str, output_dir: Optional[str] = None, 
                          openai_api_key: Optional[str] = None,
                          gemini_api_key: Optional[str] = None,
                          anthropic_api_key: Optional[str] = None):
    """Main function to administer assessments to agents from CSV"""
    
    # Use output directory from .env if not specified
    if output_dir is None:
        output_dir = os.getenv("OUTPUT_DIR", "assessment_results")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize assessment tool
        assessor = PersonalityAssessment(
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            anthropic_api_key=anthropic_api_key
        )
        
        # Print available providers
        print("Available providers:")
        if assessor.openai_client:
            print("  ✓ OpenAI")
        if assessor.anthropic_client:
            print("  ✓ Anthropic")
        if assessor.gemini_client:
            print("  ✓ Google Gemini")
        print()
        
    except ValueError as e:
        print(f"Error: {e}")
        create_env_template()
        return
    
    # Read CSV file
    results = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            agent_id = f"{row['model']}_dom{row['dominance_score']}_warm{row['warmth_score']}"
            print(f"\nProcessing agent: {agent_id}")
            
            agent_results = {
                "agent_id": agent_id,
                "model": row['model'],
                "target_dominance": int(row['dominance_score']) if row['dominance_score'] != '' else None,
                "target_warmth": int(row['warmth_score']) if row['warmth_score'] != '' else None,
                "prompt": row['prompt'],
                "assessments": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # For each assessment type
            for assessment in ["BFI", "TKI", "IAS"]:
                print(f"  Administering {assessment}...")
                
                # Administer the assessment and get results
                assessment_result = assessor.administer_single_assessment(
                    model=row['model'],
                    persona_prompt=row['prompt'],
                    assessment_type=assessment
                )
                
                # Save raw response to file
                response_file = os.path.join(output_dir, f"{agent_id}_{assessment}_response.txt")
                if assessment_result.get('raw_response'):
                    with open(response_file, 'w', encoding='utf-8') as f:
                        f.write(assessment_result['raw_response'])
                
                # Save parsed responses if available
                if assessment_result.get('parsed_responses'):
                    parsed_file = os.path.join(output_dir, f"{agent_id}_{assessment}_parsed.txt")
                    with open(parsed_file, 'w', encoding='utf-8') as f:
                        f.write(','.join(map(str, assessment_result['parsed_responses'])))
                
                # Store results
                agent_results["assessments"][assessment] = assessment_result
                
                # Print summary
                if assessment_result['status'] == 'success':
                    print(f"    ✓ {assessment} completed successfully")
                    if assessment == "IAS" and assessment_result.get('scores'):
                        scores = assessment_result['scores']
                        print(f"      Measured Dominance: {scores['dominance']:.1f} (target: {row['dominance_score']})")
                        print(f"      Measured Warmth: {scores['warmth']:.1f} (target: {row['warmth_score']})")
                else:
                    print(f"    ✗ {assessment} failed: {assessment_result.get('error', 'Unknown error')}")
                
                # Rate limiting
                time.sleep(1)
            
            results.append(agent_results)
    
    # Save comprehensive results
    results_file = os.path.join(output_dir, "assessment_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    generate_summary_report(results, output_dir)
    
    print(f"\n\nAssessment administration complete!")
    print(f"Results saved to: {results_file}")
    print(f"Summary report saved to: {os.path.join(output_dir, 'summary_report.txt')}")

def generate_summary_report(results: List[Dict], output_dir: str):
    """Generate a summary report of all assessments"""
    
    report_lines = []
    report_lines.append("PERSONALITY ASSESSMENT SUMMARY REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total agents assessed: {len(results)}")
    report_lines.append("")
    
    for agent in results:
        report_lines.append(f"\nAgent: {agent['agent_id']}")
        report_lines.append(f"Model: {agent['model']}")
        report_lines.append(f"Target - Dominance: {agent['target_dominance']}, Warmth: {agent['target_warmth']}")
        report_lines.append("-" * 30)
        
        # BFI Results
        bfi = agent['assessments'].get('BFI', {})
        if bfi.get('status') == 'success' and bfi.get('scores'):
            scores = bfi['scores']
            report_lines.append("BFI Scores:")
            report_lines.append(f"  Extraversion:     {scores['E']:.2f}")
            report_lines.append(f"  Agreeableness:    {scores['A']:.2f}")
            report_lines.append(f"  Conscientiousness: {scores['C']:.2f}")
            report_lines.append(f"  Neuroticism:      {scores['N']:.2f}")
            report_lines.append(f"  Openness:         {scores['O']:.2f}")
        else:
            report_lines.append("BFI: Failed")
        
        # TKI Results
        tki = agent['assessments'].get('TKI', {})
        if tki.get('status') == 'success' and tki.get('scores'):
            scores = tki['scores']
            report_lines.append("\nTKI Conflict Modes:")
            for mode, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"  {mode:15} {score:5.1f}%")
        else:
            report_lines.append("\nTKI: Failed")
        
        # IAS Results
        ias = agent['assessments'].get('IAS', {})
        if ias.get('status') == 'success' and ias.get('scores'):
            scores = ias['scores']
            report_lines.append("\nIAS Interpersonal Scores:")
            report_lines.append(f"  Measured Dominance: {scores['dominance']:5.1f} (target: {agent['target_dominance']})")
            report_lines.append(f"  Measured Warmth:    {scores['warmth']:5.1f} (target: {agent['target_warmth']})")
            
            # Calculate errors only if target values are not None
            if agent['target_dominance'] is not None:
                dominance_error = abs(scores['dominance'] - agent['target_dominance'])
                report_lines.append(f"  Dominance Error:    {dominance_error:5.1f}")
            else:
                report_lines.append(f"  Dominance Error:    N/A (no target)")
                
            if agent['target_warmth'] is not None:
                warmth_error = abs(scores['warmth'] - agent['target_warmth'])
                report_lines.append(f"  Warmth Error:       {warmth_error:5.1f}")
            else:
                report_lines.append(f"  Warmth Error:       N/A (no target)")
        else:
            report_lines.append("\nIAS: Failed")
        
        report_lines.append("")
    
    # Save report
    report_file = os.path.join(output_dir, "summary_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

if __name__ == "__main__":
    # Example usage
    import sys
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        create_env_template()
        print("\nPlease edit the .env file and add at least one API key.")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        #csv_file = "samples/samples_n1.csv"
        csv_file = "samples/samples_all_providers.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        sys.exit(1)
    
    administer_assessments(csv_file)