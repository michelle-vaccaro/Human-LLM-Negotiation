import csv
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os, re
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
from anthropic import Anthropic

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
        gemini_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self.gemini_client = genai
        
        # Initialize Anthropic client
        anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.anthropic_client = Anthropic(api_key=anthropic_key)
        
        # BFI-44 items
        # (question, dimension, reverse)
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
        
        # TKI items (full version with A/B choices)
        self.tki_items = [
            """A. There are times when I let others take responsibility for solving the problem.
    B. Rather than negotiate the things on which we disagree, I try to stress those things upon which we both agree.""",
            
            """A. I try to find a compromise solution.
    B. I attempt to deal with all of his/her and my concerns.""", 
            
            """A. I am usually firm in pursuing my goals.
    B. I might try to soothe the other's feelings and preserve our relationship.""", 
            
            """A. I try to find a compromise solution.
    B. I sometimes sacrifice my own wishes for the wishes of the other person.""", 
            
            """A. I consistently seek the other's help in working out a solution. 
    B. I try to do what is necessary to avoid useless tensions.""", 
            
            """A. I try to avoid creating unpleasantness for myself. 
    B. I try to win my position.""", 
            
            """A. I try to postpone the issue until I have had some time to think it over. 
    B. I give up some points in exchange for others.""", 
            
            """A. I am usually firm in pursuing my goals.
    B. I attempt to get all concerns and issues immediately out in the open.""", 
            
            """A. I feel that differences are not always worth worrying about. 
    B. I make some effort to get my way.""", 
            
            """A. I am firm in pursuing my goals.
    B. I try to find a compromise solution.""", 
            
            """A. I attempt to get all concerns and issues immediately out in the open.
    B. I might try to soothe the other's feelings and preserve our relationship.""", 
            
            """A. I sometimes avoid taking positions which would create controversy.
    B. I will let him have some of his positions if he lets me have some of mine.""", 
            
            """A. I propose a middle ground.
    B. I press to get my points made.""", 
            
            """A. I tell him my ideas and ask him for his.
    B. I try to show him the logic and benefits of my position.""", 
            
            """A. I might try to soothe the other's feelings and preserve our relationship.
    B. I try to do what is necessary to avoid tensions.""", 
            
            """A. I try not to hurt the other's feelings.
    B. I try to convince the other person of the merits of my position.""", 
            
            """A. I am usually firm in pursuing my goals.
    B. I try to do what is necessary to avoid useless tensions.""", 
            
            """A. If it makes the other person happy, I might let him maintain his views.
    B. I will let him have some of his positions if he lets me have some of mine.""", 
            
            """A. I attempt to get all concerns and issues immediately out in the open.
    B. I try to postpone the issue until I have had some time to think it over.""", 
            
            """A. I attempt to immediately work through our differences.
    B. I try to find a fair combination of gains and losses for both of us.""", 
            
            """A. In approaching negotiations, I try to be considerate of the other person's wishes.
    B. I always lean toward a direct discussion of the problem.""", 
            
            """A. I try to find a position that is intermediate between his and mine.
    B. I assert my wishes.""", 
            
            """A. I am very often concerned with satisfying all our wishes.
    B. There are times when I let others take responsibility for solving the problem.""", 
            
            """A. If the others position seems very important to him/her, I would try to meet his/her wishes.
    B. I try to get him to settle for a compromise.""", 
            
            """A. I try to show him the logic and benefits of my position.
    B. In approaching negotiations, I try to be considerate of the other person's wishes.""", 
            
            """A. I propose a middle ground.
    B. I am nearly always concerned with satisfying all our wishes.""", 
            
            """A. I sometimes avoid taking positions that would create controversy.
    B. If it makes the other person happy, I might let him maintain his views.""", 
            
            """A. I am usually firm in pursuing my goals.
    B. I usually seek the other's help in working out a solution.""", 
            
            """A. I propose a middle ground.
    B. I feel that differences are not always worth worrying about.""", 
            
            """A. I try not to hurt the other's feelings.
    B. I always share the problem with the other person so that we can work it out.""",    
        ]
        
        # IAS octants (full version with item numbers matching IAS-R scoring)
        # Item numbers correspond to the IAS-R scoring procedure
        self.ias_items = {
            "PA": ["Assertive", "Dominant", "Forceful", "Self-assured", "Domineering", "Firm", "Self-confident", "Persistent"],
            "BC": ["Boastful", "Tricky", "Calculating", "Cocky", "Wily", "Sly", "Cunning", "Crafty"],
            "DE": ["Ruthless", "Hardhearted", "Uncharitable", "Iron-hearted", "Warmthless", "Unsympathetic", "Coldhearted", "Cruel"],
            "FG": ["Unsparkling", "Unneighbourly", "Anti-social", "Dissocial", "Uncheery", "Distant", "Unsociable", "Introverted"],
            "HI": ["Timid", "Unaggressive", "Unbold", "Shy", "Meek", "Unauthoritative", "Forceless", "Bashful"],
            "JK": ["Unargumentative", "Uncunning", "Undemanding", "Unwily", "Unsly", "Uncalculating", "Uncrafty", "Boastless"],
            "LM": ["Soft-hearted", "Kind", "Tender", "Tenderhearted", "Accommodating", "Charitable", "Gentlehearted", "Sympathetic"],
            "NO": ["Cheerful", "Extraverted", "Perky", "Neighbourly", "Jovial", "Enthusiastic", "Friendly", "Outgoing"]
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
        prompt = """Here are a number of characteristics that may or may not apply to you. 
Please write a number next to each statement to indicate the extent to which you agree or disagree with that statement. 

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
        prompt = """Consider situations in which you find your wishes differing from those of another person. How do you usually respond to such situations?

On the following pages are several pairs of statements describing possible behavioral responses. For each pair, please circle the" A" or "B" statement which is most characteristic of your own behavior.

In many cases, neither the "A" nor the "B" statement may be very typical of your behavior; but please select the response which you would be more likely to use.

Please respond with ONLY the letter (A or B) for each item, separated by commas.

"""
        for i, statement_pair in enumerate(self.tki_items, 1):
            prompt += f"{i}. {statement_pair}\n"
            
        prompt += "\nYour response (e.g., A,B,A,B,A...):"
        return prompt
    
    def _create_ias_prompt(self) -> str:
        """Create IAS assessment prompt"""
        prompt = """This page lists words used to describe people’s personal characteristics.  Please rate how accurately each word describes you as a person.  Judge how accurately each word describes you on the following scale.  

1 = Extremely inaccurate
2 = Very inaccurate
3 = Quite inaccurate
4 = Slightly inaccurate
5 = Slightly accurate
6 = Quite accurate
7 = Very accurate
8 = Extremely accurate

Please respond with ONLY the number (1-8) for each item, separated by commas.

"""
        item_num = 1
        for octant, adjectives in self.ias_items.items():
            for adj in adjectives:
                prompt += f"{item_num}. {adj}\n"
                item_num += 1
                
        prompt += "\nYour response (e.g., 4,2,5,3,1...):"
        return prompt
    
    def get_agent_response(self, model: str, system_prompt: str, user_prompt: str, 
                          temperature: float = 0.0, max_retries: int = 2) -> Optional[str]:
        """Get response from the AI agent via API"""
        
        # Model provider mapping
        openai_models = ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", 
                        "gpt-4-turbo", "gpt-3.5-turbo"]
        anthropic_models = ['claude-3-haiku-20240307', 'claude-3-5-sonnet-20240620', 'claude-3-5-sonnet-20241022',
                            'claude-3-5-haiku-20241022', 'claude-3-7-sonnet-20250219', 'claude-sonnet-4-20250514',
                            'claude-opus-4-20250514']
        gemini_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro", "gemini-1.0-pro-vision",
                         "gemini-2.0-flash-live-001", "gemini-2.0-flash-lite", "gemini-2.0-flash",
                         "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
        
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
                        raise ValueError(f"Gemini client not initialized for model: {model}. Add GOOGLE_API_KEY to your .env file")
                    
                    # Create model instance
                    model_instance = self.gemini_client.GenerativeModel(model)
                    
                    # Combine system and user prompts for Gemini
                    full_prompt = f"{system_prompt}\n\n{user_prompt}"
                    
                    response = model_instance.generate_content(
                        full_prompt,
                        generation_config=self.gemini_client.GenerationConfig(
                            temperature=temperature,
                            # max_output_tokens=1000
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
    
    def parse_assessment_response(self, response_text: str, expected_items: int, assessment_type: str = None) -> Optional[List]:
        """Parse the agent's response into a list of responses (integers for BFI/IAS, strings for TKI)"""
        
        # For TKI, look for A/B responses
        if assessment_type == "TKI":
            # Look for comma-separated A/B responses
            ab_match = re.findall(r'\b[AaBb]\b', response_text)
            
            if len(ab_match) >= expected_items:
                # Take only the expected number of items and convert to uppercase
                return [ab.upper() for ab in ab_match[:expected_items]]
            
            # If not enough A/B responses found, try to parse line by line
            lines = response_text.strip().split('\n')
            parsed_responses = []
            
            for line in lines:
                # Look for patterns like "1. A" or "1: B" or just "A"
                match = re.search(r'(?:^\d+[\.\:\)\s]+)?([AaBb])\s*$', line.strip())
                if match:
                    parsed_responses.append(match.group(1).upper())
            
            if len(parsed_responses) == expected_items:
                return parsed_responses
            
            print(f"  Warning: Expected {expected_items} A/B responses but found {len(parsed_responses)}")
            return None
        
        # For BFI and IAS, look for numeric responses
        else:
            # Look for comma-separated numbers
            numbers_match = re.findall(r'\b[1-8]\b', response_text)
            
            if len(numbers_match) >= expected_items:
                # Take only the expected number of items
                return [int(n) for n in numbers_match[:expected_items]]
            
            # If not enough numbers found, try to parse line by line
            lines = response_text.strip().split('\n')
            parsed_responses = []
            
            for line in lines:
                # Look for patterns like "1. 5" or "1: 5" or just "5"
                match = re.search(r'(?:^\d+[\.\:\)\s]+)?([1-8])\s*$', line.strip())
                if match:
                    parsed_responses.append(int(match.group(1)))
            
            if len(parsed_responses) == expected_items:
                return parsed_responses
            
            print(f"  Warning: Expected {expected_items} numeric responses but found {len(parsed_responses)}")
            return None
    
    def score_bfi(self, responses: List[int]) -> Dict[str, float]:
        """Score BFI responses"""
        scores = {"E": [], "A": [], "C": [], "N": [], "O": []}
        
        for (item_id, (_, factor, reverse)), response in zip(sorted(self.bfi_items.items()), responses):
            score = response if not reverse else (6 - response)
            scores[factor].append(score)
        
        return {factor: sum(values) / len(values) for factor, values in scores.items()}
    
    def tki_scoring_util(self, a_indices, b_indices):
        """Utility function for TKI scoring"""
        rubric = np.array([""] * 30)
        rubric[a_indices] = "A"
        rubric[b_indices] = "B"
        return rubric

    def score_tki(self, responses: List[str]) -> Dict[str, float]:
        """Score TKI responses using the standard TKI scoring system"""
        # Convert responses to uppercase and validate
        answers = [x.upper()[0] for x in responses]
        assert set(answers) == {"A", "B"}, f"Invalid responses: {set(answers)}"
        
        # Define scoring indices for each conflict mode
        competing = sum(self.tki_scoring_util([2,7,9,16,24,27], [5,8,12,13,15,21]) == np.array(answers))
        collaborating = sum(self.tki_scoring_util([4,10,13,18,19,22], [1,7,20,25,27,29]) == np.array(answers))
        compromising = sum(self.tki_scoring_util([1,3,12,21,25,28], [6,9,11,17,19,23]) == np.array(answers))
        avoiding = sum(self.tki_scoring_util([0,5,6,8,11,26], [4,14,16,18,22,28]) == np.array(answers))
        accommodating = sum(self.tki_scoring_util([14,15,17,20,23,29], [0,2,3,10,24,26]) == np.array(answers))
        
        return {
            "competing": int(competing),
            "collaborating": int(collaborating), 
            "compromising": int(compromising),
            "avoiding": int(avoiding),
            "accommodating": int(accommodating)
        }

    def score_ias(self, responses: List[int]) -> Dict[str, float]:
        """Score IAS responses using the IAS-R scoring procedure"""
        # IAS-R norms (Total Sample means and standard deviations)
        ias_norms = {
            "PA": {"mean": 4.98, "sd": 0.97},
            "BC": {"mean": 3.77, "sd": 1.12},
            "DE": {"mean": 2.54, "sd": 0.85},
            "FG": {"mean": 3.35, "sd": 1.01},
            "HI": {"mean": 4.00, "sd": 1.06},
            "JK": {"mean": 4.46, "sd": 0.95},
            "LM": {"mean": 5.96, "sd": 0.81},
            "NO": {"mean": 5.59, "sd": 0.89}
        }
        
        # Calculate octant scores (mean of responses for each octant)
        octant_scores = {}
        response_idx = 0
        
        for octant, adjectives in self.ias_items.items():
            octant_sum = 0
            for _ in adjectives:
                octant_sum += responses[response_idx]
                response_idx += 1
            octant_scores[octant] = octant_sum / len(adjectives)
        
        # Convert octant scores to Z-scores
        octant_z_scores = {}
        for octant, score in octant_scores.items():
            norm = ias_norms[octant]
            z_score = (score - norm["mean"]) / norm["sd"]
            octant_z_scores[octant] = z_score
        
        # Calculate DOM and LOV factor scores using IAS-R formula
        # DOM = 0.03 * [(zPA - zHI) + 0.707(zNO + zBC - zFG - zJK)]
        # LOV = 0.03 * [(zLM - zDE) + 0.707(zNO - zBC - zFG + zJK)]
        
        zPA = octant_z_scores["PA"]
        zHI = octant_z_scores["HI"]
        zNO = octant_z_scores["NO"]
        zBC = octant_z_scores["BC"]
        zFG = octant_z_scores["FG"]
        zJK = octant_z_scores["JK"]
        zLM = octant_z_scores["LM"]
        zDE = octant_z_scores["DE"]
        
        dom_factor = 0.03 * ((zPA - zHI) + 0.707 * (zNO + zBC - zFG - zJK))
        lov_factor = 0.03 * ((zLM - zDE) + 0.707 * (zNO - zBC - zFG + zJK))
        
        # Calculate polar coordinates
        # ANGLE = tan^-1(zDOM / zLOV) with adjustments
        if lov_factor == 0:
            angle = 90 if dom_factor > 0 else 270
        else:
            angle = np.arctan(dom_factor / lov_factor) * 180 / np.pi
            
            # Apply angle adjustments
            if lov_factor < 0:
                angle += 180
            elif lov_factor > 0 and dom_factor < 0:
                angle += 360
        
        # VECTOR LENGTH = sqrt(zDOM^2 + zLOV^2)
        vector_length = np.sqrt(dom_factor**2 + lov_factor**2)
        
        # Convert to T-score: T = (vector_length * 10) + 50
        t_score = (vector_length * 10) + 50
        
        return {
            "dominance": float(dom_factor),
            "warmth": float(lov_factor),  # Using "warmth" instead of "love" as requested
            "octants": {k: float(v) for k, v in octant_scores.items()},
            "octant_z_scores": {k: float(v) for k, v in octant_z_scores.items()},
            "angle": float(angle),
            "vector_length": float(vector_length),
            "t_score": float(t_score)
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
        parsed_responses = self.parse_assessment_response(response_text, expected_items, assessment_type)
        
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