# Human-LLM Negotiation

A comprehensive research toolkit for studying negotiation dynamics between humans and Large Language Models (LLMs). This repository contains tools for administering personality assessments, running negotiation simulations, and analyzing outcomes across multiple AI providers.

## 🚀 Features

### Multi-Provider AI Support
- **OpenAI**: GPT-4, GPT-4o, GPT-3.5-turbo, and Anthropic models via OpenAI API
- **Anthropic**: Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus
- **Google Gemini**: Gemini 1.5 Pro, Gemini 1.5 Flash

### Personality Assessment Tools
- **BFI (Big Five Inventory)**: Measures the five personality dimensions
- **TKI (Thomas-Kilmann Conflict Mode)**: Measures conflict resolution styles  
- **IAS (Interpersonal Adjective Scales)**: Measures interpersonal dominance and warmth

### Negotiation Simulation
- Multi-round negotiation scenarios
- Role-based instructions for different negotiation contexts
- Outcome analysis and deal parsing

## 📁 Repository Structure

```
Human-LLM Negotiation/
├── personality_assessments.py    # Multi-provider personality assessment tool
├── simulation.py                 # Negotiation simulation framework
├── parse_deal.py                 # Deal outcome parsing and analysis
├── clean_outcomes.py             # Data cleaning utilities
├── samples/                      # Sample CSV files for testing
├── prompts/                      # Negotiation prompts and instructions
├── assessment_results/           # Generated assessment results
├── negotiations/                 # Negotiation data
├── outcomes/                     # Analysis outcomes
├── figures/                      # Generated visualizations
└── requirements.txt              # Python dependencies
```

## 🛠️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file with your API keys:
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Google Gemini API Configuration  
GOOGLE_API_KEY=your-google-api-key-here

# Anthropic Claude API Configuration
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

**Note**: You only need the API keys for the providers you want to use.

### 3. Optional Dependencies
For additional providers, install:
```bash
pip install google-generativeai anthropic
```

## 🧪 Usage

### Personality Assessments

Run personality assessments on AI models:

```bash
# Use default sample file
python personality_assessments.py

# Use custom CSV file
python personality_assessments.py samples/samples_all_providers.csv
```

**CSV Format:**
```csv
model,warmth_score,dominance_score,prompt
gpt-4o,,,You are a helpful AI assistant.
claude-3-5-sonnet-20241022,,,You are a helpful AI assistant.
gemini-1.5-pro,,,You are a helpful AI assistant.
```

### Programmatic Usage

```python
from personality_assessments import PersonalityAssessment

# Initialize with multiple providers
assessor = PersonalityAssessment(
    openai_api_key="your-openai-key",
    gemini_api_key="your-google-key", 
    anthropic_api_key="your-anthropic-key"
)

# Administer assessment
result = assessor.administer_single_assessment(
    model="gpt-4o",
    persona_prompt="You are a helpful assistant.",
    assessment_type="BFI"
)
```

## 📊 Output

Results are saved to the `assessment_results/` directory:
- **Raw API responses**: `{agent_id}_{assessment}_response.txt`
- **Parsed responses**: `{agent_id}_{assessment}_parsed.txt`
- **Summary report**: `summary_report.txt`
- **Complete results**: `assessment_results.json`

## 🔬 Research Applications

This toolkit enables research on:
- **Personality differences** across AI models
- **Negotiation strategies** and outcomes
- **Human-AI interaction patterns**
- **Model behavior consistency** across different contexts

## 📈 Supported Models

### OpenAI Models
- `gpt-4`, `gpt-4o`, `gpt-4o-mini`
- `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- `gpt-4-turbo`, `gpt-3.5-turbo`
- `o3`, `o3-mini`, `o1`, `o4-mini`, `o4`

### Anthropic Claude Models
- `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`
- `claude-3-opus-20240229`, `claude-3-sonnet-20240229`
- `claude-3-haiku-20240229`, `claude-2.1`, `claude-2.0`

### Google Gemini Models
- `gemini-1.5-pro`, `gemini-1.5-flash`
- `gemini-1.0-pro`, `gemini-1.0-pro-vision`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Big Five Inventory (BFI) for personality assessment
- Thomas-Kilmann Conflict Mode Instrument (TKI)
- Interpersonal Adjective Scales (IAS)
- OpenAI, Anthropic, and Google for their AI APIs 