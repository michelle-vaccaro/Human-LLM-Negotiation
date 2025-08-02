# Multi-Provider Personality Assessment Tool

This tool now supports multiple AI providers for administering personality assessments to AI agents.

## Supported Providers

### OpenAI Models
- `gpt-4`, `gpt-4o`, `gpt-4o-mini`
- `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- `gpt-4-turbo`, `gpt-3.5-turbo`
- `o3`, `o3-mini`, `o1`, `o4-mini`, `o4` (Anthropic models via OpenAI API)

### Anthropic Claude Models
- `claude-3-5-sonnet`, `claude-3-5-haiku`
- `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- `claude-2.1`, `claude-2.0`, `claude-instant-1.2`

### Google Gemini Models
- `gemini-1.5-pro`, `gemini-1.5-flash`
- `gemini-1.0-pro`, `gemini-1.0-pro-vision`

## Setup

1. **Install dependencies:**
   ```bash
   pip install openai python-dotenv
   ```

2. **For additional providers (optional):**
   ```bash
   pip install google-generativeai anthropic
   ```

3. **Configure API keys:**
   Create a `.env` file with at least one of the following:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   GOOGLE_API_KEY=your-google-api-key-here
   ANTHROPIC_API_KEY=your-anthropic-api-key-here
   ```

## Usage

### Basic Usage
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

### Command Line Usage
```bash
# Run with default CSV file
python personality_assessments.py

# Run with custom CSV file
python personality_assessments.py samples/samples_multi_provider.csv
```

## CSV Format

Create a CSV file with the following columns:
- `model`: The model name (e.g., "gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro")
- `warmth_score`: Target warmth score (optional)
- `dominance_score`: Target dominance score (optional)  
- `prompt`: The persona prompt for the AI agent

Example:
```csv
model,warmth_score,dominance_score,prompt
gpt-4o,,,You are a helpful AI assistant.
claude-3-5-sonnet,,,You are a helpful AI assistant.
gemini-1.5-pro,,,You are a helpful AI assistant.
```

## Assessment Types

The tool administers three personality assessments:

1. **BFI (Big Five Inventory)**: Measures the five personality dimensions
2. **TKI (Thomas-Kilmann Conflict Mode)**: Measures conflict resolution styles
3. **IAS (Interpersonal Adjective Scales)**: Measures interpersonal dominance and warmth

## Output

Results are saved to the `assessment_results/` directory:
- Raw API responses: `{agent_id}_{assessment}_response.txt`
- Parsed responses: `{agent_id}_{assessment}_parsed.txt`
- Summary report: `summary_report.txt`
- Complete results: `assessment_results.json`

## Notes

- Models are automatically routed to the appropriate API provider
- Temperature and other parameters are handled appropriately for each provider
- The tool gracefully handles missing API keys for unused providers
- Rate limiting and retry logic is implemented for all providers 