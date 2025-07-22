# Synthetic Dataset: Domain Name Generator

## Methodology

- Used Groq API with LLaMA-3 to generate 3 domain names per business description.
- Prompt enforced safe and creative names with proper JSON formatting.
- Covered 10 diverse categories:
  - Pet grooming, fintech, yoga, online therapy, etc.

## Format

Each record:
```json
{
  "business_description": "...",
  "target_domain": "..."
}
