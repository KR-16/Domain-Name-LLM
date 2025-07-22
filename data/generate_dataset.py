# data/generate_dataset.py

import os
import json
import openai
from tqdm import tqdm
from dotenv import load_dotenv

# Groq-compatible API
load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"

MODEL_NAME = "llama3-8b-8192"
# MODEL_NAME = "mixtral-8x7b-32768"
# MODEL_NAME = "gemma-7b-it"


business_categories = [
    "eco-friendly yoga accessories brand",
    "AI-powered marketing tool for small businesses",
    "fintech app for teenagers",
    "luxury pet grooming service",
    "remote coding bootcamp for underrepresented groups",
    "subscription box for organic snacks",
    "travel planner app for digital nomads",
    "cybersecurity consultancy for startups",
    "custom 3D-printed jewelry store",
    "online therapy for gamers"
]

prompt_template = (
    "You are a helpful assistant that only outputs valid JSON.\n\n"
    "Given the following business description:\n"
    "\"{desc}\"\n\n"
    "Generate exactly 3 safe and brandable domain names (including TLDs like .com, .io).\n"
    "Respond ONLY with a valid JSON list like:\n"
    "[\"domain1.com\", \"domain2.io\", \"domain3.net\"]"
)


def generate_domains(description: str):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt_template.format(desc=description)}
            ],
            temperature=0.8,
        )
        output = response['choices'][0]['message']['content']
        print(f"Raw output: {output}")
        domains = json.loads(output)
        return [{"business_description": description, "target_domain": d} for d in domains]
    except Exception as e:
        print(f" Error on '{description}': {e}")
        return []

def main():
    dataset = []
    for desc in tqdm(business_categories):
        result = generate_domains(desc)
        dataset.extend(result)

    os.makedirs("data/generated", exist_ok=True)
    with open("data/generated/domain_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Generated {len(dataset)} entries.")

if __name__ == "__main__":
    main()
