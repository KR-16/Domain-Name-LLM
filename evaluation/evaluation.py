import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import torch
import re
from dotenv import load_dotenv
load_dotenv()

# === 1. Set Groq credentials ===
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = os.getenv("GROQ_API_BASE")

# === 2. Config ===
base_model_id = "tiiuae/falcon-rw-1b"
fine_tuned_path = "training/checkpoints/v1"
groq_model = "llama3-70b-8192"

prompts = [
    "eco-friendly yoga gear startup",
    "AI interior design assistant",
    "online therapy for gamers",
    "subscription box for vegan snacks",
    "pet wellness app",
    "freelance marketplace for designers",
    "remote coding bootcamp for minorities",
    "luxury dog grooming spa",
    "AI-powered language learning app",
    "virtual event hosting platform"
]

# === 3. Load Models ===
print("\n🔄 Loading models...")
base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id).to("cpu")

fine_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
fine_model = AutoModelForCausalLM.from_pretrained(fine_tuned_path).to("cpu")

# === 4. Domain Generator ===
def generate_domains(prompt, model, tokenizer, n=3):
    input_text = f"<s>[INST] Suggest a domain name for: {prompt} [/INST]"
    inputs = tokenizer(input_text, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=20, num_return_sequences=n, do_sample=True)
    return [tokenizer.decode(out, skip_special_tokens=True).split("[/INST]")[-1].strip() for out in outputs]

# --- Utility: Extract first valid JSON block ---
def extract_first_json_object(text):
    matches = re.findall(r"\{[\s\S]+?\}", text)
    for m in matches:
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue
    print("⚠️ No valid JSON object found.")
    return {"relevance": 0, "creativity": 0, "safety": 0, "overall": 0}

# --- Groq Evaluator ---
def ask_groq(prompt, domains):
    formatted = f"""
You are evaluating domain names for a startup idea.

Business Description: "{prompt}"

Domain Names:
{chr(10).join(f"- {d}" for d in domains)}

Respond with a single JSON block:
{{
  "relevance": (1–5),
  "creativity": (1–5),
  "safety": (1–5),
  "overall": (1–5)
}}
Only include the JSON — no explanation.
"""

    try:
        response = openai.ChatCompletion.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Return a single JSON object. No explanation."},
                {"role": "user", "content": formatted},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message["content"]
        print("🔍 Raw response:", content)
        return extract_first_json_object(content)

    except Exception as e:
        print("⚠️ Groq API error:", e)
        return {"relevance": 0, "creativity": 0, "safety": 0, "overall": 0}

# === 6. Run Evaluation ===
results = []
print("\n🚀 Running Evaluation...")
for prompt in prompts:
    print(f"\n🔹 {prompt}")
    base_domains = generate_domains(prompt, base_model, base_tokenizer)
    fine_domains = generate_domains(prompt, fine_model, fine_tokenizer)

    base_score = ask_groq(prompt, base_domains)
    fine_score = ask_groq(prompt, fine_domains)

    results.append({
        "prompt": prompt,
        "base_domains": base_domains,
        "fine_domains": fine_domains,
        "base_score": base_score,
        "fine_score": fine_score
    })

# === 7. Save to CSV ===
print("\n💾 Saving results to results.csv")
rows = []
for r in results:
    rows.append({
        "prompt": r["prompt"],
        "model": "base",
        "domains": "; ".join(r["base_domains"]),
        **r["base_score"]
    })
    rows.append({
        "prompt": r["prompt"],
        "model": "fine-tuned",
        "domains": "; ".join(r["fine_domains"]),
        **r["fine_score"]
    })

pd.DataFrame(rows).to_csv("results.csv", index=False)
print("✅ Done!")
