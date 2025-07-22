
import pandas as pd
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def domain_tokens(domain):
    # Tokenize domain using regex: split by dots, hyphens, numbers, etc.
    return re.findall(r"[a-zA-Z]+", domain.lower())

# Load results CSV
df = pd.read_csv("results.csv")

# Build reference mapping using best fine-tuned domain
reference_map = {}
grouped = df[df['model'] == 'fine-tuned'].groupby('prompt')
for name, group in grouped:
    best_row = group.sort_values('overall', ascending=False).iloc[0]
    domain = best_row['domains'].split()[0] if isinstance(best_row['domains'], str) else ''
    reference_map[name] = [domain_tokens(domain)]

# Compute BLEU scores
def compute_bleu(prompt, domain):
    references = reference_map.get(prompt, [])
    candidate = domain_tokens(domain)
    if references and candidate:
        return sentence_bleu(references, candidate, smoothing_function=SmoothingFunction().method4)
    return 0.0

df['bleu_score'] = df.apply(lambda row: compute_bleu(row['prompt'], str(row['domains']).split()[0] if isinstance(row['domains'], str) else ''), axis=1)

# Save BLEU results
df.to_csv("evaluation_with_bleu.csv", index=False)
print("✅ BLEU scoring complete. Saved as evaluation_with_bleu.csv")
