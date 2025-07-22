# 🧠 Domain Name Generator (LLM Fine-Tuned)

This project uses a fine-tuned language model (LLM) to generate creative and relevant domain names from business ideas. The model is adapted with PEFT (LoRA) on a compact causal language model (e.g., Falcon-RW-1B) and is accessible through a Gradio interface embedded in a Streamlit app.

---

## 🚀 Key Features

- 🔧 Trained on business description and domain name pairs
- 🧠 Efficient parameter tuning with LoRA
- ⚡ Built on Hugging Face Transformers
- 🌐 Deployed via Streamlit Cloud with integrated Gradio UI
- 📊 Supports evaluation with BLEU scores or Groq API

---

## 📁 Project Structure

```
domain-name-llm-app/
├── app.py                # Streamlit wrapper embedding Gradio
├── main.py               # Gradio UI and model prediction
├── training/
│   └── checkpoints/v1/   # Fine-tuned model/tokenizer files
├── evaluation/
│   └── evaluation.py     # Evaluation scripts (optional)
├── requirements.txt
└── README.md
```

---

## 💡 Example Usage

> **Input**: `eco-friendly yoga gear startup`  
> **Suggestions**: `yogatotes.com`, `greenyoga.io`, `zenflow.co`

---

## 🛠️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/KR-16/domain-name-llm-app.git
cd domain-name-llm-app
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

If you need any of the following, feel free to ask:
- A `LICENSE` file
- Project badges (e.g., Hugging Face, PyPI)
- A version of the UI built solely with Streamlit
