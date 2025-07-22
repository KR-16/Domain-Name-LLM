import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load fine-tuned model and tokenizer
model_path = "training/checkpoints/v1"  # or path to your trained model
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

if torch.cuda.is_available():
    model.to("cuda")

def generate_domain(prompt):
    input_text = f"<s>[INST] Suggest a domain name for: {prompt} [/INST]"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated.replace(input_text, "").strip()

demo = gr.Interface(
    fn=generate_domain,
    inputs=gr.Textbox(label="Business Description", placeholder="e.g., eco-friendly yoga gear startup"),
    outputs=gr.Textbox(label="Suggested Domain Name"),
    title="🧠 Domain Name Generator",
    description="Enter a business idea and get a creative domain name suggestion."
)

demo.launch()
