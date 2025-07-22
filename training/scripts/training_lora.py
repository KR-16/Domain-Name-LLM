# from datasets import load_dataset, Dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# from peft import LoraConfig, get_peft_model, TaskType
# import torch
# import json
# import os

# # Load Dataset from JSON file

# def load_custom_dataset(path):
#     with open(path) as f:
#         raw = json.load(f)
#     prompts = [f"Suggest a domain name for: {x['business_description']}" for x in raw]
#     responses = [x["target_domain"] for x in raw]
#     pairs = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)]
#     return Dataset.from_list(pairs)

# # === Format for training ===
# def format(example):
#     return {"text": f"<s>[INST] {example['prompt']} [/INST] {example['response']}</s>"}

# # === Model & Tokenizer ===
# model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")

# # === LoRA config ===
# peft_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.05,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM
# )

# model = get_peft_model(model, peft_config)

# # === Dataset ===
# dataset = load_custom_dataset("data/generated/domain_dataset.json")
# dataset = dataset.map(format)

# # === Training config ===
# training_args = TrainingArguments(
#     output_dir="training/checkpoints/v1",
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=2,
#     num_train_epochs=5,
#     learning_rate=2e-4,
#     fp16=True,
#     logging_steps=10,
#     save_strategy="epoch",
#     report_to="none"
# )

# # === Data collator ===
# collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # === Trainer ===
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     tokenizer=tokenizer,
#     data_collator=collator
# )

# trainer.train()
# model.save_pretrained("training/checkpoints/v1")
# tokenizer.save_pretrained("training/checkpoints/v1")



# === Imports
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training


print("✅ Imported libraries")

# === Constants
# model_id = "microsoft/phi-2"
model_id = "tiiuae/falcon-rw-1b"
dataset_path = "data/generated/domain_dataset.json"

# === Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "right"
print("✅ Tokenizer loaded and configured")

# === Load Dataset
def load_custom_dataset(path):
    with open(path) as f:
        raw = json.load(f)
    prompts = [f"Suggest a domain name for: {x['business_description']}" for x in raw]
    responses = [x["target_domain"] for x in raw]
    return Dataset.from_list([{
        "text": f"<s>[INST] {p} [/INST] {r}</s>"
    } for p, r in zip(prompts, responses)])

def tokenize(example):
    result = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )
    result["labels"] = [
        token if token != tokenizer.pad_token_id else -100
        for token in result["input_ids"]
    ]
    return result

print("📂 Loading and tokenizing dataset...")
dataset = load_custom_dataset(dataset_path)
dataset = dataset.map(tokenize, remove_columns=["text"])
print("✅ Dataset ready with", len(dataset), "examples")

# === Load Base Model
print("🔄 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(model_id)
# fixing the gradient flow
model = prepare_model_for_kbit_training(model)
print("✅ Base model loaded")

# === Apply LoRA
print("⚙️ Applying LoRA...")
peft_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)
# ✅ Force LoRA params to require grad (bulletproof)
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True

model.print_trainable_parameters()
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

model.gradient_checkpointing_disable()
# model = model.to("cuda")
# model = model.to("cuda")
print("✅ LoRA applied and model moved to GPU")

# === Training Setup
print("⚙️ Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="domain-name-llm/checkpoints/v1",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    remove_unused_columns=False
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Custom Trainer
print("🚀 Initializing trainer...")
from transformers import Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 🔁 Move inputs to device
        for k in inputs:
            if hasattr(inputs[k], "to"):
                inputs[k] = inputs[k].to(model.device)

        # ✅ Force requires_grad on input embeddings
        inputs["input_ids"].requires_grad_ = True

        outputs = model(**inputs)
        loss = outputs.loss

        assert loss.requires_grad, "Loss has no grad_fn — check LoRA and inputs!"
        return (loss, outputs) if return_outputs else loss

model.train()


for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"✅ {name} is trainable")
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)
print(dataset[0]["labels"][:10])
print(dataset[0]["input_ids"][:10])

print("✅ Trainer ready")

# === Train
model.train()

print("💡 Confirming LoRA layers:")
for n, p in model.named_parameters():
    if "lora_" in n:
        print(f"🔎 {n} requires_grad={p.requires_grad}")

print("✅ Starting training now...")
trainer.train()

print("✅ Training complete")

# === Save Model
print("💾 Saving model...")
model.save_pretrained("training/checkpoints/v1")
tokenizer.save_pretrained("training/checkpoints/v1")
print("✅ Model saved to training/checkpoints/v1")