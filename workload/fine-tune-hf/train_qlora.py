from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

model_id = "Qwen/Qwen2.5-1.5B-Instruct"  # small, fast; swap to 7B if you want
dataset = load_dataset("trl-lib/Capybara", split="train")  # tiny multi-turn sample

# 4-bit quantization (QLoRA)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto",
)

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

cfg = SFTConfig(
    output_dir="out-qwen2.5-1.5b-qlora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_length=1024,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    logging_steps=10,
    bf16=True,
    gradient_checkpointing=True,
    packing=False,              # disabled - requires flash attention
    warmup_ratio=0.03,
    save_steps=200,
)

trainer = SFTTrainer(
    model=base,
    processing_class=tok,
    train_dataset=dataset,
    peft_config=lora,
    args=cfg,
    formatting_func=lambda ex: tok.apply_chat_template(ex["messages"], tokenize=False),
)

trainer.train()
trainer.save_model()  # saves LoRA adapters
