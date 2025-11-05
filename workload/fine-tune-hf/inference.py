from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

m_id = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(m_id)
base = AutoModelForCausalLM.from_pretrained(m_id, device_map="auto", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "out-qwen2.5-1.5b-qlora")

# Example inference
prompt = "What is machine learning?"
inputs = tok(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tok.decode(outputs[0], skip_special_tokens=True))
