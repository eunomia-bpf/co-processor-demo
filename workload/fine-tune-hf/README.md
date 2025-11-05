# QLoRA Fine-tuning with Hugging Face TRL

30-line QLoRA supervised fine-tuning using TRL on a single GPU (RTX 5090).

## Setup

Activate the virtual environment:
```bash
source ../../venv/bin/activate
```

Install dependencies:
```bash
pip install -U transformers accelerate datasets peft trl bitsandbytes
```

## Training

Run the QLoRA training script:
```bash
accelerate launch --mixed_precision bf16 train_qlora.py
```

This will:
- Load Qwen2.5-1.5B-Instruct model with 4-bit quantization
- Fine-tune using LoRA adapters (r=16)
- Train on the Capybara dataset
- Save adapters to `out-qwen2.5-1.5b-qlora/`

## Inference

Load and use the fine-tuned model:
```bash
python inference.py
```

## Customization

- **Larger model**: Change `model_id` to a 7B model (e.g., `"Qwen/Qwen2.5-7B-Instruct"`)
- **LoRA rank**: Adjust `r` in `LoraConfig` (8-32 range)
- **Sequence length**: Modify `max_seq_length` in `SFTConfig`
- **Dataset**: Replace with your own dataset in the script
