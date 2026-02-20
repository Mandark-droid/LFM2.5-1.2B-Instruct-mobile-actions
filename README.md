---
library_name: peft
license: other
base_model: LiquidAI/LFM2.5-1.2B-Instruct
tags:
  - unsloth
  - trl
  - sft
  - lora
  - function-calling
  - mobile-actions
  - on-device
  - lfm2
  - lfm2.5
  - hf_jobs
datasets:
  - google/mobile-actions
language:
  - en
pipeline_tag: text-generation
---

# LFM2.5-1.2B-Instruct-mobile-actions

A LoRA fine-tune of [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) on [google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions) for on-device function calling on Android.

The model translates natural language instructions into executable function calls for Android OS system tools (e.g. sending messages, setting alarms, making calls, web search).

## Training Details

| Parameter | Value |
|---|---|
| **Base model** | LiquidAI/LFM2.5-1.2B-Instruct (1.17B params) |
| **Method** | SFT with LoRA via Unsloth |
| **Dataset** | google/mobile-actions (8,693 train / 961 eval) |
| **Hardware** | NVIDIA L4 (22 GB VRAM) on HF Jobs |
| **Training time** | 122.9 minutes |
| **Epochs** | 3 |
| **Batch size** | 2 x 4 gradient accumulation = 8 effective |
| **Learning rate** | 2e-4 (linear decay) |
| **Optimizer** | AdamW 8-bit |
| **Max sequence length** | 2048 |
| **Precision** | 16-bit with Unsloth optimizations |

### LoRA Configuration

| Parameter | Value |
|---|---|
| **Rank (r)** | 16 |
| **Alpha** | 16 |
| **Target modules** | q_proj, k_proj, v_proj, out_proj, in_proj, w1, w2, w3 |
| **Trainable parameters** | 11,108,352 / 1,181,448,960 (0.94%) |
| **Dropout** | 0 |
| **Gradient checkpointing** | Unsloth |

### Training Results

| Metric | Value |
|---|---|
| **Final train loss** | 0.0137 |
| **Epoch 1 eval loss** | 0.0161 |
| **Epoch 2 eval loss** | 0.0155 |
| **Epoch 3 eval loss** | 0.0154 |
| **Total steps** | 3,261 |
| **Speed** | 0.442 steps/s |

Training was response-only (user/system tokens masked from loss). Eval loss tracks train loss closely with no overfitting.

## Usage

### With PEFT + Transformers

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2.5-1.2B-Instruct")
model = PeftModel.from_pretrained(base_model, "kshitijthakkar/LFM2.5-1.2B-Instruct-mobile-actions")
tokenizer = AutoTokenizer.from_pretrained("kshitijthakkar/LFM2.5-1.2B-Instruct-mobile-actions")

messages = [
    {"role": "system", "content": "You are a helpful assistant with access to the following functions. Use them if required.\n\n" + tools_json},
    {"role": "user", "content": "Send a text message to John saying I'll be late"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs, max_new_tokens=256, temperature=0.1, top_k=50, top_p=0.1, repetition_penalty=1.05)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With Unsloth (faster inference)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="kshitijthakkar/LFM2.5-1.2B-Instruct-mobile-actions",
    max_seq_length=2048,
)
FastLanguageModel.for_inference(model)
```

## Dataset Format

Each training example contains:
- **System prompt**: Available Android function definitions (JSON)
- **User message**: Natural language instruction (e.g. "Set an alarm for 7am tomorrow")
- **Assistant response**: Function call as JSON (e.g. `[{"name": "set_alarm", "arguments": {"datetime": "..."}}]`)

The model was trained on responses only, learning to map user instructions to the correct function calls given available tool definitions.

## Intended Use

- On-device function calling for Android system tools
- Translating natural language to structured API calls
- Edge deployment for mobile assistants

## Limitations

- Trained only on Android system tool functions from the google/mobile-actions dataset
- May not generalize to arbitrary function schemas outside the training distribution
- Inherits limitations of the base LFM2.5-1.2B-Instruct model

## Training Script

The training script is available at: [github.com/Mandark-droid/LFM2.5-1.2B-Instruct-mobile-actions](https://github.com/Mandark-droid/LFM2.5-1.2B-Instruct-mobile-actions)
