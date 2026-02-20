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

### Inference Results

Evaluated on 20 held-out examples from the `google/mobile-actions` eval split: **20/20 correct (100% accuracy)**

| # | User Prompt | Expected | Predicted | Time |
|---|---|---|---|---|
| 1 | Can you please save a new contact for me? The name is Lena Petrova... | `create_contact` | `create_contact` | 30.41s* |
| 2 | Please send an email to javier.ortega@ecotradeintl.com with the subject 'Update... | `send_email` | `send_email` | 5.31s |
| 3 | I need to save a new contact. The full name is Anya Sharma... | `create_contact` | `create_contact` | 2.62s |
| 4 | Please set up a new calendar event for 'Team Lunch with Marketing' on May 13... | `create_calendar_event` | `create_calendar_event` | 2.06s |
| 5 | Please send an email to Kenji Tanaka at kenji.tanaka@corpmail.jp... | `send_email` | `send_email` | 3.15s |
| 6 | Turn on the flashlight and show me the location of the Sunnyvale Library... | `turn_on_flashlight` | `turn_on_flashlight` | 1.82s |
| 7 | Can you please show me the location of the art supply store 'Canvas Creations'... | `show_map` | `show_map` | 1.77s |
| 8 | I need to set up a new calendar event. The title should be "Meeting with Dr. Che... | `create_calendar_event` | `create_calendar_event` | 2.16s |
| 9 | Please schedule a calendar event titled 'Quarterly Budget Review' for 10:30 AM... | `create_calendar_event` | `create_calendar_event` | 4.64s |
| 10 | I need to check under the sofa, please turn on the flashlight. | `turn_on_flashlight` | `turn_on_flashlight` | 0.90s |
| 11 | Please save a new contact. The name is Marcus Oliveira... | `create_contact` | `create_contact` | 2.91s |
| 12 | Can you show me where The Wandering Page bookstore is located in Brooklyn? | `show_map` | `show_map` | 1.24s |
| 13 | I'm having trouble reading this menu. Can you please turn on the flashlight... | `turn_on_flashlight` | `turn_on_flashlight` | 1.86s |
| 14 | Send an email to Aisha.Khan@fabrikam.com with the subject "Follow-up on Q1 Repor... | `send_email` | `send_email` | 3.03s |
| 15 | Please send an email to elara.pereira@email.com with the subject "Project Checkp... | `send_email` | `send_email` | 4.93s |
| 16 | Please turn off the flashlight and immediately schedule a calendar event titled... | `turn_off_flashlight` | `turn_off_flashlight` | 2.84s |
| 17 | Turn on my flashlight, and please create a calendar event titled "Morning Run"... | `turn_on_flashlight` | `turn_on_flashlight` | 2.70s |
| 18 | Please create a new contact named Aisha Khan with the phone number 987-654-3210... | `create_contact` | `create_contact` | 3.87s |
| 19 | I need to add a new contact named Elara Chen. Her phone number is +61 491 570 11... | `create_contact` | `create_contact` | 2.45s |
| 20 | Turn on the flashlight, I need to see what I dropped in this dark corner. | `turn_on_flashlight` | `turn_on_flashlight` | 0.89s |

*\*First call includes CUDA warmup. Typical inference: ~1-5s on NVIDIA L4.*

**Accuracy by function type:**

| Function | Tested | Correct | Accuracy |
|---|---|---|---|
| `create_contact` | 5 | 5 | 100% |
| `send_email` | 4 | 4 | 100% |
| `create_calendar_event` | 4 | 4 | 100% |
| `turn_on_flashlight` | 4 | 4 | 100% |
| `show_map` | 2 | 2 | 100% |
| `turn_off_flashlight` | 1 | 1 | 100% |
| **Total** | **20** | **20** | **100%** |

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
