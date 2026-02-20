# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "trl==0.22.2",
#     "huggingface_hub[hf_transfer]",
#     "trackio",
#     "tensorboard",
#     "transformers==4.57.3",
# ]
# ///
"""
Fine-tune LiquidAI/LFM2.5-1.2B-Instruct on google/mobile-actions
using Unsloth for ~2x faster training and ~60% less VRAM.

Trains the model to translate natural language instructions into
executable function calls for Android OS system tools.

Usage (HF Jobs):
    hf jobs uv run --flavor l4x1 --timeout 2h --secrets HF_TOKEN \
        "https://huggingface.co/kshitijthakkar/LFM2.5-1.2B-Instruct-mobile-actions-scripts/resolve/main/train.py"

Usage (local with GPU):
    uv run train.py
"""

import json
import logging
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
BASE_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"
DATASET = "google/mobile-actions"
OUTPUT_REPO = "kshitijthakkar/LFM2.5-1.2B-Instruct-mobile-actions"
TRACKIO_SPACE = "kshitijthakkar/trackio"

MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"
]

NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
SEED = 3407
# ─────────────────────────────────────────────────────────────────────────────


def check_cuda():
    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA not available. This script requires a GPU.")
        sys.exit(1)
    logger.info(f"CUDA: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


def format_example(example, tokenizer):
    """Convert a google/mobile-actions example into a ChatML text string.

    Each example has:
      - tools: list of function definitions
      - messages: list with role/content/tool_calls
      - metadata: "train" or "eval"

    We build a system message containing the available tools, then format
    user/assistant turns. Tool calls from the assistant are serialized as JSON.
    """
    messages = example["messages"]
    tools = example["tools"]

    formatted = []

    # System prompt with available tool definitions
    tools_json = json.dumps(tools, indent=2, default=str)
    system_content = (
        "You are a helpful assistant with access to the following functions. "
        "Use them if required.\n\n" + tools_json
    )
    formatted.append({"role": "system", "content": system_content})

    for msg in messages:
        role = msg["role"]
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")

        if role == "user":
            formatted.append({"role": "user", "content": content})

        elif role == "assistant":
            if tool_calls:
                calls = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    # Filter out None arguments and convert values to strings
                    raw_args = func.get("arguments", {})
                    args = {
                        k: str(v) for k, v in raw_args.items() if v is not None
                    }
                    calls.append({"name": func.get("name", ""), "arguments": args})
                formatted.append({
                    "role": "assistant",
                    "content": json.dumps(calls, default=str),
                })
            else:
                formatted.append({"role": "assistant", "content": content})

        elif role == "system" and content:
            # Prepend any existing system content
            formatted[0]["content"] = content + "\n\n" + formatted[0]["content"]

    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            formatted, tokenize=False, add_generation_prompt=False
        )
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
    except Exception:
        # Fallback: manual ChatML formatting
        text = ""
        for m in formatted:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"

    return {"text": text}


def main():
    print("=" * 70)
    print("LFM2.5 + Mobile Actions — Unsloth Fine-Tuning")
    print("=" * 70)
    print(f"  Model:    {BASE_MODEL}")
    print(f"  Dataset:  {DATASET}")
    print(f"  Output:   {OUTPUT_REPO}")
    print(f"  Epochs:   {NUM_EPOCHS}")
    print(f"  Batch:    {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  LR:       {LEARNING_RATE}")
    print(f"  LoRA:     r={LORA_R}, alpha={LORA_ALPHA}")
    print()

    check_cuda()

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if TRACKIO_SPACE:
        os.environ["TRACKIO_SPACE_ID"] = TRACKIO_SPACE

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from huggingface_hub import login

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        logger.info("Logged in to Hugging Face Hub")
    else:
        logger.warning("HF_TOKEN not set — model push may fail!")

    # ── 1. Load model ────────────────────────────────────────────────────────
    print("\n[1/5] Loading model...")
    start = time.time()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    print(f"  Model loaded in {time.time() - start:.1f}s")

    # ── 2. Load & format dataset ─────────────────────────────────────────────
    print("\n[2/5] Loading dataset...")
    start = time.time()

    full_dataset = load_dataset(DATASET, split="train")
    print(f"  Total examples: {len(full_dataset)}")

    # The dataset uses a 'metadata' field to mark train/eval splits
    train_ds = full_dataset.filter(lambda x: x["metadata"] == "train")
    eval_ds = full_dataset.filter(lambda x: x["metadata"] != "train")
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Format each example into ChatML text with tool definitions
    train_ds = train_ds.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=eval_ds.column_names,
    )

    print(f"  Dataset formatted in {time.time() - start:.1f}s")
    print(f"\n  --- Sample (first 400 chars) ---")
    print(f"  {train_ds[0]['text'][:400]}...")
    print()

    # ── 3. Configure trainer ─────────────────────────────────────────────────
    print("[3/5] Configuring trainer...")
    effective_batch = BATCH_SIZE * GRADIENT_ACCUMULATION
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    logging_steps = max(1, steps_per_epoch // 10)
    save_steps = max(1, steps_per_epoch // 4)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            output_dir="unsloth-output",
            dataset_text_field="text",
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            warmup_steps=5,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=SEED,
            max_length=MAX_SEQ_LENGTH,
            report_to=["tensorboard", "trackio"],
            run_name="lfm25-mobile-actions-unsloth",
            project="lfm25-mobile-actions",
            push_to_hub=True,
            hub_model_id=OUTPUT_REPO,
            eval_strategy="epoch",
        ),
    )

    # Train only on assistant responses (mask system/user tokens from loss)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # ── 4. Train ─────────────────────────────────────────────────────────────
    print(f"\n[4/5] Training {NUM_EPOCHS} epochs (~{steps_per_epoch} steps/epoch, "
          f"~{steps_per_epoch * NUM_EPOCHS} total)...")
    start = time.time()

    result = trainer.train()

    elapsed = time.time() - start
    print(f"\n  Training completed in {elapsed / 60:.1f} minutes")
    print(f"  Train loss: {result.metrics.get('train_loss', 'N/A')}")

    # Final evaluation
    print("\n  Running final evaluation...")
    try:
        ev = trainer.evaluate()
        eval_loss = ev.get("eval_loss")
        train_loss = result.metrics.get("train_loss")
        print(f"  Eval loss:  {eval_loss}")
        if eval_loss and train_loss and train_loss > 0:
            ratio = eval_loss / train_loss
            if ratio > 1.5:
                print(f"  Warning: eval/train ratio {ratio:.2f} — possible overfitting")
            else:
                print(f"  Eval/train ratio: {ratio:.2f} — looks healthy")
    except Exception as e:
        print(f"  Eval failed: {e}")

    # ── 5. Save & push ───────────────────────────────────────────────────────
    print("\n[5/5] Pushing model to Hub...")
    model.push_to_hub(OUTPUT_REPO, tokenizer=tokenizer)
    print(f"\n  Adapter available at: https://huggingface.co/{OUTPUT_REPO}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
