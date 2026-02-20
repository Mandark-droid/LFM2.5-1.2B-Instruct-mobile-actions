# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "transformers==4.57.3",
#     "huggingface_hub[hf_transfer]",
# ]
# ///
"""
Inference script for LFM2.5-1.2B-Instruct-mobile-actions.

Tests the fine-tuned LoRA adapter on sample mobile function-calling tasks.

Usage (local with GPU):
    uv run inference.py

Usage (custom prompt):
    uv run inference.py --prompt "Send an email to john@example.com about the meeting"

Usage (test against eval set):
    uv run inference.py --eval --num-samples 10
"""

import argparse
import json
import logging
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_REPO = "kshitijthakkar/LFM2.5-1.2B-Instruct-mobile-actions"
BASE_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"

# Default Android tool definitions for interactive mode
DEFAULT_TOOLS = [
    {"function": {"name": "send_message", "description": "Sends a text message to a specified phone number.", "parameters": {"type": "OBJECT", "properties": {"to": {"type": "STRING", "description": "The recipient's phone number"}, "body": {"type": "STRING", "description": "The message content"}}, "required": ["to", "body"]}}},
    {"function": {"name": "create_calendar_event", "description": "Creates a new calendar event.", "parameters": {"type": "OBJECT", "properties": {"title": {"type": "STRING", "description": "The event title"}, "datetime": {"type": "STRING", "description": "The date and time of the event"}}, "required": ["title", "datetime"]}}},
    {"function": {"name": "send_email", "description": "Sends an email to the specified address.", "parameters": {"type": "OBJECT", "properties": {"to": {"type": "STRING", "description": "The recipient email address"}, "subject": {"type": "STRING", "description": "The email subject"}, "body": {"type": "STRING", "description": "The email body"}}, "required": ["to", "subject", "body"]}}},
    {"function": {"name": "create_contact", "description": "Creates a new contact.", "parameters": {"type": "OBJECT", "properties": {"first_name": {"type": "STRING", "description": "First name"}, "last_name": {"type": "STRING", "description": "Last name"}, "phone_number": {"type": "STRING", "description": "Phone number"}, "email": {"type": "STRING", "description": "Email address"}}, "required": ["first_name"]}}},
    {"function": {"name": "show_map", "description": "Shows a location on the map.", "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING", "description": "The search query or address"}}, "required": ["query"]}}},
    {"function": {"name": "turn_on_flashlight", "description": "Turns the flashlight on.", "parameters": {"type": "OBJECT", "properties": {}, "required": []}}},
    {"function": {"name": "turn_off_flashlight", "description": "Turns the flashlight off.", "parameters": {"type": "OBJECT", "properties": {}, "required": []}}},
    {"function": {"name": "open_wifi_settings", "description": "Opens the Wi-Fi settings page.", "parameters": {"type": "OBJECT", "properties": {}, "required": []}}},
    {"function": {"name": "web_search", "description": "Searches the web for a query.", "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING", "description": "The search query"}}, "required": ["query"]}}},
]

SAMPLE_PROMPTS = [
    "Send a text message to 555-1234 saying I'll be there in 10 minutes",
    "Schedule a meeting called 'Project Review' for tomorrow at 3 PM",
    "Send an email to alice@company.com with subject 'Q4 Report' and body 'Please find the attached report'",
    "Turn on the flashlight",
    "Show me coffee shops nearby on the map",
    "Search the web for weather forecast this weekend",
    "Create a new contact for Bob Smith with phone number 555-9876",
    "Open Wi-Fi settings",
]
# ─────────────────────────────────────────────────────────────────────────────


def load_model():
    """Load the fine-tuned model with Unsloth."""
    from unsloth import FastLanguageModel

    print("Loading model...")
    start = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_REPO,
        max_seq_length=2048,
    )
    FastLanguageModel.for_inference(model)
    print(f"Model loaded in {time.time() - start:.1f}s")
    return model, tokenizer


def generate_response(model, tokenizer, user_prompt, tools=None):
    """Generate a function call response for a user prompt."""
    if tools is None:
        tools = DEFAULT_TOOLS

    tools_json = json.dumps(tools, indent=2, default=str)
    system_content = (
        "You are a helpful assistant with access to the following functions. "
        "Use them if required.\n\n" + tools_json
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    start = time.time()
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.1,
        top_k=50,
        top_p=0.1,
        repetition_penalty=1.05,
        do_sample=True,
    )
    elapsed = time.time() - start

    # Decode only the generated tokens
    generated = outputs[0][inputs.shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return response, elapsed


def run_samples(model, tokenizer):
    """Run inference on sample prompts."""
    print("\n" + "=" * 70)
    print("Sample Inference Results")
    print("=" * 70)

    for i, prompt in enumerate(SAMPLE_PROMPTS):
        response, elapsed = generate_response(model, tokenizer, prompt)
        print(f"\n[{i+1}/{len(SAMPLE_PROMPTS)}] User: {prompt}")
        print(f"  Model: {response}")
        print(f"  Time:  {elapsed:.2f}s")

        # Try to parse as JSON to validate
        try:
            parsed = json.loads(response)
            func_name = parsed[0]["name"] if isinstance(parsed, list) else parsed.get("name", "?")
            print(f"  Valid JSON, function: {func_name}")
        except (json.JSONDecodeError, KeyError, IndexError):
            print(f"  Warning: response is not valid JSON")


def run_eval(model, tokenizer, num_samples=10):
    """Test against the evaluation split of google/mobile-actions."""
    from datasets import load_dataset

    print(f"\n{'=' * 70}")
    print(f"Evaluation on google/mobile-actions (n={num_samples})")
    print("=" * 70)

    full_ds = load_dataset("google/mobile-actions", split="train")
    eval_ds = full_ds.filter(lambda x: x["metadata"] != "train")
    eval_ds = eval_ds.select(range(min(num_samples, len(eval_ds))))

    correct = 0
    total = 0

    for i, example in enumerate(eval_ds):
        tools = example["tools"]
        user_msg = next(
            (m["content"] for m in example["messages"] if m["role"] == "user"), ""
        )
        # Get expected function call
        expected_tc = next(
            (m["tool_calls"] for m in example["messages"]
             if m["role"] == "assistant" and m["tool_calls"]),
            None,
        )
        expected_name = expected_tc[0]["function"]["name"] if expected_tc else "N/A"

        response, elapsed = generate_response(model, tokenizer, user_msg, tools)

        # Check if predicted function name matches
        predicted_name = "?"
        try:
            parsed = json.loads(response)
            predicted_name = parsed[0]["name"] if isinstance(parsed, list) else parsed.get("name", "?")
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

        match = predicted_name == expected_name
        if match:
            correct += 1
        total += 1

        status = "PASS" if match else "FAIL"
        print(f"\n[{i+1}/{len(eval_ds)}] {status} | User: {user_msg[:80]}...")
        print(f"  Expected: {expected_name}")
        print(f"  Got:      {predicted_name}")
        if not match:
            print(f"  Response: {response[:200]}")
        print(f"  Time: {elapsed:.2f}s")

    print(f"\n{'=' * 70}")
    print(f"Results: {correct}/{total} correct ({100*correct/total:.1f}% accuracy)")
    print("=" * 70)


def run_interactive(model, tokenizer):
    """Interactive mode for custom prompts."""
    print("\n" + "=" * 70)
    print("Interactive Mode (type 'quit' to exit)")
    print("=" * 70)
    print("Available functions: " + ", ".join(t["function"]["name"] for t in DEFAULT_TOOLS))

    while True:
        print()
        user_input = input("User: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        response, elapsed = generate_response(model, tokenizer, user_input)
        print(f"Model: {response}")
        print(f"({elapsed:.2f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Inference for LFM2.5-1.2B-Instruct-mobile-actions"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Single prompt to test (uses default tools)",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run evaluation against the dataset eval split",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Number of eval samples to test (default: 10)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode",
    )
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This script requires a GPU.")
        sys.exit(1)

    model, tokenizer = load_model()

    if args.prompt:
        response, elapsed = generate_response(model, tokenizer, args.prompt)
        print(f"\nUser:  {args.prompt}")
        print(f"Model: {response}")
        print(f"Time:  {elapsed:.2f}s")
    elif args.eval:
        run_eval(model, tokenizer, args.num_samples)
    elif args.interactive:
        run_interactive(model, tokenizer)
    else:
        # Default: run all sample prompts
        run_samples(model, tokenizer)


if __name__ == "__main__":
    main()
