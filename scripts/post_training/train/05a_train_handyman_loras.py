"""
05a_train_handyman_loras.py — Train All 3 Handyman LoRA Adapters
==================================================================
Trains three task-specific LoRAs on the Qwen 3.5-0.8B base model:
  1. Router LoRA: intent classification + query decomposition
  2. Reranker LoRA: product relevance scoring
  3. Verifier LoRA: visual product match verification

All three share the same base weights. Served via SGLang multi-LoRA.

Usage:
    python scripts/05a_train_handyman_loras.py --lora router
    python scripts/05a_train_handyman_loras.py --lora reranker
    python scripts/05a_train_handyman_loras.py --lora verifier
    python scripts/05a_train_handyman_loras.py --lora all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(p):
    with open(p) as f:
        return yaml.safe_load(f)


LORA_CONFIGS = {
    "router": {
        "data_path": "data/training/router_lora_train.jsonl",
        "output_dir": "models/handyman_loras/router",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lr": 3e-4,
        "epochs": 3,
        "batch_size": 8,
        "max_seq_length": 1024,
    },
    "reranker": {
        "data_path": "data/training/reranker_lora_train.jsonl",
        "output_dir": "models/handyman_loras/reranker",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lr": 2e-4,
        "epochs": 3,
        "batch_size": 4,
        "max_seq_length": 2048,
    },
    "verifier": {
        "data_path": "data/training/verifier_lora_train.jsonl",
        "output_dir": "models/handyman_loras/verifier",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lr": 2e-4,
        "epochs": 5,  # More epochs since auto-generated data can be noisy
        "batch_size": 4,
        "max_seq_length": 2048,
    },
}


def train_lora(base_model_id: str, lora_name: str, lora_cfg: dict):
    """Train a single LoRA adapter."""
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"\n{'='*60}")
    print(f"Training Handyman LoRA: {lora_name}")
    print(f"{'='*60}")

    output_dir = lora_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load base model
    print(f"Loading base model: {base_model_id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=lora_cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=False,
    )

    # Apply LoRA
    print(f"Applying LoRA (rank={lora_cfg['lora_rank']}, alpha={lora_cfg['lora_alpha']})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["lora_rank"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load dataset
    print(f"Loading data from {lora_cfg['data_path']}")
    data = []
    with open(lora_cfg["data_path"]) as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"Loaded {len(data)} samples")

    def format_sample(sample):
        messages = sample["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    # Train
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=lora_cfg["epochs"],
            per_device_train_batch_size=lora_cfg["batch_size"],
            gradient_accumulation_steps=4,
            learning_rate=lora_cfg["lr"],
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            optim="adamw_8bit",
            seed=42,
            report_to="none",
        ),
        dataset_text_field="text",
        max_seq_length=lora_cfg["max_seq_length"],
        packing=True,
    )

    trainer.train()

    # Save only the LoRA adapter (not the full model)
    print(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ {lora_name} LoRA training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Handyman LoRA adapters")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--lora", choices=["router", "reranker", "verifier", "all"], default="all")
    args = parser.parse_args()

    config = load_config(args.config)
    base_model_id = config["models"]["handyman"]["model_id"]

    loras_to_train = [args.lora] if args.lora != "all" else ["router", "reranker", "verifier"]

    for lora_name in loras_to_train:
        lora_cfg = LORA_CONFIGS[lora_name]
        train_lora(base_model_id, lora_name, lora_cfg)

    print(f"\n🎯 All requested LoRAs trained!")
    print("   LoRA adapters saved to models/handyman_loras/")
    print("   Ready for multi-LoRA serving via SGLang.")


if __name__ == "__main__":
    main()
