"""
05_train_sft.py — Supervised Fine-Tuning with Unsloth
=======================================================
Trains Qwen 3.5-9B using BF16 LoRA on the SFT dataset.

Uses Unsloth for 2x faster training with 60% less memory.
LoRA config: rank=64, alpha=128, targets all attention + MLP layers.

Usage:
    python scripts/05_train_sft.py
    python scripts/05_train_sft.py --config config/settings.yaml
    
SLURM (PACE ICE cluster):
    sbatch scripts/slurm/train_sft.sbatch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SFT Training with Unsloth")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["models"]["master_brain"]
    train_cfg = config["training"]["sft"]

    # ---- Import Unsloth (must be imported before transformers) ----
    from unsloth import FastLanguageModel

    # ---- Load Model with Unsloth ----
    print(f"Loading {model_cfg['model_id']} with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_id"],
        max_seq_length=train_cfg["max_seq_length"],
        dtype=None,  # Auto-detect (will use BF16 on Ampere+)
        load_in_4bit=False,  # Train in BF16 for high-precision gradients
    )

    # ---- Apply LoRA ----
    print("Applying LoRA configuration...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=train_cfg["lora_rank"],                # 64
        lora_alpha=train_cfg["lora_alpha"],       # 128
        lora_dropout=train_cfg["lora_dropout"],   # 0.05
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",    # Attention
            "gate_proj", "up_proj", "down_proj",         # MLP
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",     # 2x longer context
        random_state=42,
    )

    # ---- Load Dataset ----
    print(f"Loading SFT data from {train_cfg['data_path']}...")
    
    from datasets import Dataset

    data = []
    with open(train_cfg["data_path"], "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    print(f"Loaded {len(data)} training samples")

    # Format into chat template
    def format_sample(sample):
        messages = sample["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    # ---- Training ----
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print("Starting SFT training...")
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        optim="adamw_8bit",
        seed=42,
        report_to="none",        # Disable wandb etc.
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=train_cfg["max_seq_length"],
        packing=True,            # Pack multiple samples into one sequence for efficiency
    )

    trainer.train()

    # ---- Save ----
    print(f"Saving adapter to {train_cfg['output_dir']}...")
    model.save_pretrained(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])

    print(f"\n✅ SFT training complete!")
    print(f"   Adapter saved to: {train_cfg['output_dir']}")
    print(f"   Samples trained on: {len(data)}")


if __name__ == "__main__":
    main()
