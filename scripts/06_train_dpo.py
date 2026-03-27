"""
06_train_dpo.py — Direct Preference Optimization with Unsloth
===============================================================
Trains on the DPO dataset (chosen vs rejected pairs) to align
the model toward High-EQ, grounded responses.

Loads the SFT-adapted model and applies DPO training on top.

Usage:
    python scripts/06_train_dpo.py
    python scripts/06_train_dpo.py --config config/settings.yaml
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
    parser = argparse.ArgumentParser(description="DPO Training with Unsloth")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["models"]["master_brain"]
    sft_cfg = config["training"]["sft"]
    dpo_cfg = config["training"]["dpo"]

    # ---- Import Unsloth ----
    from unsloth import FastLanguageModel

    # ---- Load the SFT-adapted model ----
    print(f"Loading SFT adapter from {sft_cfg['output_dir']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_cfg["output_dir"],  # Load the SFT adapter
        max_seq_length=dpo_cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=False,
    )

    # ---- Re-apply LoRA for DPO stage ----
    # The SFT adapter is already applied; we continue training it
    FastLanguageModel.for_training(model)

    # ---- Load DPO Dataset ----
    print(f"Loading DPO data from {dpo_cfg['data_path']}...")

    from datasets import Dataset

    data = []
    with open(dpo_cfg["data_path"], "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })

    print(f"Loaded {len(data)} DPO pairs")
    dataset = Dataset.from_list(data)

    # ---- DPO Training ----
    from trl import DPOTrainer, DPOConfig

    print("Starting DPO training...")
    dpo_config = DPOConfig(
        output_dir=dpo_cfg["output_dir"],
        num_train_epochs=dpo_cfg["num_epochs"],
        per_device_train_batch_size=dpo_cfg["batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        learning_rate=dpo_cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        beta=dpo_cfg["beta"],           # 0.1 — controls deviation from reference model
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        max_length=dpo_cfg["max_seq_length"],
        max_prompt_length=dpo_cfg["max_seq_length"] // 2,
    )

    # Create a reference model (frozen copy of the SFT model)
    # DPOTrainer handles this automatically when ref_model=None
    trainer = DPOTrainer(
        model=model,
        ref_model=None,     # DPOTrainer creates implicit reference from initial weights
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # ---- Save ----
    print(f"Saving DPO adapter to {dpo_cfg['output_dir']}...")
    model.save_pretrained(dpo_cfg["output_dir"])
    tokenizer.save_pretrained(dpo_cfg["output_dir"])

    print(f"\n✅ DPO training complete!")
    print(f"   Adapter saved to: {dpo_cfg['output_dir']}")
    print(f"   Pairs trained on: {len(data)}")
    print(f"   Beta: {dpo_cfg['beta']}")


if __name__ == "__main__":
    main()
