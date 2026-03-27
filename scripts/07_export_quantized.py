"""
07_export_quantized.py — Dynamic 2.0 Quantization Export
Merges LoRA adapter and exports with Unsloth Dynamic 2.0.
Usage: python scripts/07_export_quantized.py [--model master_brain|handyman|all]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_config(p):
    with open(p) as f: return yaml.safe_load(f)

def export_master_brain(config):
    from unsloth import FastLanguageModel
    dpo_cfg = config["training"]["dpo"]
    out = Path(config["training"]["quantization"]["output_dir"]) / "master_brain"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading DPO adapter from {dpo_cfg['output_dir']}...")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=dpo_cfg["output_dir"],
        max_seq_length=dpo_cfg["max_seq_length"], dtype=None, load_in_4bit=False)

    print("Merging LoRA + exporting GGUF Q4...")
    model.save_pretrained_gguf(str(out/"gguf"), tok, quantization_method="q4_k_m")
    print("Exporting merged 16-bit safetensors...")
    model.save_pretrained_merged(str(out/"merged"), tok, save_method="merged_16bit")
    print(f"✅ Master Brain exported to {out}")

def export_handyman(config):
    from unsloth import FastLanguageModel
    mid = config["models"]["handyman"]["model_id"]
    out = Path(config["training"]["quantization"]["output_dir"]) / "handyman"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {mid}...")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=mid, max_seq_length=2048, dtype=None, load_in_4bit=False)

    model.save_pretrained_gguf(str(out/"gguf"), tok, quantization_method="q4_k_m")
    model.save_pretrained_merged(str(out/"merged"), tok, save_method="merged_16bit")
    print(f"✅ Handyman exported to {out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--model", choices=["master_brain","handyman","all"], default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model in ("master_brain","all"): export_master_brain(config)
    if args.model in ("handyman","all"): export_handyman(config)

if __name__ == "__main__":
    main()
