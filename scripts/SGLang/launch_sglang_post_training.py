"""
08_launch_sglang.py — SGLang Server Launcher
Launch two SGLang Runtime instances: 0.8B Handyman + 9B Master Brain.
Configures EAGLE-2 speculative decoding and RadixAttention KV caching.
Usage: python scripts/08_launch_sglang.py
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_config(p):
    with open(p) as f: return yaml.safe_load(f)

def launch_handyman(config):
    """
    Launch the 0.8B Handyman server with multi-LoRA support.
    Serves 3 LoRA adapters on a single base model:
      - handyman-router: intent detection + query decomposition
      - handyman-reranker: product relevance scoring
      - handyman-verifier: visual product match verification
    """
    sglang = config["inference"]["sglang"]
    quant_dir = Path(config["training"]["quantization"]["output_dir"]) / "handyman" / "merged"
    model_path = str(quant_dir) if quant_dir.exists() else config["models"]["handyman"]["model_id"]

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(sglang["handyman_port"]),
        "--tp", str(sglang["tensor_parallel_size"]),
        "--mem-fraction-static", str(sglang["mem_fraction_static"]),
        "--served-model-name", "handyman",
        "--trust-remote-code",
    ]

    # Multi-LoRA: load all 3 Handyman LoRA adapters
    lora_base = Path("models/handyman_loras")
    lora_adapters = {
        "handyman-router": lora_base / "router",
        "handyman-reranker": lora_base / "reranker",
        "handyman-verifier": lora_base / "verifier",
    }

    available_loras = {name: str(path) for name, path in lora_adapters.items() if path.exists()}

    if available_loras:
        # SGLang multi-LoRA: pass adapter paths and names
        lora_paths = ",".join(available_loras.values())
        lora_names = ",".join(available_loras.keys())
        cmd.extend([
            "--lora-paths", lora_paths,
            "--lora-names", lora_names,
            "--max-loras-per-batch", "3",
        ])
        print(f"  Multi-LoRA enabled: {list(available_loras.keys())}")
    else:
        print(f"  No LoRA adapters found at {lora_base} — serving base model only")

    print(f"[Handyman] Launching on port {sglang['handyman_port']}...")
    print(f"  Command: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

def launch_master_brain(config):
    """Launch the 9B Master Brain server with EAGLE-2 speculative decoding."""
    sglang = config["inference"]["sglang"]
    spec = sglang["speculative_decoding"]
    quant_dir = Path(config["training"]["quantization"]["output_dir"]) / "master_brain" / "merged"
    model_path = str(quant_dir) if quant_dir.exists() else config["models"]["master_brain"]["model_id"]

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(sglang["master_brain_port"]),
        "--tp", str(sglang["tensor_parallel_size"]),
        "--mem-fraction-static", str(sglang["mem_fraction_static"]),
        "--served-model-name", "master_brain",
        "--trust-remote-code",
    ]
    # EAGLE-2 speculative decoding
    if spec.get("enabled") and spec.get("method") == "eagle2":
        handyman_path = str(
            Path(config["training"]["quantization"]["output_dir"]) / "handyman" / "merged"
        )
        if Path(handyman_path).exists():
            cmd.extend([
                "--speculative-algorithm", "EAGLE",
                "--speculative-draft-model-path", handyman_path,
                "--speculative-num-draft-tokens", str(spec["num_speculative_tokens"]),
            ])
            print(f"  EAGLE-2 enabled: {spec['num_speculative_tokens']} draft tokens")

    print(f"[Master Brain] Launching on port {sglang['master_brain_port']}...")
    print(f"  Command: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

def wait_for_server(port, timeout=120):
    """Wait until a server is ready on the given port."""
    import httpx
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def main():
    parser = argparse.ArgumentParser(description="Launch SGLang inference servers")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--model", choices=["handyman", "master_brain", "all"], default="all")
    args = parser.parse_args()

    config = load_config(args.config)
    sglang = config["inference"]["sglang"]
    procs = []

    if args.model in ("handyman", "all"):
        procs.append(("Handyman", launch_handyman(config), sglang["handyman_port"]))
    if args.model in ("master_brain", "all"):
        procs.append(("Master Brain", launch_master_brain(config), sglang["master_brain_port"]))

    # Wait for servers to be ready
    for name, proc, port in procs:
        print(f"Waiting for {name} on port {port}...")
        if wait_for_server(port):
            print(f"  ✅ {name} is ready!")
        else:
            print(f"  ⚠️ {name} did not start within timeout. Check logs.")

    print("\n🚀 All servers launched. Press Ctrl+C to stop.")
    try:
        for _, proc, _ in procs:
            proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        for name, proc, _ in procs:
            proc.terminate()
            print(f"  Stopped {name}")

if __name__ == "__main__":
    main()
