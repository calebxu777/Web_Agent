"""
MVP SGLang Server Launcher
Launch two SGLang Runtime instances for the Minimum Viable Prototype:
  1. Handyman (0.8B) - No LoRAs, plain base model for routing/metadata tasks
  2. Master Brain (35B) - Standard Speculative Decoding using the 0.8B as a draft model.

Usage: python scripts/SGLang/mvp_sglang_launcher.py
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path
import yaml

# Add root to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def load_config(p):
    with open(p) as f: return yaml.safe_load(f)

def launch_handyman(config):
    """
    Launch the MVP Handyman (Qwen 0.8B) server.
    We are skipping LoRA loading to evaluate the base model for routing vs generation.
    """
    sglang = config["inference"]["sglang"]
    model_path = config["models"]["handyman"]["model_id"]

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(sglang["handyman_port"]),
        "--tp", str(sglang["tensor_parallel_size"]),
        "--mem-fraction-static", str(sglang["mem_fraction_static"]),
        "--served-model-name", "handyman",
        "--trust-remote-code",
    ]

    print(f"[Handyman MVP] Launching {model_path} on port {sglang['handyman_port']}...")
    print(f"  Command: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

def launch_master_brain(config):
    """
    Launch the MVP Master Brain (Qwen 35B) server.
    Enables native speculative decoding using the same 0.8B model used for the Handyman.
    KV Caching is enabled natively in SGLang via RadixAttention automatically.
    """
    sglang = config["inference"]["sglang"]
    spec = sglang.get("speculative_decoding", {})
    model_path = config["models"]["master_brain"]["model_id"]
    draft_model_path = config["models"]["handyman"]["model_id"]

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(sglang["master_brain_port"]),
        "--tp", str(sglang["tensor_parallel_size"]),
        "--mem-fraction-static", str(sglang["mem_fraction_static"]),
        "--served-model-name", "master_brain",
        "--trust-remote-code",
    ]
    
    # In SGLang, speculative decoding natively loads the draft model into the same memory
    # space as the master model to perform insanely fast target-draft sampling.
    if spec.get("enabled"):
        cmd.extend([
            "--speculative-draft-model-path", draft_model_path,
            "--speculative-num-draft-tokens", str(spec.get("num_speculative_tokens", 5)),
        ])
        print(f"  Standard Speculative Decoding enabled via draft model: {draft_model_path}")

    print(f"[Master Brain MVP] Launching {model_path} on port {sglang['master_brain_port']}...")
    print(f"  Command: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

def wait_for_server(port, timeout=120):
    """Wait until a server is ready on the given port by pinging healthz."""
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
    parser = argparse.ArgumentParser(description="Launch MVP SGLang inference servers")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--model", choices=["handyman", "master_brain", "all"], default="all")
    args = parser.parse_args()

    # Determine absolute path to config
    config_path = Path.cwd() / args.config
    if not config_path.exists():
        # Fall back to root relative
        root = Path(__file__).parent.parent.parent
        config_path = root / args.config

    config = load_config(config_path)
    sglang = config["inference"]["sglang"]
    procs = []

    if args.model in ("handyman", "all"):
        procs.append(("Handyman MVP", launch_handyman(config), sglang["handyman_port"]))
    if args.model in ("master_brain", "all"):
        procs.append(("Master Brain MVP", launch_master_brain(config), sglang["master_brain_port"]))

    # Wait for servers to be ready
    for name, proc, port in procs:
        print(f"Waiting for {name} on port {port}...")
        if wait_for_server(port):
            print(f"  ✅ {name} is ready!")
        else:
            print(f"  ⚠️ {name} did not start within timeout. Check logs.")

    print("\n🚀 All MVP servers launched. Press Ctrl+C to stop.")
    try:
        for name, proc, _ in procs:
            proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        for name, proc, _ in procs:
            proc.terminate()
            print(f"  Stopped {name}")

if __name__ == "__main__":
    main()
