#!/usr/bin/env python3
"""Run one local-asset flat-ground stand diagnostic."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/flat_stand.json"))
    parser.add_argument("--sim-device", default=None, help="defaults to the configured sim_device")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--output-root")
    parser.add_argument("--hold-viewer", action="store_true", help="keep the finished GUI open until closed (max 20 minutes)")
    args = parser.parse_args()
    env_bin = str(Path(sys.executable).resolve().parent)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/physx_pilot_torch_extensions")
    if env_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")
    from hexapod_sim import run_stand
    out = run_stand(args.config, args.sim_device, args.headless, args.duration_s, args.output_root, args.hold_viewer)
    status = out / "status.json"
    if status.exists():
        status_data = json.loads(status.read_text())
        return 0 if status_data.get("status") == "settled" else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
