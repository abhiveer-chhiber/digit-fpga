"""Generate vectors, run the neuron simulation, and save its results."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(command):
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="neuron-sim-v1")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("Use a single report name")

    report_path = ROOT / "experiments/results" / f"{args.name}.json"
    if report_path.exists():
        raise SystemExit("Report already exists. Choose a new --name.")

    for tool in ("iverilog", "vvp"):
        if shutil.which(tool) is None:
            raise SystemExit(f"Required tool not installed: {tool}")

    sources = [
        "hardware/rtl/fixed_neuron.sv",
        "hardware/tb/fixed_neuron_tb.sv",
        "src/fixed_point/generate_neuron_vectors.py",
        "src/fixed_point/inference.py",
        "src/fixed_point/run_neuron_sim.py",
        "site_assets/network-view.json",
    ]
    hashes = {}
    for name in sources:
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()

    generated = run([
        sys.executable, "src/fixed_point/generate_neuron_vectors.py"
    ])
    categories = {
        name: int(count)
        for name, count in re.findall(
            r"^(boundary|generated|model): (\d+) cases$",
            generated.stdout, re.MULTILINE,
        )
    }
    if set(categories) != {"boundary", "generated", "model"}:
        raise RuntimeError("Missing test-vector counts")

    run([
        "iverilog", "-g2012", "-Wall", "-s", "fixed_neuron_tb",
        "-o", "build/sim/fixed_neuron_tb",
        "hardware/rtl/fixed_neuron.sv",
        "hardware/tb/fixed_neuron_tb.sv",
    ])
    simulation = run(["vvp", "build/sim/fixed_neuron_tb"])
    total = sum(categories.values())
    expected = (
        f"PASS: {total} neuron cases matched Python outputs and saturation flags"
    )
    if expected not in simulation.stdout:
        raise RuntimeError("Expected arithmetic pass message not found")
    if "PASS: reset, input gaps, saved bias/ReLU, busy-start handling, and completion pulse" not in simulation.stdout:
        raise RuntimeError("Expected control-signal pass message not found")

    version = subprocess.run(
        ["iverilog", "-V"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.splitlines()[0]
    vector_path = ROOT / "build/sim/neuron-vectors.txt"
    report = {
        "schema": 1,
        "status": "passed",
        "tool": version,
        "storage_bits": 12,
        "fractional_bits": 6,
        "accumulator_bits": 32,
        "cases": categories,
        "total_cases": total,
        "checks": [
            "integer output",
            "accumulator saturation flag",
            "output saturation flag",
            "reset during calculation",
            "gaps between valid inputs",
            "bias and ReLU captured at start",
            "start ignored while busy",
            "single-cycle completion",
            "output held while idle",
        ],
        "source_sha256": hashes,
        "vectors_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
        "simulation_output": simulation.stdout,
        "scope": "RTL simulation; no FPGA execution or physical timing measurements",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
