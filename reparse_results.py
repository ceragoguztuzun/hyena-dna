#!/usr/bin/env python3
"""
Re-parse results from existing log files

This script re-parses training.log files to extract metrics and update results.json

Usage:
    python reparse_results.py results/genomic_benchmark_human_enhancers_cohn_20251020_205809
    python reparse_results.py results/  # Re-parse all results
"""

import argparse
import json
import re
import sys
from pathlib import Path


def parse_log_file(log_file: Path) -> dict:
    """Parse a training log file to extract metrics"""
    results = {
        "accuracy": None,
        "val_accuracy": None,
        "test_accuracy": None,
        "val_loss": None,
        "test_loss": None,
    }

    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return results

    with open(log_file, 'r') as f:
        log_content = f.read()

    # Try different patterns for PyTorch Lightning output
    patterns = {
        "val_accuracy": [
            r"val/accuracy=([0-9.]+)",  # Progress bar format: val/accuracy=0.709
            r"'val/accuracy'\s+reached\s+([0-9.]+)",  # Checkpoint save format
            r"val/accuracy[:\s]+([0-9.]+)",
        ],
        "test_accuracy": [
            r"test/accuracy=([0-9.]+)",
            r"'test/accuracy'\s+reached\s+([0-9.]+)",
            r"test/accuracy[:\s]+([0-9.]+)",
        ],
        "val_loss": [
            r"val/loss=([0-9.]+)",
            r"val/loss[:\s]+([0-9.]+)",
        ],
        "test_loss": [
            r"test/loss=([0-9.]+)",
            r"test/loss[:\s]+([0-9.]+)",
        ],
    }

    # Extract metrics
    for metric, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, log_content)
            if matches:
                results[metric] = float(matches[-1])  # Take last value
                break

    # Set primary accuracy (prefer test, fall back to val)
    if results["test_accuracy"]:
        results["accuracy"] = results["test_accuracy"]
    elif results["val_accuracy"]:
        results["accuracy"] = results["val_accuracy"]

    return results


def reparse_directory(result_dir: Path):
    """Re-parse a single result directory"""
    log_file = result_dir / "training.log"
    results_file = result_dir / "results.json"

    if not log_file.exists():
        print(f"⚠️  No training.log in {result_dir}")
        return False

    # Parse metrics from log
    metrics = parse_log_file(log_file)

    # Load existing results if available
    if results_file.exists():
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = {}

    # Update with parsed metrics
    existing_results.update(metrics)

    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(existing_results, f, indent=2)

    # Print summary
    if metrics["accuracy"]:
        print(f"✓ {result_dir.name}: accuracy={metrics['accuracy']:.4f}")
        return True
    else:
        print(f"✗ {result_dir.name}: No metrics found")
        return False


def main():
    parser = argparse.ArgumentParser(description="Re-parse training logs to extract metrics")
    parser.add_argument("path", type=str, help="Path to result directory or results root")
    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path not found: {path}")
        return 1

    # Check if it's a single result directory or results root
    if (path / "training.log").exists():
        # Single result directory
        success = reparse_directory(path)
        return 0 if success else 1
    else:
        # Results root - find all result directories
        result_dirs = []
        for log_file in path.rglob("training.log"):
            result_dirs.append(log_file.parent)

        if not result_dirs:
            print(f"No training.log files found in {path}")
            return 1

        print(f"Found {len(result_dirs)} result directories\n")

        success_count = 0
        for result_dir in sorted(result_dirs):
            if reparse_directory(result_dir):
                success_count += 1

        print(f"\n✓ Successfully parsed {success_count}/{len(result_dirs)} results")
        return 0


if __name__ == "__main__":
    sys.exit(main())
