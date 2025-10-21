#!/usr/bin/env python3
"""
HyenaDNA Results Parser

This script parses training logs and checkpoints to extract benchmark results
and creates a summary table comparing to HyenaDNA's reported numbers.

Usage:
    python parse_results.py --results_dir results/
    python parse_results.py --results_dir results/ --output summary.csv
    python parse_results.py --results_dir results/ --format markdown
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from prettytable import PrettyTable

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

# HyenaDNA reported results (from paper/repo)
# These are approximate baseline numbers for comparison
HYENADNA_REPORTED = {
    "genomic_benchmark": {
        "human_enhancers_cohn": {"accuracy": 0.853, "metric": "accuracy"},
        "human_nontata_promoters": {"accuracy": 0.985, "metric": "accuracy"},
        "human_enhancers_ensembl": {"accuracy": 0.797, "metric": "accuracy"},
        "human_ensembl_regulatory": {"accuracy": 0.855, "metric": "accuracy"},
        "human_ocr_ensembl": {"accuracy": 0.863, "metric": "accuracy"},
        "dummy_mouse_enhancers_ensembl": {"accuracy": 0.762, "metric": "accuracy"},
        "demo_coding_vs_intergenomic_seqs": {"accuracy": 0.982, "metric": "accuracy"},
        "demo_human_or_worm": {"accuracy": 0.969, "metric": "accuracy"},
    },
    "nucleotide_transformer": {
        "enhancers": {"mcc": 0.631, "metric": "mcc"},
        "enhancers_types": {"mcc": 0.487, "metric": "mcc"},
        "H2AFZ": {"mcc": 0.398, "metric": "mcc"},
        "H3K4me1": {"mcc": 0.324, "metric": "mcc"},
        "H3K4me2": {"mcc": 0.274, "metric": "mcc"},
        "H3K4me3": {"mcc": 0.368, "metric": "mcc"},
        "H3K9ac": {"mcc": 0.412, "metric": "mcc"},
        "H3K9me3": {"mcc": 0.329, "metric": "mcc"},
        "H3K27ac": {"mcc": 0.445, "metric": "mcc"},
        "H3K27me3": {"mcc": 0.387, "metric": "mcc"},
        "H3K36me3": {"mcc": 0.387, "metric": "mcc"},
        "H4K20me1": {"mcc": 0.421, "metric": "mcc"},
        "promoter_all": {"f1": 0.893, "metric": "f1"},
        "promoter_no_tata": {"f1": 0.881, "metric": "f1"},
        "promoter_tata": {"f1": 0.821, "metric": "f1"},
        "splice_sites_acceptors": {"f1": 0.921, "metric": "f1"},
        "splice_sites_donors": {"f1": 0.965, "metric": "f1"},
        "splice_sites_all": {"f1": 0.940, "metric": "f1"},
    }
}

def parse_hydra_log(log_file: Path) -> Dict:
    """
    Parse PyTorch Lightning log output from Hydra

    Args:
        log_file: Path to training log file

    Returns:
        Dictionary with extracted metrics
    """
    results = {
        "val_accuracy": None,
        "test_accuracy": None,
        "val_mcc": None,
        "test_mcc": None,
        "val_f1_macro": None,
        "test_f1_macro": None,
        "val_loss": None,
        "test_loss": None,
        "epochs_completed": 0,
        "training_time": None,
        "gpu_memory": None
    }

    if not log_file.exists():
        return results

    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Parse metrics from PyTorch Lightning progress bar or logs
        # Look for patterns like:
        # val/accuracy: 0.8523
        # test/accuracy: 0.8612

        # Extract validation metrics
        val_acc_matches = re.findall(r'val[/\s]+accuracy[:\s=]+([0-9.]+)', content, re.IGNORECASE)
        if val_acc_matches:
            results["val_accuracy"] = float(val_acc_matches[-1])  # Take last value

        # Extract MCC metrics
        val_mcc_matches = re.findall(r'val[/\s]+mcc[:\s=]+([0-9.]+)', content, re.IGNORECASE)
        if val_mcc_matches:
            results["val_mcc"] = float(val_mcc_matches[-1])

        # Extract F1 macro metrics
        val_f1_matches = re.findall(r'val[/\s]+f1_macro[:\s=]+([0-9.]+)', content, re.IGNORECASE)
        if val_f1_matches:
            results["val_f1_macro"] = float(val_f1_matches[-1])

        # Extract test metrics
        test_acc_matches = re.findall(r'test[/\s]+accuracy[:\s=]+([0-9.]+)', content, re.IGNORECASE)
        if test_acc_matches:
            results["test_accuracy"] = float(test_acc_matches[-1])

        test_mcc_matches = re.findall(r'test[/\s]+mcc[:\s=]+([0-9.]+)', content, re.IGNORECASE)
        if test_mcc_matches:
            results["test_mcc"] = float(test_mcc_matches[-1])

        test_f1_matches = re.findall(r'test[/\s]+f1_macro[:\s=]+([0-9.]+)', content, re.IGNORECASE)
        if test_f1_matches:
            results["test_f1_macro"] = float(test_f1_matches[-1])

        # Extract loss
        val_loss_matches = re.findall(r'val[/\s]+loss[:\s]+([0-9.]+)', content, re.IGNORECASE)
        if val_loss_matches:
            results["val_loss"] = float(val_loss_matches[-1])

        test_loss_matches = re.findall(r'test[/\s]+loss[:\s]+([0-9.]+)', content, re.IGNORECASE)
        if test_loss_matches:
            results["test_loss"] = float(test_loss_matches[-1])

        # Extract epochs
        epoch_matches = re.findall(r'Epoch\s+(\d+)', content)
        if epoch_matches:
            results["epochs_completed"] = int(epoch_matches[-1])

        # Extract GPU memory usage
        gpu_mem_matches = re.findall(r'GPU\s+memory:\s+([0-9.]+)\s*GB', content, re.IGNORECASE)
        if gpu_mem_matches:
            results["gpu_memory"] = float(gpu_mem_matches[0])

    except Exception as e:
        print(f"Warning: Error parsing log file {log_file}: {e}")

    return results

def find_result_dirs(results_dir: Path) -> List[Path]:
    """
    Find all result directories containing results.json

    Args:
        results_dir: Root results directory

    Returns:
        List of paths to result directories
    """
    result_dirs = []

    # Look for directories with results.json
    for json_file in results_dir.rglob("results.json"):
        result_dirs.append(json_file.parent)

    # Also look for training.log files if no results.json
    for log_file in results_dir.rglob("training.log"):
        if not (log_file.parent / "results.json").exists():
            result_dirs.append(log_file.parent)

    return sorted(result_dirs)

def extract_results_from_dir(result_dir: Path) -> Optional[Dict]:
    """
    Extract results from a single result directory

    Args:
        result_dir: Path to result directory

    Returns:
        Dictionary with extracted results, or None if failed
    """
    # Try to load results.json first
    results_json = result_dir / "results.json"
    if results_json.exists():
        try:
            with open(results_json, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            print(f"Warning: Failed to load {results_json}: {e}")

    # Otherwise, try to parse from logs
    log_file = result_dir / "training.log"
    if log_file.exists():
        # Parse dataset and suite from directory name
        dir_name = result_dir.name
        parts = dir_name.split('_')

        if len(parts) >= 2:
            suite = parts[0]
            dataset = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1]

            parsed_results = parse_hydra_log(log_file)
            parsed_results["suite"] = suite
            parsed_results["dataset"] = dataset
            parsed_results["output_dir"] = str(result_dir)

            return parsed_results

    return None

def create_summary_table(all_results: List[Dict]) -> PrettyTable:
    """
    Create a formatted summary table

    Args:
        all_results: List of result dictionaries

    Returns:
        PrettyTable object
    """
    table = PrettyTable()
    table.field_names = [
        "Suite", "Dataset", "Test Score", "Val Score",
        "Baseline", "Δ", "Epochs", "Time (s)", "Status"
    ]
    table.align["Dataset"] = "l"
    table.align["Test Score"] = "r"
    table.align["Val Score"] = "r"
    table.align["Baseline"] = "r"

    for result in sorted(all_results, key=lambda x: (x.get("suite", ""), x.get("dataset", ""))):
        suite = result.get("suite", "unknown")
        dataset = result.get("dataset", "unknown")

        # Get metrics - prioritize MCC/F1 for NT tasks, accuracy for GB tasks
        val_metric = result.get("val_mcc") or result.get("val_f1_macro") or result.get("val_accuracy")
        test_metric = result.get("test_mcc") or result.get("test_f1_macro") or result.get("test_accuracy")

        training_time = result.get("training_time")
        epochs = result.get("config", {}).get("epochs", result.get("epochs_completed", "?"))
        status = result.get("status", "unknown")

        # Format metrics
        test_metric_str = f"{test_metric:.4f}" if test_metric else "N/A"
        val_metric_str = f"{val_metric:.4f}" if val_metric else "N/A"
        time_str = f"{training_time:.0f}" if training_time else "N/A"

        # Get baseline for comparison
        baseline_str = "N/A"
        delta_str = "N/A"

        if suite in HYENADNA_REPORTED and dataset in HYENADNA_REPORTED[suite]:
            baseline_info = HYENADNA_REPORTED[suite][dataset]
            metric_name = baseline_info["metric"]

            if metric_name == "accuracy" and test_metric:
                baseline = baseline_info["accuracy"]
                baseline_str = f"{baseline:.4f}"
                delta = test_metric - baseline
                delta_str = f"{delta:+.4f}"
            elif metric_name == "mcc" and test_metric:
                baseline = baseline_info.get("mcc")
                if baseline:
                    baseline_str = f"{baseline:.4f}"
                    delta = test_metric - baseline
                    delta_str = f"{delta:+.4f}"
            elif metric_name == "f1" and test_metric:
                baseline = baseline_info.get("f1")
                if baseline:
                    baseline_str = f"{baseline:.4f}"
                    delta = test_metric - baseline
                    delta_str = f"{delta:+.4f}"

        # Color code status
        if status == "success":
            status_str = f"✓ {status}"
        else:
            status_str = f"✗ {status}"

        table.add_row([
            suite[:12],
            dataset[:30],
            test_metric_str,
            val_metric_str,
            baseline_str,
            delta_str,
            epochs,
            time_str,
            status_str
        ])

    return table

def save_csv(all_results: List[Dict], output_file: Path):
    """
    Save results to CSV file

    Args:
        all_results: List of result dictionaries
        output_file: Path to output CSV file
    """
    if not all_results:
        print("Warning: No results to save")
        return

    # Create CSV with all available fields
    fieldnames = [
        "suite", "dataset", "test_accuracy", "val_accuracy",
        "test_mcc", "val_mcc", "test_f1_macro", "val_f1_macro",
        "test_loss", "val_loss", "epochs", "training_time", "status",
        "d_model", "n_layer", "batch_size", "output_dir"
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for result in all_results:
            # Flatten config into top-level fields
            row = result.copy()
            if "config" in result:
                for k, v in result["config"].items():
                    row[k] = v

            writer.writerow(row)

    print(f"Results saved to: {output_file}")

def save_markdown(all_results: List[Dict], output_file: Path):
    """
    Save results as a markdown table

    Args:
        all_results: List of result dictionaries
        output_file: Path to output markdown file
    """
    with open(output_file, 'w') as f:
        f.write("# HyenaDNA Benchmark Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Separate by suite
        genomic_results = [r for r in all_results if r.get("suite") == "genomic_benchmark"]
        nucleotide_results = [r for r in all_results if r.get("suite") == "nucleotide_transformer"]

        if genomic_results:
            f.write("## GenomicBenchmarks\n\n")
            f.write("| Dataset | Test Acc | Val Acc | Baseline | Δ | Epochs | Status |\n")
            f.write("|---------|----------|---------|----------|---|--------|--------|\n")

            for result in sorted(genomic_results, key=lambda x: x.get("dataset", "")):
                dataset = result.get("dataset", "unknown")
                test_acc = result.get("test_accuracy")
                val_acc = result.get("val_accuracy")
                epochs = result.get("config", {}).get("epochs", "?")
                status = result.get("status", "unknown")

                test_acc_str = f"{test_acc:.4f}" if test_acc else "N/A"
                val_acc_str = f"{val_acc:.4f}" if val_acc else "N/A"

                baseline_str = "N/A"
                delta_str = "N/A"
                if dataset in HYENADNA_REPORTED["genomic_benchmark"] and test_acc:
                    baseline = HYENADNA_REPORTED["genomic_benchmark"][dataset]["accuracy"]
                    baseline_str = f"{baseline:.4f}"
                    delta = test_acc - baseline
                    delta_str = f"{delta:+.4f}"

                f.write(f"| {dataset} | {test_acc_str} | {val_acc_str} | {baseline_str} | {delta_str} | {epochs} | {status} |\n")

            f.write("\n")

        if nucleotide_results:
            f.write("## Nucleotide Transformer\n\n")
            f.write("| Dataset | Test Score | Val Score | Baseline | Δ | Epochs | Status |\n")
            f.write("|---------|------------|-----------|----------|---|--------|--------|\n")

            for result in sorted(nucleotide_results, key=lambda x: x.get("dataset", "")):
                dataset = result.get("dataset", "unknown")
                test_acc = result.get("test_accuracy")
                val_acc = result.get("val_accuracy")
                epochs = result.get("config", {}).get("epochs", "?")
                status = result.get("status", "unknown")

                test_acc_str = f"{test_acc:.4f}" if test_acc else "N/A"
                val_acc_str = f"{val_acc:.4f}" if val_acc else "N/A"

                f.write(f"| {dataset} | {test_acc_str} | {val_acc_str} | N/A | N/A | {epochs} | {status} |\n")

            f.write("\n")

    print(f"Markdown summary saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Parse HyenaDNA benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse results and display summary
  python parse_results.py --results_dir results/

  # Save to CSV
  python parse_results.py --results_dir results/ --output summary.csv

  # Save as markdown
  python parse_results.py --results_dir results/ --format markdown --output summary.md
        """
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing result subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (CSV or markdown)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "markdown", "both"],
        default="csv",
        help="Output format"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    print_header("Parsing HyenaDNA Benchmark Results")

    # Find all result directories
    print(f"Searching for results in: {results_dir}")
    result_dirs = find_result_dirs(results_dir)
    print(f"Found {len(result_dirs)} result directories\n")

    # Extract results from each directory
    all_results = []
    for result_dir in result_dirs:
        result = extract_results_from_dir(result_dir)
        if result:
            all_results.append(result)
        else:
            print(f"Warning: Could not extract results from {result_dir}")

    if not all_results:
        print("No results found!")
        return 1

    print(f"Successfully extracted {len(all_results)} results\n")

    # Create and display summary table
    print_header("Results Summary")
    table = create_summary_table(all_results)
    print(table)
    print()

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.format == "markdown":
            output_path = results_dir / f"summary_{timestamp}.md"
        else:
            output_path = results_dir / f"summary_{timestamp}.csv"

    if args.format in ["csv", "both"]:
        csv_path = output_path.with_suffix('.csv')
        save_csv(all_results, csv_path)

    if args.format in ["markdown", "both"]:
        md_path = output_path.with_suffix('.md')
        save_markdown(all_results, md_path)

    # Print summary statistics
    print_header("Summary Statistics")
    successful = sum(1 for r in all_results if r.get("status") == "success")
    failed = len(all_results) - successful

    print(f"Total results: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # Average metrics
    test_accs = [r["test_accuracy"] for r in all_results if r.get("test_accuracy")]
    if test_accs:
        print(f"Average test accuracy: {sum(test_accs) / len(test_accs):.4f}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
