#!/usr/bin/env python3
"""
HyenaDNA Single Benchmark Runner

This script runs a single benchmark dataset for testing and verification.
It's designed for quick testing before running the full benchmark suite.

Usage:
    # Run with default settings (small model, 5 epochs)
    python run_single_benchmark.py --dataset human_enhancers_cohn

    # Run with custom settings
    python run_single_benchmark.py --dataset human_enhancers_cohn \\
        --epochs 10 --d_model 256 --n_layer 4 --batch_size 64

    # Run Nucleotide Transformer dataset
    python run_single_benchmark.py --suite nucleotide_transformer \\
        --dataset enhancer --epochs 10
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Dataset configurations
GENOMIC_BENCHMARKS_DATASETS = {
    "dummy_mouse_enhancers_ensembl": {"max_length": 2381, "classes": 2},
    "demo_coding_vs_intergenomic_seqs": {"max_length": 200, "classes": 2},
    "demo_human_or_worm": {"max_length": 200, "classes": 2},
    "human_enhancers_cohn": {"max_length": 500, "classes": 2},
    "human_enhancers_ensembl": {"max_length": 269, "classes": 2},
    "human_ensembl_regulatory": {"max_length": 401, "classes": 3},
    "human_nontata_promoters": {"max_length": 251, "classes": 2},
    "human_ocr_ensembl": {"max_length": 315, "classes": 2}
}

NUCLEOTIDE_TRANSFORMER_DATASETS = {
    "enhancer": {"max_length": 200, "classes": 2},
    "enhancer_types": {"max_length": 200, "classes": 3},
    "H3": {"max_length": 500, "classes": 2},
    "H3K4me1": {"max_length": 500, "classes": 2},
    "H3K4me2": {"max_length": 500, "classes": 2},
    "H3K4me3": {"max_length": 500, "classes": 2},
    "H3K9ac": {"max_length": 500, "classes": 2},
    "H3K14ac": {"max_length": 500, "classes": 2},
    "H3K36me3": {"max_length": 500, "classes": 2},
    "H3K79me3": {"max_length": 500, "classes": 2},
    "H4": {"max_length": 500, "classes": 2},
    "H4ac": {"max_length": 500, "classes": 2},
    "promoter_all": {"max_length": 300, "classes": 2},
    "promoter_non_tata": {"max_length": 300, "classes": 2},
    "promoter_tata": {"max_length": 300, "classes": 2},
    "splice_sites_acceptor": {"max_length": 600, "classes": 2},
    "splice_sites_donor": {"max_length": 600, "classes": 2}
}

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.END} {text}")

def print_error(text: str):
    print(f"{Colors.RED}✗{Colors.END} {text}")

def print_info(text: str):
    print(f"{Colors.BLUE}ℹ{Colors.END} {text}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")

def build_command(
    suite: str,
    dataset: str,
    epochs: int,
    d_model: int,
    n_layer: int,
    batch_size: int,
    max_length: Optional[int] = None,
    use_pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> list:
    """
    Build the training command for HyenaDNA

    Args:
        suite: 'genomic_benchmark' or 'nucleotide_transformer'
        dataset: Name of the dataset
        epochs: Number of training epochs
        d_model: Model dimension
        n_layer: Number of layers
        batch_size: Batch size
        max_length: Maximum sequence length (auto-set if None)
        use_pretrained: Whether to use pretrained weights
        pretrained_path: Path to pretrained model checkpoint
        output_dir: Output directory for checkpoints

    Returns:
        List of command arguments
    """
    # Get dataset info
    if suite == "genomic_benchmark":
        if dataset not in GENOMIC_BENCHMARKS_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        dataset_info = GENOMIC_BENCHMARKS_DATASETS[dataset]
        experiment = "hg38/genomic_benchmark"
    else:  # nucleotide_transformer
        if dataset not in NUCLEOTIDE_TRANSFORMER_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        dataset_info = NUCLEOTIDE_TRANSFORMER_DATASETS[dataset]
        experiment = "hg38/nucleotide_transformer"

    # Auto-set max_length if not provided
    if max_length is None:
        max_length = dataset_info["max_length"]

    # Build command
    cmd = [
        "python", "-m", "train",
        f"wandb=null",  # Disable wandb
        f"experiment={experiment}",
        f"dataset.dataset_name={dataset}",  # Fixed: need dataset. prefix
        f"dataset.max_length={max_length}",
        f"model.d_model={d_model}",
        f"model.n_layer={n_layer}",
        f"dataset.batch_size={batch_size}",
        f"trainer.max_epochs={epochs}",
        f"trainer.devices=1",
        f"model.layer.l_max={max_length + 2}",  # l_max needs +2 for special tokens
        f"train.pretrained_model_path=null",  # Disable pretrained model by default
    ]

    # Add pretrained model if specified
    if use_pretrained and pretrained_path:
        cmd.extend([
            f"train.pretrained_model_path={pretrained_path}",
            "train.pretrained_model_strict_load=False"
        ])

    # Add output directory if specified
    if output_dir:
        cmd.append(f"hydra.run.dir={output_dir}")

    return cmd

def parse_output_log(log_file: Path) -> Dict:
    """
    Parse the training output log to extract metrics

    Args:
        log_file: Path to the log file

    Returns:
        Dictionary with extracted metrics
    """
    results = {
        "accuracy": None,
        "val_accuracy": None,
        "test_accuracy": None,
        "val_loss": None,
        "test_loss": None,
        "training_time": None,
        "status": "unknown"
    }

    if not log_file.exists():
        results["status"] = "log_not_found"
        return results

    try:
        with open(log_file, 'r') as f:
            log_content = f.read()

        # PyTorch Lightning progress bar format
        # Look for patterns like: val/accuracy=0.8523
        # or 'val/accuracy': 0.8523

        # Try different patterns
        patterns = [
            r"val/accuracy[=:\s]+([0-9.]+)",
            r"'val/accuracy'[:\s]+([0-9.]+)",
            r"test/accuracy[=:\s]+([0-9.]+)",
            r"'test/accuracy'[:\s]+([0-9.]+)",
            r"val/loss[=:\s]+([0-9.]+)",
            r"'val/loss'[:\s]+([0-9.]+)",
            r"test/loss[=:\s]+([0-9.]+)",
            r"'test/loss'[:\s]+([0-9.]+)",
        ]

        import re

        # Extract validation accuracy
        val_acc_matches = re.findall(patterns[0], log_content) or re.findall(patterns[1], log_content)
        if val_acc_matches:
            results["val_accuracy"] = float(val_acc_matches[-1])  # Take last value
            results["accuracy"] = results["val_accuracy"]  # Use val as default

        # Extract test accuracy
        test_acc_matches = re.findall(patterns[2], log_content) or re.findall(patterns[3], log_content)
        if test_acc_matches:
            results["test_accuracy"] = float(test_acc_matches[-1])
            results["accuracy"] = results["test_accuracy"]  # Test overrides val

        # Extract validation loss
        val_loss_matches = re.findall(patterns[4], log_content) or re.findall(patterns[5], log_content)
        if val_loss_matches:
            results["val_loss"] = float(val_loss_matches[-1])

        # Extract test loss
        test_loss_matches = re.findall(patterns[6], log_content) or re.findall(patterns[7], log_content)
        if test_loss_matches:
            results["test_loss"] = float(test_loss_matches[-1])

        # Mark as completed if we found any metrics
        if results["val_accuracy"] or results["test_accuracy"]:
            results["status"] = "completed"
        else:
            results["status"] = "no_metrics_found"

    except Exception as e:
        results["status"] = f"parse_error: {e}"

    return results

def run_benchmark(args: argparse.Namespace) -> Dict:
    """
    Run a single benchmark

    Args:
        args: Command line arguments

    Returns:
        Dictionary with results
    """
    print_header(f"Running Benchmark: {args.dataset}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.suite}_{args.dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "training.log"

    # Build command
    try:
        cmd = build_command(
            suite=args.suite,
            dataset=args.dataset,
            epochs=args.epochs,
            d_model=args.d_model,
            n_layer=args.n_layer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_pretrained=args.use_pretrained,
            pretrained_path=args.pretrained_path,
            output_dir=str(output_dir)
        )
    except ValueError as e:
        print_error(f"Error building command: {e}")
        return {"status": "error", "error": str(e)}

    # Print configuration
    print_info("Configuration:")
    print(f"  Suite: {args.suite}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Layers: {args.n_layer}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output directory: {output_dir}")
    print()

    # Print command
    print_info("Command:")
    print(" ".join(cmd))
    print()

    # Run training
    print_info("Starting training...")
    start_time = time.time()

    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()

            process.wait()

        training_time = time.time() - start_time

        if process.returncode == 0:
            print_success(f"Training completed successfully in {training_time:.2f}s")
            status = "success"
        else:
            print_error(f"Training failed with return code {process.returncode}")
            status = "failed"

    except KeyboardInterrupt:
        print_warning("\nTraining interrupted by user")
        status = "interrupted"
        training_time = time.time() - start_time
    except Exception as e:
        print_error(f"Error during training: {e}")
        status = "error"
        training_time = time.time() - start_time

    # Parse results
    results = parse_output_log(log_file)
    results["status"] = status
    results["training_time"] = training_time
    results["dataset"] = args.dataset
    results["suite"] = args.suite
    results["config"] = {
        "epochs": args.epochs,
        "d_model": args.d_model,
        "n_layer": args.n_layer,
        "batch_size": args.batch_size
    }
    results["output_dir"] = str(output_dir)
    results["log_file"] = str(log_file)

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print_info(f"Results saved to: {results_file}")

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Run a single HyenaDNA benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small model
  python run_single_benchmark.py --dataset human_enhancers_cohn

  # Longer training with larger model
  python run_single_benchmark.py --dataset human_enhancers_cohn \\
      --epochs 50 --d_model 256 --n_layer 4

  # Run Nucleotide Transformer dataset
  python run_single_benchmark.py --suite nucleotide_transformer \\
      --dataset enhancer
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., human_enhancers_cohn)"
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="genomic_benchmark",
        choices=["genomic_benchmark", "nucleotide_transformer"],
        help="Benchmark suite"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Model dimension (default: 128)"
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=2,
        help="Number of layers (default: 2)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length (auto-set if not specified)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        help="Use pretrained model weights"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint"
    )

    args = parser.parse_args()

    # Validate dataset
    if args.suite == "genomic_benchmark":
        if args.dataset not in GENOMIC_BENCHMARKS_DATASETS:
            print_error(f"Unknown GenomicBenchmarks dataset: {args.dataset}")
            print_info("Available datasets:")
            for name in GENOMIC_BENCHMARKS_DATASETS.keys():
                print(f"  - {name}")
            return 1
    else:
        if args.dataset not in NUCLEOTIDE_TRANSFORMER_DATASETS:
            print_error(f"Unknown Nucleotide Transformer dataset: {args.dataset}")
            print_info("Available datasets:")
            for name in NUCLEOTIDE_TRANSFORMER_DATASETS.keys():
                print(f"  - {name}")
            return 1

    # Run benchmark
    results = run_benchmark(args)

    # Print summary
    print_header("Results Summary")
    print(f"Status: {results['status']}")
    if results.get('training_time'):
        print(f"Training time: {results['training_time']:.2f}s")
    if results.get('test_accuracy'):
        print(f"Test accuracy: {results['test_accuracy']:.4f}")
    if results.get('val_accuracy'):
        print(f"Validation accuracy: {results['val_accuracy']:.4f}")
    print(f"Results saved to: {results.get('output_dir', 'unknown')}")

    return 0 if results['status'] in ['success', 'completed'] else 1

if __name__ == "__main__":
    sys.exit(main())
