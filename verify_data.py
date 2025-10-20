#!/usr/bin/env python3
"""
HyenaDNA Data Verification Script

This script checks the availability of datasets for genomic benchmarks.
It verifies:
1. GenomicBenchmarks datasets (8 datasets, auto-download)
2. Nucleotide Transformer datasets (18 datasets, manual download required)

Usage:
    python verify_data.py
    python verify_data.py --download-genomic-benchmarks
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from prettytable import PrettyTable

# Try to import genomic_benchmarks
try:
    from genomic_benchmarks.data_check import is_downloaded, list_datasets
    from genomic_benchmarks.loc2seq import download_dataset
    GENOMIC_BENCHMARKS_AVAILABLE = True
except ImportError:
    GENOMIC_BENCHMARKS_AVAILABLE = False
    print("⚠️  Warning: genomic_benchmarks not installed. Run 'pip install genomic-benchmarks'")

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.END} {text}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.END} {text}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ{Colors.END} {text}")

# GenomicBenchmarks dataset information
GENOMIC_BENCHMARKS_DATASETS = {
    "dummy_mouse_enhancers_ensembl": {
        "num_seqs": 1210,
        "num_classes": 2,
        "median_len": 2381,
        "std": 984.4
    },
    "demo_coding_vs_intergenomic_seqs": {
        "num_seqs": 100000,
        "num_classes": 2,
        "median_len": 200,
        "std": 0
    },
    "demo_human_or_worm": {
        "num_seqs": 100000,
        "num_classes": 2,
        "median_len": 200,
        "std": 0
    },
    "human_enhancers_cohn": {
        "num_seqs": 27791,
        "num_classes": 2,
        "median_len": 500,
        "std": 0
    },
    "human_enhancers_ensembl": {
        "num_seqs": 154842,
        "num_classes": 2,
        "median_len": 269,
        "std": 122.6
    },
    "human_ensembl_regulatory": {
        "num_seqs": 289061,
        "num_classes": 3,
        "median_len": 401,
        "std": 184.3
    },
    "human_nontata_promoters": {
        "num_seqs": 36131,
        "num_classes": 2,
        "median_len": 251,
        "std": 0
    },
    "human_ocr_ensembl": {
        "num_seqs": 174756,
        "num_classes": 2,
        "median_len": 315,
        "std": 108.1
    }
}

# Nucleotide Transformer dataset information
NUCLEOTIDE_TRANSFORMER_DATASETS = {
    "enhancer": {"classes": 2, "max_length": 200, "metric": "mcc", "train_len": 14968},
    "enhancer_types": {"classes": 3, "max_length": 200, "metric": "mcc", "train_len": 14968},
    "H3": {"classes": 2, "max_length": 500, "metric": "mcc", "train_len": 13468},
    "H3K4me1": {"classes": 2, "max_length": 500, "metric": "mcc", "train_len": 28509},
    "H3K4me2": {"classes": 2, "max_length": 500, "metric": "mcc", "train_len": 27614},
    "H3K4me3": {"classes": 2, "max_length": 500, "metric": "mcc", "train_len": 33119},
    "H3K9ac": {"classes": 2, "max_length": 500, "metric": "mcc", "train_len": 25003},
    "H3K14ac": {"classes": 2, "max_length": 500, "metric": "mcc", "train_len": 29743},
    "H3K36me3": {"classes": 2, "max_length": 500, "metric": "mcc", "train_len": 31392},
    "H3K79me3": {"classes": 2, "max_length": 500, "metric": "mcc", "train_len": 25953},
    "H4": {"classes": 2, "max_length": 500, "metric": "mcc", "train_len": 13140},
    "H4ac": {"classes": 2, "max_length": 500, "metric": "mcc", "train_len": 30685},
    "promoter_all": {"classes": 2, "max_length": 300, "metric": "f1_macro", "train_len": 53276},
    "promoter_non_tata": {"classes": 2, "max_length": 300, "metric": "f1_macro", "train_len": 47759},
    "promoter_tata": {"classes": 2, "max_length": 300, "metric": "f1_macro", "train_len": 5517},
    "splice_sites_acceptor": {"classes": 2, "max_length": 600, "metric": "f1_macro", "train_len": 19961},
    "splice_sites_donor": {"classes": 2, "max_length": 600, "metric": "f1_macro", "train_len": 19775},
}

def check_genomic_benchmarks(data_path: str = "data/genomic_benchmark") -> Tuple[List[str], List[str]]:
    """
    Check which GenomicBenchmarks datasets are available

    Returns:
        Tuple of (available_datasets, missing_datasets)
    """
    if not GENOMIC_BENCHMARKS_AVAILABLE:
        print_error("genomic_benchmarks package not installed")
        return [], list(GENOMIC_BENCHMARKS_DATASETS.keys())

    print_header("Checking GenomicBenchmarks Datasets")

    available = []
    missing = []

    table = PrettyTable()
    table.field_names = ["Dataset", "Status", "Sequences", "Classes", "Median Len"]

    for dataset_name, info in GENOMIC_BENCHMARKS_DATASETS.items():
        if is_downloaded(dataset_name, cache_path=data_path):
            available.append(dataset_name)
            status = f"{Colors.GREEN}✓ Available{Colors.END}"
            table.add_row([
                dataset_name[:30],
                status,
                info["num_seqs"],
                info["num_classes"],
                info["median_len"]
            ])
        else:
            missing.append(dataset_name)
            status = f"{Colors.RED}✗ Missing{Colors.END}"
            table.add_row([
                dataset_name[:30],
                status,
                info["num_seqs"],
                info["num_classes"],
                info["median_len"]
            ])

    print(table)
    print(f"\nSummary: {len(available)}/{len(GENOMIC_BENCHMARKS_DATASETS)} datasets available")

    return available, missing

def check_nucleotide_transformer(data_path: str = "data/nucleotide_transformer") -> Tuple[List[str], List[str]]:
    """
    Check which Nucleotide Transformer datasets are available

    Returns:
        Tuple of (available_datasets, missing_datasets)
    """
    print_header("Checking Nucleotide Transformer Datasets")

    available = []
    missing = []

    data_path_obj = Path(data_path)

    # Check if the main directory exists
    if not data_path_obj.exists():
        print_warning(f"Data path does not exist: {data_path}")
        print_info("These datasets need to be downloaded manually")
        return [], list(NUCLEOTIDE_TRANSFORMER_DATASETS.keys())

    table = PrettyTable()
    table.field_names = ["Dataset", "Status", "Train Samples", "Max Length", "Metric"]

    for dataset_name, info in NUCLEOTIDE_TRANSFORMER_DATASETS.items():
        # Check if dataset directory exists and has train/test splits
        dataset_dir = data_path_obj / dataset_name
        train_file = dataset_dir / "train.csv"
        test_file = dataset_dir / "test.csv"

        if dataset_dir.exists() and train_file.exists():
            available.append(dataset_name)
            status = f"{Colors.GREEN}✓ Available{Colors.END}"
        else:
            missing.append(dataset_name)
            status = f"{Colors.RED}✗ Missing{Colors.END}"

        table.add_row([
            dataset_name[:25],
            status,
            info["train_len"],
            info["max_length"],
            info["metric"].upper()
        ])

    print(table)
    print(f"\nSummary: {len(available)}/{len(NUCLEOTIDE_TRANSFORMER_DATASETS)} datasets available")

    return available, missing

def download_genomic_benchmarks(data_path: str = "data/genomic_benchmark"):
    """Download all missing GenomicBenchmarks datasets"""
    if not GENOMIC_BENCHMARKS_AVAILABLE:
        print_error("genomic_benchmarks package not installed")
        return

    print_header("Downloading GenomicBenchmarks Datasets")

    for dataset_name in GENOMIC_BENCHMARKS_DATASETS.keys():
        if not is_downloaded(dataset_name, cache_path=data_path):
            print_info(f"Downloading {dataset_name}...")
            try:
                download_dataset(dataset_name, version=0, dest_path=data_path)
                print_success(f"Downloaded {dataset_name}")
            except Exception as e:
                print_error(f"Failed to download {dataset_name}: {e}")
        else:
            print_success(f"{dataset_name} already downloaded")

def print_nucleotide_transformer_instructions():
    """Print instructions for downloading Nucleotide Transformer datasets"""
    print_header("Nucleotide Transformer Dataset Instructions")

    print("The Nucleotide Transformer datasets need to be downloaded manually.")
    print("\nOption 1: Download from InstaDeep's Nucleotide Transformer repository")
    print("  Repository: https://github.com/instadeepai/nucleotide-transformer")
    print("\nOption 2: Check the original paper for data sources:")
    print("  Paper: https://doi.org/10.1101/2023.01.11.523679")
    print("\nExpected directory structure:")
    print("  data/nucleotide_transformer/")
    print("    ├── enhancer/")
    print("    │   ├── train.csv")
    print("    │   └── test.csv")
    print("    ├── H3K4me1/")
    print("    │   ├── train.csv")
    print("    │   └── test.csv")
    print("    └── ... (other datasets)")
    print("\nNote: You can still run GenomicBenchmarks without these datasets.")

def main():
    parser = argparse.ArgumentParser(description="Verify HyenaDNA benchmark data availability")
    parser.add_argument(
        "--download-genomic-benchmarks",
        action="store_true",
        help="Download missing GenomicBenchmarks datasets"
    )
    parser.add_argument(
        "--genomic-benchmarks-path",
        type=str,
        default="data/genomic_benchmark",
        help="Path to GenomicBenchmarks data directory"
    )
    parser.add_argument(
        "--nucleotide-transformer-path",
        type=str,
        default="data/nucleotide_transformer",
        help="Path to Nucleotide Transformer data directory"
    )

    args = parser.parse_args()

    print(f"{Colors.BOLD}HyenaDNA Data Verification{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")

    # Check GenomicBenchmarks
    gb_available, gb_missing = check_genomic_benchmarks(args.genomic_benchmarks_path)

    # Download if requested
    if args.download_genomic_benchmarks and gb_missing:
        response = input("\nDownload missing GenomicBenchmarks datasets? (y/n): ")
        if response.lower() == 'y':
            download_genomic_benchmarks(args.genomic_benchmarks_path)
            gb_available, gb_missing = check_genomic_benchmarks(args.genomic_benchmarks_path)

    # Check Nucleotide Transformer
    nt_available, nt_missing = check_nucleotide_transformer(args.nucleotide_transformer_path)

    if nt_missing:
        print_nucleotide_transformer_instructions()

    # Final summary
    print_header("Final Summary")
    print(f"GenomicBenchmarks: {len(gb_available)}/{len(GENOMIC_BENCHMARKS_DATASETS)} ready")
    print(f"Nucleotide Transformer: {len(nt_available)}/{len(NUCLEOTIDE_TRANSFORMER_DATASETS)} ready")

    if gb_available:
        print_success("\nYou can run GenomicBenchmarks!")
        print_info("Use: python run_single_benchmark.py --dataset human_enhancers_cohn")
        print_info("Or: bash run_all_benchmarks.sh")

    if gb_missing:
        print_warning(f"\n{len(gb_missing)} GenomicBenchmarks datasets missing")
        print_info("Run with --download-genomic-benchmarks to download them")

    # Exit code
    if gb_missing and not nt_available:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
