#!/usr/bin/env python3
"""
Convert Nucleotide Transformer CSV files to FASTA format

The verify_data.py downloads datasets as CSV, but the dataloader expects FASTA.
This script converts the CSV files to FASTA format.

Usage:
    python convert_nt_to_fasta.py
    python convert_nt_to_fasta.py --data_path data/nucleotide_transformer
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def csv_to_fasta(csv_file: Path, fasta_file: Path):
    """
    Convert CSV to FASTA format

    Expected CSV format:
    - 'sequence' or 'seq' column: DNA sequence
    - 'label' column: classification label (0 or 1)
    """
    print(f"Converting {csv_file.name} to {fasta_file.name}...")

    # Read CSV
    df = pd.read_csv(csv_file)

    # Determine column names
    seq_col = 'sequence' if 'sequence' in df.columns else 'seq'
    label_col = 'label' if 'label' in df.columns else 'target'

    if seq_col not in df.columns:
        raise ValueError(f"Could not find sequence column in {csv_file}")
    if label_col not in df.columns:
        raise ValueError(f"Could not find label column in {csv_file}")

    # Write FASTA
    with open(fasta_file, 'w') as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Writing {fasta_file.name}"):
            # FASTA format: >header\nsequence
            # Header format: >{index}_{label}
            header = f">{idx}_{row[label_col]}"
            sequence = row[seq_col]
            f.write(f"{header}\n{sequence}\n")

    print(f"✓ Created {fasta_file} ({len(df)} sequences)")


def convert_dataset(dataset_dir: Path):
    """Convert train.csv and test.csv to FASTA format"""
    train_csv = dataset_dir / "train.csv"
    test_csv = dataset_dir / "test.csv"

    train_fasta = dataset_dir / "train.fasta"
    test_fasta = dataset_dir / "test.fasta"

    if not train_csv.exists():
        print(f"⚠ Skipping {dataset_dir.name}: train.csv not found")
        return False

    try:
        # Convert train
        if train_csv.exists() and not train_fasta.exists():
            csv_to_fasta(train_csv, train_fasta)

        # Convert test
        if test_csv.exists() and not test_fasta.exists():
            csv_to_fasta(test_csv, test_fasta)

        return True
    except Exception as e:
        print(f"✗ Error converting {dataset_dir.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Nucleotide Transformer CSV files to FASTA format"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/nucleotide_transformer",
        help="Path to nucleotide transformer data directory"
    )

    args = parser.parse_args()
    data_path = Path(args.data_path)

    if not data_path.exists():
        print(f"✗ Data path does not exist: {data_path}")
        print("\nPlease download the datasets first:")
        print("  python verify_data.py --download-nucleotide-transformer")
        return 1

    print(f"Converting CSV files to FASTA in: {data_path}\n")

    # Find all dataset directories
    dataset_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    if not dataset_dirs:
        print(f"✗ No dataset directories found in {data_path}")
        return 1

    print(f"Found {len(dataset_dirs)} dataset directories\n")

    success_count = 0
    for dataset_dir in sorted(dataset_dirs):
        if convert_dataset(dataset_dir):
            success_count += 1
        print()

    print(f"\n{'='*60}")
    print(f"Conversion complete: {success_count}/{len(dataset_dirs)} datasets")
    print(f"{'='*60}\n")

    if success_count == len(dataset_dirs):
        print("✓ All datasets converted successfully!")
        print("\nYou can now run Nucleotide Transformer benchmarks:")
        print("  python run_single_benchmark.py --suite nucleotide_transformer --dataset enhancer")
        return 0
    else:
        print(f"⚠ {len(dataset_dirs) - success_count} datasets failed to convert")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
