# HyenaDNA Genomic Benchmarks - Complete Guide

This guide provides step-by-step instructions for setting up and running HyenaDNA's genomic benchmarks to establish baseline results before integrating sparse attention.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Running Benchmarks](#running-benchmarks)
5. [Results Analysis](#results-analysis)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## Overview

### Benchmark Suites

HyenaDNA includes two main benchmark suites:

#### 1. GenomicBenchmarks (8 datasets)
- **Auto-download**: Datasets download automatically
- **Sequence lengths**: 200-4776 bp
- **Tasks**: Binary and multi-class classification
- **Estimated time**: ~1-2 hours for all 8 datasets (5 epochs, small model)

| Dataset | Sequences | Classes | Median Length |
|---------|-----------|---------|---------------|
| human_enhancers_cohn | 27,791 | 2 | 500 bp |
| human_nontata_promoters | 36,131 | 2 | 251 bp |
| human_enhancers_ensembl | 154,842 | 2 | 269 bp |
| human_ensembl_regulatory | 289,061 | 3 | 401 bp |
| human_ocr_ensembl | 174,756 | 2 | 315 bp |
| dummy_mouse_enhancers_ensembl | 1,210 | 2 | 2,381 bp |
| demo_coding_vs_intergenomic_seqs | 100,000 | 2 | 200 bp |
| demo_human_or_worm | 100,000 | 2 | 200 bp |

#### 2. Nucleotide Transformer (18 datasets)
- **Manual download required**: See instructions below
- **Sequence lengths**: 200-600 bp
- **Metrics**: MCC (histone marks, enhancers), F1 (promoters, splice sites)

**Dataset categories**:
- Enhancers: `enhancer`, `enhancer_types`
- Histone marks: `H3`, `H3K4me1`, `H3K4me2`, `H3K4me3`, `H3K9ac`, `H3K14ac`, `H3K36me3`, `H3K79me3`, `H4`, `H4ac`
- Promoters: `promoter_all`, `promoter_non_tata`, `promoter_tata`
- Splice sites: `splice_sites_acceptor`, `splice_sites_donor`

---

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- ~50GB disk space for data and results

### 3-Step Setup

```bash
# 1. Install dependencies
bash setup.sh

# 2. Verify data availability
python verify_data.py --download-genomic-benchmarks

# 3. Run a quick test
python run_single_benchmark.py --dataset human_enhancers_cohn --epochs 5
```

That's it! If the test completes successfully, you're ready to run the full benchmark suite.

---

## Detailed Setup

### Step 1: Environment Setup

```bash
# Run the setup script
bash setup.sh
```

This script will:
- Install PyTorch with CUDA support
- Install required Python packages
- Initialize Flash Attention submodule (optional)
- Run environment verification tests

**Manual installation** (if setup.sh fails):

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Install additional packages
pip install genomic-benchmarks loguru prettytable
```

### Step 2: Data Verification

```bash
# Check what data is available
python verify_data.py

# Download missing GenomicBenchmarks datasets
python verify_data.py --download-genomic-benchmarks
```

**For Nucleotide Transformer datasets** (optional):

These require manual download. Options:
1. Check the [Nucleotide Transformer repository](https://github.com/instadeepai/nucleotide-transformer)
2. Refer to the [original paper](https://doi.org/10.1101/2023.01.11.523679)

Expected structure:
```
data/nucleotide_transformer/
├── enhancer/
│   ├── train.csv
│   └── test.csv
├── H3K4me1/
│   ├── train.csv
│   └── test.csv
└── ... (other datasets)
```

---

## Running Benchmarks

### Option 1: Single Dataset (Testing)

Test with one dataset to verify everything works:

```bash
# Quick test (5 epochs, small model)
python run_single_benchmark.py \
    --dataset human_enhancers_cohn \
    --epochs 5 \
    --d_model 128 \
    --n_layer 2

# Longer training (50 epochs, larger model)
python run_single_benchmark.py \
    --dataset human_enhancers_cohn \
    --epochs 50 \
    --d_model 256 \
    --n_layer 4
```

**Parameters**:
- `--dataset`: Dataset name (required)
- `--suite`: `genomic_benchmark` or `nucleotide_transformer` (default: genomic_benchmark)
- `--epochs`: Number of training epochs (default: 5)
- `--d_model`: Model dimension (default: 128)
- `--n_layer`: Number of layers (default: 2)
- `--batch_size`: Batch size (default: 128)
- `--output_dir`: Results directory (default: results)

### Option 2: Full Benchmark Suite

Run all datasets in one of the suites:

```bash
# Quick test run (5 epochs, small model, ~1-2 hours)
bash run_all_benchmarks.sh --quick

# Medium run (30 epochs, medium model, ~4-6 hours)
bash run_all_benchmarks.sh --medium

# Full training (100 epochs, large model, ~8-12 hours)
bash run_all_benchmarks.sh --full

# Run both GenomicBenchmarks and Nucleotide Transformer
bash run_all_benchmarks.sh --all --medium

# Custom configuration
bash run_all_benchmarks.sh \
    --epochs 20 \
    --d_model 256 \
    --n_layer 4 \
    --batch_size 64
```

**Key options**:
- `--quick`: 5 epochs, d_model=128, n_layer=2 (fast testing)
- `--medium`: 30 epochs, d_model=256, n_layer=4 (balanced)
- `--full`: 100 epochs, d_model=256, n_layer=4 (reproduction)
- `--all`: Run both GenomicBenchmarks and Nucleotide Transformer
- `--genomic-only`: Only GenomicBenchmarks (default)
- `--nucleotide-only`: Only Nucleotide Transformer
- `--stop-on-error`: Stop on first failure (default: continue)

### Option 3: Resume Interrupted Runs

The benchmark runner is fault-tolerant by default:

```bash
# Continues from where it left off
bash run_all_benchmarks.sh --medium
```

Results are saved incrementally, so you can:
1. Stop the script anytime (Ctrl+C)
2. Check partial results with `python parse_results.py`
3. Rerun to complete remaining datasets

---

## Results Analysis

### Parse and Summarize Results

After running benchmarks, analyze the results:

```bash
# Display summary table
python parse_results.py --results_dir results/

# Save to CSV
python parse_results.py --results_dir results/ --output summary.csv

# Generate markdown report
python parse_results.py --results_dir results/ --format markdown --output report.md

# Both CSV and markdown
python parse_results.py --results_dir results/ --format both --output results_summary
```

### Understanding the Output

The summary table includes:
- **Test Acc**: Test set accuracy/score
- **Val Acc**: Validation set accuracy/score
- **Baseline**: HyenaDNA reported results (for comparison)
- **Δ**: Difference from baseline (how close you got)
- **Epochs**: Number of training epochs
- **Time**: Training time in seconds
- **Status**: Success/failure indicator

### Expected Results

With **quick settings** (5 epochs, small model):
- Results will be lower than reported baselines
- Expect ~60-80% of baseline performance
- Good for pipeline verification

With **full settings** (100 epochs, proper model):
- Should approach or match baseline results
- Differences may occur due to random seed, hardware, etc.

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
python run_single_benchmark.py --dataset human_enhancers_cohn --batch_size 64

# Reduce model size
python run_single_benchmark.py --dataset human_enhancers_cohn --d_model 64 --n_layer 2

# Use smaller datasets first
python run_single_benchmark.py --dataset demo_human_or_worm
```

#### 2. Dataset Download Fails

**Symptom**: `genomic_benchmarks` download error

**Solutions**:
```bash
# Try manual download
python -c "from genomic_benchmarks.loc2seq import download_dataset; download_dataset('human_enhancers_cohn', version=0, dest_path='data/genomic_benchmark')"

# Check internet connection and retry
python verify_data.py --download-genomic-benchmarks
```

#### 3. Flash Attention Installation Fails

**Symptom**: Errors during `setup.sh` when installing flash-attention

**Solution**: This is optional and won't prevent benchmarks from running:
```bash
# Continue without Flash Attention
# Training will work but may be slower
```

#### 4. Hydra Configuration Errors

**Symptom**: `omegaconf.errors.ConfigAttributeError`

**Solution**: Check parameter names match the config:
```bash
# Correct parameter names
python run_single_benchmark.py \
    --dataset human_enhancers_cohn \
    --d_model 128  # not --model_dim
```

### Performance Optimization

#### GPU Memory Usage

Monitor GPU memory:
```bash
# While training is running
watch -n 1 nvidia-smi
```

Optimal batch sizes for different GPUs:
- **8GB VRAM**: batch_size=32, d_model=128
- **16GB VRAM**: batch_size=128, d_model=256
- **24GB+ VRAM**: batch_size=256, d_model=512

#### Training Speed

Expected speeds (per epoch, human_enhancers_cohn):
- **Small model** (d_model=128, n_layer=2): ~30-60 seconds
- **Medium model** (d_model=256, n_layer=4): ~60-120 seconds
- **Large model** (d_model=512, n_layer=8): ~120-300 seconds

Speed up training:
```bash
# Use mixed precision (already enabled by default)
# Check config: trainer.precision: 16

# Enable gradient accumulation for larger effective batch size
# Modify config: trainer.accumulate_grad_batches
```

---

## Advanced Usage

### Using Pretrained Models

If you have pretrained HyenaDNA weights:

```bash
# Fine-tune from pretrained checkpoint
python run_single_benchmark.py \
    --dataset human_enhancers_cohn \
    --use_pretrained \
    --pretrained_path /path/to/checkpoint.ckpt \
    --epochs 20
```

### Modifying Model Configuration

For custom model architectures, edit the config files:

```yaml
# configs/experiment/hg38/genomic_benchmark.yaml
model:
  d_model: 256      # Model dimension
  n_layer: 4        # Number of layers
  d_inner: 1024     # FFN inner dimension
  layer:
    _name_: hyena
    filter_order: 64  # Hyena filter order
```

### Hyperparameter Sweeps

Create a sweep script:

```bash
#!/bin/bash
# sweep_d_model.sh

for d_model in 64 128 256 512; do
    echo "Running with d_model=$d_model"
    python run_single_benchmark.py \
        --dataset human_enhancers_cohn \
        --d_model $d_model \
        --epochs 20
done
```

### Custom Dataset Integration

To add a new dataset:

1. Add dataset info to configs:
```yaml
# configs/dataset/genomic_benchmark.yaml
my_custom_dataset:
  train_len: 10000
  classes: 2
```

2. Update the dataset list in scripts:
```python
# run_single_benchmark.py
GENOMIC_BENCHMARKS_DATASETS = {
    "my_custom_dataset": {"max_length": 500, "classes": 2},
    # ... existing datasets
}
```

### Distributed Training

For multi-GPU training:

```bash
# Modify trainer config
python -m train \
    experiment=hg38/genomic_benchmark \
    dataset_name=human_enhancers_cohn \
    trainer.devices=4 \
    trainer.strategy=ddp
```

---

## Directory Structure

After running benchmarks, your directory will look like:

```
hyena-dna/
├── setup.sh                          # Environment setup
├── verify_data.py                    # Data verification
├── run_single_benchmark.py           # Single dataset runner
├── run_all_benchmarks.sh             # Batch runner
├── parse_results.py                  # Results parser
├── BENCHMARK_GUIDE.md                # This guide
├── data/
│   ├── genomic_benchmark/            # GenomicBenchmarks data
│   │   ├── human_enhancers_cohn/
│   │   ├── human_nontata_promoters/
│   │   └── ...
│   └── nucleotide_transformer/       # Nucleotide Transformer data
│       ├── enhancer/
│       └── ...
├── results/
│   ├── genomic_benchmark_human_enhancers_cohn_20241020_123456/
│   │   ├── training.log
│   │   ├── results.json
│   │   └── checkpoints/
│   ├── summary_20241020_143000.csv
│   └── summary_20241020_143000.md
├── logs/
│   └── run_all_benchmarks_20241020_120000.log
└── outputs/                          # Hydra outputs
    └── ...
```

---

## Next Steps

After establishing baseline results:

1. **Analyze the baselines**: Compare your results to HyenaDNA's reported numbers
2. **Integrate sparse attention**: Modify the model architecture to add sparse attention
3. **Re-run benchmarks**: Use the same scripts to evaluate sparse attention
4. **Compare results**: Use `parse_results.py` to compare before/after

---

## Resources

- **HyenaDNA Paper**: [https://arxiv.org/abs/2306.15794](https://arxiv.org/abs/2306.15794)
- **HyenaDNA Repository**: [https://github.com/HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna)
- **GenomicBenchmarks**: [https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks)
- **Nucleotide Transformer**: [https://github.com/instadeepai/nucleotide-transformer](https://github.com/instadeepai/nucleotide-transformer)

---

## Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review log files in `logs/` and `results/*/training.log`
3. Verify your environment with `python test_environment.py`
4. Check GPU memory with `nvidia-smi`

---

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@article{nguyen2023hyenadna,
  title={HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution},
  author={Nguyen, Eric and Poli, Michael and Faizi, Marjan and Thomas, Armin and Birch, Callum and Wornow, Michael and Patel, Aman and Rabideau, Clayton and Massaroli, Stefano and Bengio, Yoshua and Ermon, Stefano and Ré, Christopher and others},
  journal={arXiv preprint arXiv:2306.15794},
  year={2023}
}
```
