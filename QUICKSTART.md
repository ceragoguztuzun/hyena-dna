# HyenaDNA Benchmarks - Quick Start

Get up and running with HyenaDNA genomic benchmarks in 3 steps!

## Prerequisites
- Python 3.8+
- CUDA GPU (recommended: 16GB+ VRAM)
- ~50GB disk space

## Setup (5 minutes)

```bash
# 1. Install dependencies
bash setup.sh

# 2. Download benchmark data
python verify_data.py --download-genomic-benchmarks

# 3. Test with one dataset (5 epochs, ~2 minutes)
python run_single_benchmark.py --dataset human_enhancers_cohn --epochs 5
```

## Run All Benchmarks

### Quick Test Run (~1-2 hours)
```bash
bash run_all_benchmarks.sh --quick
```

### Full Training (~8-12 hours)
```bash
bash run_all_benchmarks.sh --full
```

## View Results

```bash
# Generate summary table
python parse_results.py --results_dir results/ --output summary.csv
```

## Available Datasets

### GenomicBenchmarks (8 datasets - auto-download)
- human_enhancers_cohn
- human_nontata_promoters
- human_enhancers_ensembl
- human_ensembl_regulatory
- human_ocr_ensembl
- dummy_mouse_enhancers_ensembl
- demo_coding_vs_intergenomic_seqs
- demo_human_or_worm

### Nucleotide Transformer (18 datasets - manual download required)
- Enhancers: enhancer, enhancer_types
- Histone marks: H3, H3K4me1, H3K4me2, H3K4me3, H3K9ac, H3K14ac, H3K36me3, H3K79me3, H4, H4ac
- Promoters: promoter_all, promoter_non_tata, promoter_tata
- Splice sites: splice_sites_acceptor, splice_sites_donor

## Common Commands

```bash
# Check data availability
python verify_data.py

# Run single dataset
python run_single_benchmark.py --dataset DATASET_NAME --epochs N

# Run all GenomicBenchmarks (default)
bash run_all_benchmarks.sh

# Run with custom settings
bash run_all_benchmarks.sh --epochs 20 --d_model 256 --n_layer 4

# Parse results
python parse_results.py --results_dir results/
```

## Configuration Presets

| Preset | Command | Epochs | Model | Time (est.) | Use Case |
|--------|---------|--------|-------|-------------|----------|
| Quick | `--quick` | 5 | Small (128/2) | 1-2h | Testing pipeline |
| Medium | `--medium` | 30 | Medium (256/4) | 4-6h | Balanced |
| Full | `--full` | 100 | Medium (256/4) | 8-12h | Reproduction |

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python run_single_benchmark.py --dataset NAME --batch_size 64

# Use smaller model
python run_single_benchmark.py --dataset NAME --d_model 64 --n_layer 2
```

### Dataset Download Failed
```bash
# Retry with force download
python verify_data.py --download-genomic-benchmarks
```

### Environment Issues
```bash
# Run environment test
python test_environment.py
```

## GPU Memory Requirements

| GPU Memory | Recommended Settings |
|------------|---------------------|
| 8GB | batch_size=32, d_model=128, n_layer=2 |
| 16GB | batch_size=128, d_model=256, n_layer=4 |
| 24GB+ | batch_size=256, d_model=512, n_layer=8 |

## More Information

For detailed documentation, see [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)

For the full HyenaDNA repository, see the main [README.md](README.md)

## Script Reference

| Script | Purpose |
|--------|---------|
| `setup.sh` | Install dependencies and set up environment |
| `verify_data.py` | Check and download benchmark datasets |
| `run_single_benchmark.py` | Run a single dataset for testing |
| `run_all_benchmarks.sh` | Run all datasets in batch |
| `parse_results.py` | Parse and summarize benchmark results |

## Expected Outputs

After running benchmarks:
```
results/
├── genomic_benchmark_human_enhancers_cohn_TIMESTAMP/
│   ├── training.log          # Full training logs
│   ├── results.json          # Metrics summary
│   └── checkpoints/          # Model checkpoints
├── summary_TIMESTAMP.csv     # Results table (CSV)
└── summary_TIMESTAMP.md      # Results table (Markdown)
```

## Support

Issues? Check:
1. [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) - Comprehensive guide
2. Training logs in `results/*/training.log`
3. GPU status with `nvidia-smi`
