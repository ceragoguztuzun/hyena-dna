#!/bin/bash
# HyenaDNA - Run All Genomic Benchmarks
#
# This script runs all 8 GenomicBenchmarks datasets sequentially
# and optionally the 18 Nucleotide Transformer datasets.
#
# Usage:
#   bash run_all_benchmarks.sh                    # Run GenomicBenchmarks only
#   bash run_all_benchmarks.sh --all              # Run both suites
#   bash run_all_benchmarks.sh --nucleotide-only  # Run Nucleotide Transformer only
#   bash run_all_benchmarks.sh --quick            # Quick test (5 epochs, small model)
#   bash run_all_benchmarks.sh --full             # Full training (100 epochs)

set -e  # Exit on error (can be disabled for fault tolerance)

# Default configuration
EPOCHS=5
D_MODEL=128
N_LAYER=2
BATCH_SIZE=128
OUTPUT_DIR="results"
RUN_GENOMIC=true
RUN_NUCLEOTIDE=false
CONTINUE_ON_ERROR=true
LOG_DIR="logs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_GENOMIC=true
            RUN_NUCLEOTIDE=true
            shift
            ;;
        --genomic-only)
            RUN_GENOMIC=true
            RUN_NUCLEOTIDE=false
            shift
            ;;
        --nucleotide-only)
            RUN_GENOMIC=false
            RUN_NUCLEOTIDE=true
            shift
            ;;
        --quick)
            EPOCHS=5
            D_MODEL=128
            N_LAYER=2
            shift
            ;;
        --medium)
            EPOCHS=30
            D_MODEL=256
            N_LAYER=4
            shift
            ;;
        --full)
            EPOCHS=100
            D_MODEL=256
            N_LAYER=4
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --d_model)
            D_MODEL="$2"
            shift 2
            ;;
        --n_layer)
            N_LAYER="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --stop-on-error)
            CONTINUE_ON_ERROR=false
            shift
            ;;
        -h|--help)
            echo "Usage: bash run_all_benchmarks.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all                 Run all benchmark suites"
            echo "  --genomic-only        Run only GenomicBenchmarks (default)"
            echo "  --nucleotide-only     Run only Nucleotide Transformer"
            echo "  --quick               Quick test (5 epochs, small model)"
            echo "  --medium              Medium training (30 epochs, medium model)"
            echo "  --full                Full training (100 epochs, large model)"
            echo "  --epochs N            Number of epochs (default: 5)"
            echo "  --d_model N           Model dimension (default: 128)"
            echo "  --n_layer N           Number of layers (default: 2)"
            echo "  --batch_size N        Batch size (default: 128)"
            echo "  --output_dir PATH     Output directory (default: results)"
            echo "  --stop-on-error       Stop on first error (default: continue)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "\n${BOLD}${BLUE}================================================================${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Get timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/run_all_benchmarks_${TIMESTAMP}.log"

# Log both to file and stdout
log() {
    echo "$1" | tee -a "$MAIN_LOG"
}

# GenomicBenchmarks datasets
GENOMIC_DATASETS=(
    "human_enhancers_cohn"
    "human_nontata_promoters"
    "human_enhancers_ensembl"
    "human_ensembl_regulatory"
    "human_ocr_ensembl"
    "dummy_mouse_enhancers_ensembl"
    "demo_coding_vs_intergenomic_seqs"
    "demo_human_or_worm"
)

# Nucleotide Transformer datasets
NUCLEOTIDE_DATASETS=(
    "enhancers"
    "enhancers_types"
    "H2AFZ"
    "H3K4me1"
    "H3K4me2"
    "H3K4me3"
    "H3K9ac"
    "H3K9me3"
    "H3K27ac"
    "H3K27me3"
    "H3K36me3"
    "H4K20me1"
    "promoter_all"
    "promoter_no_tata"
    "promoter_tata"
    "splice_sites_acceptors"
    "splice_sites_all"
    "splice_sites_donors"
)

# Track results
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
SKIPPED_RUNS=0

declare -a FAILED_DATASETS
declare -a SUCCESSFUL_DATASETS

# Run a single benchmark
run_single_benchmark() {
    local suite=$1
    local dataset=$2
    local dataset_num=$3
    local total_datasets=$4

    print_header "[$dataset_num/$total_datasets] Running $suite: $dataset"

    print_info "Configuration:"
    log "  Suite: $suite"
    log "  Dataset: $dataset"
    log "  Epochs: $EPOCHS"
    log "  Model dimension: $D_MODEL"
    log "  Layers: $N_LAYER"
    log "  Batch size: $BATCH_SIZE"
    echo ""

    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    # Run the benchmark
    local start_time=$(date +%s)

    if python run_single_benchmark.py \
        --suite "$suite" \
        --dataset "$dataset" \
        --epochs "$EPOCHS" \
        --d_model "$D_MODEL" \
        --n_layer "$N_LAYER" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$MAIN_LOG"; then

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        print_success "Completed $dataset in ${duration}s"
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        SUCCESSFUL_DATASETS+=("$dataset")

    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        print_error "Failed $dataset after ${duration}s"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        FAILED_DATASETS+=("$dataset")

        if [ "$CONTINUE_ON_ERROR" = false ]; then
            print_error "Stopping due to error (--stop-on-error flag)"
            exit 1
        else
            print_warning "Continuing to next dataset..."
        fi
    fi

    echo ""
}

# Print configuration
print_header "HyenaDNA Benchmark Suite"
log "Run started at: $(date)"
log ""
log "Configuration:"
log "  Epochs: $EPOCHS"
log "  Model dimension: $D_MODEL"
log "  Layers: $N_LAYER"
log "  Batch size: $BATCH_SIZE"
log "  Output directory: $OUTPUT_DIR"
log "  Continue on error: $CONTINUE_ON_ERROR"
log ""

if [ "$RUN_GENOMIC" = true ]; then
    log "Running GenomicBenchmarks: ${#GENOMIC_DATASETS[@]} datasets"
fi
if [ "$RUN_NUCLEOTIDE" = true ]; then
    log "Running Nucleotide Transformer: ${#NUCLEOTIDE_DATASETS[@]} datasets"
fi
log ""
log "Main log: $MAIN_LOG"
log ""

# Estimate total time
ESTIMATED_MINUTES=$((EPOCHS * (${#GENOMIC_DATASETS[@]} * $RUN_GENOMIC + ${#NUCLEOTIDE_DATASETS[@]} * $RUN_NUCLEOTIDE) / 2))
print_info "Estimated total time: ~${ESTIMATED_MINUTES} minutes (rough estimate)"
echo ""

# Run GenomicBenchmarks
if [ "$RUN_GENOMIC" = true ]; then
    print_header "Running GenomicBenchmarks Suite"

    dataset_num=0
    total_datasets=${#GENOMIC_DATASETS[@]}

    for dataset in "${GENOMIC_DATASETS[@]}"; do
        dataset_num=$((dataset_num + 1))
        run_single_benchmark "genomic_benchmark" "$dataset" "$dataset_num" "$total_datasets"
    done
fi

# Run Nucleotide Transformer
if [ "$RUN_NUCLEOTIDE" = true ]; then
    print_header "Running Nucleotide Transformer Suite"

    dataset_num=0
    total_datasets=${#NUCLEOTIDE_DATASETS[@]}

    for dataset in "${NUCLEOTIDE_DATASETS[@]}"; do
        dataset_num=$((dataset_num + 1))
        run_single_benchmark "nucleotide_transformer" "$dataset" "$dataset_num" "$total_datasets"
    done
fi

# Print summary
print_header "Benchmark Run Summary"

log "Run completed at: $(date)"
log ""
log "Results:"
log "  Total runs: $TOTAL_RUNS"
log "  Successful: $SUCCESSFUL_RUNS"
log "  Failed: $FAILED_RUNS"
log "  Skipped: $SKIPPED_RUNS"
log ""

if [ ${#SUCCESSFUL_DATASETS[@]} -gt 0 ]; then
    print_success "Successful datasets (${#SUCCESSFUL_DATASETS[@]}):"
    for dataset in "${SUCCESSFUL_DATASETS[@]}"; do
        log "  ✓ $dataset"
    done
    echo ""
fi

if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    print_error "Failed datasets (${#FAILED_DATASETS[@]}):"
    for dataset in "${FAILED_DATASETS[@]}"; do
        log "  ✗ $dataset"
    done
    echo ""
fi

log "Log file: $MAIN_LOG"
log "Results directory: $OUTPUT_DIR"
log ""

# Generate results summary
print_info "Generating results summary..."
if python parse_results.py --results_dir "$OUTPUT_DIR" --output results/summary_${TIMESTAMP}.csv; then
    print_success "Results summary generated"
    log "Summary: results/summary_${TIMESTAMP}.csv"
else
    print_warning "Failed to generate results summary"
fi

# Exit with appropriate code
if [ $FAILED_RUNS -gt 0 ]; then
    exit 1
else
    exit 0
fi
