#!/bin/bash
# HyenaDNA Genomic Benchmarks - Environment Setup Script
# This script installs all dependencies for running HyenaDNA benchmarks

set -e  # Exit on any error

echo "=========================================="
echo "HyenaDNA Environment Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[STATUS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if CUDA is available
print_status "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    CUDA_AVAILABLE=true
else
    print_warning "CUDA not detected. GPU training will not be available."
    CUDA_AVAILABLE=false
fi

# Create virtual environment (optional, uncomment if needed)
# print_status "Creating virtual environment..."
# python3 -m venv hyena_env
# source hyena_env/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (adjust version based on your CUDA version)
print_status "Installing PyTorch..."
if [ "$CUDA_AVAILABLE" = true ]; then
    # Install PyTorch with CUDA support
    # Adjust the CUDA version as needed (cu118 = CUDA 11.8)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # Install CPU-only PyTorch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install requirements from requirements.txt
print_status "Installing package requirements..."
pip install -r requirements.txt || {
    print_warning "Some packages failed to install. Trying without torchtext..."
    # Remove torchtext line and try again
    grep -v "torchtext" requirements.txt > requirements_temp.txt
    pip install -r requirements_temp.txt
    rm requirements_temp.txt
    print_info "Continuing without torchtext (not needed for genomic benchmarks)"
}

# Install additional dependencies for benchmarks
print_status "Installing benchmark-specific dependencies..."
pip install genomic-benchmarks
pip install loguru
pip install prettytable

# Initialize Flash Attention submodule (if available)
print_status "Checking Flash Attention submodule..."
if [ -d "flash-attention" ] && [ ! "$(ls -A flash-attention)" ]; then
    print_warning "Flash Attention submodule is empty. Initializing..."
    git submodule update --init --recursive
fi

# Try to install Flash Attention (optional, may fail on some systems)
if [ -d "flash-attention" ] && [ "$(ls -A flash-attention)" ]; then
    print_status "Attempting to install Flash Attention..."
    cd flash-attention
    if pip install . 2>/dev/null; then
        print_status "Flash Attention installed successfully"
    else
        print_warning "Flash Attention installation failed. Continuing without it."
        print_warning "You can still run benchmarks, but training may be slower."
    fi
    cd ..
else
    print_warning "Flash Attention directory not found. Skipping."
fi

# Verify installation
print_status "Verifying installation..."

# Test PyTorch
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || print_error "PyTorch import failed"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || print_error "CUDA check failed"

# Test key dependencies
python3 -c "import pytorch_lightning; print(f'PyTorch Lightning version: {pytorch_lightning.__version__}')" || print_error "PyTorch Lightning import failed"
python3 -c "import hydra; print('Hydra OK')" || print_error "Hydra import failed"
python3 -c "import genomic_benchmarks; print('Genomic Benchmarks OK')" || print_error "Genomic Benchmarks import failed"

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p results/genomic_benchmarks
mkdir -p results/nucleotide_transformer
mkdir -p data/genomic_benchmark
mkdir -p data/nucleotide_transformer
mkdir -p outputs

# Create a simple test to ensure the environment works
print_status "Creating test script..."
cat > test_environment.py << 'EOF'
import torch
import pytorch_lightning as pl
import hydra
from genomic_benchmarks.data_check import list_datasets
import sys

def test_environment():
    """Test that all key components are working"""
    print("\n" + "="*50)
    print("Environment Test")
    print("="*50)

    # Check PyTorch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Check PyTorch Lightning
    print(f"✓ PyTorch Lightning version: {pl.__version__}")

    # Check Genomic Benchmarks
    datasets = list_datasets()
    print(f"✓ Genomic Benchmarks available datasets: {len(datasets)}")

    print("\n" + "="*50)
    print("All tests passed! Environment is ready.")
    print("="*50 + "\n")
    return True

if __name__ == "__main__":
    try:
        test_environment()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Environment test failed: {e}")
        sys.exit(1)
EOF

# Run the test
print_status "Running environment test..."
if python3 test_environment.py; then
    print_status "Environment test passed!"
else
    print_error "Environment test failed. Please check the errors above."
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run 'python verify_data.py' to check data availability"
echo "2. Run 'python run_single_benchmark.py' to test a single dataset"
echo "3. Run 'bash run_all_benchmarks.sh' to run all benchmarks"
echo ""
echo "For more information, see BENCHMARK_GUIDE.md"
echo ""
