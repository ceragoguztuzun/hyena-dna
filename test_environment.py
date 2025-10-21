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
