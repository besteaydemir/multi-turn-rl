# Environment Setup Guide

This document provides instructions for replicating the Python environment (`env`) used for VSI-Bench multi-turn evaluation on different clusters.

## Overview

- **Environment Name**: `env`
- **Python Version**: 3.11
- **Package Manager**: Conda + pip
- **Total Packages**: ~269 (including dependencies)
- **GPU Required**: Yes (CUDA 12.x compatible)

## Quick Setup

### On Source Cluster (Current)

Export the environment to a requirements file:

```bash
# Navigate to project directory
cd /dss/dsshome1/06/di38riq/rl_multi_turn

# Export full environment (includes all dependencies)
conda env export --name env > environment.yml

# Or export minimal requirements (only explicitly installed packages)
pip list --format=freeze > requirements-full.txt
```

### On Target Cluster (New)

```bash
# Clone the repository
git clone <your-repo-url>
cd rl_multi_turn

# Option 1: Create from environment.yml (recommended - exact replication)
conda env create -f environment.yml

# Option 2: Create from scratch with pip (more flexible)
conda create -n env python=3.11 -y
conda activate env
pip install -r requirements-full.txt
```

## Core Dependencies

### Primary Packages

The following are the key packages explicitly installed (automatically pulls in ~250+ additional dependencies):

```txt
# Deep Learning & LLM
torch>=2.9.0
torchvision>=0.24.0
torchaudio>=2.9.0
transformers>=4.57.0
accelerate>=1.11.0
qwen-vl-utils>=0.0.8
vllm>=0.14.0

# 3D Processing
open3d>=0.19.0

# Data Processing
numpy>=2.2.0
pillow>=10.0.0
pandas>=2.0.0
datasets>=4.4.0

# Visualization & Logging
wandb>=0.23.0
tensorboard>=2.14.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
pyyaml>=6.0
tqdm>=4.65.0
jsonschema>=4.19.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0

# Development
ipython>=8.12.0
jupyter>=1.0.0
```

### CUDA Requirements

- CUDA 12.x (vLLM and PyTorch require CUDA 12.1+)
- GPU with at least 40GB VRAM (for Qwen3-VL 8B model)
- NVIDIA driver version 525+ recommended

## Installation Steps (Detailed)

### Step 1: Create Conda Environment

```bash
# Create environment with Python 3.11
conda create -n env python=3.11 -y
conda activate env
```

### Step 2: Install PyTorch with CUDA Support

```bash
# Install PyTorch 2.9+ with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Core ML Libraries

```bash
# Install transformers and accelerate
pip install transformers>=4.57.0 accelerate>=1.11.0

# Install Qwen VL utilities
pip install qwen-vl-utils

# Install vLLM for inference
pip install vllm>=0.14.0
```

### Step 4: Install 3D Processing

```bash
pip install open3d>=0.19.0
```

### Step 5: Install Remaining Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

### Step 6: Verify Installation

```bash
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import open3d; print(f'Open3D: {open3d.__version__}')"
```

## Cluster-Specific Configurations

### SLURM Configuration

Most evaluation scripts use SLURM for job submission. Ensure you have:

```bash
# In SLURM scripts (.sbatch files)
#SBATCH --partition=<your-gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# Activate environment
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate env
```

### Module Loading (if applicable)

Some clusters require module loading:

```bash
# Example - adjust for your cluster
module load cuda/12.1
module load gcc/11.2
module load python/3.11
```

## Environment Activation

### In Shell Scripts

```bash
#!/bin/bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate env

# Your commands here
python evaluation/sequential.py
```

### In SLURM Scripts

```bash
#!/bin/bash
#SBATCH directives here...

# Activate conda
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate env

# Run evaluation
python evaluation/sequential.py --steps 8
```

### Interactive Sessions

```bash
# Standard activation
conda activate env

# With module loading (if needed)
module load cuda/12.1
conda activate env
```

## Troubleshooting

### CUDA Version Mismatch

If you encounter CUDA version errors:

```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### vLLM Installation Issues

vLLM can be tricky to install:

```bash
# If vLLM fails, try building from source
pip install vllm --no-build-isolation

# Or use specific version
pip install vllm==0.14.0 --no-build-isolation
```

### Out of Memory (OOM) Errors

If you encounter OOM during evaluation:

```bash
# Reduce batch size in evaluation scripts
python evaluation/sequential.py --steps 4  # Instead of 8

# Or request more GPU memory in SLURM
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --mem=128G
```

### Missing System Libraries

Open3D may require system libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# CentOS/RHEL
sudo yum install mesa-libGL glib2
```

## Environment Export/Import

### Export Current Environment

```bash
# Full environment (all packages with exact versions)
conda env export --name env > environment-full.yml

# Minimal environment (only explicitly installed)
conda env export --name env --from-history > environment-minimal.yml

# Pip freeze (alternative)
pip freeze > requirements-frozen.txt
```

### Import on New Cluster

```bash
# From full export
conda env create -f environment-full.yml

# From minimal export
conda env create -f environment-minimal.yml

# From pip freeze
conda create -n env python=3.11 -y
conda activate env
pip install -r requirements-frozen.txt
```

## Data Dependencies

After setting up the environment, ensure you have:

### 1. VSI-Bench Dataset

```bash
# Download VSI-Bench
python scripts/utilities/vsi_download.py
```

### 2. ScanNet Dataset (for ScanNet questions)

```bash
# Set up ScanNet download
# See docs/DOWNLOAD_SCANNET.md for details
python data_download/download_scannet.py data_download/download_scannet.yml
```

### 3. ScanNet++ Dataset (for ScanNet++ questions)

```bash
# Set up ScanNet++ download
# See docs/DOWNLOAD_SCANNETPP.md for details
python data_download/download_scannetpp.py data_download/download_scannetpp.yml
```

### 4. Model Weights

Models are automatically downloaded from HuggingFace on first run:

```bash
# Default model: Qwen/Qwen3-VL-8B
# Set HuggingFace cache (optional)
export HF_HOME=/path/to/cache
```

## Testing the Environment

### Run Basic Tests

```bash
# Activate environment
conda activate env

# Run pytest suite
pytest test/ -v

# Run specific tests
pytest test/test_environment.py
pytest test/test_camera_utils.py
```

### Test Evaluation Pipeline

```bash
# Test mode (5 questions, no GPU required)
python evaluation/sequential.py --test --steps 8

# Single question
python evaluation/sequential.py --test --steps 8 --max-questions 1
```

## Environment Variables

Key environment variables to set:

```bash
# HuggingFace cache (to avoid re-downloading models)
export HF_HOME=/path/to/huggingface_cache

# CUDA device selection
export CUDA_VISIBLE_DEVICES=0

# Weights & Biases (optional, for logging)
export WANDB_API_KEY=your_key_here
export WANDB_PROJECT=vsi-bench-eval

# Data paths
export SCANNET_DATA=/path/to/scannet/scans
export SCANNETPP_DATA=/path/to/scannetpp/data
```

## Performance Optimization

### For Faster Inference

```bash
# Use Flash Attention (if supported)
pip install flash-attn --no-build-isolation

# Use xFormers (alternative)
pip install xformers
```

### For Multi-GPU Setup

```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## Version History

- **v1.0** (Jan 2026): Initial environment with PyTorch 2.9, vLLM 0.14, Transformers 4.57
- Python 3.11, CUDA 12.1, ~269 packages

## Contact & Support

For issues with environment setup:
1. Check troubleshooting section above
2. Verify CUDA compatibility
3. Check cluster-specific module requirements
4. Ensure sufficient disk space for model caching

## References

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Open3D Documentation](http://www.open3d.org/docs/)
