# Quick Replication Guide

This is a condensed guide for quickly replicating the environment on a new cluster.

## Prerequisites

- Linux cluster with SLURM
- NVIDIA GPU with CUDA 12.x
- Conda/Miniconda installed
- Git installed

## Step-by-Step Setup

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd rl_multi_turn
```

### 2. Setup Environment

**Recommended (Exact Replication):**
```bash
conda env create -f environment.yml
conda activate env
```

**Alternative (Fresh Install):**
```bash
./setup_env.sh
# Choose option 1, 2, or 3
```

### 3. Verify Installation

```bash
conda activate env

# Check core packages
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers, vllm, open3d"
echo "All imports successful!"
```

### 4. Configure Paths

Edit cluster-specific paths in your shell scripts:

```bash
# In SLURM scripts (.sbatch files), update:
#SBATCH --partition=<your-gpu-partition>  # e.g., gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Update conda path if needed
source /path/to/your/miniconda3/etc/profile.d/conda.sh
conda activate env
```

### 5. Download Data (Optional)

```bash
# VSI-Bench dataset (required)
python scripts/utilities/vsi_download.py

# ScanNet (for ScanNet questions)
# See docs/DOWNLOAD_SCANNET.md

# ScanNet++ (for ScanNet++ questions)
# See docs/DOWNLOAD_SCANNETPP.md
```

### 6. Run Test

```bash
# Quick test (no GPU needed)
python evaluation/sequential.py --test --steps 8

# SLURM test (with GPU)
sbatch --array=1 run_evaluation_sequential_continue.sh
```

## Common Issues

### Issue: CUDA Not Available

```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: vLLM Import Error

```bash
pip install vllm==0.14.0 --no-build-isolation
```

### Issue: Module Not Found

```bash
# Load required modules (if applicable)
module load cuda/12.1
module load gcc/11.2
```

### Issue: Permission Denied on Scripts

```bash
chmod +x setup_env.sh
chmod +x *.sh
```

## Environment Variables

Add to your `~/.bashrc` or job scripts:

```bash
# HuggingFace cache (to avoid re-downloading models)
export HF_HOME=/path/to/cache

# CUDA device
export CUDA_VISIBLE_DEVICES=0

# Weights & Biases (optional)
export WANDB_API_KEY=your_key
export WANDB_PROJECT=vsi-bench-eval
```

## File Checklist

Files to push to GitHub:
- ✅ `environment.yml` - Full conda environment
- ✅ `requirements-frozen.txt` - Frozen pip packages
- ✅ `requirements.txt` - Minimal requirements
- ✅ `setup_env.sh` - Automated setup script
- ✅ `ENVIRONMENT.md` - Detailed documentation
- ✅ `.gitignore` - Ignore patterns
- ✅ All source code (`.py`, `.sh`, `.sbatch`)
- ✅ Configuration files (`.yml`, `.yaml`)

Files NOT to push (in .gitignore):
- ❌ Model checkpoints (`checkpoints/`, `rl_checkpoints/`)
- ❌ Logs (`logs/`, `experiment_logs/`, `wandb/`)
- ❌ Data files (`.ply`, `.npy`, `.pkl`)
- ❌ Large datasets (`data_download/` contents)

## Next Steps After Setup

1. **Configure SLURM**: Edit partition names in `.sbatch` files
2. **Set Paths**: Update data paths in scripts if needed
3. **Download Data**: Download only datasets you need
4. **Test Run**: Run a small test to verify everything works
5. **Production**: Submit full evaluation jobs

## Quick Commands Reference

```bash
# Activate environment
conda activate env

# Run full evaluation (4 splits)
sbatch run_evaluation_sequential_continue.sh

# Run single split
sbatch --array=1 run_evaluation_sequential_continue.sh

# Run locally for testing
python evaluation/sequential.py --test --steps 8

# Check running jobs
squeue -u $USER

# View logs
tail -f logs/*.out

# Analyze results
python analysis/scripts/analyze_results.py
```

## Support

For detailed instructions and troubleshooting, see:
- [ENVIRONMENT.md](ENVIRONMENT.md) - Full environment setup guide
- [README.md](README.md) - Project overview
- [docs/DOWNLOAD_SCANNET.md](docs/DOWNLOAD_SCANNET.md) - ScanNet setup
- [docs/DOWNLOAD_SCANNETPP.md](docs/DOWNLOAD_SCANNETPP.md) - ScanNet++ setup
