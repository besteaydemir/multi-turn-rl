# Pushing to GitHub - Checklist

This document provides a checklist of files to push to GitHub for cluster replication.

## Files Created for Environment Replication

The following files have been created to help replicate the environment on another cluster:

### Core Environment Files
- ✅ **environment.yml** (7.3KB) - Full conda environment export with all 269 packages
- ✅ **requirements-frozen.txt** (4.4KB) - Frozen pip packages with exact versions
- ✅ **requirements.txt** (exists) - Minimal requirements for fresh install
- ✅ **setup_env.sh** (3.8KB) - Automated setup script with 3 installation options

### Documentation Files
- ✅ **ENVIRONMENT.md** (8.5KB) - Comprehensive environment setup guide with:
  - Quick setup instructions
  - Core dependencies list
  - Detailed installation steps
  - Cluster-specific configurations
  - Troubleshooting section
  - Testing instructions
  
- ✅ **REPLICATION.md** (4.2KB) - Quick reference for replication:
  - Step-by-step setup (6 steps)
  - Common issues and fixes
  - Environment variables
  - File checklist
  - Quick commands reference

### Updated Files
- ✅ **README.md** - Updated with environment setup section
- ✅ **.gitignore** - Updated to not ignore environment export files

## How to Use on Target Cluster

### Option 1: Full Replication (Recommended)
```bash
git clone <your-repo-url>
cd rl_multi_turn
conda env create -f environment.yml
conda activate env
```

### Option 2: Automated Setup
```bash
git clone <your-repo-url>
cd rl_multi_turn
./setup_env.sh
# Choose option 1, 2, or 3
```

### Option 3: Manual Fresh Install
```bash
git clone <your-repo-url>
cd rl_multi_turn
conda create -n env python=3.11 -y
conda activate env
pip install -r requirements.txt
```

## What to Push

### Essential Files (Must Push)
```bash
# Environment files
environment.yml
requirements-frozen.txt
requirements.txt
setup_env.sh

# Documentation
ENVIRONMENT.md
REPLICATION.md
README.md
docs/

# Source code
src/
rl_environment/
rl_trainer/
rl_multiturn_v2/
evaluation/
training/
utils/
scripts/
analysis/

# Configuration
config.py
example_config.yaml
*.yml
*.yaml

# SLURM scripts
*.sh
*.sbatch

# Testing
test/
pytest.ini

# Git
.gitignore
```

### Do NOT Push (Already in .gitignore)
```bash
# Large data files
data_download/
*.ply
*.npy
*.npz
*.pkl

# Logs and outputs
experiment_logs/
logs/
checkpoints/
rl_checkpoints/
rl_episodes/
wandb/
*.out
*.err

# Model caches (will be re-downloaded)
~/.cache/huggingface/

# Python artifacts
__pycache__/
*.pyc
.pytest_cache/
```

## Git Commands

### First Time Setup
```bash
cd /dss/dsshome1/06/di38riq/rl_multi_turn

# Initialize git (if not already done)
git init
git add .
git commit -m "Add environment replication files"

# Add remote
git remote add origin <your-github-url>

# Push to GitHub
git push -u origin main
```

### Update Existing Repo
```bash
cd /dss/dsshome1/06/di38riq/rl_multi_turn

# Stage environment files
git add environment.yml requirements-frozen.txt setup_env.sh
git add ENVIRONMENT.md REPLICATION.md PUSH_TO_GITHUB.md
git add README.md .gitignore

# Commit
git commit -m "Add comprehensive environment replication support

- Add environment.yml with full conda export (269 packages)
- Add requirements-frozen.txt with exact pip versions
- Add setup_env.sh for automated environment setup
- Add ENVIRONMENT.md with detailed setup instructions
- Add REPLICATION.md with quick reference guide
- Update README.md with environment setup section
- Update .gitignore to preserve environment export files"

# Push
git push origin main
```

## Verification

After pushing, verify on GitHub that these files are present:
- [ ] environment.yml
- [ ] requirements-frozen.txt
- [ ] setup_env.sh (executable)
- [ ] ENVIRONMENT.md
- [ ] REPLICATION.md
- [ ] Updated README.md

## Pulling on New Cluster

On the target cluster:
```bash
# Clone repository
git clone <your-github-url>
cd rl_multi_turn

# Quick setup
conda env create -f environment.yml
conda activate env

# Verify
python -c "import torch; print(torch.cuda.is_available())"

# Test
python evaluation/sequential.py --test --steps 8
```

## Additional Notes

### HuggingFace Models
Models will be automatically downloaded on first run:
- Qwen/Qwen3-VL-8B (~16GB)
- Qwen/Qwen3-VL-4B (~8GB)

Set cache location:
```bash
export HF_HOME=/path/to/cache
```

### Dataset Setup
After environment setup, download required datasets:
1. VSI-Bench: `python scripts/utilities/vsi_download.py`
2. ScanNet: See `docs/DOWNLOAD_SCANNET.md`
3. ScanNet++: See `docs/DOWNLOAD_SCANNETPP.md`

### Cluster-Specific Adjustments
Edit SLURM scripts for your cluster:
- Partition names: `#SBATCH --partition=<your-partition>`
- QOS: `#SBATCH --qos=<your-qos>`
- Paths: Update absolute paths in scripts
- Module loading: Add required module loads

## Questions?

Refer to:
- ENVIRONMENT.md for detailed setup
- REPLICATION.md for quick reference
- README.md for project overview
