#!/bin/bash
# setup_env.sh - Environment setup script for replication on new clusters

set -e  # Exit on error

echo "============================================"
echo "Environment Setup for VSI-Bench Evaluation"
echo "============================================"
echo ""

# Configuration
ENV_NAME="env"
PYTHON_VERSION="3.11"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
    echo ""
    echo "Download Miniconda:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "WARNING: Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Skipping environment creation. Activating existing environment."
        conda activate ${ENV_NAME}
        exit 0
    fi
fi

# Method selection
echo "Choose installation method:"
echo "  1) Full replication (from environment.yml) - Exact package versions"
echo "  2) Fresh install (from requirements.txt) - Latest compatible versions"
echo "  3) Frozen install (from requirements-frozen.txt) - Exact pip packages"
echo ""
read -p "Enter choice (1-3): " -n 1 -r
echo ""

case $REPLY in
    1)
        echo "Creating environment from environment.yml..."
        if [ ! -f "environment.yml" ]; then
            echo "ERROR: environment.yml not found!"
            exit 1
        fi
        conda env create -f environment.yml
        ;;
    2)
        echo "Creating fresh environment..."
        conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
        
        # Activate environment
        eval "$(conda shell.bash hook)"
        conda activate ${ENV_NAME}
        
        echo "Installing PyTorch with CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        
        echo "Installing requirements..."
        if [ ! -f "requirements.txt" ]; then
            echo "ERROR: requirements.txt not found!"
            exit 1
        fi
        pip install -r requirements.txt
        ;;
    3)
        echo "Creating environment from frozen requirements..."
        conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
        
        # Activate environment
        eval "$(conda shell.bash hook)"
        conda activate ${ENV_NAME}
        
        if [ ! -f "requirements-frozen.txt" ]; then
            echo "ERROR: requirements-frozen.txt not found!"
            exit 1
        fi
        pip install -r requirements-frozen.txt
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Environment Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify installation:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")'"
echo "  python -c 'import transformers; print(f\"Transformers: {transformers.__version__}\")'"
echo "  python -c 'import vllm; print(f\"vLLM: {vllm.__version__}\")'"
echo ""
echo "Next steps:"
echo "  1. Download VSI-Bench: python scripts/utilities/vsi_download.py"
echo "  2. Download datasets: See docs/DOWNLOAD_SCANNET.md and docs/DOWNLOAD_SCANNETPP.md"
echo "  3. Run tests: pytest test/ -v"
echo "  4. Test evaluation: python evaluation/sequential.py --test --steps 8"
echo ""
