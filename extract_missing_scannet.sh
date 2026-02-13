#!/bin/bash
# Extract frames from missing ScanNet scenes
# Usage: srun --cpus-per-task=4 --mem=32G --time=00:30:00 bash extract_missing_scannet.sh

source /dss/dsshome1/06/di38riq/miniconda3/etc/profile.d/conda.sh
conda activate env

SCANS_DIR="/dss/mcmlscratch/06/di38riq/scans/scans"
READER="/dss/dsshome1/06/di38riq/rl_multi_turn/analysis/scripts/scannet/reader.py"

for scene in scene0645_00 scene0704_01; do
    echo "=========================================="
    echo "Extracting $scene..."
    echo "=========================================="
    
    cd "$SCANS_DIR/$scene"
    
    if [ -d "frames/color" ] && [ "$(ls -A frames/color 2>/dev/null)" ]; then
        echo "SKIP: Frames already exist"
        continue
    fi
    
    python "$READER" \
        --filename "${scene}.sens" \
        --output_path frames \
        --export_color_images
    
    if [ $? -eq 0 ]; then
        num_frames=$(ls -1 frames/color 2>/dev/null | wc -l)
        echo "SUCCESS: Extracted $num_frames frames"
    else
        echo "ERROR: Failed to extract frames"
    fi
done

echo ""
echo "=== Verification ==="
for scene in scene0645_00 scene0704_01; do
    num=$(ls -1 "$SCANS_DIR/$scene/frames/color" 2>/dev/null | wc -l)
    echo "$scene: $num frames"
done
