#!/bin/bash

# Activate conda environment
source /dss/dsshome1/06/di38riq/miniconda3/etc/profile.d/conda.sh
conda activate env

# Extract RGB frames from all ScanNet .sens files
SCENE_LIST="vsi_bench_scannet_scenes.txt"
SCANS_DIR="/dss/mcmlscratch/06/di38riq/scans/scans"
LOG_FILE="scannet_extraction.log"

echo "Starting frame extraction at $(date)" | tee "$LOG_FILE"

total=$(wc -l < "$SCENE_LIST")
count=0

while IFS= read -r scene; do
    count=$((count + 1))
    echo "[$count/$total] Processing $scene..." | tee -a "$LOG_FILE"
    
    sens_file="$SCANS_DIR/$scene/${scene}.sens"
    output_path="$SCANS_DIR/$scene/frames"
    
    if [ ! -f "$sens_file" ]; then
        echo "  ERROR: .sens file not found: $sens_file" | tee -a "$LOG_FILE"
        continue
    fi
    
    # Check if already extracted
    if [ -d "$output_path/color" ] && [ "$(ls -A $output_path/color 2>/dev/null)" ]; then
        num_frames=$(ls -1 "$output_path/color" 2>/dev/null | wc -l)
        echo "  SKIP: Already extracted ($num_frames frames)" | tee -a "$LOG_FILE"
        continue
    fi
    
    cd /dss/dsshome1/06/di38riq/rl_multi_turn/analysis/scripts/scannet
    python reader.py --filename "$sens_file" --output_path "$output_path" --export_color_images >> "/dss/dsshome1/06/di38riq/rl_multi_turn/$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ] && [ -d "$output_path/color" ]; then
        num_frames=$(ls -1 "$output_path/color" 2>/dev/null | wc -l)
        echo "  SUCCESS: Extracted $num_frames frames" | tee -a "/dss/dsshome1/06/di38riq/rl_multi_turn/$LOG_FILE"
    else
        echo "  ERROR: Frame extraction failed" | tee -a "/dss/dsshome1/06/di38riq/rl_multi_turn/$LOG_FILE"
    fi
    
done < "/dss/dsshome1/06/di38riq/rl_multi_turn/$SCENE_LIST"

echo "" | tee -a "$LOG_FILE"
echo "Extraction complete at $(date)" | tee -a "$LOG_FILE"
