#!/bin/bash

# Batch extract RGB frames from ScanNet .sens files
# Usage: bash extract_scannet_frames.sh

SCENE_LIST="vsi_bench_scannet_scenes.txt"
SCANS_DIR="/dss/mcmlscratch/06/di38riq/scans"
READER_SCRIPT="analysis/scripts/scannet/reader.py"
LOG_FILE="scannet_extraction.log"

echo "Starting frame extraction at $(date)" | tee -a "$LOG_FILE"
echo "Processing scenes from: $SCENE_LIST" | tee -a "$LOG_FILE"

total=$(wc -l < "$SCENE_LIST")
count=0
success=0
failed=0

while IFS= read -r scene; do
    count=$((count + 1))
    echo "[$count/$total] Processing $scene..." | tee -a "$LOG_FILE"
    
    sens_file="$SCANS_DIR/$scene/${scene}.sens"
    output_dir="$SCANS_DIR/$scene/frames"
    
    # Check if .sens file exists
    if [ ! -f "$sens_file" ]; then
        echo "  ERROR: .sens file not found: $sens_file" | tee -a "$LOG_FILE"
        failed=$((failed + 1))
        continue
    fi
    
    # Check if already extracted
    if [ -d "$output_dir/color" ] && [ "$(ls -A $output_dir/color 2>/dev/null)" ]; then
        echo "  SKIP: Frames already extracted" | tee -a "$LOG_FILE"
        success=$((success + 1))
        continue
    fi
    
    # Extract frames
    cd "$SCANS_DIR/$scene"
    python "/dss/dsshome1/06/di38riq/rl_multi_turn/$READER_SCRIPT" \
        --filename "${scene}.sens" \
        --output_path "frames" \
        --export_color_images >> "/dss/dsshome1/06/di38riq/rl_multi_turn/$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        num_frames=$(ls -1 "$output_dir/color" 2>/dev/null | wc -l)
        echo "  SUCCESS: Extracted $num_frames frames" | tee -a "/dss/dsshome1/06/di38riq/rl_multi_turn/$LOG_FILE"
        success=$((success + 1))
    else
        echo "  ERROR: Frame extraction failed" | tee -a "/dss/dsshome1/06/di38riq/rl_multi_turn/$LOG_FILE"
        failed=$((failed + 1))
    fi
    
    cd "/dss/dsshome1/06/di38riq/rl_multi_turn"
    
done < "$SCENE_LIST"

echo "" | tee -a "$LOG_FILE"
echo "Extraction complete at $(date)" | tee -a "$LOG_FILE"
echo "Total: $total | Success: $success | Failed: $failed" | tee -a "$LOG_FILE"
