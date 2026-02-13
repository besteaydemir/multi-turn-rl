#!/bin/bash
# Refresh timestamps on scratch folders to prevent 30-day deletion
# This is MUCH faster than moving data back and forth

SCRATCH_DIR="/dss/mcmlscratch/06/di38riq"
LOG_FILE="$HOME/timestamp_refresh_$(date +%Y%m%d_%H%M%S).log"

echo "Starting timestamp refresh: $(date)" | tee -a "$LOG_FILE"
echo "Target: $SCRATCH_DIR" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

# Critical dataset folders that must be preserved
FOLDERS=(
    "arkit_vsi"      # ARKitScenes meshes and videos
    "data"           # ScanNet++ data
    "scans"          # ScanNet data
    "metadata"       # Metadata files
    "splits"         # Dataset splits
)

# Optional folders (comment out if not needed)
# FOLDERS+=("habitat" "habitat_semantic" "omnigibson")

for folder in "${FOLDERS[@]}"; do
    TARGET="$SCRATCH_DIR/$folder"
    
    if [ ! -d "$TARGET" ]; then
        echo "[SKIP] $folder does not exist" | tee -a "$LOG_FILE"
        continue
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo "Processing: $folder" | tee -a "$LOG_FILE"
    echo "  Path: $TARGET" | tee -a "$LOG_FILE"
    
    # Get size before
    SIZE=$(du -sh "$TARGET" 2>/dev/null | cut -f1)
    echo "  Size: $SIZE" | tee -a "$LOG_FILE"
    
    # Count files (with timeout to avoid hanging on huge dirs)
    FILE_COUNT=$(timeout 30s find "$TARGET" -type f 2>/dev/null | wc -l)
    if [ $? -eq 124 ]; then
        echo "  Files: >100k (timeout, very large)" | tee -a "$LOG_FILE"
    else
        echo "  Files: $FILE_COUNT" | tee -a "$LOG_FILE"
    fi
    
    # Update timestamps - this refreshes access time
    echo "  Updating timestamps..." | tee -a "$LOG_FILE"
    START_TIME=$(date +%s)
    
    # Use find with -exec touch to update timestamps recursively
    # This updates both the folder and all files inside
    find "$TARGET" -exec touch {} + 2>&1 | tee -a "$LOG_FILE"
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "  ✓ Completed in ${DURATION}s" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"
echo "All folders refreshed: $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"
