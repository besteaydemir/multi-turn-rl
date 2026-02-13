#!/bin/bash
# Check scratch filesystem deletion policy

echo "=== Checking Scratch Filesystem Policy ==="
echo ""

# Check if there's a policy file
if [ -f /dss/mcmlscratch/POLICY ]; then
    echo "Policy file found:"
    cat /dss/mcmlscratch/POLICY
    echo ""
fi

# Check for .cleanup or .purge files
for file in /dss/mcmlscratch/.cleanup /dss/mcmlscratch/.purge /dss/mcmlscratch/CLEANUP; do
    if [ -f "$file" ]; then
        echo "Found: $file"
        head -20 "$file"
        echo ""
    fi
done

# Show disk usage
echo "=== Disk Usage ==="
df -h /dss/mcmlscratch/06/di38riq/

echo ""
echo "=== Your Folder Stats ==="
echo "Folders in scratch:"
ls -lht /dss/mcmlscratch/06/di38riq/ | head -15

echo ""
echo "=== Check Deletion Policy ==="
echo "Files older than 30 days (if any):"
find /dss/mcmlscratch/06/di38riq/ -maxdepth 1 -type d -mtime +30 -ls 2>/dev/null | head -10
