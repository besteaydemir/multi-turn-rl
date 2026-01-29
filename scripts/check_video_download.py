#!/usr/bin/env python3
"""
Check video download progress for VSI-Bench.

Usage:
    python scripts/check_video_download.py
    python scripts/check_video_download.py --csv /path/to/your.csv
"""

import argparse
from pathlib import Path
import pandas as pd

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_CSV = "/dss/dsshome1/06/di38riq/ARKitScenes/my_filtered_raw_all.csv"
DEFAULT_DOWNLOAD_DIR = "/dss/mcmlscratch/06/di38riq/arkit_vsi/raw"


def check_download_progress(csv_path: str, download_dir: str, show_pending: bool = True):
    """Check how many videos from CSV have been downloaded."""
    
    # Get video IDs from CSV
    csv = pd.read_csv(csv_path)
    csv_video_ids = set(str(x) for x in csv['video_id'].unique())
    
    # Get downloaded video IDs (directories with vga_wide)
    base_dir = Path(download_dir)
    downloaded_video_ids = set()
    downloading_video_ids = set()  # Has directory but no vga_wide yet
    
    for split in ['Training', 'Validation']:
        split_dir = base_dir / split
        if split_dir.exists():
            for video_dir in split_dir.iterdir():
                if video_dir.is_dir():
                    if (video_dir / 'vga_wide').exists():
                        downloaded_video_ids.add(video_dir.name)
                    else:
                        downloading_video_ids.add(video_dir.name)
    
    # Calculate
    pending_video_ids = csv_video_ids - downloaded_video_ids
    in_progress = pending_video_ids & downloading_video_ids
    not_started = pending_video_ids - downloading_video_ids
    
    # Print status
    print("=" * 60)
    print("📹 VIDEO DOWNLOAD STATUS")
    print("=" * 60)
    print(f"CSV file: {csv_path}")
    print(f"Download dir: {download_dir}")
    print()
    print(f"Total video IDs in CSV:     {len(csv_video_ids):>5}")
    print(f"✅ Downloaded (vga_wide):    {len(downloaded_video_ids):>5}")
    print(f"⏳ In progress (no vga_wide): {len(in_progress):>5}")
    print(f"❌ Not started:              {len(not_started):>5}")
    print()
    
    pct = 100 * len(downloaded_video_ids) / len(csv_video_ids) if csv_video_ids else 0
    print(f"Progress: {len(downloaded_video_ids)}/{len(csv_video_ids)} ({pct:.1f}%)")
    
    # Progress bar
    bar_len = 40
    filled = int(bar_len * len(downloaded_video_ids) / len(csv_video_ids)) if csv_video_ids else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"[{bar}]")
    print()
    
    if show_pending and pending_video_ids:
        print(f"📋 Pending videos ({len(pending_video_ids)}):")
        for vid in sorted(pending_video_ids)[:15]:
            status = "⏳" if vid in in_progress else "❌"
            print(f"   {status} {vid}")
        if len(pending_video_ids) > 15:
            print(f"   ... and {len(pending_video_ids) - 15} more")
    
    return len(downloaded_video_ids), len(csv_video_ids), len(pending_video_ids)


def check_vsibench_coverage(download_dir: str):
    """Also check coverage for VSI-Bench specifically."""
    try:
        from utils.data import load_vsi_bench_questions
        from utils import MCA_QUESTION_TYPES
        
        questions = load_vsi_bench_questions(question_types=MCA_QUESTION_TYPES, dataset='arkitscenes')
        vsi_scenes = set(q['scene_name'] for q in questions)
        
        # Get downloaded
        base_dir = Path(download_dir)
        downloaded = set()
        for split in ['Training', 'Validation']:
            split_dir = base_dir / split
            if split_dir.exists():
                for video_dir in split_dir.iterdir():
                    if video_dir.is_dir() and (video_dir / 'vga_wide').exists():
                        downloaded.add(video_dir.name)
        
        vsi_downloaded = vsi_scenes & downloaded
        vsi_missing = vsi_scenes - downloaded
        
        print()
        print("=" * 60)
        print("🎯 VSI-BENCH COVERAGE")
        print("=" * 60)
        print(f"VSI-Bench needs: {len(vsi_scenes)} unique scenes")
        print(f"✅ Downloaded:   {len(vsi_downloaded)} scenes")
        print(f"❌ Missing:      {len(vsi_missing)} scenes")
        print()
        pct = 100 * len(vsi_downloaded) / len(vsi_scenes) if vsi_scenes else 0
        print(f"Coverage: {len(vsi_downloaded)}/{len(vsi_scenes)} ({pct:.1f}%)")
        
        if vsi_missing:
            print()
            print(f"Missing scenes: {sorted(vsi_missing)[:10]}")
            if len(vsi_missing) > 10:
                print(f"... and {len(vsi_missing) - 10} more")
    except Exception as e:
        print(f"\n[WARN] Could not check VSI-Bench coverage: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check video download progress")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to CSV with video IDs")
    parser.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR, help="Download directory")
    parser.add_argument("--no-pending", action="store_true", help="Don't show pending list")
    parser.add_argument("--vsibench", action="store_true", help="Also check VSI-Bench coverage")
    args = parser.parse_args()
    
    check_download_progress(args.csv, args.download_dir, show_pending=not args.no_pending)
    
    if args.vsibench:
        check_vsibench_coverage(args.download_dir)
