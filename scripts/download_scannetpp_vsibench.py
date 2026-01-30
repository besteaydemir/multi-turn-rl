#!/usr/bin/env python3
"""
Download ScanNet++ data for VSI-Bench scenes only.

This script:
1. Extracts the 50 unique ScanNet++ scenes used in VSI-Bench
2. Creates a custom config for download_scannetpp.py
3. Downloads only: meshes + RGBD frames for those scenes

Usage:
    # Step 1: Generate scene list and config
    python scripts/download_scannetpp_vsibench.py --generate-config
    
    # Step 2: Download (using the generated config)
    python data_download/download_scannetpp.py scannetpp_vsibench_config.yml
"""

import argparse
from pathlib import Path
import yaml
from datasets import load_dataset


def get_vsibench_scannetpp_scenes():
    """Extract ScanNet++ scene IDs from VSI-Bench."""
    print("📥 Loading VSI-Bench dataset...")
    vsi = load_dataset('nyu-visionx/VSI-Bench', split='test')
    
    print("🔍 Filtering ScanNet++ scenes...")
    scannetpp_data = [x for x in vsi if x['dataset'] == 'scannetpp']
    scenes = sorted(set(x['scene_name'] for x in scannetpp_data))
    
    print(f"✅ Found {len(scenes)} unique ScanNet++ scenes in VSI-Bench")
    
    # Count questions per scene
    scene_counts = {}
    for x in scannetpp_data:
        scene = x['scene_name']
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
    
    print(f"\n📊 Question distribution:")
    print(f"   Total questions: {len(scannetpp_data)}")
    print(f"   Min questions per scene: {min(scene_counts.values())}")
    print(f"   Max questions per scene: {max(scene_counts.values())}")
    print(f"   Avg questions per scene: {len(scannetpp_data) / len(scenes):.1f}")
    
    return scenes


def save_scene_list(scenes, output_path="vsi_bench_scannetpp_scenes.txt"):
    """Save scene list to text file."""
    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        f.write('\n'.join(scenes))
    print(f"\n💾 Saved scene list to: {output_file}")
    print(f"   First 5 scenes: {', '.join(scenes[:5])}")


def generate_download_config(scenes, token="<YOUR_TOKEN_HERE>", data_root="./scannetpp_data"):
    """
    Generate a custom YAML config for downloading VSI-Bench scenes only.
    
    Downloads:
    - mesh_aligned_0.05.ply (textured mesh)
    - dslr/resized_images (RGB frames)
    - dslr/resized_anon_masks (masks)
    - dslr/resized_depth (depth maps)
    """
    
    config = {
        # Authentication
        "token": token,
        "data_root": data_root,
        
        # URLs
        "root_url": "https://kaldir.vc.in.tum.de/scannetpp/download?version=v1&token=TOKEN&file=FILEPATH",
        
        # What to download
        "download_scenes": scenes,
        
        # Assets to download
        "download_assets": [
            "mesh_aligned_0_05",      # Textured mesh at 5cm resolution
            "dslr_resized_images",    # RGB frames
            "dslr_resized_masks",     # Anonymization masks
            "dslr_resized_depth",     # Depth maps
        ],
        
        # Asset definitions (from ScanNet++ release)
        "zipped_assets": [
            "dslr_resized_images",
            "dslr_resized_masks",
            "dslr_resized_depth",
        ],
        
        # Metadata files (minimal)
        "meta_files": [
            "splits/nvs_sem_train.txt",
            "splits/nvs_sem_val.txt",
        ],
        
        # Splits (will be downloaded as metadata)
        "splits": ["nvs_sem_train", "nvs_sem_val"],
        
        # Control flags
        "dry_run": False,
        "verbose": True,
        "metadata_only": False,
    }
    
    output_path = "scannetpp_vsibench_config.yml"
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Generated download config: {output_path}")
    print(f"\n📋 Configuration:")
    print(f"   Scenes: {len(scenes)}")
    print(f"   Assets: {', '.join(config['download_assets'])}")
    print(f"   Data root: {data_root}")
    print(f"\n⚠️  IMPORTANT: Edit the config file to add your download token!")
    print(f"   1. Get token from: https://kaldir.vc.in.tum.de/scannetpp/")
    print(f"   2. Edit {output_path} and replace '<YOUR_TOKEN_HERE>' with your token")
    print(f"   3. Run: python data_download/download_scannetpp.py {output_path}")
    
    return output_path


def estimate_download_size(num_scenes):
    """Estimate total download size."""
    # Rough estimates per scene (from ScanNet++ documentation)
    mesh_size_mb = 100  # Textured mesh ~100MB
    rgbd_size_mb = 500  # DSLR RGBD frames ~500MB per scene
    
    total_per_scene = mesh_size_mb + rgbd_size_mb
    total_mb = total_per_scene * num_scenes
    total_gb = total_mb / 1024
    
    print(f"\n💾 Estimated download size:")
    print(f"   Per scene: ~{total_per_scene}MB (mesh + RGBD)")
    print(f"   Total ({num_scenes} scenes): ~{total_gb:.1f}GB")
    print(f"\n   Note: Actual size may vary. The full dataset is >10TB,")
    print(f"         but we're only downloading {num_scenes}/230 scenes.")


def main():
    parser = argparse.ArgumentParser(
        description="Download ScanNet++ data for VSI-Bench scenes only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate config (with default data root)
  python scripts/download_scannetpp_vsibench.py --generate-config
  
  # Generate config with custom data root
  python scripts/download_scannetpp_vsibench.py --generate-config --data-root /path/to/data
  
  # Just show the scene list
  python scripts/download_scannetpp_vsibench.py --list-scenes
  
After generating config:
  1. Edit scannetpp_vsibench_config.yml to add your token
  2. Run: python data_download/download_scannetpp.py scannetpp_vsibench_config.yml
        """
    )
    
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate download configuration file"
    )
    
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="Just list the scenes (don't generate config)"
    )
    
    parser.add_argument(
        "--data-root",
        type=str,
        default="./scannetpp_vsibench",
        help="Root directory for downloaded data (default: ./scannetpp_vsibench)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default="<YOUR_TOKEN_HERE>",
        help="ScanNet++ download token (get from https://kaldir.vc.in.tum.de/scannetpp/)"
    )
    
    parser.add_argument(
        "--save-scene-list",
        type=str,
        help="Save scene list to this file (default: vsi_bench_scannetpp_scenes.txt)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ScanNet++ VSI-Bench Downloader")
    print("=" * 70)
    
    # Get scenes from VSI-Bench
    scenes = get_vsibench_scannetpp_scenes()
    
    # Estimate size
    estimate_download_size(len(scenes))
    
    # Save scene list if requested
    if args.save_scene_list:
        save_scene_list(scenes, args.save_scene_list)
    
    # Just list scenes
    if args.list_scenes:
        print(f"\n📋 VSI-Bench ScanNet++ scenes ({len(scenes)}):")
        for i, scene in enumerate(scenes, 1):
            print(f"   {i:2d}. {scene}")
        return
    
    # Generate config
    if args.generate_config:
        config_path = generate_download_config(
            scenes,
            token=args.token,
            data_root=args.data_root
        )
        
        print(f"\n✅ Ready to download!")
        print(f"\nNext steps:")
        print(f"   1. Edit {config_path}")
        print(f"   2. Add your token (replace '<YOUR_TOKEN_HERE>')")
        print(f"   3. Run: python data_download/download_scannetpp.py {config_path}")
    else:
        print("\n💡 Use --generate-config to create download configuration")
        print("   Use --list-scenes to just see the scene list")


if __name__ == "__main__":
    main()
