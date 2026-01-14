import pandas as pd
from datasets import load_dataset

###########################################
# 1. Load VSI-Bench table + apply filters
###########################################

print("🔍 Loading VSI-Bench metadata...")
vsi = load_dataset("nyu-visionx/VSI-Bench", split="test")

print(f"ℹ️ VSI-Bench table rows: {len(vsi)}")

filtered = vsi.filter(
    lambda x: x["dataset"] == "arkitscenes"
)

print(f"🔎 Filtered rows: {len(filtered)}\n")

# Extract unique scene_ids (VSI calls it scene_name)
vsi_scene_ids = set(row["scene_name"] for row in filtered)
print("📌 Unique VSI-Bench scene_ids needed:", vsi_scene_ids, "\n")


###########################################
# 2. Load ARKitScenes CSV
###########################################

arkit_csv_path = "/dss/dsshome1/06/di38riq/ARKitScenes/raw/raw_train_val_splits.csv"

print(f"📁 Loading ARKit CSV: {arkit_csv_path}")
arkit_df = pd.read_csv(arkit_csv_path, names=["video_id", "visit_id", "fold"])

print(f"ℹ️ ARKit CSV rows: {len(arkit_df)}")
print("📄 Columns:", list(arkit_df.columns), "\n")


###########################################
# 3. Join VSI scene_ids → ARKit visit_id
###########################################

output_rows = []
unmatched = []

print("🔗 Matching VSI scene_ids → ARKit video_ids…\n")

for scene_id in vsi_scene_ids:
    matches = arkit_df[arkit_df["video_id"] == scene_id]



    print(f"=== scene_id: {scene_id} ===")

    if matches.empty:
        print("❌ No matching visit_id in ARKit CSV\n")
        unmatched.append(scene_id)
        continue

    print(f"✔️ Found {len(matches)} matches:")
    print(matches, "\n")

    for _, row in matches.iterrows():
        output_rows.append({
            "video_id": row["video_id"],
            "visit_id": row["visit_id"],
            "fold": row["fold"]
        })


###########################################
# 4. Write output CSV
###########################################

output_df = pd.DataFrame(output_rows)

output_csv_path = "my_filtered_raw_all.csv"
output_df.to_csv(output_csv_path, index=False)

print("=======================================")
print(f"📁 Saved filtered CSV: {output_csv_path}")
print(f"📝 Rows written: {len(output_df)}")
print(f"❗ Unmatched scenes: {len(unmatched)} → {unmatched}")
print("=======================================\n")

print("✅ You can now run:\n")
print(
    "python3 download_data.py raw "
    "--video_id_csv my_filtered_raw.csv "
    "--download_dir /tmp/arkit_faro "
    "--download_laser_scanner_point_cloud\n"
)

#/dss/mcmlscratch/06/di38riq/arkit_vsi
python3 download_data.py raw \
--video_id_csv my_filtered_raw_all.csv \
--download_dir /dss/mcmlscratch/06/di38riq/arkit_vsi \
--raw_dataset_assets mesh

# Works
# python3 download_data.py raw --split Validation --video_id 41069043 \
# --download_dir /dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir --raw_dataset_assets mesh

# Works
# python3 download_data.py raw --split Training --video_id 47333462 \
# --download_dir /dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir --download_laser_scanner_point_cloud


# Works
# python3 download_data.py raw --split Training --video_id 47333462 \
# --download_dir /dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir --raw_dataset_assets mesh

# Does not work
# python3 download_data.py raw --split Validation --video_id 41069043 \
# --download_dir /dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir --download_laser_scanner_point_cloud

# Does not work
# python3 download_data.py raw --split Validation --video_id 42898527 \
# --download_dir /dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir --download_laser_scanner_point_cloud