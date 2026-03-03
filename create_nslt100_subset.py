import os
import json
import shutil
from collections import defaultdict

def create_subset():
    # Configuration
    json_path = 'nslt_100.json'
    class_list_path = 'wlasl_class_list.txt'
    videos_dir = 'videos'
    output_root = 'nslt100_subset'
    videos_out_dir = os.path.join(output_root, 'videos')
    
    clean_json_out = os.path.join(output_root, 'subset_nslt100.json')
    class_dist_out = os.path.join(output_root, 'class_distribution.json')
    
    # Initialize output directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(videos_out_dir, split), exist_ok=True)
    
    # Load gloss mapping (ID -> Gloss Name)
    gloss_map = {}
    if not os.path.exists(class_list_path):
        print(f"Error: {class_list_path} not found.")
        return
        
    with open(class_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                # Store gloss in uppercase as in example BOOK
                gloss_map[int(parts[0])] = parts[1].strip().upper()
                
    # Load NSLT-100 metadata
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
        
    with open(json_path, 'r') as f:
        nslt_data = json.load(f)
        
    # Counters and Tracking
    total_copied = 0
    missing_videos = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    class_dist = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    successful_entries = []
    
    # Process entries
    for video_id, info in nslt_data.items():
        subset = info['subset']
        gloss_id = info['action'][0]
        gloss_name = gloss_map.get(gloss_id)
        
        if not gloss_name:
            continue
            
        src_path = os.path.join(videos_dir, f"{video_id}.mp4")
        
        # Check if source video exists
        if not os.path.exists(src_path):
            missing_videos += 1
            continue
            
        # Target path structure: nslt100_subset/videos/<subset>/<gloss>/<video_id>.mp4
        target_gloss_dir = os.path.join(videos_out_dir, subset, gloss_name)
        os.makedirs(target_gloss_dir, exist_ok=True)
        
        target_path = os.path.join(target_gloss_dir, f"{video_id}.mp4")
        
        # Copy file if not already present
        try:
            if not os.path.exists(target_path):
                shutil.copy2(src_path, target_path)
            
            total_copied += 1
            split_counts[subset] += 1
            class_dist[gloss_name][subset] += 1
            
            # Metadata for clean JSON
            # Relative path relative to nslt100_subset
            rel_path = os.path.relpath(target_path, output_root).replace("\\", "/")
            successful_entries.append({
                "video_id": video_id,
                "gloss": gloss_name,
                "subset": subset,
                "new_path": rel_path
            })
        except Exception as e:
            print(f"Error copying {video_id}: {e}")
            
    # Save clean JSON
    with open(clean_json_out, 'w') as f:
        json.dump(successful_entries, f, indent=2)
        
    # Save class distribution
    with open(class_dist_out, 'w') as f:
        json.dump(class_dist, f, indent=2)
        
    # Final Summary
    print("\n" + "="*40)
    print("      NSLT-100 SUBSET CREATION SUMMARY")
    print("="*40)
    print(f"Total videos successfully copied: {total_copied}")
    print(f"Total videos missing in source  : {missing_videos}")
    print("-" * 20)
    for subset, count in split_counts.items():
        print(f"Videos in {subset:<10}: {count}")
    print("-" * 20)
    
    if class_dist:
        avg_samples = total_copied / len(class_dist)
        print(f"Total gloss classes      : {len(class_dist)}")
        print(f"Avg samples per class    : {avg_samples:.2f}")
    
    print("="*40)
    print(f"Output saved to: {output_root}")
    print(f"Clean metadata: {clean_json_out}")
    print(f"Distribution:   {class_dist_out}")
    print("="*40)

if __name__ == "__main__":
    create_subset()
