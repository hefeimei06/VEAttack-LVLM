import json
import os
import argparse

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument("--result-file", type=str)
args = parser.parse_args()
output_file_path = args.result_file + '_sorted.jsonl'

coco_files_mapping = {
    "coco_pope_adversarial.json": 0,  # IDs 1-3000
    "coco_pope_random.json": 10000000,  # IDs 10000001-10002910
    "coco_pope_popular.json": 20000000  # IDs 20000001-20003000
}
input_coco_dir = "./pope_eval/coco"
output_coco_dir = "./pope_eval/coco_sample"

# Create output directory
os.makedirs(output_coco_dir, exist_ok=True)

# Step 1: Extract sampled question_ids and categorize them
sampled_ids = {
    "adversarial": [],
    "random": [],
    "popular": []
}

with open(output_file_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        qid = data["question_id"]

        if 1 <= qid <= 3000:
            sampled_ids["adversarial"].append(qid)
        elif 10000001 <= qid <= 10002910:
            sampled_ids["random"].append(qid - 10000000)  # Convert to 1-2910
        elif 20000001 <= qid <= 20003000:
            sampled_ids["popular"].append(qid - 20000000)  # Convert to 1-3000
        else:
            print(f"Warning: Unexpected question_id {qid}")

print(f"Found {sum(len(v) for v in sampled_ids.values())} sampled question_ids")

# Step 2: Process each COCO file with proper ID mapping
for coco_file, offset in coco_files_mapping.items():
    input_path = os.path.join(input_coco_dir, coco_file)
    output_path = os.path.join(output_coco_dir, coco_file)

    # Determine which IDs to look for in this file
    if "adversarial" in coco_file:
        target_ids = sampled_ids["adversarial"]
    elif "random" in coco_file:
        target_ids = sampled_ids["random"]
    elif "popular" in coco_file:
        target_ids = sampled_ids["popular"]

    target_ids_set = set(target_ids)

    # Read and filter the file
    filtered_data = []
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data["question_id"] in target_ids_set:
                filtered_data.append(data)

    # Verify we found all expected IDs
    found_ids = {item["question_id"] for item in filtered_data}
    missing_ids = set(target_ids) - found_ids
    if missing_ids:
        print(f"Warning: {len(missing_ids)} IDs not found in {coco_file}")

    # Write the filtered data
    with open(output_path, 'w') as f_out:
        # Maintain original order from sampled file
        ordered_data = []
        for qid in target_ids:
            # Find the item with this qid
            item = next((x for x in filtered_data if x["question_id"] == qid), None)
            if item:
                ordered_data.append(item)

        # Write to file
        for item in ordered_data:
            f_out.write(json.dumps(item) + "\n")

    print(f"Created {coco_file} with {len(ordered_data)} entries")

print("Processing complete. Files saved to:", output_coco_dir)