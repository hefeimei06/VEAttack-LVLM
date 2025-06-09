import json
import argparse

# Input file paths
parser = argparse.ArgumentParser()
parser.add_argument("--result-file", type=str)
args = parser.parse_args()
result_jsonl_path = args.result_file + '.jsonl'  # Your sampled results file
original_jsonl_path = "./pope_eval/llava_pope_test.jsonl"  # Original POPE test file
output_jsonl_path = "./pope_eval/llava_pope_test_sample.jsonl"  # Output file path

# 1. Extract all question_ids from result.jsonl
selected_question_ids = set()
with open(result_jsonl_path, "r") as f_result:
    for line in f_result:
        data = json.loads(line)
        selected_question_ids.add(data["question_id"])

print(f"Found {len(selected_question_ids)} unique question_ids")

# 2. Filter matching entries from the original file
matched_count = 0
with open(original_jsonl_path, "r") as f_original, open(output_jsonl_path, "w") as f_out:
    for line in f_original:
        entry = json.loads(line)
        if entry["question_id"] in selected_question_ids:
            f_out.write(line)
            matched_count += 1

print(f"Successfully matched and wrote {matched_count} records to {output_jsonl_path}")

# Verify if counts match
if matched_count == len(selected_question_ids):
    print("Validation passed: Output count matches input count")
else:
    print(f"Warning: Only matched {matched_count} records, expected {len(selected_question_ids)}")