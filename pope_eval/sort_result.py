import json
import argparse

def sort_jsonl_by_question_id(input_file, output_file):
    # Read all records from the input file
    with open(input_file, 'r') as f:
        records = [json.loads(line) for line in f]

    # Sort records by question_id in ascending order
    sorted_records = sorted(records, key=lambda x: x['question_id'])

    # Write sorted records to a new JSONL file
    with open(output_file, 'w') as f:
        for record in sorted_records:
            f.write(json.dumps(record) + '\n')



parser = argparse.ArgumentParser()
parser.add_argument("--result-file", type=str)
args = parser.parse_args()

# Usage example
input_file = args.result_file + '.jsonl' # Your original file
output_file = args.result_file + '_sorted.jsonl'  # Output file with sorted records

sort_jsonl_by_question_id(input_file, output_file)
print(f"Successfully generated sorted file: {output_file}")