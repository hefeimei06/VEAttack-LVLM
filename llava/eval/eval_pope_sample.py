import os
import json
import argparse

avg_f1_score = 0

def eval_pope(answers, label_file, category):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    # Process answers to convert text to "yes" or "no"
    for answer in answers:
        text = answer['text'].lower().strip()
        # Simplify text to "yes" or "no"
        if 'no' in text or 'not' in text:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    # Convert labels to 0 (no) or 1 (yes)
    for i in range(len(label_list)):
        label_list[i] = 0 if label_list[i].lower() == 'no' else 1

    # Create prediction list
    pred_list = [0 if answer['text'] == 'no' else 1 for answer in answers]

    # Ensure pred_list and label_list have the same length
    if len(pred_list) != len(label_list):
        print(f"Warning: Mismatch in number of predictions ({len(pred_list)}) and labels ({len(label_list)}) for category {category}")
        min_len = min(len(pred_list), len(label_list))  # Fixed here
        pred_list = pred_list[:min_len]
        label_list = label_list[:min_len]

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list) if pred_list else 0.0

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    str_to_log = f'Category: {category}, # samples: {len(pred_list)}\n'
    str_to_log += 'TP\tFP\tTN\tFN\n'
    str_to_log += f'{TP}\t{FP}\t{TN}\t{FN}\n'

    # Avoid division by zero
    precision = float(TP) / float(TP + FP) if (TP + FP) > 0 else 0.0
    recall = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    global avg_f1_score
    avg_f1_score += f1

    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    str_to_log += f'Accuracy: {acc:.3f}\n'
    str_to_log += f'Precision: {precision:.3f}\n'
    str_to_log += f'Recall: {recall:.3f}\n'
    str_to_log += f'F1 score: {f1:.3f}\n'
    str_to_log += f'Yes ratio: {yes_ratio:.3f}\n'
    str_to_log += f'%.3f, %.3f, %.3f, %.3f, %.3f\n' % (f1, acc, precision, recall, yes_ratio)
    print(str_to_log)
    return str_to_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--model-name", type=str, default='')
    args = parser.parse_args()

    # Load questions and create a dictionary
    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}

    # Load answers
    answers = [json.loads(q) for q in open(args.result_file)]

    # Validate question IDs
    answer_question_ids = set(answer['question_id'] for answer in answers)
    question_ids = set(questions.keys())
    missing_questions = answer_question_ids - question_ids
    if missing_questions:
        print(f"Warning: Answer file contains question IDs not found in question file: {missing_questions}")
        answers = [answer for answer in answers if answer['question_id'] in question_ids]

    outputs = ''
    category_count = 0

    for file in os.listdir(args.annotation_dir):
        if not (file.startswith('coco_pope_') and file.endswith('.json')):
            continue
        category = file[10:-5]
        # Filter answers for the current category
        cur_answers = [x for x in answers if x['question_id'] in questions and questions[x['question_id']]['category'] == category]
        if not cur_answers:
            print(f"Warning: No answers found for category {category}")
            continue
        outputs += f'Category: {category}, # samples: {len(cur_answers)}\n'
        print(f'Category: {category}, # samples: {len(cur_answers)}')
        outputs += eval_pope(cur_answers, os.path.join(args.annotation_dir, file), category)
        print("====================================")
        category_count += 1

    if category_count > 0:
        print(f"Average F1-score: {avg_f1_score/category_count:.4f}")
        outputs += f"Average F1-score: {avg_f1_score/category_count:.4f}\n"
    else:
        print("No categories evaluated.")
        outputs += "No categories evaluated.\n"

    # Save results
    os.makedirs("./pope_eval", exist_ok=True)
    with open(f"./pope_eval/{args.model_name}.txt", 'w') as f:
        f.write(outputs)