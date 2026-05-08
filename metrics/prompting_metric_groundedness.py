import json
import os
import csv
import collections
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score
from scipy.stats import spearmanr
import numpy as np

def calculate_metrics(y_true, y_pred, y_true_binary=None, y_scores=None):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'yes' and yp == 'yes')
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'no' and yp == 'no')
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'no' and yp == 'yes')
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'yes' and yp == 'no')
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    precision_no = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_no = 2 * (precision_no * specificity) / (precision_no + specificity) if (precision_no + specificity) > 0 else 0.0
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    po = (tp + tn) / total if total > 0 else 0.0
    pe_yes = ((tp + fp) / total) * ((tp + fn) / total)
    pe_no = ((tn + fn) / total) * ((tn + fp) / total)
    pe = pe_yes + pe_no
    
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "f1_score": f1,
        "f1_score_no": f1_no,
        "cohens_kappa": kappa,
        "total_eval": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

    return metrics

def fleiss_kappa(ratings, n_categories=2):
    """
    Computes Fleiss' kappa for a set of ratings.
    ratings: array of shape (N_items, N_raters) containing category indices
    n_categories: total number of distinct categories
    """
    if not ratings:
        return 0.0
    N = len(ratings)
    k = len(ratings[0])
    
    counts = np.zeros((N, n_categories))
    for i, item_ratings in enumerate(ratings):
        for rating in item_ratings:
            counts[i, rating] += 1
            
    P_i = (np.sum(counts**2, axis=1) - k) / (k * (k - 1))
    P_bar = np.mean(P_i)
    
    p_j = np.sum(counts, axis=0) / (N * k)
    P_e_bar = np.sum(p_j**2)
    
    if P_e_bar == 1:
        return 1.0
        
    return (P_bar - P_e_bar) / (1 - P_e_bar)

def main():
    base_dir = 'Qwen-VL-Series-Finetune/groundedness/ocr/results'
    csv_file = os.path.join(base_dir, 'annotations_merged.csv')
    jsonl_file = os.path.join(base_dir, 'sampled_210_original_evaluated2.jsonl')
    
    # 1. Read annotators' groundedness labels
    annotators_data_by_path = {}
    annotators_data_by_id = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_key = "/".join(row['image'].split('/')[-2:])
            csv_fp = row.get('file_path', '').strip()
            
            if csv_fp == 'pos_neg/visual_elements.json':
                dict_fp = 'visual.json'
            else:
                dict_fp = csv_fp.split('/')[-1]
                
            ann = {
                'u1': row['groundedness_user1'].strip().lower(),
                'u2': row['groundedness_user2'].strip().lower(),
                'u3': row['groundedness_user3'].strip().lower()
            }
            annotators_data_by_path[(dict_fp, img_key)] = ann
            if 'original_index' in row and row['original_index'].isdigit():
                annotators_data_by_id[int(row['original_index'])] = ann
            
    # 2. Read model predictions
    y_true_majority = []
    y_pred = []
    output_results = []
    
    u1_labels_list = []
    u2_labels_list = []
    u3_labels_list = []
    pred_labels_list = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            idx = int(item.get('id', -1))
            img_key = item.get('image', '')
            
            # parse evaluated_reason
            eval_reason = item.get('evaluated_reason', '{}')
            try:
                parsed_eval = json.loads(eval_reason)
                pred_label = parsed_eval.get('label', '').strip().lower()
            except:
                pred_label = ''
                
            item_fp = item.get('file_path')
            
            if item_fp is not None:
                if (item_fp, img_key) not in annotators_data_by_path:
                    continue
                ann = annotators_data_by_path[(item_fp, img_key)]
            else:
                if idx in annotators_data_by_id:
                    ann = annotators_data_by_id[idx]
                else:
                    continue
                
            u1, u2, u3 = ann['u1'], ann['u2'], ann['u3']
            
            # Ensure valid labels to maintain consistent subset across all metrics
            if not all(label in {'yes', 'no'} for label in (u1, u2, u3, pred_label)):
                continue
            
            # Majority vote
            votes = [u1, u2, u3]
            majority_vote = collections.Counter(votes).most_common(1)[0][0]
            
            # Save for metric calc
            u1_labels_list.append(u1)
            u2_labels_list.append(u2)
            u3_labels_list.append(u3)
            y_true_majority.append(majority_vote)
            y_pred.append(pred_label)
            pred_labels_list.append(pred_label)
            
            output_results.append({
                "id": idx,
                "image": img_key,
                "ground_truth_majority": majority_vote,
                "predicted": pred_label,
                "u1": u1,
                "u2": u2,
                "u3": u3
            })
            
    # 3. Calculate Annotator Agreement (pairwise Cohen's Kappa)
    print("--- Annotator Agreement Metrics ---")
    kappa_12 = cohen_kappa_score(u1_labels_list, u2_labels_list)
    kappa_23 = cohen_kappa_score(u2_labels_list, u3_labels_list)
    kappa_13 = cohen_kappa_score(u1_labels_list, u3_labels_list)
    
    # 4. Calculate Fleiss Kappa for all 3 annotators
    # Map 'no' -> 0, 'yes' -> 1
    label_to_idx = {'no': 0, 'yes': 1}
    fleiss_ratings = []
    fleiss_ratings_with_model = []
    for u1, u2, u3, pred in zip(u1_labels_list, u2_labels_list, u3_labels_list, pred_labels_list):
        if u1 in label_to_idx and u2 in label_to_idx and u3 in label_to_idx:
            fleiss_ratings.append([label_to_idx[u1], label_to_idx[u2], label_to_idx[u3]])
        if u1 in label_to_idx and u2 in label_to_idx and u3 in label_to_idx and pred in label_to_idx:
            fleiss_ratings_with_model.append([label_to_idx[u1], label_to_idx[u2], label_to_idx[u3], label_to_idx[pred]])
            
    fleiss_k = fleiss_kappa(fleiss_ratings, n_categories=2)
    fleiss_k_with_model = fleiss_kappa(fleiss_ratings_with_model, n_categories=2)
    
    print(f"Cohen's Kappa (User1 vs User2): {kappa_12:.4f}")
    print(f"Cohen's Kappa (User2 vs User3): {kappa_23:.4f}")
    print(f"Cohen's Kappa (User1 vs User3): {kappa_13:.4f}")
    print(f"Fleiss' Kappa (3 Users only): {fleiss_k:.4f}")
    print(f"Fleiss' Kappa (3 Users + Model): {fleiss_k_with_model:.4f}")
    print("-" * 40)
    
    # 5. Evaluate Model vs Annotators
    print("--- Model Performance vs Ground Truth ---")
    metrics_majority = calculate_metrics(y_true_majority, y_pred)
    print("Model vs Majority Vote:")
    print(json.dumps(metrics_majority, indent=4))
    
    print("-" * 40)
    kappa_majority = cohen_kappa_score(y_true_majority, y_pred)
    kappa_pred_u1 = cohen_kappa_score(u1_labels_list, pred_labels_list)
    kappa_pred_u2 = cohen_kappa_score(u2_labels_list, pred_labels_list)
    kappa_pred_u3 = cohen_kappa_score(u3_labels_list, pred_labels_list)
    print(f"Model Cohen's Kappa against Majority Vote: {kappa_majority:.4f}")
    print(f"Model Cohen's Kappa against User1: {kappa_pred_u1:.4f}")
    print(f"Model Cohen's Kappa against User2: {kappa_pred_u2:.4f}")
    print(f"Model Cohen's Kappa against User3: {kappa_pred_u3:.4f}")

    # Write the predictions to an output file
    out_file = os.path.join(base_dir, 'gpt_metric_evaluated_reason_results.jsonl')
    with open(out_file, 'w', encoding='utf-8') as f:
        for r in output_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\nWritten detailed predictions and ground truths to {out_file}")

if __name__ == '__main__':
    main()
