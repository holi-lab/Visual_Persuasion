import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    model_name = "cross-encoder/nli-deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    # cross-encoder/nli-deberta-v3-base label mapping:
    # {"contradiction": 0, "entailment": 1, "neutral": 2} usually, but we should make sure
    # using model.config.label2id
    entailment_idx = -1
    for k, v in model.config.label2id.items():
        if "entail" in k.lower():
            entailment_idx = v
            break
            
    if entailment_idx == -1:
        entailment_idx = 1
        
    input_file = "your_input_file"
    output_file = "your_output_file"
    
    hyp_a = "This image conveys the intended message persuasively."
    hyp_b = "This image fails to convey the intended message persuasively."
    
    results = []
    
    y_true = []
    y_pred = []
    tp = tn = fp = fn = 0
    
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        data = json.loads(line)
        reasoning = data.get("reasoning", "")
        
        # perform NLI for both hypotheses
        inputs_a = tokenizer(reasoning, hyp_a, return_tensors="pt", truncation=True)
        inputs_b = tokenizer(reasoning, hyp_b, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs_a = model(**inputs_a).logits
            outputs_b = model(**inputs_b).logits
            
        probs_a = torch.nn.functional.softmax(outputs_a, dim=-1)[0]
        probs_b = torch.nn.functional.softmax(outputs_b, dim=-1)[0]
        
        prob_persuasive = probs_a[entailment_idx].item()
        prob_not_persuasive = probs_b[entailment_idx].item()
        
        pred_inferability = "Yes" if prob_persuasive > prob_not_persuasive else "No"
        
        data["prob_persuasive"] = prob_persuasive
        data["prob_not_persuasive"] = prob_not_persuasive
        data["pred_inferability"] = pred_inferability
        
        results.append(data)
        
        # Calculate metrics data
        gt_inf = str(data.get("inferability", "")).lower()
        pred_inf = pred_inferability.lower()
        
        gt_label = 1 if 'yes' in gt_inf or gt_inf == 'true' else 0
        pred_label = 1 if 'yes' in pred_inf or pred_inf == 'true' else 0
        
        y_true.append(gt_label)
        y_pred.append(pred_label)
        
        if gt_label == 1 and pred_label == 1:
            tp += 1
        elif gt_label == 0 and pred_label == 0:
            tn += 1
        elif gt_label == 0 and pred_label == 1:
            fp += 1
        elif gt_label == 1 and pred_label == 0:
            fn += 1
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(lines)}")
            
    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"Saved results to {output_file}")
    
    total_eval = len(y_true)
    if total_eval > 0:
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = (sens + spec) / 2.0
        
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        po = (tp + tn) / total_eval
        pe_p = ((tp + fp) / total_eval) * ((tp + fn) / total_eval)
        pe_n = ((fn + tn) / total_eval) * ((fp + tn) / total_eval)
        pe = pe_p + pe_n
        
        kappa = (po - pe) / (1 - pe) if pe != 1.0 else 0.0
        
        result = {
            "model": "nli-deberta-v3-base",
            "balanced_accuracy": round(bal_acc, 4),
            "f1_score": round(f1, 4),
            "cohens_kappa": round(kappa, 4),
            "total_eval": total_eval,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }
        
        print("\nEvaluation Metrics:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
