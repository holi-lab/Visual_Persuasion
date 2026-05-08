import json
import os

# 1. Provide the file paths to your 6 JSONL files here.
# Replace these placeholder paths with your actual filenames/paths.
file_paths = [
    '/Qwen-VL-Series-Finetune/groundedness/ocr/paper_results/results/phi_base_groundedness.jsonl',
    '/Qwen-VL-Series-Finetune/groundedness/ocr/paper_results/results/phi_grpo_groundedness.jsonl',
    '/Qwen-VL-Series-Finetune/groundedness/ocr/paper_results/results/phi_merged_groundedness.jsonl',
    # Add the remaining three paths below
    '/Qwen-VL-Series-Finetune/groundedness/ocr/paper_results/results/qwen_base_groundedness.jsonl',
    '/Qwen-VL-Series-Finetune/groundedness/ocr/paper_results/results/qwen_grpo_groundedness.jsonl',
    '/Qwen-VL-Series-Finetune/groundedness/ocr/paper_results/results/qwen_merged_groundedness.jsonl',
]

results = []

print("Starting groundedness calculation...\n")

# 2. Iterate through each file and calculate the metric
for file_path in file_paths:
    total_entries = 0
    yes_count = 0
    
    filename = os.path.basename(file_path) # Get just the filename for display
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            print(f"Processing file: {filename}...")
            
            for line in f:
                try:
                    # Parse each line as a JSON object
                    entry = json.loads(line.strip())
                    total_entries += 1
                    
                    # 3. Use the structure you described: check evaluated_reason -> label
                    evaluated_reason = entry.get("evaluated_reason")
                    if evaluated_reason:
                        # Sometimes 'evaluated_reason' is a string containing a JSON object,
                        # and sometimes it might be the dictionary itself depending on your 
                        # exact data storage. We need to handle both cases.
                        
                        # Case A: evaluated_reason is a JSON string (most likely)
                        if isinstance(evaluated_reason, str):
                            reason_data = json.loads(evaluated_reason)
                            label = reason_data.get("label")
                        # Case B: evaluated_reason is already a dictionary
                        elif isinstance(evaluated_reason, dict):
                            label = evaluated_reason.get("label")
                        else:
                            label = None
                        
                        # Count the "Yes" evaluations
                        if label == "Yes":
                            yes_count += 1
                            
                except json.JSONDecodeError:
                    print(f"  Error: Could not parse JSON on line {total_entries} in {filename}. Skipping line.")
                except Exception as e:
                    print(f"  An unexpected error occurred while processing line {total_entries} in {filename}: {e}")
                    
        # 4. Calculate Groundedness (with check for empty files)
        if total_entries > 0:
            groundedness_score = yes_count / total_entries
            # Format to 3 decimal places
            formatted_score = f"{groundedness_score:.3f}" 
        else:
            groundedness_score = 0.0
            formatted_score = "0.000"
            print(f"  Warning: File {filename} was empty.")
            
        print(f"  -> Total Entries: {total_entries}, 'Yes' Count: {yes_count}, Score: {formatted_score}")
        
        results.append({
            "file": filename,
            "total_entries": total_entries,
            "yes_count": yes_count,
            "groundedness": formatted_score
        })
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check your path and try again.")
    except Exception as e:
        print(f"An unexpected error occurred while opening the file {file_path}: {e}")

# 5. Display the final results in a table format
print("\n" + "="*50)
print(f"{'Filename':<30} | {'Groundedness':>15}")
print("-" * 50)
for result in results:
    print(f"{result['file']:<30} | {result['groundedness']:>15}")
print("="*50)
print("\nYou can use the formatted scores above to fill in the Groundedness column in Table 3!")