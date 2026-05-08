# visual-persuasion
Official repository containing the evaluation scripts for the paper: "Can MLLMs Reason About Visual Persuasion? Evaluating the Efficacy and Faithfulness of Reasoning."

This repository focuses on our three-dimensional evaluation framework for measuring model-rationale faithfulness: 
<br>Rationale-to-Decision Consistency, Rationale-to-Image Groundedness, and Rationale-to-Decision Sensitivity.

📊 Evaluation Metrics
The included scripts correspond to the automated evaluation pipelines described in Section 4 of the paper:

Consistency:
<br>run_nli.py: Evaluates whether the predicted persuasiveness decision is derivable from the generated rationale.

Groundedness:
<br>groundedness_calculate.py: Main script for aggregating groundedness metrics.
<br>prompting_metric_groundedness.py: Groundedness evaluation using zero-shot LLM prompting.
<br>atomic_facts_groundedness.py: Groundedness evaluation using the decomposed atomic facts methodology.
<br>clip_metric_groundedness.py: Baseline groundedness calculations using CLIP text/image similarity.

Sensitivity:
<br>sensitivity.py & sensitivity.sh: Pipeline for measuring rationale-to-decision sensitivity via counterfactual image editing.

🏋️ Training Code
The supervised fine-tuning (Reasoning-SFT) and Group Relative Policy Optimization (GRPO) training codes are not hosted in this repository. Our training environments were built by cloning and adapting the following open-source repositories:
<br>Qwen-2.5-VL Training: 2U1/Qwen-VL-Series-Finetune
<br>Phi-3.5-vision Training: 2U1/Phi3-Vision-Finetune

📄 Dataset Structure
<br>The `qwen.json` and the `phi.json` file contains the generated rationales and formatted instruction data. Each entry includes the following fields:
<br>`id`: A unique integer identifier for the data instance.
<br>`image`: The file path to the corresponding image being evaluated.
<br>`reasoning`: The generated rationale assessing the image's persuasiveness
<br>`answer`: The binary decision ("Yes" or "No") indicating if the image persuasively conveys the target message.
<br>`conversations`: A structured array of dialogue turns (alternating between "human" and "gpt") formatted for multi-turn instruction tuning.


Running the .py files
<br> python run_nli.py

Running the .sh files
<br> bash sensitivity.sh
