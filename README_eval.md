## 📊 Evaluation & Benchmarking

KG-Inspect provides evaluation utilities to benchmark the inspection pipeline on **MMAD-style datasets** such as **DS-MVTec** and **VisA**.

The evaluation workflow is designed to:

- Run KG-Inspect (CNN + RAG/VLM) on image–question pairs
- Save per-question predictions to CSV
- Compute dataset-level metrics
- Support **resume**, **aggregation**, and **re-evaluation** of failed cases

---

## 1️⃣ Initial Evaluation (MMAD-style)

The main evaluation script runs the full **InspectionPipeline** on MMAD-style JSON annotations.

### Script

```

eval_kg_inspect_mmad.py

```

### What it does

- Loads MMAD-style `mmad.json`
- Runs **CNNInspect once per image** (cached)
- Answers all questions associated with that image
- Ensures MCQ questions always include `Options: A/B/C/D`
- Retries automatically if the model returns:
  - `SORRY_TEXT`
  - empty / unparsable answers
- Saves:
  - per-question answers CSV
  - final metrics CSV
  - accuracy trend plots (PNG)

### Example command

```bash
python eval_kg_inspect_mmad.py \
  --data_path   /path/to/MMAD_root \
  --json_path   /path/to/mmad.json \
  --output_dir  ./kg_inspect_eval \
  --mode        hybrid \
  --device      cuda \
  --save_every  500 \
  --max_retries 5
```

### Outputs

```text
kg_inspect_eval/
├── answers_kg_inspect_DS-MVTec_VisA.csv
├── metrics_kg_inspect_MVTec-AD_VisA.csv
├── accuracy_until_00100_images.png
├── accuracy_until_00200_images.png
└── kg_inspect_eval_log.txt
```

---

## 2️⃣ Aggregating Evaluation Results (Multiple Machines)

When evaluation is run on **multiple machines** or **multiple subsets**, you can merge all results and recompute global metrics.

### Script

```
aggregate_eval_results.py
```

### What it does

- Merges multiple `answers_*.csv`
- Cleans redundant columns (`Unnamed:*`, `retries`, `retry_status`)
- Drops duplicates by `(dataset, image, question_index)`
- Recomputes metrics on selected datasets

### Example command

```bash
python aggregate_eval_results.py \
  --answers \
    run1/answers.csv \
    run2/answers.csv \
    run3/answers.csv \
  --combined_output answers_kg_inspect_merged.csv \
  --metrics_output  metrics_kg_inspect_merged.csv \
  --datasets_to_keep MVTec-AD VisA \
  --normal_flag good
```

---

## 3️⃣ Re-evaluating Failed Rows (Retry SORRY / Empty Answers)

If some rows failed during the first evaluation (e.g. returned `SORRY_TEXT`),
you can selectively **re-evaluate only those rows** without rerunning everything.

### Script

```
eval_retry_failed_rows.py
```

### What it does

- Loads an existing `answers_*.csv`
- Finds rows where:
  - `raw_output == SORRY_TEXT`, or
  - `model_answer` is empty

- Re-runs only those rows
- Rebuilds **visual context once per image**
- Injects MCQ options from `mmad.json` if missing
- Updates answers and recomputes metrics

### Example command

```bash
python eval_retry_failed_rows.py \
  --answers_csv ./kg_inspect_eval/answers_kg_inspect_DS-MVTec_VisA.csv \
  --data_path   /path/to/MMAD_root \
  --json_path   /path/to/mmad.json \
  --output_dir  ./kg_inspect_reval \
  --mode        hybrid \
  --max_retries 5
```

### Outputs

```text
kg_inspect_reval/
├── answers_kg_inspect_DS-MVTec_VisA_reval.csv
├── metrics_kg_inspect_reval.csv
└── reval_log.txt
```

---

## 📁 CSV Schema (Answers)

Each answers CSV follows this schema:

| Column         | Description                        |
| -------------- | ---------------------------------- |
| dataset        | Dataset name (e.g. MVTec-AD, VisA) |
| image          | Relative image path                |
| question_index | Question index per image           |
| question_type  | Question type                      |
| question_text  | Original question text             |
| correct_answer | Ground-truth answer                |
| model_answer   | Parsed model answer                |
| is_correct     | Boolean correctness                |
| raw_output     | Full raw model output              |
| retries        | (optional) number of retries       |
| retry_status   | (optional) retry status            |

---

## 🧪 Notes & Best Practices

- **Pretrained models are required** for evaluation
  See the _Pretrained Models_ section above.
- Text-only RAG evaluation will still work without vision models.
- Use `--resume` in the initial eval to safely continue long runs.
- Use re-eval instead of full reruns to save compute.
- Always aggregate before reporting final metrics.

---

## 🧭 Typical Evaluation Workflow

```text
Initial Eval (single / multi machine)
        ↓
Aggregate CSVs
        ↓
Re-evaluate failed rows
        ↓
Final metrics & plots
```

---
