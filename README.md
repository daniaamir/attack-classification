from pathlib import Path

readme = r'''# Robust Attack-Type Classification on UNSW-NB15 Network Traffic

A machine-learning-based Network Intrusion Detection System (NIDS) that classifies network-flow records into **10 traffic categories** using the UNSW-NB15 benchmark dataset. The project evaluates data quality, duplicate-aware validation, model performance, feature reliance, and robustness to corrupted input.

> **Project:** CSCY413 – Special Topics in Cybersecurity  
> **Institution:** British University in Dubai  
> **Author:** Dania Elnour  
> **Student ID:** 23002102

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [System Workflow](#system-workflow)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Robustness and Feature Analysis](#robustness-and-feature-analysis)
- [Using the Prototype](#using-the-prototype)
- [Running the Project in Google Colab](#running-the-project-in-google-colab)
- [Repository Contents](#repository-contents)
- [Limitations](#limitations)
- [Reproducibility and Testing](#reproducibility-and-testing)
- [References](#references)

---

## Project Overview

Network environments generate large volumes of traffic records, making manual identification of suspicious activity difficult. This project develops a supervised, multi-class NIDS prototype that analyses network-flow features and predicts whether a record belongs to normal traffic or one of nine attack categories.

The work focuses not only on classification performance, but also on whether the evaluation is affected by duplicate records, conflicting labels, extreme feature values, class imbalance, and reliance on dominant features.

The final selected classifier is **LightGBM with mild class weighting**. The prototype includes a Gradio interface for classifying an individual traffic record or an uploaded CSV file. It is an offline, file-based prototype: it does not capture live network packets, block connections, or automatically respond to attacks.

## Objectives

1. Audit the UNSW-NB15 training and testing partitions for data-quality issues.
2. Build and compare baseline, tree-ensemble, and gradient-boosting classifiers.
3. Compare numeric preprocessing and class-weighting strategies.
4. Use duplicate-grouped validation to reduce the risk of identical feature vectors appearing in both training and validation folds.
5. Evaluate the selected model on the official testing partition.
6. Examine performance by attack category, including false-positive behaviour.
7. Test robustness against Gaussian noise, extreme values, and zero replacement.
8. Analyse feature reliance through permutation importance and TTL-feature ablation.
9. Provide a simple interface for single-record and batch CSV predictions.

## Dataset

The project uses the **UNSW-NB15** network intrusion detection dataset and its official training/testing partition.

| Partition | Records |
|---|---:|
| Training | 175,341 |
| Testing | 82,332 |
| Total | 257,673 |

The classification target is `attack_cat`, containing ten classes:

- Normal
- Generic
- Exploits
- Fuzzers
- DoS
- Reconnaissance
- Analysis
- Backdoor
- Shellcode
- Worms

The original partition files contain 42 predictor columns: 39 numeric features and three categorical features (`proto`, `service`, and `state`). The identifier (`id`), binary label (`label`), and target (`attack_cat`) are excluded from the model predictors. Seven engineered traffic features are added, resulting in **49 model predictors**.

### Data-quality audit

The audit identified several issues that are important when interpreting the results:

- 42.4% of training rows repeat the feature values of another row.
- 17.4% of training rows belong to feature groups associated with conflicting class labels.
- 8,541 testing records (10.37%) have an exact feature-vector match in the training partition.
- Seventeen features have more than 50% zero values.
- Nine features contain values above one million.
- The `sttl` feature is especially concentrated in Normal traffic.

Duplicate groups are used when creating validation folds. The post-evaluation test-overlap analysis is descriptive; it is not used to tune or reselect the model.

**Dataset source:** [UNSW-NB15 dataset – Australian Centre for Cyber Security](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

The dataset files are not reproduced in this README. Use the official dataset files or the copies provided with the project, subject to the dataset's terms and citation requirements.

## System Workflow

The implemented workflow is:

1. Load the official training and testing CSV files.
2. Validate the required columns and dataset schema.
3. Exclude identifier and target-leakage columns from the predictors.
4. Audit duplicates, conflicting labels, zero-heavy features, and extreme values.
5. Generate seven additional traffic features from byte counts, packet counts, and flow duration.
6. Preprocess numeric and categorical columns.
7. Compare candidate models using duplicate-grouped validation on training data.
8. Select a model using the accuracy guardrail and Macro-F1.
9. Refit the selected configuration on the complete training partition.
10. Evaluate on the official testing partition and generate metrics and diagnostic outputs.
11. Run training-only robustness and feature-reliance diagnostics.
12. Use the Gradio interface for single-record or batch CSV predictions.

## Methodology

### Feature engineering

Seven deterministic features are derived from the original traffic fields:

- Total bytes
- Source/destination byte log ratio
- Total packets
- Source/destination packet log ratio
- Mean bytes per packet
- Bytes per duration
- Packets per duration

The same feature-preparation logic is used for training and inference. The target and identifier are not used to create predictors.

### Preprocessing strategies

| Strategy | Description |
|---|---|
| S0 – Raw | Keeps numeric values unchanged, with fitted preprocessing safeguards. |
| S1 – Clipping | Clips numeric values to the training-fold 1st and 99th percentiles. |
| S2 – Signed-log | Applies a signed logarithmic transformation and zero indicators for zero-heavy features. |

Categorical variables are imputed and one-hot encoded. Unseen categorical levels are ignored by the encoder.

### Class weighting

Three principal weighting approaches were considered:

- **Plain:** no class weighting.
- **Mild:** class weights use an exponent of 0.35 to give additional influence to minority classes without fully balancing all classes.
- **Balanced:** stronger weighting that gives each class equal total weight.

### Validation and model selection

Model selection uses the training partition only. Feature-row hashes are used as groups so identical feature vectors are not split across training and validation folds.

In the reported fast screening run, nine candidate configurations were compared using two sampled duplicate-grouped folds. Candidates within 1.5 percentage points of the best mean validation accuracy were eligible; the eligible model with the highest Macro-F1 was selected.

The selected model was then refitted on all 175,341 training records. The official testing partition was evaluated after the model configuration had been locked.

## Models

The candidate screening included:

- Majority-class baseline
- Logistic Regression
- Random Forest
- Extra Trees
- Histogram-based Gradient Boosting
- XGBoost
- LightGBM
- Soft-voting ensemble

**Selected model:** LightGBM with mild class weights.

Reported final configuration:

| Parameter | Value |
|---|---:|
| Boosting estimators | 520 |
| Learning rate | 0.035 |
| Number of leaves | 47 |
| Minimum child samples | 45 |
| Row subsampling | 0.90 |
| Column subsampling | 0.90 |
| L2 regularisation | 4.0 |
| Class-weight exponent | 0.35 |
| Numeric preprocessing | S0 – Raw |

## Results

The following values are from the final model's evaluation on the official testing partition.

| Metric | Result |
|---|---:|
| Accuracy | **74.33%** |
| Macro precision | 0.5323 |
| Macro recall | 0.6100 |
| Macro-F1 | **0.5442** |
| Weighted F1 | **0.7778** |
| Macro one-vs-rest FPR | 0.0281 |
| Normal-versus-attack FPR | 0.2897 |
| Attack detection recall | **98.47%** |
| Matthews correlation coefficient | 0.6761 |
| Macro ROC-AUC | 0.9617 |
| Top-2 accuracy | 0.9166 |

The evaluation covers 82,332 testing records. The final model training took approximately 235.9 seconds, and prediction over the official testing partition took approximately 106.9 seconds (about 1.299 milliseconds per record) in the reported run.

### Interpreting the results

The high attack recall indicates that the model identifies most attack records in the binary Normal-versus-attack view. However, the Normal-versus-attack false-positive rate is approximately 29%, meaning a substantial proportion of normal traffic is incorrectly flagged as an attack.

The Macro-F1 score is lower than the Weighted F1 score because Macro-F1 gives each of the ten classes equal importance, including rare attack categories. The results should therefore be interpreted using several metrics rather than accuracy alone.

## Robustness and Feature Analysis

The following diagnostics were performed on a training-only grouped validation fold. They are **not official testing-set results**.

### Corruption tests

| Condition | Macro-F1 | Retained Macro-F1 |
|---|---:|---:|
| Clean validation data | 0.6453 | 100.0% |
| Gaussian noise (1% of training-fold standard deviation) | 0.3148 | 48.8% |
| Extreme values (1% of numeric cells) | 0.6356 | 98.5% |
| Zero replacement (5% of numeric cells) | 0.5911 | 91.6% |

Gaussian noise caused the largest performance reduction in these experiments. This indicates that the selected model is sensitive to small numeric perturbations under the tested condition.

### TTL-feature ablation

The features `sttl`, `dttl`, and `ct_state_ttl` were removed together for an ablation experiment.

| Condition | Macro-F1 |
|---|---:|
| Original feature group | 0.6453 |
| Without TTL features | 0.6330 |
| Difference | -0.0123 |

The result suggests that the selected model retains most of its diagnostic-fold Macro-F1 without the TTL feature group. Permutation importance and the confusion matrix provide additional views of model behaviour.

### Duplicate-overlap audit

Among the 8,541 official testing records with an exact feature-vector match in the training data:

- Accuracy on matching records: 51.47%.
- Accuracy on testing records without an exact training match: 76.98%.

This is a descriptive post-evaluation audit and was not used for model selection.

## Using the Prototype

The Gradio interface provides four views:

1. **Single traffic record:** submit one JSON record containing the required raw traffic predictor fields.
2. **Batch CSV:** upload a CSV file containing one traffic record per row.
3. **Evaluation dashboard:** inspect available model evaluation information and results.
4. **Dataset/model overview:** view information about the dataset and selected model.

For a single record, the interface returns:

- Predicted attack category
- Risk level (`Normal` or `Attack`)
- Prediction confidence
- Probability for each of the ten classes

For batch CSV input, the output retains the input columns and adds:

- `predicted_attack_cat`
- `risk_level`
- `prediction_confidence`

Provide the raw predictor columns; the system derives the seven engineered features internally. Prediction confidence is the model's maximum class probability and is not a guarantee that the prediction is correct.

## Running the Project in Google Colab

The executable prototype is provided as a Google Colab notebook.

1. Open the notebook stored in this repository.
2. Select **Runtime → Run all** (or execute the cells in order).
3. Allow the notebook to install or import the required dependencies.
4. When prompted, upload the official `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` files.
5. Wait for data auditing, feature preparation, model screening, final training, evaluation, and export steps to finish.
6. Open the Gradio interface when the notebook launches it.
7. Upload a CSV file or enter a single traffic record to obtain predictions.
8. Retain the generated results directory and test logs together.

The report describes `QUICK_RUN=False` as retaining the complete official training partition for the final refit. `FAST_MODE=True` uses a reduced screening workload; it does not mean that the final model is trained on only the sampled screening groups. Setting `FAST_MODE=False` activates the longer screening and confirmation protocol described in the report.

> **Important:** The exact notebook filename and dependency versions should be taken from the files in this repository. Do not assume that a results archive alone contains the notebook source.

## Repository Contents

The repository contains the project implementation, dataset files, experiment outputs, and academic deliverables. Use the folders and filenames shown in the GitHub repository as the source of truth.

Expected project artefacts include:

- Google Colab notebook / implementation source
- UNSW-NB15 training and testing CSV files
- Saved LightGBM model (`.joblib`)
- Official testing predictions
- Metrics in CSV and JSON formats
- Confusion matrix and diagnostic figures
- Automated test output and summary
- `requirements.txt`
- Project reports and supplementary results

The supplementary execution-results archive documents a saved model, predictions, metrics, figures, README, requirements file, manifest, and automated test log. The notebook source should be available separately.

## Reproducibility and Testing

The implementation defines 19 automated checks. The archived execution summary reports:

- 19 tests run
- 0 failures
- 0 errors
- Status: **PASS**

The checks cover pipeline and software properties such as leakage-column exclusion, schema consistency, duplicate-group separation, deterministic feature engineering, class-weight validity, probability alignment, model save/reload, unseen categories, metric validity, and the existence of required result artefacts.

Passing software tests does not establish real-world detection performance.

## Limitations

- The project uses the UNSW-NB15 benchmark and does not establish performance on live organisational traffic.
- The prototype is file-based and does not capture live packets, block network connections, or automatically respond to threats.
- Minority attack categories remain difficult to classify reliably.
- The Normal-versus-attack false-positive rate is approximately 29% on the official testing partition.
- Gaussian noise at the tested level substantially reduced Macro-F1.
- Confidence values are not calibrated guarantees.
- The historical development runs used the same official testing partition; they are not three independent holdout evaluations.
- Cross-dataset validation and live deployment testing remain future work.

## References

1. N. Moustafa and J. Slay, “UNSW-NB15: A comprehensive data set for network intrusion detection systems,” 2015. [UNSW-NB15 project page](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
2. M. T. Ribeiro et al. / relevant explainability and machine-learning references are listed in the project's IEEE report. See the submitted report's reference section for the complete bibliography.

For the full literature review, methodology, detailed per-class metrics, confusion matrix, feature analysis, and discussion, consult the submitted **Deliverable 2 IEEE report** in this repository.

---

## Academic Use

This repository was developed as part of the CSCY413 Special Topics in Cybersecurity course at the British University in Dubai. The implementation is an academic prototype and should not be treated as a production-ready intrusion prevention system.
'''
path = Path("/mnt/data/README.md")
path.write_text(readme, encoding="utf-8")
print(f"Created {path} ({len(readme.splitlines())} lines, {path.stat().st_size} bytes)")
