# Robust Attack-Type Classification on Noisy Network Traffic Data

CSCY413 Special Topics in Cybersecurity | Track: Trustworthy & Robust AI for Security Operations

## Overview

Cyber attacks are becoming more frequent and sophisticated, and organizations
generate far more network traffic than analysts can inspect manually. Machine-learning
classifiers can help detect and categorise threats early, but real traffic data is
often messy: it contains extreme values, impossible values (such as negative
counts), and placeholder zeros that act as disguised missing data.

This project builds and evaluates classifiers that predict the attack type of a
network traffic record, and studies how reliable they are when the data is noisy.

Research question: How accurately can a machine-learning model classify attack
types from network traffic features, how robust is it to extreme and corrupted
values, and which features does it actually rely on?

Planned approach:
- Explore the data and document its quality issues
- Build baseline models (majority class, logistic regression)
- Build stronger models (e.g. Random Forest, gradient boosting)
- Compare preprocessing strategies for extreme values
- Test robustness with ablation studies and explainability (feature importance, SHAP)

## Dataset

Source: [kaggle]

| File | Rows | Description |
|------|------|-------------|
| train.csv | 10,684 | Labelled training data |
| test.csv | 2,672 | Unlabelled test data (no target column) |
| sample_submission.csv | 5,344 | Example submission format |

Features: 13 numeric features, plus an ID and a Timestamp (1 Jan - 25 Feb 2023):
Suspicious_Port_Activity, Traffic_Volume_Variation, Packet_Length_Anomaly,
Malware_Score, Threat_Level_Index, User_Behavior_Score, Geo_Dispersion,
Payload_Entropy, Login_Attempts, Device_Response_Time, Session_Duration,
Packet_Retry_Rate, Anomaly_Tendency.

Target:`Attack Type` with three classes: 0 (50%), 1 (25%), 2 (25%).


Known data issues:
- Login_Attempts has extreme values (about +/- 8e8), and about a quarter are negative
- Device_Response_Time is exactly 0 for all class-0 records, which acts as a near-perfect separator
- Session_Duration and Packet_Retry_Rate contain a few negative values
- Several features show almost no correlation with the target, which suggests the data may be synthetic

**Note:** The raw data files are not stored in this repository.
Place train.csv and test.csv in the `data/` folder to run the code.

## Project Structure

- `docs/` - project proposal and reports
- `data/` - dataset notes (raw files are not committed)
- `notebooks/` - data exploration and experiments
- `src/` - preprocessing, models, and evaluation code
- `tests/` - automated tests
- `results/` - metrics, figures, and confusion matrices
- `requirements.txt` - Python dependencies

## Ethics and Legal Compliance

- Only a public or instructor-provided dataset is used.
- No live network traffic is captured and no systems are tested without authorisation.
- No personal data is collected or stored.
- All datasets, tools, and libraries are cited.
- Use of generative AI follows the module's AIRA Level 3 policy: it is limited to
  planning, feedback, and language refinement. All code is written and verified by the author.

## Author

[Dania Elnour] ([23002102])
Individual project, CSCY413 [semester: 1/year: 4]
