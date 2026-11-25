"""Evaluation metrics and prediction helpers copied from the project notebook."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support, matthews_corrcoef, log_loss, roc_auc_score, average_precision_score)
def align_model_probabilities(model, X, labels):
    """Align classifier probability columns with the canonical training-label order."""
    raw_prob = np.asarray(model.predict_proba(X), dtype=float)
    estimator = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    model_classes = [str(v) for v in getattr(estimator, "classes_")]
    aligned = np.zeros((len(X), len(labels)), dtype=float)
    label_positions = {str(label): i for i, label in enumerate(labels)}
    for source_index, class_name in enumerate(model_classes):
        if class_name in label_positions:
            aligned[:, label_positions[class_name]] = raw_prob[:, source_index]
    row_sums = aligned.sum(axis=1, keepdims=True)
    aligned = np.divide(aligned, row_sums, out=np.zeros_like(aligned), where=row_sums > 0)
    return aligned


def calculate_metrics(y_true, y_pred, labels, y_proba=None):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    total = int(cm.sum())
    per_class_rows = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = total - tp - fp - fn
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        per_class_rows.append({
            "class": str(label), "precision": float(precision[i]),
            "recall": float(recall[i]), "f1_score": float(f1[i]),
            "specificity": float(specificity),
            "false_positive_rate_ovr": float(fpr), "support": int(support[i])
        })

    normal_label = "Normal" if "Normal" in labels else None
    if normal_label is not None:
        true_attack = y_true != normal_label
        predicted_attack = y_pred != normal_label
        fp_attack = int(np.sum(~true_attack & predicted_attack))
        fn_attack = int(np.sum(true_attack & ~predicted_attack))
        tn_attack = int(np.sum(~true_attack & ~predicted_attack))
        tp_attack = int(np.sum(true_attack & predicted_attack))
        binary_fpr = fp_attack / (fp_attack + tn_attack) if (fp_attack + tn_attack) else 0.0
        attack_recall = tp_attack / (tp_attack + fn_attack) if (tp_attack + fn_attack) else 0.0
    else:
        binary_fpr, attack_recall = float("nan"), float("nan")

    summary = {
        "n_samples": total,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(np.mean(recall)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f_weighted),
        "fpr_macro_ovr": float(np.mean([r["false_positive_rate_ovr"] for r in per_class_rows])),
        "specificity_macro_ovr": float(np.mean([r["specificity"] for r in per_class_rows])),
        "normal_vs_attack_fpr": float(binary_fpr),
        "attack_detection_recall": float(attack_recall),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred))
    }

    if y_proba is not None:
        probabilities = np.asarray(y_proba, dtype=float)
        if probabilities.shape != (len(y_true), len(labels)):
            raise ValueError(f"Probability matrix shape {probabilities.shape} does not match {(len(y_true), len(labels))}.")
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        y_onehot = np.column_stack([y_true == label for label in labels]).astype(float)
        summary["log_loss"] = float(log_loss(y_true, probabilities, labels=labels))
        summary["brier_score_multiclass"] = float(np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1)))
        try:
            summary["roc_auc_macro_ovr"] = float(roc_auc_score(
                y_true, probabilities, labels=labels, multi_class="ovr", average="macro"
            ))
        except Exception:
            summary["roc_auc_macro_ovr"] = float("nan")
        try:
            summary["average_precision_macro_ovr"] = float(average_precision_score(
                y_onehot, probabilities, average="macro"
            ))
        except Exception:
            summary["average_precision_macro_ovr"] = float("nan")
        if len(labels) >= 2:
            top2_indices = np.argpartition(probabilities, -2, axis=1)[:, -2:]
            true_index = {label: i for i, label in enumerate(labels)}
            summary["top2_accuracy"] = float(np.mean([
                true_index[label] in top2_indices[row]
                for row, label in enumerate(y_true)
            ]))

    return summary, pd.DataFrame(per_class_rows), cm


def risk_level(category):
    return "Normal" if str(category).strip().lower() == "normal" else "Attack"
