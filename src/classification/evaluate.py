"""Clinical diagnostic evaluation module for brain tumor classification models.

Provides standardized evaluation routines:
- Multi-class accuracy, loss, and Scikit-Learn classification reports
- Clean confusion matrix visualization via ConfusionMatrixDisplay
- Medical diagnostic safety metrics: False Negatives, Sensitivity, and Specificity
- Serialization of evaluation metrics and metadata to JSON
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
import tensorflow as tf


def plot_confusion_matrix(y_true, y_pred, class_names, title="Test Confusion Matrix (GoogLeNet)", save_path=None):
    """Plot and optionally save a publication-quality confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    return fig, cm


def compute_clinical_safety_metrics(y_true, y_pred, class_names, healthy_class_name="notumor"):
    """Compute critical medical safety metrics: False Negatives, Tumor Sensitivity, and Specificity."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if healthy_class_name not in class_names:
        raise ValueError(f"Healthy class '{healthy_class_name}' not found in class_names: {class_names}")

    notumor_idx = class_names.index(healthy_class_name)
    tumor_indices = [i for i, c in enumerate(class_names) if c != healthy_class_name]

    is_tumor_true = np.isin(y_true, tumor_indices)
    is_tumor_pred = np.isin(y_pred, tumor_indices)

    true_positives = int(np.sum(is_tumor_true & is_tumor_pred))
    false_negatives = int(np.sum(is_tumor_true & ~is_tumor_pred))  # Missed tumor!
    false_positives = int(np.sum(~is_tumor_true & is_tumor_pred))
    true_negatives = int(np.sum(~is_tumor_true & ~is_tumor_pred))

    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0

    summary = {
        "total_test_cases": int(len(y_true)),
        "total_tumor_cases": int(np.sum(is_tumor_true)),
        "total_healthy_cases": int(np.sum(~is_tumor_true)),
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "tumor_sensitivity": float(sensitivity),
        "tumor_specificity": float(specificity),
    }

    print("=" * 60)
    print("CLINICAL DIAGNOSTIC SAFETY SUMMARY")
    print("=" * 60)
    print(f"Total Test Cases:            {summary['total_test_cases']}")
    print(f"Tumor Cases (True Positives): {summary['true_positives']}")
    print(f"Healthy Controls (No Tumor): {summary['true_negatives']}")
    print(f"False Negatives (Missed):    {summary['false_negatives']}  <-- Critically important")
    print(f"False Positives:             {summary['false_positives']}")
    print(f"Overall Tumor Sensitivity:   {summary['tumor_sensitivity'] * 100:.2f}%")
    print(f"Overall Tumor Specificity:   {summary['tumor_specificity'] * 100:.2f}%")
    print("=" * 60)

    return summary


def evaluate_model_on_test(model, test_ds, class_names, save_dir=None):
    """Run complete diagnostic evaluation on test dataset: loss, accuracy, report, and confusion matrix."""
    print("Running inference on test dataset...")
    y_true, y_pred, y_probs = [], [], []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        if isinstance(preds, list):
            preds = preds[0]  # Take main output if auxiliary classifiers present
        y_probs.extend(preds)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    evaluation = model.evaluate(test_ds, verbose=0)
    test_loss = float(evaluation[0] if isinstance(evaluation, list) else evaluation)
    test_acc = float(evaluation[1] if isinstance(evaluation, list) else accuracy_score(y_true, y_pred))

    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

    cm_path = Path(save_dir) / "confusion_matrix.png" if save_dir else None
    fig, cm = plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path)
    plt.show()

    safety_summary = compute_clinical_safety_metrics(y_true, y_pred, class_names)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        metrics_summary = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
            "clinical_safety": safety_summary,
            "class_names": class_names,
        }
        metrics_path = save_dir / "evaluation_results.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"Evaluation metrics saved to: {metrics_path}")

    return {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "clinical_safety": safety_summary,
    }


def evaluate_saved_model(checkpoint_path: Path, test_ds, class_names, save_dir=None):
    """Load a saved Keras model checkpoint and evaluate it on test dataset."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    print(f"Loading checkpoint: {checkpoint_path}")
    model = tf.keras.models.load_model(str(checkpoint_path))
    return evaluate_model_on_test(model, test_ds, class_names, save_dir=save_dir)
