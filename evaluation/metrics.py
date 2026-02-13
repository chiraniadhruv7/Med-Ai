import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64


def compute_auc(y_true: np.ndarray, y_scores: np.ndarray, labels: list) -> dict:
    results = {}
    for i, label in enumerate(labels):
        try:
            auc = roc_auc_score(y_true[:, i], y_scores[:, i])
            results[label] = round(auc, 4)
        except ValueError:
            results[label] = None
    valid = [v for v in results.values() if v is not None]
    results["mean_auc"] = round(np.mean(valid), 4) if valid else None
    return results


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "label": label,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0,
    }


def plot_confusion_matrix_b64(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix: {label}", fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    tick_labels = ["Negative", "Positive"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def run_sample_evaluation():
    np.random.seed(42)
    labels = ["Pneumonia", "Effusion", "Cardiomegaly", "Atelectasis"]
    n_samples = 100

    y_true = np.random.randint(0, 2, size=(n_samples, len(labels)))
    y_scores = np.clip(y_true + np.random.normal(0, 0.3, size=(n_samples, len(labels))), 0, 1)

    auc_results = compute_auc(y_true, y_scores, labels)

    pneumonia_cm = compute_confusion_matrix(
        y_true[:, 0],
        (y_scores[:, 0] > 0.5).astype(int),
        "Pneumonia"
    )

    return {
        "auc_scores": auc_results,
        "sample_confusion_matrix": pneumonia_cm,
        "n_samples": n_samples,
        "labels_evaluated": labels,
    }
