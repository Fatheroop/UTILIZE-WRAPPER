"""
Training Script: Moon Dataset with Constructive Layer-Wise Model
Follows ML best practices:
  - Generates synthetic Moon classification dataset.
  - Strict featurization ordering (split train/test BEFORE standard scaling).
  - Trains layer-by-layer until accuracy saturates per layer.
  - Generates comprehensive plots and saves checkpoint.
"""

import os
import argparse
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_models.core_model import ConstructiveLayerwiseModel


def plot_decision_boundary(
    model: ConstructiveLayerwiseModel,
    X: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    title: str,
    up_to_layer: int,
):
    """Plot the decision boundary of the constructive model up to a specific layer."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid_points, up_to_layer=up_to_layer)[:, 1]
    Z = probs.reshape(xx.shape)

    contour = ax.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.75)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5, linestyles="--")

    # Scatter points
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolors="k",
        s=30,
        alpha=0.9,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Feature 1 (Standardized)")
    ax.set_ylabel("Feature 2 (Standardized)")
    return contour


def main():
    parser = argparse.ArgumentParser(description="Train Constructive Layer-Wise Model on Moons Dataset")
    parser.add_argument("--samples", type=int, default=1200, help="Number of samples in moons dataset")
    parser.add_argument("--noise", type=float, default=0.10, help="Gaussian noise standard deviation")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden units per constructive layer")
    parser.add_argument("--max_layers", type=int, default=4, help="Maximum number of constructive layers")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate for layer training")
    parser.add_argument("--patience", type=int, default=12, help="Saturation patience epochs")
    parser.add_argument("--epochs", type=int, default=250, help="Max epochs per layer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save generated plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n========================================================")
    print("🌕 Generating Synthetic Moon Dataset")
    print("========================================================")
    X_raw, y_raw = make_moons(n_samples=args.samples, noise=args.noise, random_state=args.seed)

    # ML Best Practice: Strict featurization ordering (split BEFORE preprocessing)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.25, random_state=args.seed, stratify=y_raw
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"Dataset generated: Total={args.samples} (Train={len(X_train)}, Test={len(X_test)})")
    print(f"Features: 2 | Classes: 2 | Noise: {args.noise}")

    # Build constructive model
    model = ConstructiveLayerwiseModel(
        input_dim=2,
        num_classes=2,
        hidden_dim=args.hidden_dim,
        max_layers=args.max_layers,
        activation="tanh",
        learning_rate=args.lr,
        weight_decay=1e-4,
        patience=args.patience,
        min_delta=1e-3,
        max_epochs_per_layer=args.epochs,
        target_accuracy=0.99,
        include_input_skips=True,
        random_state=args.seed,
    )

    # Train model following layerwise rule
    model.fit(
        X=X_train,
        y=y_train,
        X_val=X_test,
        y_val=y_test,
        verbose=True,
    )

    # Save model checkpoint
    checkpoint_path = "constructive_model.pt"
    model.save(checkpoint_path)
    print(f"\nSaved model checkpoint to: {checkpoint_path}")

    # Evaluate layer-by-layer progression
    print("\n========================================================")
    print("📊 Constructive Layer-by-Layer Progression Table:")
    print("========================================================")
    print(f"{'Layer':<8} | {'Saturated Epoch':<18} | {'Train Acc (%)':<15} | {'Test Acc (%)':<15}")
    print("-" * 65)

    layer_train_accs = []
    layer_test_accs = []
    num_layers = len(model.layers)

    for l_idx in range(num_layers):
        tr_acc = model.evaluate(X_train, y_train, up_to_layer=l_idx)["accuracy"] * 100
        te_acc = model.evaluate(X_test, y_test, up_to_layer=l_idx)["accuracy"] * 100
        sat_ep = model.training_history[l_idx]["saturated_epoch"]
        layer_train_accs.append(tr_acc)
        layer_test_accs.append(te_acc)
        print(f"Layer {l_idx + 1:<2} | Epoch {sat_ep:<12} | {tr_acc:>12.2f}% | {te_acc:>12.2f}%")

    print("-" * 65)

    # Final overall evaluation
    y_pred = model.predict(X_test)
    final_test_acc = accuracy_score(y_test, y_pred)
    final_test_f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    print("\n========================================================")
    print("🏆 Final Evaluation on Independent Test Set:")
    print("========================================================")
    print(f"Test Accuracy: {final_test_acc * 100:.2f}%")
    print(f"Test F1-Score: {final_test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Visualization 1: Decision Boundary Progression
    cols = min(num_layers + 1, 4)
    rows = (num_layers + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    # Subplot 0: Raw Training Data
    axes[0].scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="coolwarm",
        edgecolors="k",
        s=30,
        alpha=0.8,
    )
    axes[0].set_title("Ground Truth Training Data", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Feature 1 (Standardized)")
    axes[0].set_ylabel("Feature 2 (Standardized)")

    # Subplots 1..N: Decision Boundaries per Layer
    for l_idx in range(num_layers):
        ax = axes[l_idx + 1]
        sat_epoch = model.training_history[l_idx]["saturated_epoch"]
        t_acc = layer_test_accs[l_idx]
        title = f"Layer {l_idx + 1} Boundary (Sat @ ep {sat_epoch})\nTest Acc: {t_acc:.1f}%"
        plot_decision_boundary(model, X_train, y_train, ax, title, up_to_layer=l_idx)

    # Hide unused subplots
    for ax_idx in range(num_layers + 1, len(axes)):
        axes[ax_idx].axis("off")

    plt.tight_layout()
    boundaries_path = os.path.join(args.output_dir, "decision_boundaries_evolution.png")
    plt.savefig(boundaries_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {boundaries_path}")

    # Visualization 2: Training Dynamics & Metrics
    fig, (ax_loss, ax_acc, ax_cm) = plt.subplots(1, 3, figsize=(18, 5))

    # Loss Curves
    for l_idx, hist in enumerate(model.training_history):
        epochs = list(range(1, len(hist["loss"]) + 1))
        ax_loss.plot(epochs, hist["loss"], label=f"Layer {l_idx + 1} Loss", linewidth=2)
        sat = hist["saturated_epoch"]
        if sat is not None and sat <= len(hist["loss"]):
            ax_loss.scatter([sat], [hist["loss"][sat - 1]], color="red", zorder=5)
            ax_loss.annotate(f"Sat L{l_idx+1}", (sat, hist["loss"][sat - 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    ax_loss.set_title("Layer-Wise Loss Curves (with Saturation Point)", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch within Layer")
    ax_loss.set_ylabel("Cross Entropy Loss")
    ax_loss.legend()
    ax_loss.grid(True, linestyle="--", alpha=0.5)

    # Accuracy Progression
    layer_labels = [f"Layer {i+1}" for i in range(num_layers)]
    ax_acc.plot(layer_labels, layer_train_accs, marker="o", linewidth=2.5, label="Train Accuracy")
    ax_acc.plot(layer_labels, layer_test_accs, marker="s", linewidth=2.5, linestyle="--", label="Test Accuracy")
    ax_acc.set_title("Accuracy Progression Across Layers", fontsize=12, fontweight="bold")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim([80, 102])
    ax_acc.legend()
    ax_acc.grid(True, linestyle="--", alpha=0.5)

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Moon 0", "Moon 1"])
    disp.plot(ax=ax_cm, cmap="Blues", values_format="d")
    ax_cm.set_title(f"Test Confusion Matrix\n(Accuracy: {final_test_acc*100:.1f}%)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    metrics_path = os.path.join(args.output_dir, "training_saturation_and_metrics.png")
    plt.savefig(metrics_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {metrics_path}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
