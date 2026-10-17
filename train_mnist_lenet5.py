"""
Training Script: Constructive LeNet-5 on MNIST Dataset with In-Memory Features and Prefetching
Executes constructive layer-wise training on MNIST with dynamic stage resolution,
in-memory feature map caching, and asynchronous stage prefetching for prediction.
"""

import os
import argparse
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from mnist_dataset import get_normalized_mnist
from lenet5_disk_model import DiskConstructiveLeNet5


def main():
    parser = argparse.ArgumentParser(
        description="Train Constructive LeNet-5 on MNIST with Minimal Memory / Disk Offloading"
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=20000,
        help="Number of MNIST training samples (up to 60,000)",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=5000,
        help="Number of MNIST testing samples (up to 10,000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Maximum epochs per constructive stage",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Patience epochs before declaring accuracy saturated",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.003,
        help="Learning rate for stage optimization",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Mini-batch size for training and streaming",
    )
    parser.add_argument(
        "--storage_dir",
        type=str,
        default="disk_lenet5",
        help="Disk directory for checkpoints and model metadata",
    )
    parser.add_argument(
        "--layers_per_stage",
        type=int,
        default=3,
        help="Number of computational layers trained jointly per constructive stage",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save generated metric plots",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n========================================================")
    print("📦 Loading and Normalizing MNIST Dataset")
    print("========================================================")
    train_images, train_labels, test_images, test_labels = get_normalized_mnist(
        num_train=args.num_train,
        num_test=args.num_test,
    )

    print(f"Train split: {train_images.shape} (Range: [{train_images.min():.2f}, {train_images.max():.2f}])")
    print(f"Test split:  {test_images.shape} (Range: [{test_images.min():.2f}, {test_images.max():.2f}])")
    print(f"Unique digit classes: {np.unique(train_labels)}")

    # Reserve 10% of training data as validation split to monitor saturation
    val_split_size = int(len(train_images) * 0.1)
    val_images = train_images[:val_split_size]
    val_labels = train_labels[:val_split_size]
    sub_train_images = train_images[val_split_size:]
    sub_train_labels = train_labels[val_split_size:]

    print(f"Sub-train: {len(sub_train_images)} samples | Validation: {len(val_images)} samples")

    # Instantiate Disk-Backed Constructive LeNet-5 with dynamically inferred input shape
    model = DiskConstructiveLeNet5(
        in_shape=tuple(train_images.shape[1:]),
        num_classes=len(np.unique(train_labels)),
        activation="tanh",
        learning_rate=args.lr,
        weight_decay=1e-4,
        batch_size=args.batch_size,
        layers_per_stage=args.layers_per_stage,
        patience=args.patience,
        min_delta=0.002,
        max_epochs_per_stage=args.epochs,
        target_accuracy=0.995,
        storage_dir=args.storage_dir,
        random_state=42,
    )

    # Train layer-by-layer following the constructive rule with disk offloading
    model.fit(
        train_images=sub_train_images,
        train_labels=sub_train_labels,
        val_images=val_images,
        val_labels=val_labels,
        verbose=True,
    )

    # Sequential disk-streaming evaluation on the independent test set
    print("\n========================================================")
    print("📊 Evaluating Stage Progression on Test Set:")
    print("========================================================")
    stage_test_accuracies = []

    for s_idx in range(model.num_stages):
        stage_name = model.get_stage_name(s_idx)
        test_acc = model.evaluate(test_images, test_labels, up_to_stage=s_idx)
        stage_test_accuracies.append(test_acc)
        sat_epoch = model.training_history[s_idx]["saturated_epoch"]
        print(f"After {stage_name:<42} (Sat @ Ep {sat_epoch:2d}) -> Test Acc: {test_acc * 100:6.2f}%")

    # Final overall prediction on test set using asynchronous prefetching
    test_preds = model.predict(test_images, use_prefetching=True)
    final_test_acc = float(np.mean(test_preds == test_labels))

    print("\n========================================================")
    print(f"🏆 Final LeNet-5 Test Accuracy: {final_test_acc * 100:.2f}%")
    print("========================================================")
    print("\nClassification Report across all Digits:")
    print(classification_report(test_labels, test_preds, digits=4))

    # Generate Comprehensive Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Stage Accuracy Progression
    ax_acc = axes[0, 0]
    stage_short_labels = [f"Stage {i+1}\n({model.get_stage_name(i).split('(')[1].split(')')[0]})" for i in range(model.num_stages)]
    bars = ax_acc.bar(stage_short_labels, [a * 100 for a in stage_test_accuracies], color="#2b5c8f", width=0.55, edgecolor="black")
    ax_acc.set_title("Test Accuracy Progression Across LeNet-5 Stages", fontsize=12, fontweight="bold")
    ax_acc.set_ylabel("Test Accuracy (%)")
    ax_acc.set_ylim([70, 102])
    ax_acc.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax_acc.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # 2. Stage Loss Curves and Saturation Points
    ax_loss = axes[0, 1]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for s_idx, hist in enumerate(model.training_history):
        eps = list(range(1, len(hist["loss"]) + 1))
        col = palette[s_idx % len(palette)]
        ax_loss.plot(eps, hist["loss"], label=f"Stage {s_idx + 1} Loss", color=col, linewidth=2)
        sat_ep = hist["saturated_epoch"]
        if sat_ep is not None and sat_ep <= len(hist["loss"]):
            ax_loss.scatter([sat_ep], [hist["loss"][sat_ep - 1]], color="red", s=50, zorder=5)
            ax_loss.annotate(f"Sat S{s_idx+1}", (sat_ep, hist["loss"][sat_ep - 1]),
                             textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")

    ax_loss.set_title("Layer-Wise Loss Dynamics & Saturation Points", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch within Stage")
    ax_loss.set_ylabel("Cross Entropy Loss")
    ax_loss.legend()
    ax_loss.grid(True, linestyle="--", alpha=0.5)

    # 3. Confusion Matrix
    ax_cm = axes[1, 0]
    cm = confusion_matrix(test_labels, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(model.num_classes)))
    disp.plot(ax=ax_cm, cmap="Blues", values_format="d")
    ax_cm.set_title(f"Confusion Matrix (Final Test Acc: {final_test_acc * 100:.2f}%)", fontsize=12, fontweight="bold")

    # 4. Sample Test Predictions
    axes[1, 1].axis("off")
    sub_gs = axes[1, 1].inset_axes([0, 0, 1, 1])
    sub_gs.axis("off")

    sample_indices = np.random.RandomState(42).choice(len(test_images), 16, replace=False)
    for i, idx in enumerate(sample_indices):
        r, c = divmod(i, 4)
        sub_ax = sub_gs.inset_axes([c * 0.25, (3 - r) * 0.25, 0.22, 0.22])
        # De-normalize image for display
        img = (test_images[idx, 0] * 0.3081) + 0.1307
        true_lbl = test_labels[idx]
        pred_lbl = test_preds[idx]
        color = "green" if true_lbl == pred_lbl else "red"
        sub_ax.imshow(img, cmap="gray")
        sub_ax.set_title(f"T:{true_lbl} P:{pred_lbl}", color=color, fontsize=9, fontweight="bold")
        sub_ax.axis("off")

    axes[1, 1].set_title("Sample Predictions (Green: Correct, Red: Error)", fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "lenet5_mnist_results.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comprehensive LeNet-5 metric plot to: {plot_path}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
