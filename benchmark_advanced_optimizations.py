"""
Comprehensive Benchmark: Advanced Optimizations in Constructive Learning
========================================================================
Compares:
1. Standard Constructive Wrapper (Baseline)
2. Selective Gradient Backprop (Neuron Responsibility Gating, ratio=0.80)
3. Loss-Gated Selective Sampling (Sample-level early exit, threshold=0.95)
4. Adaptive Curvature Early Stopping (Dynamic loss delta plateau detection)
5. Combined Synergistic Suite (Selective Grad + Loss Gated + Curvature Early Stop)

Generates empirical metrics and visualization plot: plots/benchmark_selective_optimizations.png
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from constructive_torch import ConstructiveWrapper
from benchmark_four_datasets import get_fashion_mnist_dataset

def build_conv_network():
    s1 = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    s2 = nn.Sequential(
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten()
    )
    s3 = nn.Sequential(
        nn.Linear(64 * 4 * 4, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return [s1, s2, s3]

def run_benchmarks():
    print("=" * 90)
    print("🚀 EVALUATING ADVANCED OPTIMIZATION TECHNIQUES IN CONSTRUCTIVE LEARNING")
    print("=" * 90)

    # 1. Load data
    x_tr, x_te, y_tr, y_te = get_fashion_mnist_dataset(n_train=6000, n_test=1500)
    print(f"Dataset: Fashion-MNIST | Train: {len(x_tr)} samples | Test: {len(x_te)} samples")

    configurations = [
        {
            "name": "1. Standard Baseline",
            "short": "Baseline",
            "selective_grad": False,
            "selective_ratio": 1.0,
            "loss_gated": False,
            "loss_thresh": 0.95,
            "early_stopping_curvature": False,
            "color": "#64748b"
        },
        {
            "name": "2. Selective Gradient (Ratio=0.80)",
            "short": "Selective Grad",
            "selective_grad": True,
            "selective_ratio": 0.80,
            "loss_gated": False,
            "loss_thresh": 0.95,
            "early_stopping_curvature": False,
            "color": "#3b82f6"
        },
        {
            "name": "3. Loss-Gated Sampling (Tau=0.95)",
            "short": "Loss-Gated",
            "selective_grad": False,
            "selective_ratio": 1.0,
            "loss_gated": True,
            "loss_thresh": 0.95,
            "early_stopping_curvature": False,
            "color": "#10b981"
        },
        {
            "name": "4. Curvature Early Stop (Delta=0.002)",
            "short": "Curvature Stop",
            "selective_grad": False,
            "selective_ratio": 1.0,
            "loss_gated": False,
            "loss_thresh": 0.95,
            "early_stopping_curvature": True,
            "color": "#f59e0b"
        },
        {
            "name": "5. Combined Synergistic Suite",
            "short": "Combined Suite",
            "selective_grad": True,
            "selective_ratio": 0.80,
            "loss_gated": True,
            "loss_thresh": 0.95,
            "early_stopping_curvature": True,
            "color": "#8b5cf6"
        },
    ]

    results = []

    for cfg in configurations:
        print(f"\n▶ Running: {cfg['name']}...")
        torch.manual_seed(42)
        np.random.seed(42)

        stages = build_conv_network()
        wrapper = ConstructiveWrapper(
            stages=stages,
            in_shape=(1, 28, 28),
            num_classes=10,
            batch_size=128,
            learning_rate=0.003,
            max_epochs_per_stage=4,
            fine_tune_epochs=2,
            fine_tune_lr=0.001,
            auto_tune_hyperparams=False,
            use_ema=False,
            selective_gradient=cfg["selective_grad"],
            selective_gradient_ratio=cfg["selective_ratio"],
            loss_gated_sampling=cfg["loss_gated"],
            loss_gated_threshold=cfg["loss_thresh"],
            early_stopping_curvature=cfg["early_stopping_curvature"],
            early_stopping_curvature_delta=0.002,
            random_state=42,
        )

        t0 = time.time()
        wrapper.fit(x_tr, y_tr, verbose=False)
        train_time = time.time() - t0

        t_inf0 = time.time()
        preds = wrapper.predict(x_te)
        inf_time = time.time() - t_inf0

        acc = accuracy_score(y_te, preds) * 100
        f1 = f1_score(y_te, preds, average="macro")
        throughput = len(x_tr) / max(0.001, train_time)

        # Count total training epochs across all stages
        total_epochs = sum(len(h.get("loss", [])) for h in wrapper.training_history)

        res = {
            "name": cfg["name"],
            "short": cfg["short"],
            "color": cfg["color"],
            "accuracy": acc,
            "f1": f1,
            "train_time": train_time,
            "inf_time": inf_time,
            "throughput": throughput,
            "total_epochs": total_epochs,
        }
        results.append(res)
        print(f"   ✓ Accuracy: {acc:.2f}% | Macro F1: {f1:.4f} | Time: {train_time:.2f}s | Throughput: {throughput:.0f} spl/s | Total Epochs: {total_epochs}")

    # Calculate relative speedups and accuracy differences
    base_time = results[0]["train_time"]
    base_acc = results[0]["accuracy"]

    for r in results:
        r["speedup"] = (base_time / r["train_time"])
        r["time_reduction"] = ((base_time - r["train_time"]) / base_time) * 100
        r["acc_gain"] = r["accuracy"] - base_acc

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Configuration':<35} | {'Acc (%)':<8} | {'F1':<7} | {'Time (s)':<9} | {'Speedup':<8} | {'Epochs':<6}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<35} | {r['accuracy']:<8.2f} | {r['f1']:<7.4f} | {r['train_time']:<9.2f} | {r['speedup']:<8.2f}x | {r['total_epochs']:<6d}")
    print("=" * 90)

    # Plot results
    plot_results(results)
    return results

def plot_results(results):
    os.makedirs("plots", exist_ok=True)
    out_path = "plots/benchmark_selective_optimizations.png"

    names = [r["short"] for r in results]
    colors = [r["color"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["f1"] * 100 for r in results]
    times = [r["train_time"] for r in results]
    speedups = [r["speedup"] for r in results]
    epochs = [r["total_epochs"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0f172a")

    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Panel 1: Test Accuracy and Macro F1
    ax1 = axes[0, 0]
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax1.bar(x - width/2, accs, width, label="Test Accuracy (%)", color="#3b82f6", alpha=0.9, edgecolor="#60a5fa")
    bars2 = ax1.bar(x + width/2, f1s, width, label="Macro F1 (x100)", color="#10b981", alpha=0.9, edgecolor="#34d399")
    ax1.set_title("Classification Accuracy & Macro F1 Score", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right", color="#cbd5e1")
    ax1.set_ylim(min(accs) - 2.0, max(accs) + 2.0)
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.2, color="#94a3b8")
    for b in bars1:
        ax1.annotate(f"{b.get_height():.2f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", color="#93c5fd", fontsize=8, fontweight="bold")

    # Panel 2: Total Training Time (Wall Clock)
    ax2 = axes[0, 1]
    bars_time = ax2.bar(names, times, color=colors, alpha=0.85, edgecolor="#94a3b8", width=0.55)
    ax2.set_title("Wall-Clock Training Time (seconds)", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticklabels(names, rotation=15, ha="right", color="#cbd5e1")
    ax2.set_ylabel("Seconds (Lower is Better)", color="#cbd5e1")
    ax2.grid(axis="y", linestyle="--", alpha=0.2, color="#94a3b8")
    for b in bars_time:
        ax2.annotate(f"{b.get_height():.2f}s", (b.get_x() + b.get_width() / 2, b.get_height() + 0.3),
                     ha="center", va="bottom", color="#f8fafc", fontsize=9, fontweight="bold")

    # Panel 3: Relative Speedup (Multiplier)
    ax3 = axes[1, 0]
    bars_sp = ax3.bar(names, speedups, color="#8b5cf6", alpha=0.85, edgecolor="#c084fc", width=0.55)
    ax3.axhline(1.0, color="#ef4444", linestyle="--", label="Baseline (1.0x)")
    ax3.set_title("Training Speedup Multiplier vs. Baseline", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xticklabels(names, rotation=15, ha="right", color="#cbd5e1")
    ax3.set_ylabel("Speedup Multiplier (Higher is Better)", color="#cbd5e1")
    ax3.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="upper left")
    ax3.grid(axis="y", linestyle="--", alpha=0.2, color="#94a3b8")
    for b in bars_sp:
        ax3.annotate(f"{b.get_height():.2f}x", (b.get_x() + b.get_width() / 2, b.get_height() + 0.03),
                     ha="center", va="bottom", color="#e9d5ff", fontsize=9, fontweight="bold")

    # Panel 4: Epoch Convergence Efficiency
    ax4 = axes[1, 1]
    bars_ep = ax4.bar(names, epochs, color="#f59e0b", alpha=0.85, edgecolor="#fbbf24", width=0.55)
    ax4.set_title("Total Epochs Trained Across Stages", color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xticklabels(names, rotation=15, ha="right", color="#cbd5e1")
    ax4.set_ylabel("Epochs (Adaptive Saturation)", color="#cbd5e1")
    ax4.grid(axis="y", linestyle="--", alpha=0.2, color="#94a3b8")
    for b in bars_ep:
        ax4.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height() + 0.2),
                     ha="center", va="bottom", color="#fde68a", fontsize=9, fontweight="bold")

    plt.suptitle("Advanced Optimization Techniques in Constructive Deep Learning\nNeuron Responsibility Gating, Loss Gating & Curvature Early Stopping",
                 color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n📊 Benchmark chart successfully saved to: {out_path}")

if __name__ == "__main__":
    run_benchmarks()
