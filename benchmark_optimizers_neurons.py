"""
Benchmark: Different Optimizers and Neurons vs. Auto-Constructive Model
======================================================================
Compares standard PyTorch models trained with different fixed optimizers
(SGD, Adam, AdamW, RMSprop) and different fixed neuron dimensions against
the AutoConstructiveModel which dynamically searches and selects the best
neurons, activation functions, and optimizers according to model accuracy.

Evaluates on two distinct benchmarks:
  1. Real-World Biomedical: Wisconsin Breast Cancer Diagnostic (30 continuous features, 2 classes)
  2. Multi-Class Non-Linear Benchmark: Synthetic non-linear dataset (2,000 samples, 20 features, 3 classes)
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from resource_monitor import ResourceTracker
from constructive_torch import AutoConstructiveModel, ConstructiveWrapper, ArchitecturalAutoTuner


class StandardPyTorchMLP(nn.Module):
    """Standard end-to-end PyTorch MLP."""
    def __init__(self, in_features: int, hidden_dims: list, num_classes: int, activation: str = "relu"):
        super().__init__()
        layers = []
        curr_in = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(curr_in, h))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            curr_in = h
        layers.append(nn.Linear(curr_in, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_standard_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    optimizer_name: str = "adam",
    learning_rate: float = 0.002,
    epochs: int = 25,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict:
    """Train standard PyTorch model and record metrics."""
    model = model.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    opt_name = optimizer_name.lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate * 5, momentum=0.9)
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    tracker = ResourceTracker(f"Standard_{optimizer_name}")
    with tracker:
        for epoch in range(epochs):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad(set_to_none=True)
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

    model.eval()
    with torch.no_grad():
        test_out = model(test_x)
        test_loss = criterion(test_out, test_y).item()
        test_preds = torch.argmax(test_out, dim=1).cpu().numpy()
        y_true = test_y.cpu().numpy()
        test_acc = accuracy_score(y_true, test_preds)
        test_f1 = f1_score(y_true, test_preds, average="macro")

    return {
        "test_acc": test_acc,
        "test_loss": test_loss,
        "macro_f1": test_f1,
        "duration_sec": tracker.total_time_sec,
        "peak_ram_mb": tracker.peak_ram_mb,
    }


def evaluate_dataset(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    standard_configs: list,
    candidate_neurons: list,
    candidate_activations: list,
    candidate_optimizers: list,
    max_stages: int = 2,
    epochs: int = 25,
) -> dict:
    in_features = X.shape[1]
    num_classes = len(np.unique(y))

    # Stratified Split (70% train, 15% val, 15% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, stratify=y_train_val, random_state=42
    )

    # Standardize using training statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    tx_train = torch.from_numpy(X_train).float()
    ty_train = torch.from_numpy(y_train).long()
    tx_val = torch.from_numpy(X_val).float()
    ty_val = torch.from_numpy(y_val).long()
    tx_test = torch.from_numpy(X_test).float()
    ty_test = torch.from_numpy(y_test).long()

    print("\n" + "=" * 80)
    print(f"📊 DATASET BENCHMARK: {dataset_name.upper()}")
    print(f"   Features: {in_features} | Classes: {num_classes} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print("=" * 80)

    dataset_results = {}

    # 1. Evaluate Standard Models
    for label, h_dims, act, opt in standard_configs:
        print(f"Evaluating: {label}...")
        torch.manual_seed(42)
        std_model = StandardPyTorchMLP(in_features, h_dims, num_classes, activation=act)
        res = train_standard_model(
            std_model,
            tx_train, ty_train,
            tx_val, ty_val,
            tx_test, ty_test,
            optimizer_name=opt,
            learning_rate=0.002,
            epochs=epochs,
            batch_size=64,
        )
        dataset_results[label] = {
            **res,
            "architecture": f"{h_dims}",
            "activation": act.upper(),
            "optimizer": opt.upper(),
            "is_auto": False,
        }
        print(f"   -> Test Acc: {res['test_acc']*100:.2f}% | Loss: {res['test_loss']:.4f} | Time: {res['duration_sec']:.3f}s | RAM: {res['peak_ram_mb']:.1f} MB")

    # 2. Evaluate Auto-Constructive Model
    print(f"\nEvaluating: AutoConstructiveModel (Dynamic Search on {dataset_name})...")
    torch.manual_seed(42)
    tracker_auto = ResourceTracker("AutoConstructiveModel")
    auto_model = AutoConstructiveModel(
        in_shape=in_features,
        num_classes=num_classes,
        max_stages=max_stages,
        candidate_neurons=candidate_neurons,
        candidate_activations=candidate_activations,
        candidate_optimizers=candidate_optimizers,
        max_epochs_per_stage=12,
        fine_tune_epochs=5,
        target_accuracy=1.0,  # Allow full multi-stage learning
        learning_rate=0.002,
        batch_size=64,
        label_smoothing=0.05,
        use_ema=True,
    )

    with tracker_auto:
        auto_model.fit(X_train, y_train, X_val, y_val, verbose=False)

    test_preds_auto = auto_model.predict(X_test)
    test_acc_auto = accuracy_score(y_test, test_preds_auto)
    test_f1_auto = f1_score(y_test, test_preds_auto, average="macro")

    # Export to standard PyTorch and verify parity
    exported_model = auto_model.get_standard_model()
    exported_model.eval()
    with torch.no_grad():
        exp_out = exported_model(tx_test)
        exp_preds = torch.argmax(exp_out, dim=1).numpy()
        criterion = nn.CrossEntropyLoss()
        test_loss_auto = criterion(exp_out, ty_test).item()

    assert np.array_equal(test_preds_auto, exp_preds), "Exported standard model predictions must match exactly!"

    summary = auto_model.get_architecture_summary()
    chosen_desc = " -> ".join([f"S{s['stage_idx']+1}({s['neurons']}n, {s['activation'].upper()}, {s['optimizer'].upper()})" for s in summary])

    dataset_results["Auto-Constructive (Dynamic Search)"] = {
        "test_acc": test_acc_auto,
        "test_loss": test_loss_auto,
        "macro_f1": test_f1_auto,
        "duration_sec": tracker_auto.total_time_sec,
        "peak_ram_mb": tracker_auto.peak_ram_mb,
        "architecture": chosen_desc,
        "activation": "Dynamic",
        "optimizer": "Dynamic",
        "is_auto": True,
        "summary": summary,
    }

    print(f"   -> Auto-Constructive Test Acc: {test_acc_auto*100:.2f}% | Loss: {test_loss_auto:.4f} | Time: {tracker_auto.total_time_sec:.3f}s | RAM: {tracker_auto.peak_ram_mb:.1f} MB")
    print(f"   -> Selected Architecture: {chosen_desc}")

    return dataset_results


def run_full_suite():
    standard_configs = [
        ("Standard PyTorch (SGD, [64, 32], ReLU)", [64, 32], "relu", "sgd"),
        ("Standard PyTorch (RMSprop, [64, 32], ReLU)", [64, 32], "relu", "rmsprop"),
        ("Standard PyTorch (Adam, [32, 16], ReLU)", [32, 16], "relu", "adam"),
        ("Standard PyTorch (Adam, [64, 32], ReLU)", [64, 32], "relu", "adam"),
        ("Standard PyTorch (Adam, [128, 64], ReLU)", [128, 64], "relu", "adam"),
        ("Standard PyTorch (AdamW, [64, 32], ReLU)", [64, 32], "relu", "adamw"),
    ]

    candidate_neurons = [16, 32, 64, 128]
    candidate_activations = ["relu", "gelu", "silu", "tanh"]
    candidate_optimizers = ["adamw", "adam", "sgd", "rmsprop"]

    # 1. Breast Cancer Dataset (Real Biomedical)
    bc_data = load_breast_cancer()
    results_bc = evaluate_dataset(
        "Wisconsin Breast Cancer (Binary Diagnosis)",
        bc_data.data, bc_data.target,
        standard_configs,
        candidate_neurons,
        candidate_activations,
        candidate_optimizers,
        max_stages=2,
        epochs=25,
    )

    # 2. Multi-Class Complex Non-Linear Dataset
    X_mc, y_mc = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=14,
        n_redundant=4,
        n_classes=3,
        n_clusters_per_class=2,
        flip_y=0.03,
        random_state=42,
    )
    results_mc = evaluate_dataset(
        "Complex Multi-Class (3-Class Non-Linear, 2k samples)",
        X_mc, y_mc,
        standard_configs,
        candidate_neurons,
        candidate_activations,
        candidate_optimizers,
        max_stages=2,
        epochs=25,
    )

    # 3. Generate Visual Plots
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    plt.subplots_adjust(hspace=0.38, wspace=0.25)

    short_labels = [
        "Std SGD\n[64,32]",
        "Std RMSprop\n[64,32]",
        "Std Adam\n[32,16]",
        "Std Adam\n[64,32]",
        "Std Adam\n[128,64]",
        "Std AdamW\n[64,32]",
        "Auto-Constructive\n(Dynamic Search)",
    ]

    # Panel 1: Breast Cancer Test Accuracy
    ax1 = axes[0, 0]
    labels_bc = list(results_bc.keys())
    accs_bc = [results_bc[m]["test_acc"] * 100.0 for m in labels_bc]
    colors_bc = ["#4A90E2" if not results_bc[m]["is_auto"] else "#E94E77" for m in labels_bc]
    bars1 = ax1.bar(short_labels, accs_bc, color=colors_bc, width=0.58, edgecolor="black", linewidth=1.2)
    ax1.set_title("Breast Cancer: Test Accuracy Comparison (%)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim([min(accs_bc) - 2.5, 101.0])
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # Panel 2: Multi-Class Non-Linear Test Accuracy
    ax2 = axes[0, 1]
    labels_mc = list(results_mc.keys())
    accs_mc = [results_mc[m]["test_acc"] * 100.0 for m in labels_mc]
    colors_mc = ["#4A90E2" if not results_mc[m]["is_auto"] else "#E94E77" for m in labels_mc]
    bars2 = ax2.bar(short_labels, accs_mc, color=colors_mc, width=0.58, edgecolor="black", linewidth=1.2)
    ax2.set_title("Multi-Class 3-Way: Test Accuracy Comparison (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim([min(accs_mc) - 3.0, max(accs_mc) + 2.5])
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # Panel 3: Training Time Comparison (Multi-Class)
    ax3 = axes[1, 0]
    times_mc = [results_mc[m]["duration_sec"] for m in labels_mc]
    bars3 = ax3.bar(short_labels, times_mc, color=colors_mc, width=0.58, edgecolor="black", linewidth=1.2)
    ax3.set_title("Multi-Class: Wall-Clock Training Time (s)", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Time (seconds)", fontsize=11)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05, f"{yval:.2f}s", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # Panel 4: Peak RAM Footprint
    ax4 = axes[1, 1]
    rams_mc = [results_mc[m]["peak_ram_mb"] for m in labels_mc]
    bars4 = ax4.bar(short_labels, rams_mc, color=colors_mc, width=0.58, edgecolor="black", linewidth=1.2)
    ax4.set_title("Peak RAM Footprint (RSS MB)", fontsize=13, fontweight="bold")
    ax4.set_ylabel("Resident Set Size (MB)", fontsize=11)
    ax4.set_ylim([min(rams_mc) - 20, max(rams_mc) + 30])
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f} MB", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plot_path = "plots/benchmark_optimizers_neurons.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n📊 4-Panel comparison chart saved to: {plot_path}")

    # 4. Print Summary Tables
    print("\n" + "=" * 95)
    print("📈 FINAL BENCHMARK SUMMARY TABLE: WISCONSIN BREAST CANCER")
    print("=" * 95)
    header = f"{'Model':<44} | {'Test Acc':<9} | {'Loss':<8} | {'Macro F1':<9} | {'Time (s)':<9} | {'RAM (MB)':<8}"
    print(header)
    print("-" * len(header))
    for name, res in results_bc.items():
        print(f"{name:<44} | {res['test_acc']*100:>7.2f}% | {res['test_loss']:>8.4f} | {res['macro_f1']*100:>7.2f}% | {res['duration_sec']:>8.3f}s | {res['peak_ram_mb']:>7.1f}M")
    print("=" * 95)

    print("\n" + "=" * 95)
    print("📈 FINAL BENCHMARK SUMMARY TABLE: COMPLEX MULTI-CLASS (3 CLASSES, 2,000 SAMPLES)")
    print("=" * 95)
    print(header)
    print("-" * len(header))
    for name, res in results_mc.items():
        print(f"{name:<44} | {res['test_acc']*100:>7.2f}% | {res['test_loss']:>8.4f} | {res['macro_f1']*100:>7.2f}% | {res['duration_sec']:>8.3f}s | {res['peak_ram_mb']:>7.1f}M")
    print("=" * 95)


if __name__ == "__main__":
    run_full_suite()
