"""
Experimental Validation of Further Optimization Techniques
==========================================================
Tests:
  1. Selective Gradient Backpropagation (Neuron/Filter Responsibility Gating)
  2. Loss-Gated Sample-Level Selective Backpropagation (Hard Example Mining)
  3. Dynamic Curvature Early Stopping (Stage-level saturation gating)
  4. Lookahead Meta-Optimizer (Slow-fast weight synchronization)
"""

import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmark_four_datasets import get_fashion_mnist_dataset


def apply_selective_gradient_mask(parameters, ratio: float = 0.75):
    """
    Selective Gradient Masking:
    Identifies neuron/filter output slices and zeros out gradients for the
    bottom (1 - ratio) neurons with the lowest gradient norm.
    Only the top `ratio` neurons responsible for the prediction are updated!
    """
    with torch.no_grad():
        for p in parameters:
            if p.grad is None or p.ndim < 2:
                continue
            # p.shape[0] is the number of output neurons/filters
            out_dim = p.shape[0]
            if out_dim <= 4:
                continue

            # Compute L2 norm of gradient per output neuron/channel
            flat_grads = p.grad.view(out_dim, -1)
            norms = torch.norm(flat_grads, p=2, dim=1)  # shape: (out_dim,)

            # Determine top-k threshold
            k = max(1, int(round(out_dim * ratio)))
            if k >= out_dim:
                continue

            topk_vals, _ = torch.topk(norms, k)
            threshold = topk_vals[-1]

            # Mask out neurons with gradient norm strictly below threshold
            mask = (norms >= threshold).view(out_dim, *([1] * (p.ndim - 1)))
            p.grad.mul_(mask.float())


def test_selective_gradient_on_stage():
    print("=" * 70)
    print("🔬 EXPERIMENT 1: SELECTIVE GRADIENT (NEURON RESPONSIBILITY GATING)")
    print("=" * 70)

    x_tr, x_te, y_tr, y_te = get_fashion_mnist_dataset(n_train=4000, n_test=1000)
    tx = torch.from_numpy(x_tr).float()
    ty = torch.from_numpy(y_tr).long()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx, ty), batch_size=128, shuffle=True)

    ratios = [1.0, 0.85, 0.75, 0.60]  # 1.0 = standard all neurons updated

    for r in ratios:
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        criterion = nn.CrossEntropyLoss()

        t0 = time.time()
        for ep in range(5):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()

                if r < 1.0:
                    apply_selective_gradient_mask(model.parameters(), ratio=r)

                optimizer.step()

        elapsed = time.time() - t0
        model.eval()
        with torch.inference_mode():
            preds = torch.argmax(model(torch.from_numpy(x_te).float()), dim=1).numpy()
        acc = accuracy_score(y_te, preds) * 100
        f1 = f1_score(y_te, preds, average="macro")
        tag = "Standard (100% Neurons)" if r == 1.0 else f"Selective Gradient ({int(r*100)}% Neurons Updated)"
        print(f"   [{tag:<38}] Test Acc: {acc:.2f}% | Macro F1: {f1:.4f} | Time: {elapsed:.2f}s")


def test_loss_gated_sample_backprop():
    print("\n" + "=" * 70)
    print("🔬 EXPERIMENT 2: LOSS-GATED SELECTIVE SAMPLE BACKPROPAGATION")
    print("=" * 70)

    x_tr, x_te, y_tr, y_te = get_fashion_mnist_dataset(n_train=4000, n_test=1000)
    tx = torch.from_numpy(x_tr).float()
    ty = torch.from_numpy(y_tr).long()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx, ty), batch_size=128, shuffle=True)

    confidence_thresholds = [1.0, 0.95, 0.90, 0.85]  # 1.0 = all samples backpropagated

    for tau in confidence_thresholds:
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        criterion = nn.CrossEntropyLoss(reduction="none")

        t0 = time.time()
        skipped_samples = 0
        total_samples = 0

        for ep in range(5):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                raw_losses = criterion(out, by)

                if tau < 1.0:
                    with torch.no_grad():
                        probs = torch.softmax(out, dim=1)
                        conf = probs.gather(1, by.unsqueeze(1)).squeeze(1)
                        mask = (conf < tau).float()
                        skipped_samples += int((conf >= tau).sum().item())
                        total_samples += len(by)

                    active_loss = (raw_losses * mask).sum() / max(1.0, mask.sum().item())
                    if mask.sum() > 0:
                        active_loss.backward()
                        optimizer.step()
                else:
                    raw_losses.mean().backward()
                    optimizer.step()
                    total_samples += len(by)

        elapsed = time.time() - t0
        model.eval()
        with torch.inference_mode():
            preds = torch.argmax(model(torch.from_numpy(x_te).float()), dim=1).numpy()
        acc = accuracy_score(y_te, preds) * 100
        f1 = f1_score(y_te, preds, average="macro")
        skip_pct = (skipped_samples / max(1, total_samples)) * 100
        tag = "Standard (All Samples)" if tau == 1.0 else f"Loss-Gated (Tau={tau}, Skipped {skip_pct:.1f}%)"
        print(f"   [{tag:<46}] Test Acc: {acc:.2f}% | Macro F1: {f1:.4f} | Time: {elapsed:.2f}s")


if __name__ == "__main__":
    test_selective_gradient_on_stage()
    test_loss_gated_sample_backprop()
