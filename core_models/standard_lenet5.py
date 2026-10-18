"""
Standard End-to-End LeNet-5 Architecture (Standard Baseline)
Features:
  - Dynamically inferred flattened dimension (supports any resolution/channels).
  - Modern LeNet-5 defaults: ReLU activation and MaxPool2d (achieving >99% accuracy on MNIST).
  - Configurable pooling ('max' or 'avg') and activation ('relu', 'tanh', etc.).
  - Cosine annealing learning rate scheduler for smooth convergence.
  - Consistent post-epoch eval-mode metric tracking.
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def _get_activation(act_name: str) -> nn.Module:
    name = act_name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "gelu":
        return nn.GELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    elif name == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation: '{act_name}'")


def _get_pool2d(pool_type: str, kernel_size: int = 2, stride: int = 2) -> nn.Module:
    p = pool_type.lower()
    if p == "max":
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    elif p == "avg":
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    else:
        raise ValueError(f"Unsupported pooling: '{pool_type}'")


class StandardLeNet5(nn.Module):
    """
    Standard LeNet-5 model trained end-to-end via standard backpropagation.
    All layers remain in active memory throughout the entire training and inference lifecycle.
    """

    def __init__(
        self,
        in_shape: Tuple[int, int, int] = (1, 28, 28),
        num_classes: int = 10,
        c1_channels: int = 6,
        c2_channels: int = 16,
        c5_dim: int = 120,
        f6_dim: int = 84,
        activation: str = "relu",
        pool_type: str = "max",
    ):
        super().__init__()
        self.in_shape = in_shape
        self.num_classes = num_classes
        act = _get_activation(activation)
        pool1 = _get_pool2d(pool_type, kernel_size=2, stride=2)
        pool2 = _get_pool2d(pool_type, kernel_size=2, stride=2)

        padding = 2 if in_shape[1:] == (28, 28) else 0

        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], c1_channels, kernel_size=5, padding=padding),
            act,
            pool1,
            nn.Conv2d(c1_channels, c2_channels, kernel_size=5),
            act,
            pool2,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *in_shape)
            flattened_dim = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, c5_dim),
            act,
            nn.Linear(c5_dim, f6_dim),
            act,
            nn.Linear(f6_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits


class StandardLeNet5Trainer:
    """Trainer for the standard end-to-end LeNet-5 baseline."""

    def __init__(
        self,
        in_shape: Tuple[int, int, int] = (1, 28, 28),
        num_classes: int = 10,
        c1_channels: int = 6,
        c2_channels: int = 16,
        c5_dim: int = 120,
        f6_dim: int = 84,
        activation: str = "relu",
        pool_type: str = "max",
        learning_rate: float = 0.0015,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 15,
        patience: int = 5,
        min_delta: float = 0.0001,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.in_shape = in_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = StandardLeNet5(
            in_shape=in_shape,
            num_classes=num_classes,
            c1_channels=c1_channels,
            c2_channels=c2_channels,
            c5_dim=c5_dim,
            f6_dim=f6_dim,
            activation=activation,
            pool_type=pool_type,
        ).to(self.device)

        self.training_history: Dict[str, Any] = {
            "loss": [],
            "train_acc": [],
            "val_acc": [],
            "saturated_epoch": None,
        }

    def _evaluate_loader(self, loader: DataLoader) -> Tuple[float, float]:
        """Compute accurate eval-mode loss and accuracy across a dataloader."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                logits = self.model(bx)
                loss = criterion(logits, by)
                total_loss += loss.item() * len(bx)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == by).sum().item()
                total += len(bx)

        return (total_loss / total), (correct / total)

    def fit(
        self,
        train_images: Union[np.ndarray, torch.Tensor],
        train_labels: Union[np.ndarray, torch.Tensor],
        val_images: Optional[Union[np.ndarray, torch.Tensor]] = None,
        val_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        verbose: bool = True,
    ) -> "StandardLeNet5Trainer":
        """Train standard LeNet-5 with all parameters in memory."""
        train_x = torch.from_numpy(train_images).float() if isinstance(train_images, np.ndarray) else train_images.float()
        train_y = torch.from_numpy(train_labels).long().squeeze() if isinstance(train_labels, np.ndarray) else train_labels.long().squeeze()

        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        eval_train_loader = DataLoader(train_dataset, batch_size=self.batch_size * 2, shuffle=False)

        if val_images is not None and val_labels is not None:
            val_x = torch.from_numpy(val_images).float() if isinstance(val_images, np.ndarray) else val_images.float()
            val_y = torch.from_numpy(val_labels).long().squeeze() if isinstance(val_labels, np.ndarray) else val_labels.long().squeeze()
            val_dataset = TensorDataset(val_x, val_y)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False)
        else:
            val_loader = None

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        best_acc = 0.0
        epochs_no_improve = 0
        best_state = None

        if verbose:
            print("=" * 65)
            print("🌐 Standard LeNet-5 End-to-End Training Started")
            print(f"   Train samples: {len(train_x)} | Classes: {self.num_classes}")
            print(f"   Device: {self.device} | Mode: Full Backprop in RAM")
            print("=" * 65)

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Accurate eval-mode evaluation
            train_loss, epoch_train_acc = self._evaluate_loader(eval_train_loader)
            self.training_history["loss"].append(train_loss)
            self.training_history["train_acc"].append(epoch_train_acc)

            if val_loader is not None:
                _, epoch_val_acc = self._evaluate_loader(val_loader)
                self.training_history["val_acc"].append(epoch_val_acc)
                val_str = f" | Val Acc: {epoch_val_acc * 100:.2f}%"
                target_metric = epoch_val_acc
            else:
                val_str = ""
                target_metric = epoch_train_acc

            if verbose:
                print(
                    f"  [Standard LeNet-5] Epoch {epoch:2d}/{self.max_epochs} "
                    f"- Loss: {train_loss:.4f} - Train Acc: {epoch_train_acc * 100:.2f}%{val_str}"
                )

            # Plateau detection
            if target_metric > (best_acc + self.min_delta):
                best_acc = target_metric
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                if verbose:
                    print(f"  [Standard LeNet-5] Saturated (Plateaued for {self.patience} epochs) at epoch {epoch}.")
                self.training_history["saturated_epoch"] = epoch
                break

        if self.training_history["saturated_epoch"] is None:
            self.training_history["saturated_epoch"] = self.max_epochs

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return self

    def predict(self, test_images: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Standard model inference with all layers in RAM."""
        self.model.eval()
        test_x = torch.from_numpy(test_images).float() if isinstance(test_images, np.ndarray) else test_images.float()
        dataset = TensorDataset(test_x)
        loader = DataLoader(dataset, batch_size=max(self.batch_size * 2, 256), shuffle=False)

        all_preds = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                logits = self.model(bx)
                all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    def evaluate(self, test_images: Union[np.ndarray, torch.Tensor], test_labels: Union[np.ndarray, torch.Tensor]) -> float:
        preds = self.predict(test_images)
        labels_np = test_labels.cpu().numpy().squeeze() if isinstance(test_labels, torch.Tensor) else np.asarray(test_labels).squeeze()
        return float(np.mean(preds == labels_np))
