"""
Standard End-to-End Multi-Layer Perceptron (MLP) Baseline
Trains standard multi-layer fully-connected architecture end-to-end via conventional backpropagation.
Used as the benchmark baseline against Constructive Layer-Wise Models for vector/tabular tasks.
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _get_activation(act_name: str) -> nn.Module:
    name = act_name.lower()
    if name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    elif name == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation: {act_name}")


class StandardMLP(nn.Module):
    """
    Standard Multi-Layer Perceptron (MLP) trained end-to-end.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        num_classes: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        layers = []
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(_get_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(_get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.classifier(h)


class StandardMLPTrainer:
    """
    Trainer for Standard MLP with early stopping, resource evaluation, and prediction.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        num_classes: int = 2,
        activation: str = "relu",
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        max_epochs: int = 300,
        patience: int = 20,
        min_delta: float = 1e-3,
        target_accuracy: float = 0.999,
        device: Optional[str] = None,
        random_state: Optional[int] = 42,
    ):
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = StandardMLP(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            activation=activation,
        ).to(self.device)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.target_accuracy = target_accuracy

        self.training_history: Dict[str, Any] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "StandardMLPTrainer":
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.long, device=self.device)

        X_val_t = (
            torch.tensor(X_val, dtype=torch.float32, device=self.device)
            if X_val is not None
            else None
        )
        y_val_t = (
            torch.tensor(y_val, dtype=torch.long, device=self.device)
            if y_val is not None
            else None
        )

        best_acc = 0.0
        best_loss = float("inf")
        epochs_no_improve = 0
        best_weights = None

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            logits = self.model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

            # Train metrics
            self.model.eval()
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                train_acc = (preds == y_t).float().mean().item()
                train_loss = loss.item()

            self.training_history["loss"].append(train_loss)
            self.training_history["accuracy"].append(train_acc)

            # Validation metrics
            if X_val_t is not None and y_val_t is not None:
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                    val_preds = torch.argmax(val_logits, dim=1)
                    val_acc = (val_preds == y_val_t).float().mean().item()
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_accuracy"].append(val_acc)
            else:
                val_acc = train_acc
                val_loss = train_loss

            # Check improvements
            improvement = (train_acc - best_acc) >= self.min_delta
            if train_acc > best_acc or (train_acc == best_acc and train_loss < best_loss):
                best_acc = max(best_acc, train_acc)
                best_loss = min(best_loss, train_loss)
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if improvement:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                epochs_no_improve += 1

            if verbose and (epoch % 20 == 0 or epoch == 1 or train_acc >= self.target_accuracy):
                val_str = f" | Val Acc: {val_acc:.4f}" if X_val_t is not None else ""
                print(f"  [Standard MLP] Epoch {epoch:3d}/{self.max_epochs} - Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}{val_str}")

            if train_acc >= self.target_accuracy:
                if verbose:
                    print(f"  [Standard MLP] Target accuracy {self.target_accuracy*100:.1f}% reached at epoch {epoch}!")
                break

            if epochs_no_improve >= self.patience:
                if verbose:
                    print(f"  [Standard MLP] Saturated at epoch {epoch}.")
                break

        if best_weights is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        acc = float(np.mean(preds == y))
        return {"accuracy": acc}
