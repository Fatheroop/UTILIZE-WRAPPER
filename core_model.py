"""
Core Model: Constructive Layer-Wise Neural Network

Implements the constructive layer-wise learning rule specified in readme.md:
  1. Create first layer in network.
  2. Pass all training data through first layer.
  3. Train first layer until accuracy is saturated.
  4. Store all outputs for every training data sample into a feature map.
  5. Instead of full end-to-end retraining, combine the frozen feature map and
     original training data to train the new layer weights until accuracy saturates again.
     Update the stored feature map with the new layer's outputs.
  6. Repeat for subsequent layers.
"""

from typing import List, Optional, Tuple, Dict, Any, Union
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


class SingleConstructiveLayer(nn.Module):
    """
    Constructive Computational Block for Tabular/Dense Networks.
    Supports single-layer or multi-layer computational blocks (e.g. 3 layers).
    Includes intra-block residual skip connections for deeper blocks to preserve
    gradient highways and accelerate convergence.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_layers: int = 1,
        num_classes: int = 2,
        activation: str = "tanh",
        dropout: float = 0.0,
        use_gating: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.activation = activation
        self.use_gating = use_gating

        # First linear layer maps from in_features to hidden_dim
        first_mods = [
            nn.Linear(in_features, hidden_dim),
            _get_activation(activation),
        ]
        if dropout > 0.0:
            first_mods.append(nn.Dropout(dropout))
        self.first_layer = nn.Sequential(*first_mods)

        # Subsequent intermediate layers with residual skip connections
        if num_layers > 1:
            res_blocks = []
            for _ in range(num_layers - 1):
                mods = [
                    nn.Linear(hidden_dim, hidden_dim),
                    _get_activation(activation),
                ]
                if dropout > 0.0:
                    mods.append(nn.Dropout(dropout))
                res_blocks.append(nn.Sequential(*mods))
            self.residual_layers = nn.ModuleList(res_blocks)
        else:
            self.residual_layers = nn.ModuleList()

        # RNN-style Highway / GRU Update Gate: z = sigmoid(W_z x + U_z h + b_z)
        if self.use_gating:
            self.skip_proj = (
                nn.Linear(in_features, hidden_dim, bias=False)
                if in_features != hidden_dim
                else nn.Identity()
            )
            self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
            # Initialize negative bias so gate starts near zero (identity pass-through)
            nn.init.constant_(self.gate.bias, -2.5)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through this constructive block with residual skip connections and optional RNN gating.
        Args:
            x: Input tensor [batch_size, in_features]
        Returns:
            feature_map: Hidden activations [batch_size, hidden_dim]
            logits: Classification logits [batch_size, num_classes]
        """
        h = self.first_layer(x)
        for layer in self.residual_layers:
            h = h + layer(h)
        if self.use_gating:
            base = self.skip_proj(x)
            z = torch.sigmoid(self.gate(torch.cat([base, h], dim=-1)))
            h = (1.0 - z) * base + z * h
        logits = self.classifier(h)
        return h, logits


class ConstructiveLayerwiseModel(nn.Module):
    """
    Constructive Layer-Wise Neural Network trained iteratively layer-by-layer.
    
    Earlier layers' weights are frozen once their accuracy saturates. Their output
    activations (feature maps) are cached and passed alongside the original input
    features to train the next layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dim: Union[int, List[int]] = 16,
        max_layers: int = 4,
        layers_per_stage: int = 1,
        activation: str = "tanh",
        learning_rate: float = 0.02,
        weight_decay: float = 1e-4,
        patience: int = 15,
        min_delta: float = 1e-3,
        max_epochs_per_layer: int = 300,
        target_accuracy: float = 0.995,
        eval_probe_batches: int = 4,
        max_retrains_per_layer: int = 1,
        include_input_skips: bool = True,
        use_gating: bool = False,
        batch_size: Optional[int] = None,
        candidate_hidden_dims: Optional[List[int]] = None,
        candidate_activations: Optional[List[str]] = None,
        candidate_optimizers: Optional[List[str]] = None,
        device: Optional[str] = None,
        random_state: Optional[int] = 42,
    ):
        super().__init__()
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.max_layers = max_layers
        self.layers_per_stage = layers_per_stage
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.max_epochs_per_layer = max_epochs_per_layer
        self.target_accuracy = target_accuracy
        self.eval_probe_batches = eval_probe_batches
        self.max_retrains_per_layer = max_retrains_per_layer
        self.include_input_skips = include_input_skips
        self.use_gating = use_gating
        self.batch_size = batch_size
        self.candidate_hidden_dims = candidate_hidden_dims
        self.candidate_activations = candidate_activations
        self.candidate_optimizers = candidate_optimizers
        self.selected_layer_hyperparameters: List[Dict[str, Any]] = []

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.layers: nn.ModuleList = nn.ModuleList()
        self.training_history: List[Dict[str, Any]] = []
        
        # Stored feature maps for training data (per readme.md step 4 & 5)
        self.train_feature_maps: List[np.ndarray] = []
        self.current_feature_map: Optional[torch.Tensor] = None

        self.to(self.device)

    def _get_hidden_dim(self, layer_idx: int) -> int:
        """Resolve hidden dimension for a specific layer."""
        if isinstance(self.hidden_dim, list):
            if layer_idx < len(self.hidden_dim):
                return self.hidden_dim[layer_idx]
            return self.hidden_dim[-1]
        return self.hidden_dim

    def _make_layer(self, layer_idx: int) -> SingleConstructiveLayer:
        """
        Construct layer k.
        - Layer 0 receives input_dim.
        - Subsequent layers receive:
            [original input, previous feature map] if include_input_skips is True
            [previous feature map] if include_input_skips is False
        """
        layer_hdim = self._get_hidden_dim(layer_idx)

        if layer_idx == 0:
            in_dim = self.input_dim
        else:
            prev_hdim = self._get_hidden_dim(layer_idx - 1)
            in_dim = (self.input_dim + prev_hdim) if self.include_input_skips else prev_hdim

        layer = SingleConstructiveLayer(
            in_features=in_dim,
            hidden_dim=layer_hdim,
            num_layers=self.layers_per_stage,
            num_classes=self.num_classes,
            activation=self.activation,
            use_gating=self.use_gating,
        )
        return layer.to(self.device)

    def forward(
        self, x: torch.Tensor, up_to_layer: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequential inference through frozen cascade of layers up to layer `up_to_layer`.
        Returns:
            (final_feature_map, final_logits)
        """
        if len(self.layers) == 0:
            raise RuntimeError("Model has no layers trained yet.")

        num_eval_layers = (
            len(self.layers)
            if up_to_layer is None
            else min(up_to_layer + 1, len(self.layers))
        )

        curr_x = x.to(self.device)
        feature_map = None
        logits = None

        for idx in range(num_eval_layers):
            layer = self.layers[idx]
            if idx == 0:
                layer_in = curr_x
            else:
                if self.include_input_skips:
                    layer_in = torch.cat([curr_x, feature_map], dim=-1)
                else:
                    layer_in = feature_map

            feature_map, logits = layer(layer_in)

        return feature_map, logits

    def _explore_layer_hyperparameters(
        self,
        layer_in: torch.Tensor,
        targets: torch.Tensor,
        val_in: Optional[torch.Tensor],
        val_targets: Optional[torch.Tensor],
        layer_idx: int,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run rapid probe tournament on current layer's frozen input to find
        the optimal hidden_dim, activation, and optimizer for this layer.
        """
        cand_dims = self.candidate_hidden_dims or [self._get_hidden_dim(layer_idx)]
        cand_acts = self.candidate_activations or [self.activation]
        cand_opts = self.candidate_optimizers or ["adam", "adamw", "sgd", "rmsprop"]
        probe_limit = (self.batch_size * 2) if self.batch_size else 256
        probe_steps = getattr(self, "tournament_steps", 5)
        loss_penalty_weight = getattr(self, "loss_penalty_weight", 0.02)

        probe_n = min(len(layer_in), probe_limit)
        p_idx = torch.linspace(0, len(layer_in) - 1, probe_n, device=self.device).long()
        p_x = layer_in[p_idx]
        p_y = targets[p_idx]

        target_vx = val_in if val_in is not None else p_x
        target_vy = val_targets if val_targets is not None else p_y
        probe_vn = min(len(target_vx), probe_limit)
        pv_idx = torch.linspace(0, len(target_vx) - 1, probe_vn, device=self.device).long()
        p_vx = target_vx[pv_idx]
        p_vy = target_vy[pv_idx]

        in_dim = layer_in.shape[-1]
        criterion = nn.CrossEntropyLoss()

        best_score = -float("inf")
        best_cfg = {
            "hidden_dim": cand_dims[0],
            "activation": cand_acts[0],
            "optimizer": cand_opts[0],
            "val_acc": 0.0,
            "val_loss": float("inf"),
        }

        for hdim in cand_dims:
            for act_name in cand_acts:
                for opt_name in cand_opts:
                    cand_layer = SingleConstructiveLayer(
                        in_features=in_dim,
                        hidden_dim=hdim,
                        num_layers=self.layers_per_stage,
                        num_classes=self.num_classes,
                        activation=act_name,
                        use_gating=self.use_gating,
                    ).to(self.device)

                    if opt_name == "adamw":
                        opt = optim.AdamW(cand_layer.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                    elif opt_name == "sgd":
                        opt = optim.SGD(cand_layer.parameters(), lr=self.learning_rate * 5.0, momentum=0.9, weight_decay=self.weight_decay)
                    elif opt_name == "rmsprop":
                        opt = optim.RMSprop(cand_layer.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                    else:
                        opt = optim.Adam(cand_layer.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

                    cand_layer.train()
                    for _ in range(probe_steps):
                        opt.zero_grad(set_to_none=True)
                        _, out = cand_layer(p_x)
                        loss = criterion(out, p_y)
                        loss.backward()
                        opt.step()

                    cand_layer.eval()
                    with torch.no_grad():
                        _, v_out = cand_layer(p_vx)
                        v_loss = criterion(v_out, p_vy).item()
                        v_acc = float((torch.argmax(v_out, dim=1) == p_vy).float().mean().item())
                        score = v_acc - loss_penalty_weight * min(v_loss, 10.0)

                    if score > best_score:
                        best_score = score
                        best_cfg = {
                            "hidden_dim": hdim,
                            "activation": act_name,
                            "optimizer": opt_name,
                            "val_acc": v_acc,
                            "val_loss": v_loss,
                        }

        if verbose:
            print(f"   🏆 [Layer {layer_idx + 1} Auto-Tune] Selected: Neurons={best_cfg['hidden_dim']} | Act={best_cfg['activation'].upper()} | Opt={best_cfg['optimizer'].upper()} | Probe Acc: {best_cfg['val_acc']*100:.2f}%")

        return best_cfg

    def _train_single_layer(
        self,
        layer: SingleConstructiveLayer,
        layer_input: torch.Tensor,
        targets: torch.Tensor,
        val_input: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
        verbose: bool = True,
        layer_idx: int = 0,
        optimizer_name: str = "adam",
    ) -> Dict[str, Any]:
        """
        Train a single layer until accuracy saturates.
        """
        criterion = nn.CrossEntropyLoss()
        if optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(
                layer.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                foreach=True,
            )
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(
                layer.parameters(),
                lr=self.learning_rate * 5.0,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                layer.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                foreach=True,
            )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs_per_layer, eta_min=self.learning_rate * 0.1
        )

        best_acc = 0.0
        best_loss = float("inf")
        epochs_no_improve = 0
        saturated = False
        best_state = None

        history: Dict[str, Any] = {
            "layer_idx": layer_idx,
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "saturated_epoch": None,
            "saturation_reason": "",
        }

        n_samples = len(layer_input)
        effective_batch_size = (
            self.batch_size
            if (self.batch_size is not None and self.batch_size < n_samples)
            else (min(64, n_samples) if n_samples > 64 else n_samples)
        )
        use_mini_batches = self.batch_size is not None and n_samples > self.batch_size

        # Probe batch sample size: test on 3 to 4 batches only (not all training samples)
        if self.eval_probe_batches is not None and self.eval_probe_batches > 0:
            probe_samples = min(n_samples, effective_batch_size * self.eval_probe_batches)
        else:
            probe_samples = n_samples

        probe_indices = torch.arange(probe_samples, device=self.device)

        if val_input is not None and val_targets is not None:
            val_n = len(val_input)
            val_probe_samples = (
                min(val_n, effective_batch_size * self.eval_probe_batches)
                if (self.eval_probe_batches and self.eval_probe_batches > 0)
                else val_n
            )
            val_probe_in = val_input[:val_probe_samples]
            val_probe_tgt = val_targets[:val_probe_samples]
        else:
            val_probe_in = None
            val_probe_tgt = None

        retrain_passes = 0
        total_allowed_retrains = self.max_retrains_per_layer
        current_max_epochs = self.max_epochs_per_layer
        epoch = 0

        while epoch < current_max_epochs:
            epoch += 1
            layer.train()
            if use_mini_batches:
                perm = torch.randperm(n_samples, device=self.device)
                total_loss = 0.0
                for start_idx in range(0, n_samples, self.batch_size):
                    batch_idx = perm[start_idx : start_idx + self.batch_size]
                    bx = layer_input[batch_idx]
                    by = targets[batch_idx]
                    optimizer.zero_grad(set_to_none=True)
                    _, logits_b = layer(bx)
                    loss_b = criterion(logits_b, by)
                    loss_b.backward()
                    optimizer.step()
                    total_loss += loss_b.detach() * len(bx)
                scheduler.step()
                loss_val = (total_loss / n_samples).item()
                layer.eval()
                with torch.inference_mode():
                    if probe_samples < n_samples:
                        _, logits = layer(layer_input[probe_indices])
                        preds = torch.argmax(logits, dim=1)
                        acc = (preds == targets[probe_indices]).float().mean().item()
                    else:
                        _, logits = layer(layer_input)
                        preds = torch.argmax(logits, dim=1)
                        acc = (preds == targets).float().mean().item()
            else:
                optimizer.zero_grad(set_to_none=True)
                h, logits = layer(layer_input)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
                layer.eval()
                with torch.inference_mode():
                    if probe_samples < n_samples:
                        _, probe_logits = layer(layer_input[probe_indices])
                        preds = torch.argmax(probe_logits, dim=1)
                        acc = (preds == targets[probe_indices]).float().mean().item()
                    else:
                        preds = torch.argmax(logits, dim=1)
                        acc = (preds == targets).float().mean().item()
                    loss_val = loss.item()

            history["loss"].append(loss_val)
            history["accuracy"].append(acc)

            # Validation tracking with probe batches
            if val_probe_in is not None:
                with torch.inference_mode():
                    _, val_logits = layer(val_probe_in)
                    val_loss = criterion(val_logits, val_probe_tgt).item()
                    val_preds = torch.argmax(val_logits, dim=1)
                    val_acc = (val_preds == val_probe_tgt).float().mean().item()
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                target_metric = val_acc
            else:
                target_metric = acc
                val_loss = loss_val

            # Saturation check: improvement > min_delta or significant loss reduction
            improvement = (target_metric - best_acc) >= self.min_delta
            loss_imprv = (loss_val < best_loss - 0.005) and (target_metric >= best_acc - 0.02)

            if improvement or loss_imprv:
                if target_metric > best_acc:
                    best_acc = target_metric
                if loss_val < best_loss:
                    best_loss = loss_val
                best_state = {k: v.cpu().clone() for k, v in layer.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and (epoch % 20 == 0 or epoch == 1 or target_metric >= self.target_accuracy):
                val_str = f" | Val Acc (probe {self.eval_probe_batches}b): {val_acc:.4f}" if val_probe_in is not None else ""
                probe_str = f" (probe {self.eval_probe_batches}b)" if probe_samples < n_samples else ""
                print(
                    f"  [Layer {layer_idx + 1}] Epoch {epoch:3d}/{current_max_epochs} "
                    f"- Loss: {loss_val:.4f} - Train Acc{probe_str}: {acc:.4f}{val_str}"
                )

            # Check saturation termination
            if target_metric >= self.target_accuracy:
                msg = f"Target accuracy reached ({target_metric:.4f} >= {self.target_accuracy})"
                if verbose:
                    print(f"  [Layer {layer_idx + 1}] Saturated: {msg} at epoch {epoch}.")
                history["saturated_epoch"] = epoch
                history["saturation_reason"] = msg
                saturated = True
                break

            effective_patience = self.patience

            if epochs_no_improve >= effective_patience:
                msg = f"Plateaued (no improvement >= {self.min_delta} for {effective_patience} epochs)"
                if verbose:
                    print(f"  [Layer {layer_idx + 1}] Saturated: {msg} at epoch {epoch}.")
                history["saturated_epoch"] = epoch
                history["saturation_reason"] = msg
                saturated = True
                break

            # If current epoch limit reached: check "if needed train again"
            if epoch == current_max_epochs and retrain_passes < total_allowed_retrains and target_metric < self.target_accuracy:
                if epochs_no_improve <= max(2, self.patience // 2):
                    retrain_passes += 1
                    extra_epochs = max(20, self.max_epochs_per_layer // 2)
                    current_max_epochs += extra_epochs
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = max(self.learning_rate * 0.5, 1e-4)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=extra_epochs, eta_min=self.learning_rate * 0.05
                    )
                    epochs_no_improve = 0
                    if verbose:
                        print(
                            f"  [Layer {layer_idx + 1}] 🔄 Probe Accuracy ({target_metric*100:.2f}%) below target ({self.target_accuracy*100:.2f}%) "
                            f"but still gaining momentum — training again for {extra_epochs} epochs (Retrain pass {retrain_passes}/{total_allowed_retrains})..."
                        )

        if not saturated:
            history["saturated_epoch"] = current_max_epochs
            history["saturation_reason"] = f"Max epochs ({current_max_epochs}) reached"
            if verbose:
                print(
                    f"  [Layer {layer_idx + 1}] Saturated: {history['saturation_reason']}. Best Metric: {best_acc:.4f}"
                )

        # Restore optimal weights for this layer
        if best_state is not None:
            layer.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        history["final_best_acc"] = best_acc
        return history

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "ConstructiveLayerwiseModel":
        """
        Executes the layer-wise training rule:
          1. Create first layer in network.
          2. Pass all training data through first layer.
          3. Train first layer until accuracy is saturated.
          4. Store all outputs for every training data sample into a feature map.
          5. Combine feature map and training data to train new layer weights until
             accuracy is saturated again (freezing earlier layers).
             Update stored feature map with new layer's outputs.
          6. Repeat for subsequent layers.
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)

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

        train_feature_map: Optional[torch.Tensor] = None
        val_feature_map: Optional[torch.Tensor] = None

        if verbose:
            print("=" * 60)
            print("🚀 Constructive Layer-Wise Model Training Started")
            print(f"   Input Dim: {self.input_dim} | Hidden Dim: {self.hidden_dim} | Classes: {self.num_classes}")
            print(f"   Max Layers: {self.max_layers} | Device: {self.device}")
            print("=" * 60)

        for layer_idx in range(self.max_layers):
            if verbose:
                print(f"\n--- [Step 1 & 2] Constructing Layer {layer_idx + 1} ---")

            # 1. Prepare layer inputs:
            # First layer uses original inputs.
            # Subsequent layers use [X, frozen_feature_map] (or frozen_feature_map).
            if layer_idx == 0:
                layer_in = X_t
                val_layer_in = X_val_t
            else:
                if self.include_input_skips:
                    layer_in = torch.cat([X_t, train_feature_map.detach()], dim=-1)
                    val_layer_in = (
                        torch.cat([X_val_t, val_feature_map.detach()], dim=-1)
                        if X_val_t is not None
                        else None
                    )
                else:
                    layer_in = train_feature_map.detach()
                    val_layer_in = val_feature_map.detach() if val_feature_map is not None else None

            # 2. Explore or resolve optimal layer hyperparameters
            layer_opt_name = "adam"
            if self.candidate_hidden_dims or self.candidate_activations or self.candidate_optimizers:
                best_cfg = self._explore_layer_hyperparameters(
                    layer_in, y_t, val_layer_in, y_val_t, layer_idx, verbose=verbose
                )
                self.selected_layer_hyperparameters.append({"layer_idx": layer_idx, **best_cfg})
                new_layer = SingleConstructiveLayer(
                    in_features=layer_in.shape[-1],
                    hidden_dim=best_cfg["hidden_dim"],
                    num_layers=self.layers_per_stage,
                    num_classes=self.num_classes,
                    activation=best_cfg["activation"],
                    use_gating=self.use_gating,
                ).to(self.device)
                layer_opt_name = best_cfg["optimizer"]
            else:
                new_layer = self._make_layer(layer_idx)

            # 3. Train layer until accuracy is saturated
            if verbose:
                print(f"--- [Step 3] Training Layer {layer_idx + 1} until saturation ---")
            hist = self._train_single_layer(
                layer=new_layer,
                layer_input=layer_in,
                targets=y_t,
                val_input=val_layer_in,
                val_targets=y_val_t,
                verbose=verbose,
                layer_idx=layer_idx,
                optimizer_name=layer_opt_name,
            )

            # Freeze newly trained layer so its weights won't be modified
            for param in new_layer.parameters():
                param.requires_grad = False
            new_layer.eval()

            # Append layer to cascade
            self.layers.append(new_layer)
            self.training_history.append(hist)

            # Fast probe testing on 3-4 batches only (avoiding evaluating every training sample)
            with torch.inference_mode():
                probe_samples = (
                    min(len(layer_in), (self.batch_size or 32) * self.eval_probe_batches)
                    if (self.eval_probe_batches and self.eval_probe_batches > 0)
                    else len(layer_in)
                )
                _, probe_logits = new_layer(layer_in[:probe_samples])
                probe_preds = torch.argmax(probe_logits, dim=1)
                current_acc = (probe_preds == y_t[:probe_samples]).float().mean().item()

                if val_layer_in is not None:
                    val_probe_samples = (
                        min(len(val_layer_in), (self.batch_size or 32) * self.eval_probe_batches)
                        if (self.eval_probe_batches and self.eval_probe_batches > 0)
                        else len(val_layer_in)
                    )
                    _, val_logits = new_layer(val_layer_in[:val_probe_samples])
                    val_preds = torch.argmax(val_logits, dim=1)
                    current_val_acc = (val_preds == y_val_t[:val_probe_samples]).float().mean().item()
                    val_str = f" | Val Acc (probe {self.eval_probe_batches}b): {current_val_acc:.4f}"
                    eval_metric = current_val_acc
                else:
                    val_str = ""
                    eval_metric = current_acc

            if verbose:
                print(
                    f"✔ Layer {layer_idx + 1} finalized: Train Acc (probe {self.eval_probe_batches}b) = {current_acc:.4f}{val_str}"
                )

            # Stopping condition: target accuracy reached (stop adding new layers early!)
            if eval_metric >= self.target_accuracy or current_acc >= 1.0:
                if verbose:
                    print(
                        f"\n🎯 [Convergence] Target accuracy {self.target_accuracy*100:.2f}% reached after {layer_idx + 1} layer(s)!"
                        f"\n   Stopping constructive growth early — no new layers needed."
                    )
                with torch.no_grad():
                    train_feature_map, _ = new_layer(layer_in)
                    self.train_feature_maps.append(train_feature_map.cpu().numpy())
                    self.current_feature_map = train_feature_map
                break

            # Else: Accuracy is low — add new layer if within max_layers
            if layer_idx < self.max_layers - 1:
                if verbose:
                    print(
                        f"   Layer {layer_idx + 1} accuracy ({eval_metric * 100:.2f}%) below target ({self.target_accuracy * 100:.2f}%)."
                        f" Accuracy is low — adding new layer {layer_idx + 2}..."
                    )
                with torch.no_grad():
                    train_feature_map, _ = new_layer(layer_in)
                    if val_layer_in is not None:
                        val_feature_map, _ = new_layer(val_layer_in)
                    self.train_feature_maps.append(train_feature_map.cpu().numpy())
                    self.current_feature_map = train_feature_map
                if verbose:
                    print(f"--- [Step 4 & 5] Layer {layer_idx + 1} feature map cached: shape {list(train_feature_map.shape)} ---")
            else:
                if verbose:
                    print(f"   Max layers ({self.max_layers}) reached. Stopping constructive growth.")
                with torch.no_grad():
                    train_feature_map, _ = new_layer(layer_in)
                    self.train_feature_maps.append(train_feature_map.cpu().numpy())
                    self.current_feature_map = train_feature_map

        if verbose:
            print("\n" + "=" * 60)
            print(f"Training Complete! Total Active Layers: {len(self.layers)}")
            print("=" * 60)

        return self

    def predict_proba(
        self,
        X: np.ndarray,
        up_to_layer: Optional[int] = None,
        early_exit_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute predicted class probabilities.
        Supports Adaptive Computation Time (ACT) / Dynamic Halting if early_exit_threshold is provided.
        """
        self.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = len(X_t)

        with torch.inference_mode():
            if early_exit_threshold is None:
                _, logits = self.forward(X_t, up_to_layer=up_to_layer)
                probs = torch.softmax(logits, dim=-1)
                return probs.cpu().numpy()

            # ACT Dynamic Halting mode
            final_probs = torch.zeros((n, self.num_classes), device=self.device)
            unhalted_mask = torch.ones(n, dtype=torch.bool, device=self.device)

            curr_x = X_t
            feature_map = None
            num_eval = len(self.layers) if up_to_layer is None else min(up_to_layer + 1, len(self.layers))

            for idx in range(num_eval):
                layer = self.layers[idx]
                if idx == 0:
                    layer_in = curr_x
                else:
                    if self.include_input_skips:
                        layer_in = torch.cat([curr_x, feature_map], dim=-1)
                    else:
                        layer_in = feature_map

                feature_map, logits = layer(layer_in)
                probs = torch.softmax(logits, dim=-1)
                max_p, _ = torch.max(probs, dim=-1)

                halt_now = (max_p >= early_exit_threshold) & unhalted_mask
                if halt_now.any():
                    final_probs[halt_now] = probs[halt_now]
                    unhalted_mask[halt_now] = False

                if not unhalted_mask.any():
                    break

            if unhalted_mask.any():
                final_probs[unhalted_mask] = probs[unhalted_mask]

            return final_probs.cpu().numpy()

    def predict(
        self,
        X: np.ndarray,
        up_to_layer: Optional[int] = None,
        early_exit_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predict class labels with optional ACT dynamic halting."""
        probs = self.predict_proba(X, up_to_layer=up_to_layer, early_exit_threshold=early_exit_threshold)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray, up_to_layer: Optional[int] = None) -> Dict[str, float]:
        """Evaluate accuracy on a dataset."""
        preds = self.predict(X, up_to_layer=up_to_layer)
        acc = float(np.mean(preds == y))
        return {"accuracy": acc}

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Scikit-learn compatible score method."""
        return self.evaluate(X, y)["accuracy"]

    def get_selected_hyperparameters(self) -> List[Dict[str, Any]]:
        """Return the list of optimal hyperparameters selected per layer."""
        return self.selected_layer_hyperparameters

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "num_classes": self.num_classes,
                "hidden_dim": self.hidden_dim,
                "max_layers": self.max_layers,
                "layers_per_stage": self.layers_per_stage,
                "activation": self.activation,
                "include_input_skips": self.include_input_skips,
                "batch_size": self.batch_size,
                "eval_probe_batches": self.eval_probe_batches,
                "max_retrains_per_layer": self.max_retrains_per_layer,
                "num_trained_layers": len(self.layers),
            },
            "training_history": self.training_history,
        }, filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "ConstructiveLayerwiseModel":
        """Load model checkpoint from disk."""
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        cfg = checkpoint["config"]
        model = cls(
            input_dim=cfg["input_dim"],
            num_classes=cfg["num_classes"],
            hidden_dim=cfg["hidden_dim"],
            max_layers=cfg["max_layers"],
            layers_per_stage=cfg.get("layers_per_stage", 3),
            activation=cfg["activation"],
            include_input_skips=cfg["include_input_skips"],
            batch_size=cfg.get("batch_size", None),
            eval_probe_batches=cfg.get("eval_probe_batches", 4),
            max_retrains_per_layer=cfg.get("max_retrains_per_layer", 1),
            device=device,
        )
        for i in range(cfg["num_trained_layers"]):
            model.layers.append(model._make_layer(i))
        model.load_state_dict(checkpoint["state_dict"])
        model.training_history = checkpoint["training_history"]
        return model
