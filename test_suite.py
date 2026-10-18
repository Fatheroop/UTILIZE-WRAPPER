"""
Comprehensive Test and Verification Suite
Validates correctness, dynamic shape adaptability, checkpoint serialization,
thread-safe resource tracking, and learning mechanics across all modules.
"""

import os
import shutil
import unittest
import numpy as np
import torch

from core_model import ConstructiveLayerwiseModel
from lenet5_disk_model import DiskConstructiveLeNet5
from standard_lenet5 import StandardLeNet5, StandardLeNet5Trainer
from resource_monitor import ResourceTracker
from mnist_dataset import get_normalized_mnist
from constructive_torch import ConstructiveWrapper


class TestConstructiveModules(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_scratch_checkpoints"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_core_model_fit_save_load(self):
        """Test Core Model learning, checkpointing, and evaluation."""
        X = np.random.randn(80, 4).astype(np.float32)
        y = np.random.randint(0, 2, size=(80,)).astype(np.int64)

        model = ConstructiveLayerwiseModel(
            input_dim=4,
            num_classes=2,
            hidden_dim=8,
            max_layers=2,
            max_epochs_per_layer=5,
            patience=2,
            random_state=42,
        )
        model.fit(X, y, verbose=False)

        preds = model.predict(X)
        self.assertEqual(len(preds), 80)
        self.assertIn(model.evaluate(X, y)["accuracy"], [preds == y] and [float(np.mean(preds == y))])

        ckpt_path = os.path.join(self.test_dir, "core_test.pt")
        model.save(ckpt_path)
        self.assertTrue(os.path.exists(ckpt_path))

        loaded = ConstructiveLayerwiseModel.load(ckpt_path)
        loaded_preds = loaded.predict(X)
        np.testing.assert_array_equal(preds, loaded_preds)

    def test_lenet5_dynamic_resolutions(self):
        """Test LeNet-5 adaptation to different input resolutions without hardcoding."""
        resolutions = [(1, 28, 28), (1, 32, 32), (3, 32, 32)]

        for in_shape in resolutions:
            model = DiskConstructiveLeNet5(
                in_shape=in_shape,
                num_classes=5,
                c1_channels=4,
                c2_channels=8,
                c5_dim=32,
                f6_dim=16,
                storage_dir=os.path.join(self.test_dir, f"lenet_{in_shape[0]}_{in_shape[1]}"),
            )
            # Verify 4 stages exist and have consistent forward passes
            self.assertEqual(model.num_stages, 4)
            dummy = torch.randn(2, *in_shape)
            s0 = model._instantiate_stage(0)
            feats, logits = s0(dummy)
            self.assertEqual(logits.shape, (2, 5))

    def test_lenet5_training_and_prefetching(self):
        """Test mini training, disk checkpointing, and prefetching prediction."""
        X_train = np.random.randn(64, 1, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 4, size=(64,)).astype(np.int64)
        X_test = np.random.randn(20, 1, 28, 28).astype(np.float32)

        storage_dir = os.path.join(self.test_dir, "lenet_train_test")
        model = DiskConstructiveLeNet5(
            in_shape=(1, 28, 28),
            num_classes=4,
            max_epochs_per_stage=3,
            patience=2,
            storage_dir=storage_dir,
            random_state=42,
        )
        model.fit(X_train, y_train, verbose=False)

        # Predict with prefetching
        preds_prefetch = model.predict(X_test, use_prefetching=True)
        # Predict synchronous
        preds_sync = model.predict(X_test, use_prefetching=False)

        self.assertEqual(len(preds_prefetch), 20)
        np.testing.assert_array_equal(preds_prefetch, preds_sync)

        # Test model reload from disk configuration
        loaded_model = DiskConstructiveLeNet5.load(storage_dir)
        self.assertEqual(loaded_model.layers_per_stage, model.layers_per_stage)
        loaded_preds = loaded_model.predict(X_test, use_prefetching=True)
        np.testing.assert_array_equal(preds_prefetch, loaded_preds)

    def test_layers_per_stage_configurations(self):
        """Test Constructive LeNet-5 initialization and forward pass with 1 and 3 layers per stage."""
        for num_layers in [1, 3]:
            m = DiskConstructiveLeNet5(
                in_shape=(1, 28, 28),
                num_classes=10,
                layers_per_stage=num_layers,
                storage_dir=os.path.join(self.test_dir, f"test_layers_{num_layers}"),
            )
            self.assertEqual(m.layers_per_stage, num_layers)
            x = torch.randn(4, 1, 28, 28)
            s0 = m._instantiate_stage(0)
            f0, l0 = s0(x)
            self.assertEqual(l0.shape, (4, 10))

            s1 = m._instantiate_stage(1)
            f1, l1 = s1(f0)
            self.assertEqual(l1.shape, (4, 10))

            s2 = m._instantiate_stage(2)
            f2, l2 = s2(f1)
            self.assertEqual(l2.shape, (4, 10))

            s3 = m._instantiate_stage(3)
            f3, l3 = s3(f2)
            self.assertEqual(l3.shape, (4, 10))

    def test_standard_lenet5_baseline(self):
        """Test Standard LeNet-5 baseline execution."""
        X_train = np.random.randn(50, 1, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 3, size=(50,)).astype(np.int64)
        X_test = np.random.randn(15, 1, 28, 28).astype(np.float32)

        trainer = StandardLeNet5Trainer(
            in_shape=(1, 28, 28),
            num_classes=3,
            max_epochs=2,
            random_state=42,
        )
        trainer.fit(X_train, y_train, verbose=False)
        preds = trainer.predict(X_test)
        self.assertEqual(len(preds), 15)

    def test_resource_tracker(self):
        """Test resource tracker monitoring accuracy and memory metrics."""
        with ResourceTracker("Unit Test Tracker", sample_interval_sec=0.01) as tracker:
            # Allocate temporary memory
            arr = np.ones((500, 500), dtype=np.float64)
            _ = np.dot(arr, arr)
            tracker.log_results(train_acc=0.95, test_acc=0.92, macro_f1=0.91, inference_latency_ms=12.5)

        summary = tracker.summary()
        self.assertEqual(summary["name"], "Unit Test Tracker")
        self.assertEqual(summary["test_accuracy_pct"], 92.0)
        self.assertGreater(summary["peak_ram_mb"], 0.0)

    def test_chunked_feature_streaming_and_prefetching(self):
        """Test 2MB chunked feature store and asynchronous prefetch streamer."""
        from lenet5_disk_model import ChunkedFeatureMapStore, PrefetchedChunkStreamer

        chunk_dir = os.path.join(self.test_dir, "test_chunks")
        # Use small chunk size (16 KB) to test splitting 100 samples into multiple chunks
        store = ChunkedFeatureMapStore(chunk_dir, chunk_size_bytes=16 * 1024)
        feats = torch.randn(100, 6, 14, 14)
        labels = torch.randint(0, 10, (100,))

        store.save_from_tensors(feats, labels)
        self.assertGreater(len(store.chunk_files), 1)

        # Test prefetching streamer
        streamer = PrefetchedChunkStreamer(store.chunk_files, batch_size=16, shuffle=True)
        total_streamed = 0
        for bx, by in streamer:
            self.assertEqual(bx.shape[1:], (6, 14, 14))
            total_streamed += len(bx)
        self.assertEqual(total_streamed, 100)

        # Cleanup
        store.clear()
        self.assertEqual(len(os.listdir(chunk_dir)), 0)

    def test_4xor_constructive_and_standard(self):
        """Test 4-bit XOR parity learning on Constructive Model and Standard MLP."""
        from standard_mlp import StandardMLPTrainer
        from benchmark_4xor import generate_4xor_data

        X_can, y_can, X_train, y_train, X_test, y_test = generate_4xor_data(
            num_samples=300, noise=0.02, random_state=42
        )

        # 1. Test Updated Constructive Model (3-layer blocks)
        c_model = ConstructiveLayerwiseModel(
            input_dim=4,
            num_classes=2,
            hidden_dim=16,
            max_layers=2,
            layers_per_stage=3,
            activation="relu",
            learning_rate=0.02,
            max_epochs_per_layer=80,
            patience=20,
            random_state=42,
        )
        c_model.fit(X_train, y_train, verbose=False)
        c_preds = c_model.predict(X_can)
        self.assertEqual(len(c_preds), 16)
        self.assertGreaterEqual(float(np.mean(c_preds == y_can)), 0.85)

        # Test save and load with layers_per_stage
        ckpt_path = os.path.join(self.test_dir, "xor_cmodel.pt")
        c_model.save(ckpt_path)
        loaded = ConstructiveLayerwiseModel.load(ckpt_path)
        self.assertEqual(loaded.layers_per_stage, 3)
        np.testing.assert_array_equal(c_preds, loaded.predict(X_can))

        # 2. Test Standard MLP Baseline
        std_mlp = StandardMLPTrainer(
            in_features=4,
            hidden_dim=16,
            num_layers=3,
            num_classes=2,
            activation="relu",
            learning_rate=0.02,
            max_epochs=100,
            patience=20,
            random_state=42,
        )
        std_mlp.fit(X_train, y_train, verbose=False)
        std_preds = std_mlp.predict(X_can)
        self.assertEqual(len(std_preds), 16)
        self.assertGreaterEqual(float(np.mean(std_preds == y_can)), 0.85)

    def test_constructive_torch_wrapper_api(self):
        """Test ConstructiveWrapper: auto-partitioning, training, forward, and standard model export."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 2),
        )

        X = torch.randn(80, 8)
        y = torch.randint(0, 2, (80,))

        wrapper = ConstructiveWrapper(
            model=model,
            in_shape=(8,),
            num_classes=2,
            max_epochs_per_stage=3,
            patience=2,
            random_state=42,
        )
        wrapper.fit(X, y, verbose=False)

        # Test forward as nn.Module
        wrapper.eval()
        with torch.inference_mode():
            out = wrapper(X[:5])
            self.assertEqual(out.shape, (5, 2))

        # Test export to standard model
        std_model = wrapper.get_standard_model()
        self.assertIsInstance(std_model, torch.nn.Sequential)

        # Verify exact output match between wrapper and standard model
        with torch.inference_mode():
            std_out = std_model(X[:5])
            torch.testing.assert_close(out, std_out, rtol=1e-5, atol=1e-5)

    def test_network_with_wrapper_vs_without_wrapper(self):
        """Direct comparison test: Network With Wrapper vs. Network Without Wrapper on identical architecture."""
        def make_net():
            return torch.nn.Sequential(
                torch.nn.Linear(10, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 2),
            )

        torch.manual_seed(42)
        np.random.seed(42)
        X = torch.randn(120, 10)
        y = (X[:, 0] + X[:, 1] > 0).long()
        X_train, y_train = X[:100], y[:100]
        X_test, y_test = X[100:], y[100:]

        # 1. Network With Wrapper
        net_wrap = make_net()
        wrapper = ConstructiveWrapper(
            model=net_wrap,
            in_shape=(10,),
            num_classes=2,
            learning_rate=0.01,
            max_epochs_per_stage=10,
            patience=5,
            random_state=42,
        )
        wrapper.fit(X_train, y_train, verbose=False)
        acc_wrap = wrapper.evaluate(X_test, y_test)
        self.assertGreaterEqual(acc_wrap, 0.75)

        # Exported standard model matches wrapper exactly
        std_exported = wrapper.get_standard_model()
        with torch.inference_mode():
            wrap_out = wrapper(X_test)
            std_out = std_exported(X_test)
            torch.testing.assert_close(wrap_out, std_out, rtol=1e-5, atol=1e-5)

        # 2. Network Without Wrapper (Standard PyTorch Training)
        net_std = make_net()
        opt = torch.optim.Adam(net_std.parameters(), lr=0.01)
        crit = torch.nn.CrossEntropyLoss()
        for _ in range(25):
            net_std.train()
            opt.zero_grad()
            out = net_std(X_train)
            loss = crit(out, y_train)
            loss.backward()
            opt.step()

        net_std.eval()
        with torch.inference_mode():
            std_preds = torch.argmax(net_std(X_test), dim=1)
            acc_std = float((std_preds == y_test).float().mean().item())
            self.assertGreaterEqual(acc_std, 0.70)

    def test_insitu_hyperparameter_auto_tuner(self):
        """Test continuous online hyperparameter auto-tuning without extra epochs."""
        torch.manual_seed(42)
        np.random.seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
        )
        X = torch.randn(100, 8)
        y = (X[:, 0] > 0).long()

        wrapper = ConstructiveWrapper(
            model=model,
            in_shape=(8,),
            num_classes=2,
            learning_rate=0.01,
            max_epochs_per_stage=8,
            auto_tune_hyperparams=True,
            fine_tune_epochs=2,
            random_state=42,
        )
        wrapper.fit(X, y, verbose=False)

        tuned_records = wrapper.get_tuned_hyperparameters()
        self.assertGreater(len(tuned_records), 0)
        for rec in tuned_records:
            self.assertIn("epoch", rec)
            self.assertIn("chosen_lr", rec)
            self.assertIn("chosen_wd", rec)
            self.assertIn("candidate_lrs", rec)
            self.assertGreater(rec["chosen_lr"], 0.0)

        acc = wrapper.evaluate(X, y)
        self.assertGreaterEqual(acc, 0.80)

    def test_insitu_tuner_advanced_mechanics(self):
        """Test optimizer-aware curvature vectors, adaptive clipping, and parabolic vertex refinement."""
        from constructive_torch import InSituHyperparameterTuner
        torch.manual_seed(42)
        net = torch.nn.Sequential(torch.nn.Linear(6, 12), torch.nn.ReLU(), torch.nn.Linear(12, 2))
        opt = torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=1e-4)
        crit = torch.nn.CrossEntropyLoss()

        tuner = InSituHyperparameterTuner(
            module=net,
            optimizer=opt,
            criterion=crit,
            min_lr=1e-5,
            max_lr=0.05,
            enabled=True,
        )

        X = torch.randn(64, 6)
        y = (X[:, 0] > 0).long()

        # Step 1: probe step
        rec = tuner.step_epoch(1, X, y, X, y, verbose=False)
        self.assertIn("grad_clip", rec)
        self.assertGreater(rec["grad_clip"], 0.0)
        self.assertGreater(rec["chosen_lr"], 0.0)

        # Test gradient clipping
        out = net(X)
        loss = crit(out, y)
        loss.backward()
        clipped_norm = tuner.clip_gradients(net.parameters())
        self.assertGreater(clipped_norm, 0.0)
        opt.step()

        # Step 2: verify AdamW moments are tracked in directions
        params = dict(net.named_parameters())
        dirs = tuner._get_update_directions(params)
        self.assertEqual(len(dirs), len(params))
        for k, d in dirs.items():
            self.assertEqual(d.shape, params[k].shape)

    def test_architectural_auto_tuning_neurons_activation_optimizer(self):
        """Test automated selection of neurons, activation functions, and optimizers."""
        from constructive_torch import AutoConstructiveModel, ArchitecturalAutoTuner

        torch.manual_seed(42)
        X = torch.randn(80, 6)
        y = (X[:, 0] > 0).long()

        # 1. Test AutoConstructiveModel
        auto_model = AutoConstructiveModel(
            in_shape=6,
            num_classes=2,
            max_stages=2,
            candidate_neurons=[16, 32],
            candidate_activations=["relu", "gelu"],
            candidate_optimizers=["adam", "adamw"],
            max_epochs_per_stage=3,
            fine_tune_epochs=1,
            random_state=42,
        )
        auto_model.fit(X, y, verbose=False)

        summary = auto_model.get_architecture_summary()
        self.assertEqual(len(summary), 2)
        for stage_info in summary:
            self.assertIn("neurons", stage_info)
            self.assertIn("activation", stage_info)
            self.assertIn("optimizer", stage_info)
            self.assertIn(stage_info["neurons"], [16, 32])
            self.assertIn(stage_info["activation"], ["relu", "gelu"])
            self.assertIn(stage_info["optimizer"], ["adam", "adamw"])

        # Test prediction and exported model
        preds = auto_model.predict(X[:10])
        self.assertEqual(len(preds), 10)
        std_model = auto_model.get_standard_model()
        self.assertIsInstance(std_model, torch.nn.Sequential)

        # 2. Test ConstructiveLayerwiseModel candidate search
        core_model = ConstructiveLayerwiseModel(
            input_dim=6,
            num_classes=2,
            max_layers=2,
            candidate_hidden_dims=[16, 32],
            candidate_activations=["relu", "tanh"],
            candidate_optimizers=["adam", "adamw"],
            max_epochs_per_layer=5,
            random_state=42,
        )
        core_model.fit(X.numpy(), y.numpy(), verbose=False)
        core_summary = core_model.get_selected_hyperparameters()
        self.assertGreater(len(core_summary), 0)
        for layer_info in core_summary:
            self.assertIn(layer_info["hidden_dim"], [16, 32])
            self.assertIn(layer_info["activation"], ["relu", "tanh"])
            self.assertIn(layer_info["optimizer"], ["adam", "adamw"])

    def test_flexible_hyperparameter_configuration(self):
        """Test non-hardcoded configurations: BatchNorm, Dropout, custom optimizer kwargs, and pooling guards."""
        from constructive_torch import AutoConstructiveModel, ArchitecturalAutoTuner
        import torch.nn as nn

        torch.manual_seed(42)
        # Test 1: Dense network with batch norm, dropout, and custom optimizer kwargs
        X = torch.randn(60, 5)
        y = (X[:, 0] > 0).long()

        auto_model = AutoConstructiveModel(
            in_shape=5,
            num_classes=2,
            max_stages=1,
            candidate_neurons=[16],
            candidate_activations=["gelu"],
            candidate_optimizers=["sgd", "rmsprop"],
            use_batch_norm=True,
            dropout_rate=0.2,
            val_split=0.2,
            tournament_steps=2,
            probe_batch_multiplier=1,
            loss_penalty_weight=0.05,
            optimizer_kwargs={"momentum": 0.85},
            random_state=42,
        )
        auto_model.fit(X, y, verbose=False)
        std_model = auto_model.get_standard_model()

        # Verify BatchNorm1d and Dropout are in exported model
        has_bn = any(isinstance(m, nn.BatchNorm1d) for m in std_model.modules())
        has_dropout = any(isinstance(m, nn.Dropout) for m in std_model.modules())
        self.assertTrue(has_bn, "Exported model should contain BatchNorm1d when use_batch_norm=True")
        self.assertTrue(has_dropout, "Exported model should contain Dropout when dropout_rate > 0")

        # Test 2: Custom optimizer kwargs in ArchitecturalAutoTuner
        dummy_layer = nn.Linear(5, 2)
        opt = ArchitecturalAutoTuner.create_optimizer(
            "sgd", dummy_layer.parameters(), lr=0.01, optimizer_kwargs={"momentum": 0.77}
        )
        self.assertEqual(opt.param_groups[0]["momentum"], 0.77)

        # Test 3: Spatial Conv guard on small dimensions (2x2) without crashing
        best_cfg_spatial = ArchitecturalAutoTuner.run_tournament(
            in_shape=(1, 2, 2),
            out_dim=2,
            task="classification",
            probe_train_x=torch.randn(10, 1, 2, 2),
            probe_train_y=torch.randint(0, 2, (10,)),
            probe_val_x=torch.randn(5, 1, 2, 2),
            probe_val_y=torch.randint(0, 2, (5,)),
            candidate_neurons=[8],
            candidate_activations=["relu"],
            candidate_optimizers=["adam"],
            use_pooling=True,
            steps=2,
            verbose=False,
        )
        self.assertEqual(best_cfg_spatial["neurons"], 8)

    def test_zero_copy_chunked_inference_and_proba(self):
        """Test zero-copy chunk streaming for predict and predict_proba with parity against standard model."""
        torch.manual_seed(42)
        np.random.seed(42)
        X = np.random.randn(250, 6).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

        model = torch.nn.Sequential(
            torch.nn.Linear(6, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 2),
        )
        wrapper = ConstructiveWrapper(
            model=model,
            in_shape=(6,),
            num_classes=2,
            max_epochs_per_stage=3,
            patience=2,
            random_state=42,
        )
        wrapper.fit(X, y, verbose=False)

        # 1. predict_proba
        probs = wrapper.predict_proba(X, batch_size=64)
        self.assertEqual(probs.shape, (250, 2))
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

        # 2. predict
        preds = wrapper.predict(X, batch_size=64)
        self.assertEqual(len(preds), 250)
        np.testing.assert_array_equal(preds, np.argmax(probs, axis=1))

        # 3. Parity with exported standard model
        std_model = wrapper.get_standard_model()
        with torch.inference_mode():
            std_logits = std_model(torch.from_numpy(X).float())
            std_probs = torch.softmax(std_logits, dim=1).numpy()
            np.testing.assert_allclose(probs, std_probs, atol=1e-5)

    def test_export_traced_and_quantized_models(self):
        """Test TorchScript JIT tracing, dynamic quantization, and file export parity."""
        from constructive_torch import AutoConstructiveModel

        torch.manual_seed(42)
        np.random.seed(42)
        X = np.random.randn(80, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        auto_model = AutoConstructiveModel(
            in_shape=5,
            num_classes=2,
            max_stages=1,
            candidate_neurons=[16],
            candidate_activations=["relu"],
            candidate_optimizers=["adam"],
            max_epochs_per_stage=3,
            fine_tune_epochs=1,
            random_state=42,
        )
        auto_model.fit(X, y, verbose=False)

        # 1. Traced model export
        traced = auto_model.export_traced_model()
        self.assertIsInstance(traced, torch.jit.ScriptModule)
        with torch.inference_mode():
            test_x = torch.from_numpy(X[:10]).float()
            ref_out = auto_model.get_standard_model()(test_x)
            traced_out = traced(test_x)
            torch.testing.assert_close(traced_out, ref_out, rtol=1e-5, atol=1e-5)

        # 2. Dynamic quantized model export
        quantized = auto_model.export_quantized_model()
        self.assertIsInstance(quantized, torch.nn.Module)
        with torch.inference_mode():
            q_out = quantized(test_x.cpu())
            self.assertEqual(q_out.shape, (10, 2))

        # 3. File export tests (TorchScript & ONNX)
        ts_path = os.path.join(self.test_dir, "model_export.pt")
        onnx_path = os.path.join(self.test_dir, "model_export.onnx")
        auto_model.export_torchscript(ts_path)
        self.assertTrue(os.path.exists(ts_path))
        self.assertGreater(os.path.getsize(ts_path), 0)

        auto_model.export_onnx(onnx_path)
        try:
            import onnx
            import onnxscript
            self.assertTrue(os.path.exists(onnx_path))
            self.assertGreater(os.path.getsize(onnx_path), 0)
        except ImportError:
            pass

    def test_gradient_accumulation_and_warmup(self):
        """Test gradient accumulation and learning rate warmup scheduler across stages and fine-tuning."""
        torch.manual_seed(42)
        np.random.seed(42)
        X = torch.randn(128, 4)
        y = (X[:, 0] > 0).long()

        wrapper = ConstructiveWrapper(
            stages=[
                torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(8, 2)),
            ],
            in_shape=(4,),
            num_classes=2,
            batch_size=16,
            gradient_accumulation_steps=4,
            warmup_epochs=1,
            max_epochs_per_stage=3,
            fine_tune_epochs=2,
            use_ema=True,
            random_state=42,
        )
        wrapper.fit(X, y, verbose=False)
        acc = wrapper.evaluate(X, y)
        self.assertGreaterEqual(acc, 0.75)

    def test_selective_gradient_and_loss_gating(self):
        """Test Selective Gradient optimization, Loss-Gated sampling, and Curvature Early Stopping."""
        torch.manual_seed(42)
        np.random.seed(42)
        X = torch.randn(160, 6)
        y = (X[:, 0] + X[:, 1] > 0).long()

        wrapper = ConstructiveWrapper(
            stages=[
                torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(16, 2)),
            ],
            in_shape=(6,),
            num_classes=2,
            batch_size=32,
            learning_rate=0.01,
            max_epochs_per_stage=4,
            fine_tune_epochs=2,
            selective_gradient=True,
            selective_gradient_ratio=0.80,
            loss_gated_sampling=True,
            loss_gated_threshold=0.90,
            early_stopping_curvature=True,
            early_stopping_curvature_delta=0.005,
            random_state=42,
        )
        wrapper.fit(X, y, verbose=False)
        acc = wrapper.evaluate(X, y)
        self.assertGreaterEqual(acc, 0.70)

        # Verify export parity
        std_model = wrapper.get_standard_model()
        std_model.eval()
        with torch.inference_mode():
            std_out = std_model(X[:20])
            std_preds = torch.argmax(std_out, dim=1).numpy()
            wrapper_preds = wrapper.predict(X[:20].numpy())
            np.testing.assert_array_equal(wrapper_preds, std_preds)

    def test_wrapper_save_load_and_guards(self):
        """Test ConstructiveWrapper save, load, and safety guards on uninitialized models."""
        X = np.random.randn(80, 4).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)

        # 1. Test safety guard: uninitialized get_standard_model raises RuntimeError when in_shape is missing
        empty_wrapper = ConstructiveWrapper(task="classification")
        with self.assertRaises(RuntimeError):
            empty_wrapper.get_standard_model()

        # 2. Train a small wrapper model
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
        )
        wrapper = ConstructiveWrapper(
            model=model,
            in_shape=(4,),
            num_classes=2,
            max_epochs_per_stage=3,
            fine_tune_epochs=1,
            random_state=42,
        )
        wrapper.fit(X, y, verbose=False)
        orig_preds = wrapper.predict(X)

        # 3. Test instance save & load
        save_path = os.path.join(self.test_dir, "wrapper_checkpoint.pt")
        wrapper.save(save_path)
        self.assertTrue(os.path.exists(save_path))

        # Re-load into fresh wrapper
        fresh_model = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
        )
        new_wrapper = ConstructiveWrapper(
            model=fresh_model,
            in_shape=(4,),
            num_classes=2,
            random_state=42,
        )
        new_wrapper.fit(X[:2], y[:2], verbose=False) # initialize stages
        new_wrapper.load(save_path)
        loaded_preds = new_wrapper.predict(X)
        np.testing.assert_array_equal(orig_preds, loaded_preds)

        # 4. Test classmethod load_from_checkpoint
        loaded_from_class = ConstructiveWrapper.load_from_checkpoint(
            save_path,
            model=torch.nn.Sequential(
                torch.nn.Linear(4, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 2),
            ),
        )
        class_preds = loaded_from_class.predict(X)
        np.testing.assert_array_equal(orig_preds, class_preds)

    def test_breakthrough_techniques_suite(self):
        """
        Test modern AI breakthrough techniques suite:
          1. AirLLM FP16 activation streaming (RAM and chunked disk)
          2. 2-layer expressive auxiliary probe heads (Belilovsky DGL)
          3. Barlow Twins / VICReg feature covariance decorrelation
          4. Intermediate manifold mixup augmentation
          5. Gradient Centralization (GC)
          6. Test-Time Augmentation (predict_tta and predict_proba_tta)
          7. Exact numerical parity when exported to native nn.Sequential
        """
        np.random.seed(42)
        torch.manual_seed(42)
        X = np.random.randn(64, 1, 14, 14).astype(np.float32)
        y = np.random.randint(0, 3, size=(64,)).astype(np.int64)

        stages = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            ),
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(8 * 7 * 7, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 3),
            ),
        ]

        wrapper = ConstructiveWrapper(
            stages=stages,
            in_shape=(1, 14, 14),
            num_classes=3,
            batch_size=16,
            max_epochs_per_stage=2,
            fine_tune_epochs=1,
            activation_precision="float16",
            probe_head_depth=2,
            feature_decorrelation=True,
            feature_decorrelation_weight=0.01,
            mixup_alpha=0.2,
            gradient_centralization=True,
            selective_gradient=True,
            feature_storage="chunked_disk",
            storage_dir=os.path.join(self.test_dir, "breakthrough_cache"),
            random_state=42,
        )

        wrapper.fit(X, y, verbose=False)
        self.assertEqual(wrapper.activation_dtype, torch.float16)
        self.assertEqual(wrapper.probe_head_depth, 2)
        self.assertTrue(wrapper.feature_decorrelation)
        self.assertTrue(wrapper.gradient_centralization)

        # Test predictions and TTA
        preds = wrapper.predict(X)
        self.assertEqual(len(preds), 64)
        probs_tta = wrapper.predict_proba_tta(X)
        self.assertEqual(probs_tta.shape, (64, 3))
        preds_tta = wrapper.predict_tta(X)
        self.assertEqual(len(preds_tta), 64)

        # Test exact export parity
        std_model = wrapper.get_standard_model()
        with torch.no_grad():
            t_X = torch.from_numpy(X).float()
            std_out = std_model(t_X).numpy()
            wrap_out = wrapper(t_X).numpy()
            np.testing.assert_allclose(std_out, wrap_out, atol=1e-5)


if __name__ == "__main__":
    unittest.main()


