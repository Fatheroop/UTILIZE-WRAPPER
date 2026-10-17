"""
MNIST Dataset Loader for Minimal-Resource Disk-Backed Training
Loads and parses the official MNIST idx-ubyte binary files directly with NumPy.
Features:
  - Atomic downloads with temporary files to prevent corrupted caches.
  - Fallback mirror support.
  - Configurable normalization and slicing.
  - Input validation and robust error handling.
"""

import os
import gzip
import shutil
import urllib.request
import urllib.error
import numpy as np
from typing import Tuple, Optional


MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist_if_needed(data_dir: str = "data/mnist", timeout_sec: int = 20) -> None:
    """Download MNIST gzip files if not already cached locally, using atomic downloads."""
    os.makedirs(data_dir, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (ConstructiveLeNet5/1.0)"}

    for key, filename in MNIST_FILES.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            continue

        temp_path = filepath + ".tmp"
        downloaded = False

        for mirror in MNIST_MIRRORS:
            url = mirror + filename
            try:
                print(f"Downloading {filename} from {url}...")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout_sec) as response, open(temp_path, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)

                # Atomic replace
                os.replace(temp_path, filepath)
                downloaded = True
                break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print(f"Warning: Failed downloading from {url}: {e}. Trying fallback mirror...")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

        if not downloaded and not (os.path.exists(filepath) and os.path.getsize(filepath) > 0):
            raise RuntimeError(
                f"Failed to download {filename} from all available mirrors. Please check internet connection."
            )


def load_mnist_raw(
    data_dir: str = "data/mnist",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw MNIST dataset from idx-ubyte gzip files.
    Returns:
        (train_images, train_labels, test_images, test_labels)
    """
    download_mnist_if_needed(data_dir)

    train_img_path = os.path.join(data_dir, MNIST_FILES["train_images"])
    train_lbl_path = os.path.join(data_dir, MNIST_FILES["train_labels"])
    test_img_path = os.path.join(data_dir, MNIST_FILES["test_images"])
    test_lbl_path = os.path.join(data_dir, MNIST_FILES["test_labels"])

    try:
        with gzip.open(train_img_path, "rb") as f:
            train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
        with gzip.open(train_lbl_path, "rb") as f:
            train_labels = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)

        with gzip.open(test_img_path, "rb") as f:
            test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
        with gzip.open(test_lbl_path, "rb") as f:
            test_labels = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    except Exception as e:
        raise IOError(f"Error parsing MNIST gzip files: {e}. Consider deleting {data_dir} to re-download.")

    return train_images, train_labels, test_images, test_labels


def get_normalized_mnist(
    data_dir: str = "data/mnist",
    num_train: Optional[int] = None,
    num_test: Optional[int] = None,
    mean: float = 0.1307,
    std: float = 0.3081,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns float32 normalized MNIST:
      Images: shape (N, 1, 28, 28), standardized by ((x / 255.0) - mean) / std
      Labels: shape (N,), int64 in [0..9]
    """
    train_images, train_labels, test_images, test_labels = load_mnist_raw(data_dir)

    train_images = ((train_images.astype(np.float32) / 255.0) - mean) / std
    test_images = ((test_images.astype(np.float32) / 255.0) - mean) / std

    if num_train is not None and 0 < num_train < len(train_images):
        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(len(train_images))[:num_train]
        train_images = train_images[indices]
        train_labels = train_labels[indices]

    if num_test is not None and 0 < num_test < len(test_images):
        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(len(test_images))[:num_test]
        test_images = test_images[indices]
        test_labels = test_labels[indices]

    return train_images, train_labels, test_images, test_labels
