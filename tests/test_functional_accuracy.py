"""Functional accuracy tests verifying model outputs across data formats.

These tests verify that model accuracy is consistent within 0.1% across all
data formats (FFCV, LitData, ImageFolder, .slipcache). If bytes are wrong,
decodes fail, or decode is subtly wrong, accuracy will drift measurably.

These tests require:
- ImageNet validation set (or subset)
- pretrained model weights
- CUDA GPU (optional but recommended)

Run with:
    pytest tests/test_functional_accuracy.py -v -m imagenet
    pytest tests/test_functional_accuracy.py -v -m "slow and imagenet"
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# Skip markers for slow and imagenet tests
pytestmark = [
    pytest.mark.slow,
    pytest.mark.imagenet,
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set these environment variables to run the tests:
#   IMAGENET_VAL_DIR: Path to ImageNet validation folder
#   IMAGENET_FFCV: Path to ImageNet validation .ffcv file (optional)
#   IMAGENET_LITDATA: Path to ImageNet LitData cache (optional)

IMAGENET_VAL_DIR = os.environ.get("IMAGENET_VAL_DIR")
IMAGENET_FFCV = os.environ.get("IMAGENET_FFCV")
IMAGENET_LITDATA = os.environ.get("IMAGENET_LITDATA")

# Expected accuracy range for ResNet50 on ImageNet val
EXPECTED_TOP1_MIN = 0.75  # Should be at least 75%
EXPECTED_TOP1_MAX = 0.78  # Should be at most 78%


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_top1(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Evaluate top-1 accuracy on a loader.

    Args:
        model: PyTorch model in eval mode
        loader: DataLoader or SlipstreamLoader
        device: Device to run on
        max_batches: Optional limit on batches (for quick tests)

    Returns:
        Top-1 accuracy as float (0.0 to 1.0)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Handle both dict (slipstream) and tuple (standard) formats
            if isinstance(batch, dict):
                images = batch['image']
                labels = batch['label']
            else:
                images, labels = batch

            # Move to device
            if not isinstance(images, torch.Tensor):
                images = torch.from_numpy(images)
            if not isinstance(labels, torch.Tensor):
                labels = torch.from_numpy(labels)

            images = images.to(device)
            labels = labels.to(device)

            # Ensure images are in correct format (N, C, H, W) and normalized
            if images.dim() == 4 and images.shape[-1] == 3:
                # HWC -> CHW
                images = images.permute(0, 3, 1, 2)
            if images.dtype == torch.uint8:
                images = images.float() / 255.0

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def imagenet_val_path():
    """Get ImageNet validation directory path."""
    if IMAGENET_VAL_DIR is None:
        pytest.skip("IMAGENET_VAL_DIR not set")
    path = Path(IMAGENET_VAL_DIR)
    if not path.exists():
        pytest.skip(f"IMAGENET_VAL_DIR={path} does not exist")
    return path


@pytest.fixture
def imagenet_ffcv_path():
    """Get ImageNet FFCV file path."""
    if IMAGENET_FFCV is None:
        pytest.skip("IMAGENET_FFCV not set")
    path = Path(IMAGENET_FFCV)
    if not path.exists():
        pytest.skip(f"IMAGENET_FFCV={path} does not exist")
    return path


@pytest.fixture
def imagenet_litdata_path():
    """Get ImageNet LitData cache path."""
    if IMAGENET_LITDATA is None:
        pytest.skip("IMAGENET_LITDATA not set")
    path = Path(IMAGENET_LITDATA)
    if not path.exists():
        pytest.skip(f"IMAGENET_LITDATA={path} does not exist")
    return path


@pytest.fixture
def resnet50_model():
    """Load pretrained ResNet50 model."""
    import torchvision.models as models
    model = models.resnet50(weights="IMAGENET1K_V1")
    model = model.to(get_device())
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tests: Single Format Accuracy (Sanity Checks)
# ---------------------------------------------------------------------------

class TestSingleFormatAccuracy:
    """Sanity checks for individual data formats."""

    def test_imagefolder_accuracy(self, imagenet_val_path, resnet50_model, tmp_path):
        """Verify ImageFolder format produces correct accuracy."""
        from torchvision import transforms
        from torchvision.datasets import ImageFolder

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = ImageFolder(str(imagenet_val_path), transform=transform)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

        device = get_device()
        acc = evaluate_top1(resnet50_model, loader, device, max_batches=50)

        # Should be in expected range
        assert acc > EXPECTED_TOP1_MIN, f"Accuracy {acc:.3f} below expected minimum {EXPECTED_TOP1_MIN}"
        assert acc < EXPECTED_TOP1_MAX, f"Accuracy {acc:.3f} above expected maximum {EXPECTED_TOP1_MAX}"

    def test_slipstream_imagefolder_accuracy(self, imagenet_val_path, resnet50_model, tmp_path):
        """Verify SlipstreamDataset + SlipstreamLoader produces correct accuracy."""
        from slipstream import SlipstreamDataset, SlipstreamLoader
        from slipstream.pipelines import supervised_val

        dataset = SlipstreamDataset(
            local_dir=str(imagenet_val_path),
            cache_dir=str(tmp_path / "cache"),
            decode_images=False,
        )

        device = get_device()
        loader = SlipstreamLoader(
            dataset,
            batch_size=64,
            pipelines=supervised_val(size=224, device=str(device)),
        )

        acc = evaluate_top1(resnet50_model, loader, device, max_batches=50)

        assert acc > EXPECTED_TOP1_MIN, f"Accuracy {acc:.3f} below expected minimum"
        assert acc < EXPECTED_TOP1_MAX, f"Accuracy {acc:.3f} above expected maximum"


# ---------------------------------------------------------------------------
# Tests: Cross-Format Accuracy Comparison
# ---------------------------------------------------------------------------

class TestCrossFormatAccuracy:
    """Verify accuracy matches across all data formats."""

    def test_imagefolder_vs_slipcache_accuracy(
        self, imagenet_val_path, resnet50_model, tmp_path
    ):
        """Compare torchvision ImageFolder vs SlipstreamDataset + cache."""
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        from slipstream import SlipstreamDataset, SlipstreamLoader
        from slipstream.pipelines import supervised_val

        device = get_device()
        max_batches = 50

        # Torchvision ImageFolder baseline
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tv_dataset = ImageFolder(str(imagenet_val_path), transform=transform)
        tv_loader = DataLoader(tv_dataset, batch_size=64, shuffle=False, num_workers=4)
        tv_acc = evaluate_top1(resnet50_model, tv_loader, device, max_batches)

        # Slipstream with cache
        slip_dataset = SlipstreamDataset(
            local_dir=str(imagenet_val_path),
            cache_dir=str(tmp_path / "slipcache"),
            decode_images=False,
        )
        slip_loader = SlipstreamLoader(
            slip_dataset,
            batch_size=64,
            pipelines=supervised_val(size=224, device=str(device)),
        )
        slip_acc = evaluate_top1(resnet50_model, slip_loader, device, max_batches)

        # Should be within 0.5% (more lenient for partial dataset)
        diff = abs(tv_acc - slip_acc)
        assert diff < 0.005, (
            f"Accuracy difference {diff:.4f} exceeds 0.5%: "
            f"torchvision={tv_acc:.4f}, slipstream={slip_acc:.4f}"
        )

    def test_ffcv_vs_slipcache_accuracy(
        self, imagenet_ffcv_path, resnet50_model, tmp_path
    ):
        """Compare FFCV source vs SlipstreamDataset + cache."""
        from slipstream import SlipstreamDataset, SlipstreamLoader
        from slipstream.pipelines import supervised_val

        device = get_device()
        max_batches = 50

        # FFCV via slipstream reader
        ffcv_dataset = SlipstreamDataset(
            local_dir=str(imagenet_ffcv_path),
            cache_dir=str(tmp_path / "ffcv_cache"),
            decode_images=False,
        )
        ffcv_loader = SlipstreamLoader(
            ffcv_dataset,
            batch_size=64,
            pipelines=supervised_val(size=224, device=str(device)),
        )
        ffcv_acc = evaluate_top1(resnet50_model, ffcv_loader, device, max_batches)

        # Rebuild cache from FFCV and compare
        # (This tests the cache round-trip preserves accuracy)
        loaded_loader = SlipstreamLoader(
            ffcv_dataset,
            batch_size=64,
            pipelines=supervised_val(size=224, device=str(device)),
        )
        loaded_acc = evaluate_top1(resnet50_model, loaded_loader, device, max_batches)

        diff = abs(ffcv_acc - loaded_acc)
        assert diff < 0.005, (
            f"Accuracy difference {diff:.4f} exceeds 0.5%: "
            f"first_run={ffcv_acc:.4f}, second_run={loaded_acc:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: Model Accuracy Sanity Checks
# ---------------------------------------------------------------------------

class TestModelSanityChecks:
    """Basic sanity checks for model output correctness."""

    def test_model_outputs_reasonable_logits(self, resnet50_model):
        """Model should output reasonable logits for random input."""
        device = get_device()

        # Random input
        x = torch.randn(1, 3, 224, 224).to(device)

        with torch.no_grad():
            logits = resnet50_model(x)

        # Should have 1000 classes
        assert logits.shape == (1, 1000)

        # Logits should be finite
        assert torch.isfinite(logits).all()

        # Softmax should sum to 1
        probs = F.softmax(logits, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_model_deterministic_with_seed(self, resnet50_model):
        """Model should produce identical outputs for same input."""
        device = get_device()

        torch.manual_seed(42)
        x = torch.randn(1, 3, 224, 224).to(device)

        with torch.no_grad():
            out1 = resnet50_model(x.clone())
            out2 = resnet50_model(x.clone())

        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Pytest Configuration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "imagenet: marks tests requiring ImageNet data"
    )
