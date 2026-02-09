#!/usr/bin/env python
"""Model accuracy verification across all data formats.

This script verifies that model predictions are consistent across all supported
data formats (ImageFolder, LitData, FFCV, SlipCache). It uses ImageFolder as
the "gold standard" reference and compares per-image predictions.

Two levels of verification:
1. Aggregate accuracy: Top-1/Top-5 should match within 0.1%
2. Per-image agreement: Predictions should match exactly (or nearly so)

Usage:
    # Full verification (ResNet50 + AlexNet)
    python scripts/verify_model_accuracy.py

    # Quick test (first 1000 samples only)
    python scripts/verify_model_accuracy.py --quick

    # Specific model only
    python scripts/verify_model_accuracy.py --model resnet50

    # Skip SlipCache building (use existing caches)
    python scripts/verify_model_accuracy.py --skip-cache-build

Requirements:
    - GPU recommended (CPU is very slow)
    - AWS credentials for S3 access
    - ~10GB disk space for caches
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Data source paths
IMAGENET_VAL_S3_TAR = "s3://visionlab-datasets/imagenet1k-raw/val.tar.gz"
LITDATA_VAL_S3 = "s3://visionlab-datasets/imagenet1k/pre-processed/s256-l512-jpgbytes-q100-streaming/val/"
FFCV_VAL_S3 = (
    "s3://visionlab-datasets/imagenet1k/pre-processed/"
    "s256-l512-jpgbytes-q100-ffcv/imagenet1k-s256-l512-jpg-q100-cs100-val-7ac6386e.ffcv"
)

# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def extract_filename(path: str) -> str:
    """Extract canonical filename from path (e.g., 'n01440764/ILSVRC2012_val_00000293.JPEG')."""
    # Handle various path formats
    # LitData: 'val/n01440764/ILSVRC2012_val_00000293.JPEG'
    # ImageFolder: 'n01440764/ILSVRC2012_val_00000293.JPEG'
    # Just use class/filename as the canonical key
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"  # class/filename
    return parts[-1]


@dataclass
class FormatResult:
    """Results for a single format."""
    name: str
    top1_correct: int = 0
    top5_correct: int = 0
    total: int = 0
    predictions: list[int] = field(default_factory=list)
    filenames: list[str] = field(default_factory=list)  # For alignment
    inference_time: float = 0.0

    @property
    def top1_accuracy(self) -> float:
        return self.top1_correct / self.total if self.total > 0 else 0.0

    @property
    def top5_accuracy(self) -> float:
        return self.top5_correct / self.total if self.total > 0 else 0.0

    def get_prediction_map(self) -> dict[str, int]:
        """Return filename -> prediction mapping."""
        return dict(zip(self.filenames, self.predictions))


def get_model(name: str, device: torch.device) -> nn.Module:
    """Load a pretrained model."""
    import torchvision.models as models

    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown model: {name}")

    model = model.to(device)
    model.eval()
    return model


def get_val_transform() -> transforms.Compose:
    """Standard ImageNet validation transform."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ImageFolderDataset(Dataset):
    """Wrapper for SlipstreamImageFolder that applies transforms."""

    def __init__(self, reader: Any, transform: transforms.Compose):
        self.reader = reader
        self.transform = transform

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        import io
        from PIL import Image

        sample = self.reader[idx]
        img_bytes = sample["image"]
        label = sample["label"]
        path = sample.get("path", str(idx))

        # Decode and transform
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = self.transform(img)

        return img_tensor, label, extract_filename(path)


class LitDataDataset(Dataset):
    """Wrapper for StreamingReader that applies transforms."""

    def __init__(self, reader: Any, transform: transforms.Compose):
        self.reader = reader
        self.transform = transform

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        import io
        from PIL import Image

        sample = self.reader[idx]
        img_bytes = sample["image"]
        label = sample["label"]
        path = sample.get("path", str(idx))

        # Decode and transform
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = self.transform(img)

        return img_tensor, label, extract_filename(path)


class FFCVDataset(Dataset):
    """Wrapper for FFCVFileReader that applies transforms.

    Note: FFCV files typically don't store filenames, so we use index-based
    alignment. FFCV files are assumed to be in the same order as ImageFolder.
    """

    def __init__(self, reader: Any, transform: transforms.Compose):
        self.reader = reader
        self.transform = transform

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        import io
        from PIL import Image

        sample = self.reader[idx]
        img_bytes = sample["image"]
        label = sample["label"]
        # FFCV doesn't store paths, use index as identifier
        # This works because FFCV is created in same order as ImageFolder
        path = sample.get("path", f"__idx__{idx}")

        # Decode and transform
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = self.transform(img)

        return img_tensor, label, path


class SlipCacheDataset(Dataset):
    """Wrapper for OptimizedCache that applies transforms."""

    def __init__(self, cache: Any, transform: transforms.Compose):
        self.cache = cache
        self.transform = transform
        self.image_format = cache.get_image_format("image")
        self.has_path = "path" in cache.fields

    def __len__(self) -> int:
        return self.cache.num_samples

    def _get_image_bytes(self, idx: int) -> bytes:
        """Extract raw image bytes from cache."""
        storage = self.cache.fields["image"]
        meta = storage._metadata[idx]
        ptr = int(meta["data_ptr"])
        size = int(meta["data_size"])
        return bytes(storage._data_mmap[ptr:ptr + size])

    def _get_path(self, idx: int) -> str:
        """Get path for sample, or index-based identifier if no path field."""
        if self.has_path:
            # StringStorage uses _offsets array with (offset, length) pairs
            path_storage = self.cache.fields["path"]
            offset, length = path_storage._offsets[idx]
            path_bytes = bytes(path_storage._data_mmap[offset:offset + length])
            return path_bytes.decode("utf-8")
        return f"__idx__{idx}"

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        import io
        from PIL import Image

        img_bytes = self._get_image_bytes(idx)
        label = int(self.cache.fields["label"]._data[idx])
        path = self._get_path(idx)

        if self.image_format == "yuv420":
            # YUV420 requires special decoding
            from slipstream.decoders.yuv420_decoder import YUV420NumbaBatchDecoder

            h, w = self.cache.get_image_dims("image", idx)
            decoder = YUV420NumbaBatchDecoder(num_threads=1)
            try:
                yuv_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                batch_data = np.zeros((1, len(yuv_arr)), dtype=np.uint8)
                batch_data[0, :len(yuv_arr)] = yuv_arr
                sizes = np.array([len(img_bytes)], dtype=np.uint64)
                heights = np.array([h], dtype=np.uint32)
                widths = np.array([w], dtype=np.uint32)
                rgb = decoder.decode_batch(batch_data, sizes, heights, widths)[0]
                img = Image.fromarray(rgb)
            finally:
                decoder.shutdown()
        else:
            # JPEG format
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img_tensor = self.transform(img)
        return img_tensor, label, extract_filename(path)


def run_inference(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    max_samples: int | None = None,
    desc: str = "Inference",
) -> FormatResult:
    """Run inference and collect predictions with filenames for alignment."""
    result = FormatResult(name=desc)

    # Limit samples if requested
    if max_samples is not None and max_samples < len(dataset):
        indices = list(range(max_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            # Unpack batch - now includes filenames
            images, labels, filenames = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Top-1 predictions
            _, preds = outputs.topk(1, dim=1)
            preds = preds.squeeze(1)
            result.predictions.extend(preds.cpu().tolist())
            result.filenames.extend(filenames)

            # Top-1 accuracy
            result.top1_correct += (preds == labels).sum().item()

            # Top-5 accuracy
            _, top5_preds = outputs.topk(5, dim=1)
            top5_correct = top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds))
            result.top5_correct += top5_correct.any(dim=1).sum().item()

            result.total += labels.size(0)

    result.inference_time = time.time() - start_time
    return result


def compare_predictions(
    gold: FormatResult,
    other: FormatResult,
) -> dict[str, Any]:
    """Compare predictions between gold standard and another format.

    Uses filename-based alignment to handle shuffled datasets like LitData.
    For formats without filenames (using __idx__), falls back to position-based.
    """
    gold_map = gold.get_prediction_map()
    other_map = other.get_prediction_map()

    # Check if we're using index-based alignment (no real filenames)
    uses_index = any(f.startswith("__idx__") for f in other_map.keys())

    if uses_index:
        # Index-based comparison (assumes same order)
        common_keys = sorted(gold_map.keys() & other_map.keys())
        if not common_keys:
            # Fallback to position-based for __idx__ keys
            agreements = sum(
                g == o for g, o in zip(gold.predictions, other.predictions)
            )
            disagreements = []
            for idx, (g, o) in enumerate(zip(gold.predictions, other.predictions)):
                if g != o:
                    disagreements.append({"index": idx, "gold": g, "other": o})
            return {
                "total": len(gold.predictions),
                "agreements": agreements,
                "agreement_rate": agreements / len(gold.predictions),
                "disagreements": disagreements[:20],
                "num_disagreements": len(disagreements),
                "alignment": "position",
            }
    else:
        # Filename-based comparison
        common_keys = sorted(gold_map.keys() & other_map.keys())

    if not common_keys:
        return {
            "total": 0,
            "agreements": 0,
            "agreement_rate": 0.0,
            "disagreements": [],
            "num_disagreements": 0,
            "alignment": "none",
            "error": "No common filenames found",
        }

    agreements = 0
    disagreements = []
    for filename in common_keys:
        g = gold_map[filename]
        o = other_map[filename]
        if g == o:
            agreements += 1
        else:
            disagreements.append({"filename": filename, "gold": g, "other": o})

    return {
        "total": len(common_keys),
        "agreements": agreements,
        "agreement_rate": agreements / len(common_keys),
        "disagreements": disagreements[:20],  # First 20 only
        "num_disagreements": len(disagreements),
        "alignment": "filename",
        "gold_total": len(gold_map),
        "other_total": len(other_map),
    }


def load_imagefolder(verbose: bool = True) -> Any:
    """Load ImageFolder reader."""
    from slipstream.readers.imagefolder import open_imagefolder

    if verbose:
        print("Loading ImageFolder (gold standard)...")
    return open_imagefolder(IMAGENET_VAL_S3_TAR, verbose=verbose)


def load_litdata(verbose: bool = True) -> Any:
    """Load LitData reader."""
    from slipstream.readers.streaming import StreamingReader

    if verbose:
        print("Loading LitData...")
    return StreamingReader(remote_dir=LITDATA_VAL_S3, max_cache_size="50GB")


def load_ffcv(verbose: bool = True) -> Any:
    """Load FFCV reader."""
    from slipstream.readers.ffcv import FFCVFileReader

    if verbose:
        print("Loading FFCV...")
    return FFCVFileReader(FFCV_VAL_S3, verbose=verbose)


def build_or_load_slipcache(
    reader: Any, name: str, verbose: bool = True, rebuild: bool = False
) -> Any:
    """Build or load SlipCache for a reader."""
    import shutil
    from slipstream.cache import OptimizedCache

    cache_path = reader.cache_path
    manifest_path = cache_path / ".slipstream" / "manifest.json"

    # Delete existing cache if rebuild requested
    if rebuild and cache_path.exists():
        if verbose:
            print(f"Deleting existing SlipCache for {name}: {cache_path}")
        shutil.rmtree(cache_path)

    if manifest_path.exists():
        if verbose:
            print(f"Loading existing SlipCache for {name}: {cache_path}")
        return OptimizedCache.load(cache_path, verbose=verbose)

    if verbose:
        print(f"Building SlipCache for {name}: {cache_path}")
    return OptimizedCache.build(reader, output_dir=cache_path, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(description="Verify model accuracy across formats")
    parser.add_argument("--model", choices=["resnet50", "alexnet", "both"], default="both",
                        help="Model to use for verification")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with first 1000 samples only")
    parser.add_argument("--skip-cache-build", action="store_true",
                        help="Skip SlipCache building (use existing caches)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--rebuild-caches", action="store_true",
                        help="Delete and rebuild all SlipCaches (useful if caches are stale)")
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cpu":
        print("WARNING: Running on CPU will be very slow. GPU recommended.")

    max_samples = 1000 if args.quick else None
    if args.quick:
        print(f"Quick mode: using first {max_samples} samples only")

    # Determine which models to run
    models_to_run = ["resnet50", "alexnet"] if args.model == "both" else [args.model]

    # Load all data sources
    print("\n" + "=" * 60)
    print("LOADING DATA SOURCES")
    print("=" * 60)

    transform = get_val_transform()

    # ImageFolder (gold standard)
    imagefolder_reader = load_imagefolder()
    imagefolder_ds = ImageFolderDataset(imagefolder_reader, transform)

    # LitData
    litdata_reader = load_litdata()
    litdata_ds = LitDataDataset(litdata_reader, transform)

    # FFCV
    ffcv_reader = load_ffcv()
    ffcv_ds = FFCVDataset(ffcv_reader, transform)

    # SlipCaches (build or load)
    slipcache_datasets = {}
    if not args.skip_cache_build:
        print("\n" + "=" * 60)
        print("BUILDING/LOADING SLIPCACHES")
        print("=" * 60)

        for name, reader in [
            ("ImageFolder", imagefolder_reader),
            ("LitData", litdata_reader),
            ("FFCV", ffcv_reader),
        ]:
            cache = build_or_load_slipcache(reader, name, rebuild=args.rebuild_caches)
            slipcache_datasets[f"SlipCache-{name}"] = SlipCacheDataset(cache, transform)

    # All datasets to evaluate
    datasets = {
        "ImageFolder": imagefolder_ds,
        "LitData": litdata_ds,
        "FFCV": ffcv_ds,
        **slipcache_datasets,
    }

    # Results storage
    all_results = {}

    for model_name in models_to_run:
        print("\n" + "=" * 60)
        print(f"RUNNING {model_name.upper()}")
        print("=" * 60)

        model = get_model(model_name, device)

        results: dict[str, FormatResult] = {}

        # Run inference on all formats
        for ds_name, dataset in datasets.items():
            print(f"\n{ds_name}:")
            result = run_inference(
                model=model,
                dataset=dataset,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=max_samples,
                desc=ds_name,
            )
            result.name = ds_name
            results[ds_name] = result

            print(f"  Top-1: {result.top1_accuracy * 100:.2f}%")
            print(f"  Top-5: {result.top5_accuracy * 100:.2f}%")
            print(f"  Time: {result.inference_time:.1f}s")

        # Compare all formats to gold standard (ImageFolder)
        print("\n" + "-" * 40)
        print("COMPARISON TO GOLD STANDARD (ImageFolder)")
        print("-" * 40)

        gold = results["ImageFolder"]
        comparisons = {}

        for ds_name, result in results.items():
            if ds_name == "ImageFolder":
                continue

            comparison = compare_predictions(gold, result)
            comparisons[ds_name] = comparison

            status = "✅" if comparison["agreement_rate"] > 0.999 else "⚠️"
            alignment = comparison.get("alignment", "position")
            print(f"\n{ds_name}:")
            print(f"  Alignment: {alignment}")
            print(f"  Agreement: {comparison['agreements']}/{comparison['total']} "
                  f"({comparison['agreement_rate'] * 100:.3f}%) {status}")
            print(f"  Top-1 diff: {abs(gold.top1_accuracy - result.top1_accuracy) * 100:.3f}%")

            if comparison["num_disagreements"] > 0:
                print(f"  First disagreements: {comparison['disagreements'][:5]}")

        # Summary
        print("\n" + "-" * 40)
        print("SUMMARY")
        print("-" * 40)

        accuracies = [r.top1_accuracy for r in results.values()]
        max_diff = (max(accuracies) - min(accuracies)) * 100

        print(f"\nTop-1 Accuracy Range: {min(accuracies) * 100:.2f}% - {max(accuracies) * 100:.2f}%")
        print(f"Max Difference: {max_diff:.3f}%")

        if max_diff < 0.1:
            print("✅ PASS: All formats within 0.1% accuracy")
        else:
            print("⚠️ WARNING: Accuracy difference exceeds 0.1%")

        # Store results for this model
        all_results[model_name] = {
            "results": {
                name: {
                    "top1_accuracy": r.top1_accuracy,
                    "top5_accuracy": r.top5_accuracy,
                    "total": r.total,
                    "inference_time": r.inference_time,
                }
                for name, r in results.items()
            },
            "comparisons": {
                name: {
                    "agreement_rate": c["agreement_rate"],
                    "num_disagreements": c["num_disagreements"],
                }
                for name, c in comparisons.items()
            },
            "max_accuracy_diff": max_diff,
        }

    # Save results to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    all_pass = True
    for model_name, model_results in all_results.items():
        max_diff = model_results["max_accuracy_diff"]
        status = "✅ PASS" if max_diff < 0.1 else "❌ FAIL"
        print(f"{model_name}: {status} (max diff: {max_diff:.3f}%)")
        if max_diff >= 0.1:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
