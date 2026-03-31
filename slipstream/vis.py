"""Visualization helpers for slipstream batches."""

import numpy as np
import torch


def show_batch(batch, view_names=None, n_cols=8, suptitle=None,
               row_labels=None, col_labels=None, mean=None, std=None):
    """Show batch tensors as a grid of images.

    Args:
        batch: One of:
            - dict: batch dict from loader; use ``view_names`` to pick rows.
            - [B, C, H, W] tensor: single batch shown as one row.
            - list of [B, C, H, W] tensors: each tensor is a row.
        view_names: List of keys to extract from a dict batch (one row each).
            Also used as default ``row_labels`` when ``row_labels`` is None.
        n_cols: Max images per row (default 8).
        suptitle: Figure title.
        row_labels: Labels for each row (left margin). Defaults to
            ``view_names`` when batch is a dict.
        col_labels: Labels for each column (top).
        mean: Per-channel mean for denormalization (list or tuple of floats).
        std: Per-channel std for denormalization (list or tuple of floats).

    Example::

        from slipstream import show_batch, IMAGENET_MEAN, IMAGENET_STD

        # Dict batch from loader with named views
        show_batch(batch, ['global_view', 'local_view'],
                   mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Single tensor
        show_batch(images, suptitle='My batch')

        # List of tensors with custom labels
        show_batch([view1, view2], row_labels=['large', 'small'])
    """
    import matplotlib.pyplot as plt

    # ── Normalize input to list-of-tensors ──
    if isinstance(batch, dict):
        if view_names is None:
            raise ValueError("view_names is required when batch is a dict")
        images = [batch[k] for k in view_names]
        if row_labels is None:
            row_labels = list(view_names)
    elif isinstance(batch, torch.Tensor) and batch.ndim == 4:
        images = [batch]
    elif isinstance(batch, list):
        images = batch
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    n_rows = len(images)
    n_cols = min(n_cols, images[0].shape[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    for r, row_batch in enumerate(images):
        for c in range(n_cols):
            img = row_batch[c].clone()
            if mean is not None and std is not None:
                for ch in range(min(3, img.shape[0])):
                    img[ch] = img[ch] * std[ch] + mean[ch]
            img = img[:3].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
            axes[r, c].imshow(img)
            axes[r, c].axis('off')
        if row_labels:
            axes[r, 0].set_ylabel(row_labels[r], fontsize=10, rotation=0,
                                   labelpad=60, va='center')
    if col_labels:
        for c in range(min(n_cols, len(col_labels))):
            axes[0, c].set_title(col_labels[c], fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def show_rgba(rgba_batch, n_cols=8, suptitle=None, row_labels=None):
    """Show RGBA numpy arrays [B, H, W, 4] with alpha overlay.

    Composites each RGBA image onto a light gray checkerboard so that
    transparent regions are visible.

    Args:
        rgba_batch: [B, H, W, 4] numpy array, or list of such arrays
            (each becomes a row).
        n_cols: Max images per row (default 8).
        suptitle: Figure title.
        row_labels: Labels for each row (left margin).
    """
    import matplotlib.pyplot as plt

    if isinstance(rgba_batch, np.ndarray) and rgba_batch.ndim == 4:
        rgba_batch = [rgba_batch]
    n_rows = len(rgba_batch)
    n_cols = min(n_cols, rgba_batch[0].shape[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    for r, batch in enumerate(rgba_batch):
        for c in range(n_cols):
            h, w = batch.shape[1], batch.shape[2]
            checker = np.zeros((h, w, 3), dtype=np.uint8) + 200
            checker[::16, :, :] = 220
            checker[:, ::16, :] = 220
            alpha = batch[c, :, :, 3:4].astype(np.float32) / 255.0
            rgb = batch[c, :, :, :3].astype(np.float32)
            composite = (alpha * rgb + (1 - alpha) * checker).astype(np.uint8)
            axes[r, c].imshow(composite)
            axes[r, c].axis('off')
        if row_labels:
            axes[r, 0].set_ylabel(row_labels[r], fontsize=10, rotation=0,
                                   labelpad=60, va='center')
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
