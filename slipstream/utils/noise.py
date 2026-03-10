import torch


def power_law_noise_orig(size, alpha=2.0, out_channels=1, scale=(-1, 1), device='cpu', batch_size=None, generator=None, independent_channels=False):
    """
    Original power-law (1/f^α) noise implementation.

    Preserved as reference. Requires square, even-sized output.
    See :func:`power_law_noise` for the current implementation with
    non-square support and cleaner frequency grid math.

    Args:
        size: Size of the square output (must be even)
        alpha: Exponent controlling frequency emphasis (0 < alpha <= 3)
               α=0: white noise, α=1: pink noise, α=2: Brownian noise
        out_channels: Number of output channels
        scale: Tuple (min, max) for output range. Default (-1, 1). Use (0, 1) for [0,1] range.
               Set to None to skip scaling and return raw values.
        device: torch device
        batch_size: If provided, generates a batch of independent noise samples
        generator: Optional torch.Generator for reproducible noise
        independent_channels: If True, generate independent noise per channel
            (colored noise). If False (default), duplicate a single noise
            field across channels (grayscale noise).

    Returns:
        Tensor of shape (out_channels, size, size) or (batch_size, out_channels, size, size)
    """
    assert size % 2 == 0, "Size must be even"

    # Determine shape — include channels when generating independently
    if independent_channels and out_channels > 1:
        if batch_size is not None:
            shape = (batch_size, out_channels, size, size)
        else:
            shape = (out_channels, size, size)
    else:
        if batch_size is not None:
            shape = (batch_size, size, size)
        else:
            shape = (size, size)

    # 1. Random noise (Gaussian white noise)
    m = torch.randn(shape, device=device, generator=generator)

    # 2. Fourier transform and shift
    mf = torch.fft.fftshift(torch.fft.fft2(m), dim=(-2, -1))

    # 3. Create frequency filter (1/f^α)
    d = (torch.arange(size, device=device) - size//2 - 1) ** 2
    dd = torch.sqrt(d.unsqueeze(1) + d.unsqueeze(0))
    filt = dd ** (-alpha)
    filt[torch.isinf(filt)] = 1.0  # Handle DC component

    # 4. Apply filter to frequencies
    ff = mf * filt

    # 5. Inverse Fourier transform
    b = torch.fft.ifft2(torch.fft.ifftshift(ff, dim=(-2, -1)))
    b = b.real  # Take real part (imaginary should be ~0)

    # Scale to desired range
    if scale is not None:
        b_min = b.amin(dim=(-2, -1), keepdim=True)
        b_max = b.amax(dim=(-2, -1), keepdim=True)
        # Normalize to [0, 1] then scale to [scale[0], scale[1]]
        b = (b - b_min) / (b_max - b_min)
        b = b * (scale[1] - scale[0]) + scale[0]

    # Add channel dimension and repeat (only when not already independent)
    if independent_channels and out_channels > 1:
        pass  # channels already in the tensor
    else:
        if batch_size is not None:
            b = b.unsqueeze(1).expand(-1, out_channels, -1, -1)
        else:
            b = b.unsqueeze(0).expand(out_channels, -1, -1)

    return b


def power_law_noise(
    h,
    w=None,
    alpha=2.0,
    out_channels=1,
    scale=(-1, 1),
    device='cpu',
    batch_size=None,
    generator=None,
    independent_channels=False,
):
    """
    Generate power-law (1/f^α) noise.

    Uses ``torch.fft.fftfreq`` for frequency grid computation and supports
    non-square output sizes.

    Args:
        h: Output height (any positive int; even recommended for FFT speed).
        w: Output width.  Defaults to ``h`` (square).
        alpha: Exponent controlling frequency emphasis.
            α=0: white, α=1: pink, α=2: Brownian.
        out_channels: Number of output channels.
        scale: ``(min, max)`` for output range, or ``None`` to skip.
        device: Torch device.
        batch_size: If provided, generate a batch of independent samples.
        generator: Optional ``torch.Generator`` for reproducibility.
        independent_channels: If ``True``, generate independent noise per
            channel (colorful).  If ``False``, duplicate one field (grayscale).

    Returns:
        Tensor of shape ``(C, H, W)`` or ``(B, C, H, W)``.
    """
    if w is None:
        w = h

    # Determine generation shape
    gen_channels = out_channels if independent_channels else 1
    if batch_size is not None:
        shape = (batch_size, gen_channels, h, w)
    else:
        shape = (gen_channels, h, w)

    # 1. White noise → FFT
    white = torch.randn(shape, device=device, generator=generator)
    noise_fft = torch.fft.fftshift(torch.fft.fft2(white, dim=(-2, -1)), dim=(-2, -1))

    # 2. Frequency grid via fftfreq
    freq_y = torch.fft.fftshift(torch.fft.fftfreq(h, device=device)) * h
    freq_x = torch.fft.fftshift(torch.fft.fftfreq(w, device=device)) * w
    yy, xx = torch.meshgrid(freq_y, freq_x, indexing='ij')
    rho = torch.sqrt(xx ** 2 + yy ** 2)

    # 3. 1/f^alpha filter, DC = 0
    filt = rho ** (-alpha)
    filt[rho == 0] = 0.0  # zero DC — removes mean bias before scaling

    # Reshape for broadcasting
    if batch_size is not None:
        filt = filt.view(1, 1, h, w)
    else:
        filt = filt.view(1, h, w)

    # 4. Apply filter and inverse FFT
    filtered = noise_fft * filt
    noise = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1)).real

    # 5. Scale to desired range (per-sample normalization)
    if scale is not None:
        n_min = noise.amin(dim=(-2, -1), keepdim=True)
        n_max = noise.amax(dim=(-2, -1), keepdim=True)
        noise = (noise - n_min) / (n_max - n_min + 1e-8)
        noise = noise * (scale[1] - scale[0]) + scale[0]

    # 6. Expand non-independent channels (view, no memory copy)
    if not independent_channels and out_channels > 1:
        if batch_size is not None:
            noise = noise.expand(-1, out_channels, -1, -1)
        else:
            noise = noise.expand(out_channels, -1, -1)

    return noise
