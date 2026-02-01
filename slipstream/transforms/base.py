"""Base classes for GPU batch augmentations with replay support for SSL."""

import torch


class BatchAugment:
    """Base class for GPU batch augmentations with replay.

    Subclasses implement before_call() to sample random parameters,
    and apply_last() to apply those parameters. __call__ does both.
    This enables SSL replay: call t(x) on view 1, then t.apply_last(y)
    on view 2 to apply the same augmentation.
    """

    def before_call(self, b: torch.Tensor, **kwargs) -> None:
        """Sample random parameters for this batch."""
        pass

    def last_params(self) -> dict:
        """Return stored params for replay."""
        return {}

    def apply_last(self, b: torch.Tensor) -> torch.Tensor:
        """Apply using stored params (replay mode for SSL)."""
        return b

    def __call__(self, b: torch.Tensor, **kwargs) -> torch.Tensor:
        self.before_call(b, **kwargs)
        return self.apply_last(b)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def apply_last(self, b):
        for t in self.transforms:
            b = t.apply_last(b)
        return b

    def __call__(self, b, replay=False):
        for t in self.transforms:
            b = t.apply_last(b) if replay else t(b)
        return b

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class RandomApply:
    """Randomly apply transforms with probability `p`."""

    def __init__(self, transforms, p, seed=None, device=None):
        self.transforms = transforms
        self.p = p
        self.do = None
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator("cpu" if device is None else device)
            self.rng.manual_seed(self.seed)

    def before_call(self, b, **kwargs):
        self.do = self.p == 1.0 or torch.rand(1, generator=self.rng).item() < self.p

    def last_params(self):
        return {"do": self.do}

    def apply_last(self, b):
        if self.last_params()["do"]:
            for t in self.transforms:
                b = t.apply_last(b)
        return b

    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        if self.do:
            for t in self.transforms:
                b = t(b)
        return b

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        for t in self.transforms:
            format_string += ",\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class MultiSample:
    """Performs transforms multiple times, returning multiple copies of the input.

    Args:
        transforms: List of transforms (or a Compose)
        num_copies: number of copies to produce
        return_input: whether to return the input as first output
        clone_input: whether to clone the input before each application
    """

    def __init__(self, transforms, num_copies, return_input=False, clone_input=True):
        self.transforms = transforms if isinstance(transforms, Compose) else Compose(transforms)
        self.num_copies = num_copies
        self.return_input = return_input
        self.clone_input = clone_input

    def __call__(self, b, replay=False):
        output = []
        if self.return_input:
            output.append(b)
        for _ in range(self.num_copies):
            output.append(self.transforms(b.clone() if self.clone_input else b))
        return tuple(output)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += f"num_copies={self.num_copies}"
        format_string += f", return_input={self.return_input}"
        format_string += f", clone_input={self.clone_input}, transforms="
        lines = [f"{line}" for line in self.transforms.__repr__().split("\n")]
        format_string += "\n".join(lines)
        return format_string
