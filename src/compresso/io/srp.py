from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from compresso.params.srp import SRPTensor


PathLike = Union[str, Path]


def save_srp_tensor(path: PathLike, srp: SRPTensor) -> None:
    """Save SRPTensor payload with torch.save."""
    torch.save(srp.to_dict(), Path(path))


def load_srp_tensor(path: PathLike, map_location=None, *, validate: bool = True) -> SRPTensor:
    """Load SRPTensor payload produced by :func:`save_srp_tensor`."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    return SRPTensor.from_dict(payload, validate=validate)
