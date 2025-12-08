import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence
from compresso.params.masked import MaskedParam


class MaskedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that uses MaskedParam internally.

    From the outside:
      - Same constructor signature (plus sparsity args)
      - Same forward: y = x @ W^T + b

    Inside:
      - Weight is managed by MaskedParam (dynamic top-k, schedule, rewind, freeze).
      - Bias is a plain dense learnable vector (not pruned).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k_target: int,
        bias: bool = True,
        k_schedule: Optional[Sequence[int]] = None,
        num_stages: int = 10,
        stability_window: int = 5,
        change_threshold: float = 0.01,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # --- init dense weight exactly like nn.Linear ----------------------
        # Create raw weight tensor (not a Parameter yet)
        weight = torch.empty(out_features, in_features)

        # Same as nn.Linear.reset_parameters:
        # kaiming_uniform_(weight, a=math.sqrt(5))
        a = math.sqrt(5)
        nn.init.kaiming_uniform_(weight, a=a)

        # --- wrap in MaskedParam ------------------------------------------
        # Import your MaskedParam (adjust the import to your project layout)
        # from your_module import MaskedParam
        self.mparam = MaskedParam(
            weight=weight,
            k_target=k_target,
            k_schedule=k_schedule,
            num_stages=num_stages,
            stability_window=stability_window,
            change_threshold=change_threshold,
        )

        # --- bias: follow nn.Linear exactly --------------------------------
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            # fan_in = in_features for a (out_features, in_features) matrix
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, in_features)
        returns: (batch_size, out_features)
        """
        # MaskedParam.__call__ -> MaskedParam.forward()
        # returns effective weight matrix (out_features, in_features)
        W_eff = self.mparam()
        return F.linear(x, W_eff, self.bias)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class MaskedEmbedding(nn.Module):
    """
    Embedding layer with MaskedParam-managed sparsity on the embedding table.

    Weight shape: (num_embeddings, embedding_dim)
    Sparsity is applied per-row over the embedding_dim:
      - each token embedding keeps k non-zero components (per row k)
      - k follows the schedule defined in MaskedParam

    Forward uses a *masked dense* weight via F.embedding, so:
      - behavior is identical to nn.Embedding from the outside
      - all sparsity + rewinding logic lives inside MaskedParam
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        k_target: int,
        k_schedule: Optional[Sequence[int]] = None,
        num_stages: int = 10,
        stability_window: int = 5,
        change_threshold: float = 0.01,
        padding_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        """
        Args:
            num_embeddings:
                Vocabulary size (number of rows).
            embedding_dim:
                Dimension of each embedding vector (number of columns).
            k_target:
                Final per-row k (non-zeros per embedding row).
            k_schedule / num_stages / stability_window / change_threshold:
                Passed down to MaskedParam (same semantics as for MaskedLinear).
            padding_idx:
                Optional index that will always produce a zero embedding.
                We enforce padding row = 0.0 in the initial weight.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = padding_idx

        # --- Initialize dense weight like nn.Embedding (simple version) ---
        # You can tweak this to exactly match PyTorch's init if you want.
        weight = torch.empty(self.num_embeddings, self.embedding_dim, **factory_kwargs)
        # Normal init (Embedding uses something similar; not critical here)
        nn.init.normal_(weight, mean=0.0, std=1.0)

        # Zero out padding row in the *initial* weight, if any
        if self.padding_idx is not None:
            if not (0 <= self.padding_idx < self.num_embeddings):
                raise ValueError(
                    f"padding_idx {self.padding_idx} is out of range "
                    f"[0, {self.num_embeddings})"
                )
            weight[self.padding_idx].zero_()

        # --- Wrap in MaskedParam (sparsity over embedding_dim per row) ---
        self.mparam = MaskedParam(
            weight=weight,
            k_target=k_target,
            k_schedule=k_schedule,
            num_stages=num_stages,
            stability_window=stability_window,
            change_threshold=change_threshold,
        )

        # Optional marker so you can detect this type robustly later
        self._masked_embedding = True

    @property
    def weight(self) -> torch.Tensor:
        """
        Expose underlying dense weight parameter (masked version is returned in forward).
        This mirrors nn.Embedding API (module.weight).
        """
        return self.mparam.weight

    def extra_repr(self) -> str:
        s = (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"k_target={self.mparam.k_target}"
        )
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        return s

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: LongTensor of shape (...), same as nn.Embedding.

        We:
          - get the *current* masked dense weight from MaskedParam
          - call F.embedding with that
        """
        # mparam() returns masked dense weight (respecting current k / mask_frozen)
        W_eff = self.mparam()  # shape: (num_embeddings, embedding_dim)

        return F.embedding(
            input,
            W_eff,
            padding_idx=self.padding_idx,
            # You can add more args later: max_norm, norm_type, etc.
        )