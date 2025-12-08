import torch
import torch.nn as nn
from typing import List
from compresso.layers.coo import CooSparseLinear, CooSparseEmbedding
from compresso.layers.masked import MaskedLinear, MaskedEmbedding
from compresso.params.coo import CooSparseParam
from compresso.params.masked import MaskedParam

def masked_parameters(model: nn.Module) -> List["MaskedParam"]:
    """
    Yield all MaskedParam instances inside the model.
    """
    for m in model.modules():
        if isinstance(m, MaskedParam):
            yield m

def convert_masked_to_coo_inplace(model: nn.Module):
    """
    Recursively walk `model` and replace all MaskedLinear modules
    with CooSparseLinear built from their final MaskedParam.
    """
    for name, child in list(model.named_children()):
        # Recurse first
        convert_masked_to_coo_inplace(child)

        # Then handle this child
        if isinstance(child, MaskedLinear):
            mparam = child.mparam   # your MaskedParam instance

            # 1) Build CooSparseParam from final mask + weights
            coo_param = mparam.maskedparam_to_coo()  # function we sketched before

            # 2) Create new sparse linear with same bias
            bias = child.bias
            new_layer = CooSparseLinear(coo_param, bias=bias)

            # 3) Replace in parent
            setattr(model, name, new_layer)