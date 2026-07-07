import torch
import torch.nn as nn
from typing import Optional, Set, Tuple, List
from compresso.layers.coo import CooSparseLinear, CooSparseEmbedding
from compresso.layers.masked import MaskedLinear, MaskedEmbedding
from compresso.layers.gated import GatedMaskedParam, GatedMaskedAttentionParam
from compresso.params.coo import CooSparseParam
from compresso.params.masked import MaskedParam, SharedMaskedParam

def masked_parameters(model: nn.Module) -> List["MaskedParam"]:
    """
    Yield all MaskedParam instances inside the model.
    """
    for m in model.modules():
        if isinstance(m, MaskedParam) or isinstance(m, SharedMaskedParam) or isinstance(m, GatedMaskedParam) or isinstance(m, GatedMaskedAttentionParam):
            yield m

def convert_masked_to_coo_inplace(model: nn.Module):
    """
    Recursively walk `model` and replace all MaskedLinear modules
    with CooSparseLinear built from their final MaskedParam.
    """
    #print("Running convert_masked_to_coo_inplace")
    n = 0
    for name, child in list(model.named_children()):
        # Recurse first
        n+=convert_masked_to_coo_inplace(child)

        # Then handle this child
        if isinstance(child, MaskedLinear):
            mparam = child.mparam   # your MaskedParam instance
            if mparam.mask_frozen:
                # 1) Build CooSparseParam from final mask + weights
                coo_param = mparam.maskedparam_to_coo()  # function we sketched before

                # 2) Create new sparse linear with same bias
                bias = child.bias
                new_layer = CooSparseLinear(coo_param, bias=bias)

                # 3) Replace in parent
                setattr(model, name, new_layer)
                n+=1
        
        if isinstance(child, MaskedEmbedding):
            mparam = child.mparam   # your MaskedParam instance
            #print(mparam)
            if mparam.k_current == mparam.k_target:
                # 1) Build CooSparseParam from final mask + weights
                coo_param = mparam.maskedparam_to_coo()  # function we sketched before
                new_layer = CooSparseEmbedding(coo_param, padding_idx=child.padding_idx)
                # 3) Replace in parent
                setattr(model, name, new_layer)
                n+=1
    return n

@torch.no_grad()
def compact_gated_modules(
    model: nn.Module,
    *,
    only_gate_ids: Optional[Set[int]] = None,
    require_gate_attr: str = "gate",
    verbose: bool = False,
) -> int:
    """
    Traverse the whole model, find submodules that contain a gate (default attr name 'gate'),
    and replace that submodule with gate.compact() output.

    - Works regardless of where BlockMLP lives (transformer.h, nested ModuleLists, etc.).
    - Replacement happens in the parent via setattr.

    Args:
        model:
            Root module.
        only_gate_ids:
            If provided, only compact modules whose gate satisfies id(gate) in this set.
            (Useful for "compact after rewind only if advanced".)
        require_gate_attr:
            Attribute name on the module that holds the gate (default 'gate').
        verbose:
            Print replacements.

    Returns:
        Number of modules compacted (replaced).
    """

    # Build mapping: module full-name -> module instance
    name_to_module = dict(model.named_modules())

    # We must avoid mutating while iterating name_to_module directly, so iterate over a snapshot list.
    items = list(model.named_modules())

    compacted = 0

    for full_name, sub in items:
        if full_name == "":
            continue  # root

        # gate must exist as an attribute on the submodule
        gate = getattr(sub, require_gate_attr, None)
        if gate is None:
            continue

        # Optional filtering: only compact gates that were marked "advanced" by controller
        if only_gate_ids is not None and id(gate) not in only_gate_ids:
            continue

        # Safety: the gate must have compact()
        if not hasattr(gate, "compact"):
            continue

        # If you want: require mask to be already set by rewind before compaction
        # (Do NOT recompute in here unless you mean to.)
        if hasattr(gate, "mask_frozen"):
            gate.mask_frozen = True  # optional; sub will be replaced anyway

        new_module = gate.compact()

        # Replace submodule in its parent
        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
            parent = name_to_module[parent_name]
        else:
            parent = model
            child_name = full_name

        # setattr works for normal submodules, ModuleList elements are stored under numeric names
        # like '0', '1', ... which setattr also handles because ModuleList registers them as attributes.
        setattr(parent, child_name, new_module)

        compacted += 1
        if verbose:
            print(f"[compact] Replaced '{full_name}' -> {type(new_module).__name__}")

    return compacted

@torch.no_grad()
def compact_all_gated_mlps_after_rewind(model: nn.Module) -> int:
    """
    Find any module that looks like a BlockMLP (has fc1, fc2, act, gate)
    where gate is a GatedMaskedMLPParam that just advanced on the last rewind,
    and compact it in-place *without removing the gate* (we spawn a new gate).

    Returns number of MLPs compacted.
    """
    n = 0
    for m in model.modules():
        gate = getattr(m, "gate", None)
        fc1 = getattr(m, "fc1", None)
        fc2 = getattr(m, "fc2", None)

        if gate is None or fc1 is None or fc2 is None:
            continue
        if not isinstance(gate, GatedMaskedParam):
            continue

        if not getattr(gate, "_just_advanced", False):
            continue  # only compact when we actually moved to a new stage

        fc1_new, fc2_new, gate_new = gate.spawn_compacted_gate()

        if gate.k_current==gate.k_next:
            print("Final k reached for gate.")
        # swap linears + gate in the owning module (BlockMLP)
        m.fc1 = fc1_new
        m.fc2 = fc2_new
        m.gate = gate_new

        n += 1

    return n