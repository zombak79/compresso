import torch
import torch.nn as nn
from compresso.utils.helpers import masked_parameters

class SparsityController:
    """
    Global dispatcher for MaskedParam-based pruning.

    Responsibilities:
      - Keep track of global step.
      - Every `mask_update_interval` steps, call `step_mask()` on all MaskedParams.
      - If any MaskedParam's stage is completed, call `rewind()` on all of them
        and signal that the optimizer should be reset.
      - Once all MaskedParams have `schedule_done=True`, optionally freeze masks
        and stop scheduling.
    """

    def __init__(
        self,
        model: nn.Module,
        mask_update_interval: int = 10,
        freeze_at_schedule_end: bool = True,
        method = "all", # or one
    ):
        self.model = model
        #self.init_state = copy.deepcopy(model.state_dict())
        
        self.mask_update_interval = int(mask_update_interval)
        self.freeze_at_schedule_end = bool(freeze_at_schedule_end)
        
        self.global_step: int = 0
        self.phase: str = "mask_search"  # or "final"
        self.num_restarts: int = 0
        self.method = method

    def step(self) -> dict:
        """
        Call this *once per optimizer step*.

        Returns:
            info dict with:
              - 'rewind_triggered': bool, whether a global rewind happened
              - 'all_schedules_done': bool, whether all MaskedParams finished schedule
        """
        self.global_step += 1
        
        info = {
            "rewind_triggered": False,
            "all_schedules_done": False,
            "phase": self.phase
        }

        # Once we are in final phase, do nothing here
        if self.phase != "mask_search":
            return info

        # Only update masks every mask_update_interval steps
        if self.global_step % self.mask_update_interval != 0:
            return info

        # 1) Step masks for all MaskedParams
        any_stage_completed = False
        all_schedules_done = True
        all_stages_completed = True
        check_if_any_masked_params_present=False
        
        for mp in masked_parameters(self.model):
            check_if_any_masked_params_present=True
            delta = mp.step_mask()  # average unstable_fraction
            # mp.stage_completed is set inside step_mask
            if mp.stage_completed:
                any_stage_completed = True
            else:
                all_stages_completed = False
            if not mp.schedule_done:
                all_schedules_done = False
        
        if not check_if_any_masked_params_present:
            # Nothing to do
            return info

        # 2) If at least one param completed its current stage -> rewind all
        if all_stages_completed and self.method=="all":
            self.num_restarts += 1
            #self.model.load_state_dict(self.init_state)
            for mp in masked_parameters(self.model):
                stats = mp.rewind()
                # you can log stats here if you want
                print(f"[SparsityController] Rewind: {stats}")
            info["rewind_triggered"] = True
        
        
        if any_stage_completed and self.method=="one":
            self.num_restarts += 1
            #self.model.load_state_dict(self.init_state)
            for mp in masked_parameters(self.model):
                stats = mp.rewind()
                # you can log stats here if you want
                print(f"[SparsityController] Rewind: {stats}")
            info["rewind_triggered"] = True

        # 3) If all schedules done -> optionally freeze masks and switch to final
        if all_schedules_done:
            info["all_schedules_done"] = True
            if self.freeze_at_schedule_end:
                for mp in masked_parameters(self.model):
                    mp.freeze_mask()
            self.phase = "final"

        stats=[]
        for mp in masked_parameters(self.model):
            stats.append(mp.get_stats()["last_num_changes"])
        info["num_changes"]=stats
        self.num_changes=stats
        return info