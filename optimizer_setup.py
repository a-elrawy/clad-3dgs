"""Turn on dense fused Adam for the pinned trainer."""
import torch


def fused_adam(optimizer):
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError("--fused-adam requires the default dense Adam optimizer")
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not parameter.is_cuda or parameter.dtype != torch.float32:
                raise ValueError("--fused-adam requires CUDA float32 Gaussian parameters")
            step = optimizer.state.get(parameter, {}).get("step")
            if step is not None and (not torch.is_tensor(step) or step.dtype != torch.float32):
                raise ValueError("--fused-adam requires native float32 Adam step counters")
    options = dict(optimizer.defaults, fused=True, foreach=False)
    groups = [dict(group, fused=True, foreach=False) for group in optimizer.param_groups]
    selected = torch.optim.Adam(groups, **options)
    saved = optimizer.state_dict()
    for group in saved["param_groups"]:
        group.update(fused=True, foreach=False)
    # Fused Adam wants step counters on the parameter device.
    # Parameters, moments, and counter values stay as loaded.
    selected.load_state_dict(saved)
    return selected


def enable_fused_adam(model_type):
    setup = model_type.training_setup
    restore = model_type.restore

    def training_setup(model, *args, **kwargs):
        setup(model, *args, **kwargs)
        model.optimizer = fused_adam(model.optimizer)

    def restore_checkpoint(model, *args, **kwargs):
        restore(model, *args, **kwargs)
        model.optimizer = fused_adam(model.optimizer)

    model_type.training_setup = training_setup
    model_type.restore = restore_checkpoint
