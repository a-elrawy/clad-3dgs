"""First-order activation adjoints, applied by dense Adam."""

import math
import torch

NAMES = ("rotation", "scaling", "opacity")
ATTRS = ("_rotation", "_scaling", "_opacity")
OPTIONS = (
    "lr",
    "betas",
    "eps",
    "fused",
    "foreach",
    "weight_decay",
    "amsgrad",
    "maximize",
    "capturable",
    "differentiable",
)


class _ConsumeActivations(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rotation, scaling, opacity, owner):
        norm = rotation.norm(2, dim=1, keepdim=True)
        activated_scale, activated_opacity = scaling.exp(), opacity.sigmoid()
        ctx.save_for_backward(
            rotation, scaling, opacity, norm, activated_scale, activated_opacity
        )
        ctx.owner = owner
        ctx.set_materialize_grads(False)
        return (
            rotation / norm.clamp_min(1e-12).expand_as(rotation),
            activated_scale,
            activated_opacity,
        )

    @staticmethod
    def backward(ctx, *seeds):
        owner = ctx.owner
        if torch.is_grad_enabled() or owner.consumed:
            raise RuntimeError(
                "Activation consumption requires one first-order backward"
            )
        if any(seed is None for seed in seeds):
            raise RuntimeError(
                "Activation consumption requires all three output adjoints"
            )
        tensors = ctx.saved_tensors
        raw, saved = list(tensors[:3]), list(tensors[3:])
        owner.validate_live()
        owner.extension.consume_activations(
            raw,
            list(seeds),
            saved,
            [state["exp_avg"] for state in owner.states],
            [state["exp_avg_sq"] for state in owner.states],
            owner.rates,
            owner.bias1,
            owner.bias2,
        )
        for state in owner.states:
            state["step"].add_(1)
        torch.autograd.graph.increment_version(
            raw
            + [
                state[key]
                for state in owner.states
                for key in ("exp_avg", "exp_avg_sq")
            ]
        )
        owner.consumed = True
        return None, None, None, None


class _ActivationStep:
    def __init__(self, model, extension):
        self.model, self.optimizer, self.extension = model, model.optimizer, extension
        if type(self.optimizer) is not torch.optim.Adam:
            raise ValueError("unsupported optimizer")
        self.parameters = [getattr(model, name) for name in ATTRS]
        self.groups, self.states, self.values, self.versions = [], [], [], []
        self.options, self.state_versions = [], []
        for name, raw, width in zip(NAMES, self.parameters, (4, 3, 1)):
            groups = [g for g in self.optimizer.param_groups if g.get("name") == name]
            if len(groups) != 1:
                raise ValueError("Expected one optimizer group for " + name)
            group = groups[0]
            if (
                len(group["params"]) != 1
                or group["params"][0] is not raw
                or raw.grad is not None
                or not raw.requires_grad
                or not raw.is_cuda
                or raw.dtype != torch.float32
                or not raw.is_contiguous()
                or raw.ndim != 2
                or raw.shape != (self.parameters[0].shape[0], width)
                or raw.device != self.parameters[0].device
            ):
                raise ValueError("Unsupported activation ownership or layout")
            if (
                group["betas"] != (0.9, 0.999)
                or group["eps"] != 1e-15
                or group.get("fused") is not True
                or group.get("foreach")
                or group.get("weight_decay", 0) != 0
                or any(
                    group.get(k, False)
                    for k in ("amsgrad", "maximize", "capturable", "differentiable")
                )
            ):
                raise ValueError("Unsupported activation optimizer options")
            rate = group["lr"]
            if (
                not isinstance(rate, (float, int))
                or not math.isfinite(rate)
                or rate < 0
            ):
                raise ValueError("unsupported activation learning rate")
            state = self.optimizer.state.get(raw)
            if not state:
                state = dict(
                    step=torch.zeros((), device=raw.device),
                    exp_avg=torch.zeros_like(raw),
                    exp_avg_sq=torch.zeros_like(raw),
                )
            for key in ("exp_avg", "exp_avg_sq"):
                value = state[key]
                if (
                    value.shape != raw.shape
                    or value.device != raw.device
                    or value.dtype != raw.dtype
                    or not value.is_contiguous()
                ):
                    raise ValueError("Unsupported activation moment layout")
            step = state["step"]
            if (
                step.device != raw.device
                or step.dtype != torch.float32
                or step.numel() != 1
            ):
                raise ValueError("Unsupported activation counter")
            self.groups.append(group)
            self.states.append(state)
            self.values.append({k: state[k] for k in ("exp_avg", "exp_avg_sq", "step")})
            self.versions.append(raw._version)
            self.state_versions.append(
                {k: v._version for k, v in self.values[-1].items()}
            )
            self.options.append(tuple(group.get(k) for k in OPTIONS))
        writable = self.parameters + [
            state[k] for state in self.states for k in ("exp_avg", "exp_avg_sq", "step")
        ]
        regions = sorted(
            (v.data_ptr(), v.data_ptr() + v.numel() * v.element_size())
            for v in writable
            if v.numel()
        )
        if any(left[1] > right[0] for left, right in zip(regions, regions[1:])):
            raise ValueError("overlapping activation parameters or moments")
        # One D2H read for the three counters.
        counters = (
            torch.stack([s["step"].reshape(()) for s in self.states]).cpu().tolist()
        )
        if any(
            not math.isfinite(c) or c < 0 or c != int(c) or c >= 2**24 for c in counters
        ):
            raise ValueError("Invalid activation counters")
        for raw, state in zip(self.parameters, self.states):
            self.optimizer.state[raw] = state
        self.rates = [group["lr"] for group in self.groups]
        self.bias1 = [1 - 0.9 ** (int(c) + 1) for c in counters]
        self.bias2 = [math.sqrt(1 - 0.999 ** (int(c) + 1)) for c in counters]
        self.consumed = False

    def validate_live(self):
        if self.model.optimizer is not self.optimizer:
            raise RuntimeError("Activation optimizer was replaced")
        for attr, raw, group, state, values, version, options, state_versions in zip(
            ATTRS,
            self.parameters,
            self.groups,
            self.states,
            self.values,
            self.versions,
            self.options,
            self.state_versions,
        ):
            if (
                getattr(self.model, attr) is not raw
                or len(group["params"]) != 1
                or group["params"][0] is not raw
                or raw._version != version
                or raw.grad is not None
                or not raw.requires_grad
                or self.optimizer.state.get(raw) is not state
                or any(state[k] is not value for k, value in values.items())
                or any(state[k]._version != v for k, v in state_versions.items())
                or tuple(group.get(k) for k in OPTIONS) != options
                or not any(g is group for g in self.optimizer.param_groups)
            ):
                raise RuntimeError("Activation ownership changed after forward")


class DenseActivationConsumer:
    def __init__(self, extension):
        if not hasattr(extension, "consume_activations"):
            raise ValueError("Build the combined activation consumer first")
        self.extension = extension
        self.active = None
        self.last_reason = None

    def bind(
        self,
        model,
        policy,
        *,
        exclusive_use=False,
        gradient_processing=False,
        higher_order=False
    ):
        reason = "consumption disabled" if policy is None else policy.reason()
        if not torch.is_grad_enabled():
            reason = "evaluation without gradients"
        elif not exclusive_use:
            reason = "activation parameters have other consumers"
        elif gradient_processing:
            reason = "activation gradients require native processing"
        elif higher_order:
            reason = "higher-order derivatives require native backward"
        if reason is not None:
            self.last_reason = reason
            return None
        if self.active is not None:
            raise RuntimeError("finish the previous activation step first")
        try:
            frame = _ActivationStep(model, self.extension)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            self.last_reason = str(error)
            return None
        self.active = frame
        self.last_reason = None
        return _ConsumeActivations.apply(*frame.parameters, frame)

    def end_step(self):
        if self.active is not None and not self.active.consumed:
            raise RuntimeError("activation frame has no completed backward")
        self.active = None
