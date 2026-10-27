"""Research integration with dense Graphdeco Adam; no optimizer state cache."""

from dataclasses import dataclass
import math
from types import SimpleNamespace
import torch


class ConsumerRender(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, means, screen, dc, rest, opacity, scale, rotation, settings, owner
    ):
        empty = torch.empty(0, device=means.device)
        direct = getattr(owner, "direct_inputs", False)
        sh = empty if direct else torch.cat((dc, rest), 1)
        args = (
            settings.bg,
            means,
            empty,
            opacity,
            scale,
            rotation,
            settings.scale_modifier,
            empty,
            settings.viewmatrix,
            settings.projmatrix,
            settings.tanfovx,
            settings.tanfovy,
            settings.image_height,
            settings.image_width,
            sh,
            settings.sh_degree,
            settings.campos,
            settings.prefiltered,
            settings.antialiasing,
            settings.debug,
        )
        if direct:
            args = args[:14] + (dc, rest) + args[15:]
            function = owner.extension.rasterize_gaussians_direct
        else:
            function = owner.extension.rasterize_gaussians
        count, color, radii, geom, binning, image, depth = function(*args)
        ctx.settings = settings
        ctx.owner = owner
        ctx.count = count
        ctx.save_for_backward(
            means, opacity, scale, rotation, radii, sh, geom, binning, image, dc, rest
        )
        return color, radii, depth

    @staticmethod
    def backward(ctx, seed, _, depth_seed):
        if getattr(ctx.owner, "native_only", False) and torch.is_grad_enabled():
            raise RuntimeError("Direct SH inputs support first-order backward only")
        means, opacity, scale, rotation, radii, sh, geom, binning, image, dc, rest = (
            ctx.saved_tensors
        )
        s = ctx.settings
        o = ctx.owner
        empty = torch.empty(0, device=means.device)
        args = (
            s.bg,
            means,
            radii,
            empty,
            opacity,
            scale,
            rotation,
            s.scale_modifier,
            empty,
            s.viewmatrix,
            s.projmatrix,
            s.tanfovx,
            s.tanfovy,
            seed.contiguous(),
            depth_seed.contiguous(),
            sh,
            s.sh_degree,
            s.campos,
            geom,
            ctx.count,
            binning,
            image,
            s.antialiasing,
            s.debug,
        )
        if getattr(o, "native_only", False):
            grads = o.extension.rasterize_gaussians_backward_direct(
                *(args[:15] + (dc, rest) + args[16:])
            )
            gm2, gcol, gopacity, gm, gcov, gsh, gs, grot = grads
            return gm, gm2, gsh[:, :1], gsh[:, 1:], gopacity, gs, grot, None, None
        a, b = o.moments["dc"]
        c, d = o.moments["rest"]
        step = o.iteration
        function = (
            o.extension.rasterize_gaussians_backward_consume_direct
            if getattr(o, "direct_inputs", False)
            else o.extension.rasterize_gaussians_backward_consume
        )
        grads = function(
            *args,
            o.params["dc"],
            o.params["rest"],
            a,
            b,
            c,
            d,
            1 - 0.9**step,
            math.sqrt(1 - 0.999**step),
        )
        gm2, gcol, gopacity, gm, gcov, gsh, gs, grot = grads
        return gm, gm2, None, None, gopacity, gs, grot, None, None


def render_direct_native(
    extension, means, screen, dc, rest, opacity, scale, rotation, settings
):
    """Use direct inputs with native SH backward as a matched layout control."""
    owner = SimpleNamespace(extension=extension, direct_inputs=True, native_only=True)
    return ConsumerRender.apply(
        means, screen, dc, rest, opacity, scale, rotation, settings, owner
    )


@dataclass(frozen=True)
class StepPolicy:
    optimizer_step: bool = True
    topology_change: bool = False
    observation_before_step: bool = False
    single_backward: bool = True
    unscaled_gradients: bool = True
    exclusive_sh_use: bool = True

    def reason(self):
        if not self.optimizer_step:
            return "optimizer step skipped"
        if self.topology_change:
            return "topology changes before optimizer step"
        if self.observation_before_step:
            return "parameters observed before optimizer step"
        if not self.single_backward:
            return "gradient accumulation or multiple backward calls"
        if not self.unscaled_gradients:
            return "scaled gradients require native optimizer processing"
        if not self.exclusive_sh_use:
            return "SH parameters used by another loss or consumer"
        return None


class BoundStep:
    def __init__(self, groups, states, extension):
        self.groups = groups
        self.states = states
        self.extension = extension
        self.params = {
            key: group["params"][0] for key, group in zip(["dc", "rest"], groups)
        }
        self.moments = {
            key: (state["exp_avg"], state["exp_avg_sq"])
            for key, state in zip(["dc", "rest"], states)
        }
        self.iteration = int(states[0]["step"].item()) + 1
        self.consumed = False
        self.versions = [p._version for p in self.params.values()]

    def validate(self):
        if self.consumed:
            raise RuntimeError("SH consumption permits one backward per optimizer step")
        for group, p, version in zip(self.groups, self.params.values(), self.versions):
            if group["params"][0] is not p or p._version != version:
                raise RuntimeError("SH parameter changed between forward and backward")

    @torch.no_grad()
    def finish(self):
        for state in self.states:
            state["step"].add_(1)
        torch.autograd.graph.increment_version(tuple(self.params.values()))
        self.consumed = True


class LifecycleRender(ConsumerRender):
    @staticmethod
    def backward(ctx, seed, radii_seed, depth_seed):
        if torch.is_grad_enabled():
            raise RuntimeError("SH consumption supports first-order backward only")
        ctx.owner.validate()
        result = ConsumerRender.backward(ctx, seed, radii_seed, depth_seed)
        ctx.owner.finish()
        return result


class DenseSHConsumer:
    def __init__(self, optimizer, *, native, extension, direct_inputs=False):
        if direct_inputs and not all(
            hasattr(extension, name)
            for name in (
                "rasterize_gaussians_direct",
                "rasterize_gaussians_backward_consume_direct",
            )
        ):
            raise ValueError("Build the extension with direct-input support first")
        self.native = native
        self.extension = extension
        self.direct_inputs = direct_inputs
        self.optimizer = optimizer
        self.active = None
        self.last_reason = None

    def bind(self, dc, rest, policy):
        if self.active is not None:
            raise RuntimeError("finish the previous optimizer step first")
        reason = policy.reason()
        if not torch.is_grad_enabled():
            reason = "evaluation without gradients"
        if torch.cuda.current_stream() != torch.cuda.default_stream():
            raise RuntimeError("the pinned rasterizer requires the default CUDA stream")
        if type(self.optimizer) is not torch.optim.Adam:
            reason = "unsupported optimizer"
        if reason is not None:
            self.last_reason = reason
            return None
        groups = []
        for name, param, shape, rate in [
            ("f_dc", dc, (dc.shape[0], 1, 3), 0.0025),
            ("f_rest", rest, (dc.shape[0], 15, 3), 0.000125),
        ]:
            matches = [g for g in self.optimizer.param_groups if g.get("name") == name]
            if len(matches) != 1:
                reason = "missing or duplicate SH optimizer group"
                break
            g = matches[0]
            groups.append(g)
            if len(g["params"]) != 1 or g["params"][0] is not param:
                reason = "SH parameter ownership mismatch"
            if (
                tuple(param.shape) != shape
                or not param.is_contiguous()
                or param.dtype != torch.float32
                or not param.is_cuda
            ):
                reason = "unsupported SH tensor layout"
            if not param.requires_grad or param.grad is not None:
                reason = "frozen or accumulated SH gradients"
            if g["lr"] != rate or g["betas"] != (0.9, 0.999) or g["eps"] != 1e-15:
                reason = "unsupported SH Adam hyperparameters"
            if g.get("fused") is not True or g.get("foreach"):
                reason = "SH consumption requires fused Adam without foreach"
            if (
                any(
                    g.get(k, False)
                    for k in ["amsgrad", "maximize", "differentiable", "capturable"]
                )
                or g.get("weight_decay", 0) != 0
            ):
                reason = "unsupported Adam option"
        if dc.device != rest.device:
            reason = "SH device mismatch"
        if reason is not None:
            self.last_reason = reason
            return None
        states = []
        for group in groups:
            param = group["params"][0]
            state = self.optimizer.state.get(param)
            if not state:
                state = dict(
                    step=torch.tensor(
                        0.0, device=param.device if group.get("fused") else "cpu"
                    ),
                    exp_avg=torch.zeros_like(param),
                    exp_avg_sq=torch.zeros_like(param),
                )
            for key in ["exp_avg", "exp_avg_sq"]:
                value = state.get(key)
                if (
                    value is None
                    or value.shape != param.shape
                    or value.dtype != param.dtype
                    or value.device != param.device
                    or not value.is_contiguous()
                ):
                    self.last_reason = "unsupported SH moment layout"
                    return None
            step = state["step"].item()
            if not math.isfinite(step) or step < 0 or step != int(step):
                self.last_reason = "invalid SH step counter"
                return None
            states.append(state)
        if states[0]["step"].item() != states[1]["step"].item():
            self.last_reason = "different SH Adam step counters"
            return None
        for group, state in zip(groups, states):
            self.optimizer.state[group["params"][0]] = state
        self.active = BoundStep(groups, states, self.extension)
        self.active.direct_inputs = self.direct_inputs
        self.last_reason = None
        return self.active

    def render(
        self, means, screen, dc, rest, opacity, scale, rotation, settings, policy
    ):
        owner = self.bind(dc, rest, policy)
        if owner is not None:
            return LifecycleRender.apply(
                means, screen, dc, rest, opacity, scale, rotation, settings, owner
            )
        return self.native.GaussianRasterizer(settings)(
            means3D=means,
            means2D=screen,
            opacities=opacity,
            shs=torch.cat((dc, rest), 1),
            scales=scale,
            rotations=rotation,
        )

    def end_step(self):
        if self.active is not None and not self.active.consumed:
            raise RuntimeError("consumer frame has no completed backward")
        self.active = None


def official_step_policy(
    iteration,
    options,
    *,
    white_background=False,
    testing_iterations=(),
    saving_iterations=()
):
    """Match the pinned trainer's operations between backward and Adam."""
    topology = False
    if iteration < options.densify_until_iter:
        topology = (
            iteration > options.densify_from_iter
            and iteration % options.densification_interval == 0
        )
        topology = topology or iteration % options.opacity_reset_interval == 0
        topology = topology or (
            white_background and iteration == options.densify_from_iter
        )
    return StepPolicy(
        optimizer_step=iteration < options.iterations,
        topology_change=topology,
        observation_before_step=iteration in testing_iterations
        or iteration in saving_iterations,
    )
