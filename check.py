"""Compare consumers with native gradients, Adam, and layouts."""

from pathlib import Path
import argparse
import math
import sys
import subprocess
import tempfile
import torch
from types import SimpleNamespace


def strided(tensor):
    storage = torch.full(
        (*tensor.shape, 4), -17, device=tensor.device, dtype=tensor.dtype
    )
    storage[..., 0] = tensor
    result = storage[..., 0]
    assert not result.is_contiguous()
    return result


def check_normalization(extension):
    from normalization import normalize_quaternion
    import torch.nn.functional as functional

    torch.manual_seed(42)
    checks, worst = 0, 0.0
    for n in (0, 1, 35, 4097):
        for scale in (0.0, 1e-14, 1e-12, 1.0, 1e4):
            raw = torch.randn((n, 4), device="cuda") * scale
            seed = torch.randn_like(raw)
            for packed in (True, False):
                if not packed and raw.numel():
                    raw, seed = strided(raw).detach(), strided(seed)
                raw.requires_grad_(True)
                forward = functional.normalize(raw, dim=1, eps=1e-12)
                expected = torch.autograd.grad(forward, raw, seed)[0]
                actual_forward = normalize_quaternion(raw, extension)
                actual = torch.autograd.grad(actual_forward, raw, seed)[0]
                assert torch.equal(actual_forward, forward)
                assert torch.isfinite(actual).all()
                difference = actual.double() - expected.double()
                relative = difference.norm() / expected.double().norm().clamp_min(1e-30)
                assert relative < 2e-6
                # Bound uses the VJP inputs. The output can be ~0.
                bound = seed.double().norm(dim=1, keepdim=True) / raw.double().norm(
                    dim=1, keepdim=True
                ).clamp_min(1e-12)
                assert (difference.abs() <= 2e-6 * bound + 1e-7).all()
                checks += raw.numel()
                worst = max(worst, float(relative))

    eps = torch.tensor(1e-12, device="cuda")
    raw = torch.zeros((4, 4), device="cuda")
    raw[:3, 0] = torch.stack(
        (
            torch.nextafter(eps, torch.zeros_like(eps)),
            eps,
            torch.nextafter(eps, torch.full_like(eps, float("inf"))),
        )
    )
    raw.requires_grad_(True)
    seed = torch.ones_like(raw)
    expected = torch.autograd.grad(functional.normalize(raw, dim=1), raw, seed)[0]
    actual = extension.backward(raw, seed, raw.norm(2, dim=1, keepdim=True))
    assert (
        (actual - expected).abs() <= 2e-6 * seed.norm(dim=1, keepdim=True) / eps
    ).all()
    checks += raw.numel()

    raw = torch.randn((35, 4), device="cuda") + 0.2
    seed = torch.randn_like(raw)
    actual = extension.backward(raw, seed, raw.norm(2, dim=1, keepdim=True)).double()
    finite = torch.empty_like(actual)
    for j in range(4):
        plus, minus = raw.double(), raw.double()
        plus[:, j] += 1e-5
        minus[:, j] -= 1e-5
        finite[:, j] = (
            (functional.normalize(plus, dim=1) * seed).sum(1)
            - (functional.normalize(minus, dim=1) * seed).sum(1)
        ) / 2e-5
    torch.testing.assert_close(actual, finite, rtol=2e-5, atol=2e-6)
    checks += actual.numel()

    # A few components can land on either side of the epsilon clamp.
    raw = (
        functional.normalize(
            torch.randn((4097, 4), device="cuda", dtype=torch.float64), dim=1
        )
        .float()
        .mul_(1e-12)
        .requires_grad_()
    )
    seed = torch.randn_like(raw)
    expected = torch.autograd.grad(functional.normalize(raw, dim=1), raw, seed)[0]
    actual = torch.autograd.grad(normalize_quaternion(raw, extension), raw, seed)[0]
    bound = 2e-6 * seed.double().norm(dim=1, keepdim=True) / 1e-12
    assert ((actual.double() - expected.double()).abs() <= bound).all()
    checks += actual.numel()

    raw = torch.randn((35, 4), device="cuda", requires_grad=True)
    try:
        torch.autograd.grad(
            normalize_quaternion(raw, extension).sum(), raw, create_graph=True
        )
    except RuntimeError as error:
        assert "first-order" in str(error)
    else:
        raise AssertionError("Quaternion activation accepted higher-order backward")
    with torch.cuda.stream(torch.cuda.Stream()):
        raw = torch.randn((35, 4), device="cuda", requires_grad=True)
        actual = torch.autograd.grad(normalize_quaternion(raw, extension).sum(), raw)[0]
        expected = torch.autograd.grad(functional.normalize(raw, dim=1).sum(), raw)[0]
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.cuda.synchronize()
    print(f"{checks} quaternion comparisons; worst relative L2 {worst:.3g}")


def check_activation_consumer(extension):
    torch.manual_seed(42)
    rates = [0.001, 0.005, 0.025]
    cases = comparisons = 0
    for n in (0, 1, 35, 257):
        for scale in (0.0, 1e-14, 1e-12, 1.0, 1e4):
            for counter in (0, 99, 29854):
                q = torch.randn((n, 4), device="cuda") * scale
                raw_scale = torch.rand((n, 3), device="cuda") * 20 - 15
                extremes = torch.tensor(
                    [
                        -120.0,
                        -100.0,
                        -90.0,
                        -40.0,
                        -20.0,
                        -10.0,
                        0.0,
                        10.0,
                        16.0,
                        20.0,
                        40.0,
                        100.0,
                        120.0,
                    ],
                    device="cuda",
                )
                opacity = extremes.repeat((n + len(extremes) - 1) // len(extremes))[
                    :n, None
                ].contiguous()
                raw = [q, raw_scale, opacity]
                seeds = [torch.randn_like(value) * 0.001 for value in raw]
                saved = [
                    q.norm(2, dim=1, keepdim=True),
                    raw_scale.exp(),
                    opacity.sigmoid(),
                ]
                reference = [torch.nn.Parameter(value.clone()) for value in raw]
                optimizer = torch.optim.Adam(
                    [
                        dict(params=[value], lr=rate)
                        for value, rate in zip(reference, rates)
                    ],
                    eps=1e-15,
                    fused=True,
                    foreach=False,
                )
                counters = [counter, counter + 3, counter + 7]
                first, second = [], []
                for value, step in zip(reference, counters):
                    m = (
                        torch.randn_like(value) * 0.02
                        if counter
                        else torch.zeros_like(value)
                    )
                    v = (
                        torch.rand_like(value) * 0.03
                        if counter
                        else torch.zeros_like(value)
                    )
                    optimizer.state[value] = dict(
                        step=torch.tensor(float(step), device="cuda"),
                        exp_avg=m.clone(),
                        exp_avg_sq=v.clone(),
                    )
                    first.append(m)
                    second.append(v)
                outputs = [
                    torch.nn.functional.normalize(reference[0], dim=1, eps=1e-12),
                    reference[1].exp(),
                    reference[2].sigmoid(),
                ]
                grads = list(torch.autograd.grad(outputs, reference, seeds))
                generated_q = extension.backward(reference[0], seeds[0], saved[0])
                bound = (
                    2e-6
                    * seeds[0].double().norm(dim=1, keepdim=True)
                    / q.double().norm(dim=1, keepdim=True).clamp_min(1e-12)
                )
                assert (
                    (generated_q.double() - grads[0].double()).abs() <= bound + 1e-7
                ).all()
                # Step Adam with the generated quaternion grad so this checks the optimizer.
                grads[0] = generated_q
                for value, grad in zip(reference, grads):
                    value.grad = grad
                optimizer.step()
                # Strided read-only inputs, including a partial block.
                incoming = [strided(value) for value in seeds] if n == 35 else seeds
                residuals = [strided(value) for value in saved] if n == 35 else saved
                extension.consume_activations(
                    raw,
                    incoming,
                    residuals,
                    first,
                    second,
                    rates,
                    [1 - 0.9 ** (step + 1) for step in counters],
                    [math.sqrt(1 - 0.999 ** (step + 1)) for step in counters],
                )
                for value, m, v, expected in zip(raw, first, second, reference):
                    for actual, wanted in (
                        (value, expected),
                        (m, optimizer.state[expected]["exp_avg"]),
                        (v, optimizer.state[expected]["exp_avg_sq"]),
                    ):
                        assert torch.isfinite(actual).all()
                        torch.testing.assert_close(actual, wanted, rtol=3e-6, atol=2e-7)
                        comparisons += actual.numel()
                cases += 1
    for invalid in ("cross-group", "moments", "seed", "saved", "bias", "rate"):
        raw = [torch.randn((35, width), device="cuda") for width in (4, 3, 1)]
        seeds = [torch.randn_like(value) for value in raw]
        saved = [raw[0].norm(2, dim=1, keepdim=True), raw[1].exp(), raw[2].sigmoid()]
        first, second = [torch.zeros_like(v) for v in raw], [
            torch.ones_like(v) for v in raw
        ]
        if invalid == "cross-group":
            first[1] = raw[0].view(-1)[:105].view(35, 3)
        elif invalid == "moments":
            second[2] = first[2]
        elif invalid == "seed":
            seeds[0] = raw[0]
        elif invalid == "saved":
            saved[1] = raw[1]
        bias, rate = [0.1] * 3, list(rates)
        if invalid == "bias":
            bias[1] = 0
        if invalid == "rate":
            rate[2] = float("nan")
        before = [v.clone() for v in raw + first + second]
        try:
            extension.consume_activations(
                raw, seeds, saved, first, second, rate, bias, [math.sqrt(0.001)] * 3
            )
        except RuntimeError:
            for value, previous in zip(raw + first + second, before):
                assert torch.equal(value, previous)
        else:
            raise AssertionError(f"Invalid activation inputs accepted: {invalid}")
    torch.cuda.synchronize()
    print(
        f"{cases} combined activation cases; {comparisons} native-AD parameter/moment comparisons; six mutation rejections"
    )


def activation_model():
    model = SimpleNamespace()
    groups = []
    for name, attr, width, rate in zip(
        ("rotation", "scaling", "opacity"),
        ("_rotation", "_scaling", "_opacity"),
        (4, 3, 1),
        (0.001, 0.005, 0.025),
    ):
        raw = torch.nn.Parameter(torch.randn((35, width), device="cuda"))
        setattr(model, attr, raw)
        groups.append(dict(name=name, params=[raw], lr=rate))
    model.optimizer = torch.optim.Adam(groups, eps=1e-15, fused=True, foreach=False)
    for group, counter in zip(model.optimizer.param_groups, (99, 3, 29854)):
        raw = group["params"][0]
        model.optimizer.state[raw] = dict(
            step=torch.tensor(float(counter), device="cuda"),
            exp_avg=torch.randn_like(raw) * 0.02,
            exp_avg_sq=torch.rand_like(raw) * 0.03,
        )
    return model


def activation_capture(model):
    values = []
    for group in model.optimizer.param_groups:
        raw = group["params"][0]
        state = model.optimizer.state[raw]
        values.extend([raw] + [state[k] for k in ("exp_avg", "exp_avg_sq", "step")])
    return [v.detach().clone() for v in values]


def activation_unchanged(model, before):
    for actual, expected in zip(activation_capture(model), before):
        assert torch.equal(actual, expected)


def check_activation_binding(extension):
    from activation import DenseActivationConsumer, ATTRS
    from adapter import StepPolicy
    from normalization import normalize_quaternion

    torch.manual_seed(42)
    model = activation_model()
    reference = activation_model()
    with torch.no_grad():
        for group, ref_group in zip(
            model.optimizer.param_groups, reference.optimizer.param_groups
        ):
            raw, ref = group["params"][0], ref_group["params"][0]
            ref.copy_(raw)
            for key, val in model.optimizer.state[raw].items():
                reference.optimizer.state[ref][key].copy_(val)
    expected_outputs = (
        normalize_quaternion(reference._rotation, extension),
        reference._scaling.exp(),
        reference._opacity.sigmoid(),
    )
    consumer = DenseActivationConsumer(extension)
    actual_outputs = consumer.bind(model, StepPolicy(), exclusive_use=True)
    for actual, expected in zip(actual_outputs, expected_outputs):
        assert torch.equal(actual, expected)
    seeds = [torch.randn_like(v) * 0.001 for v in actual_outputs]
    torch.autograd.backward(expected_outputs, seeds)
    reference.optimizer.step()
    torch.autograd.backward(actual_outputs, seeds, retain_graph=True)
    assert consumer.active.consumed
    for attr, counter in zip(ATTRS, (100, 4, 29855)):
        raw = getattr(model, attr)
        assert raw.grad is None
        assert int(model.optimizer.state[raw]["step"]) == counter
    for actual, expected in zip(
        activation_capture(model), activation_capture(reference)
    ):
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=2e-7)
    before = activation_capture(model)
    model.optimizer.step()
    activation_unchanged(model, before)
    try:
        torch.autograd.backward(actual_outputs, seeds)
    except RuntimeError:
        activation_unchanged(model, before)
    else:
        raise AssertionError("Repeated consumption was accepted")
    for invalid in (
        "higher-order",
        "missing-seed",
        "late-gradient",
        "late-freeze",
        "parameter",
        "state",
        "counter",
        "options",
        "group",
    ):
        model = activation_model()
        consumer = DenseActivationConsumer(extension)
        outputs = consumer.bind(model, StepPolicy(), exclusive_use=True)
        if invalid == "late-gradient":
            model._opacity.grad = torch.ones_like(model._opacity)
        if invalid == "late-freeze":
            model._opacity.requires_grad_(False)
        if invalid == "parameter":
            model._scaling = torch.nn.Parameter(model._scaling.detach().clone())
        if invalid == "state":
            state = model.optimizer.state[model._opacity]
            state["exp_avg"] = state["exp_avg"].clone()
        if invalid == "counter":
            model.optimizer.state[model._scaling]["step"].add_(1)
        if invalid == "options":
            model.optimizer.param_groups[1]["eps"] = 1e-8
        if invalid == "group":
            model.optimizer.param_groups[1] = dict(model.optimizer.param_groups[1])
        before = activation_capture(model)
        try:
            if invalid == "higher-order":
                torch.autograd.grad(
                    sum(v.sum() for v in outputs), model._rotation, create_graph=True
                )
            elif invalid == "missing-seed":
                outputs[0].sum().backward()
            else:
                torch.autograd.backward(outputs, [torch.ones_like(v) for v in outputs])
        except RuntimeError:
            activation_unchanged(model, before)
        else:
            raise AssertionError("Unsafe ownership was accepted: " + invalid)
    torch.cuda.synchronize()
    # Fallback leaves optimizer state alone.
    fallback_cases = (
        "disabled",
        "skipped",
        "topology",
        "observation",
        "accumulation",
        "scaled",
        "exclusive",
        "processing",
        "declared-higher",
        "optimizer",
        "foreach",
        "frozen",
        "accumulated",
        "learning-rate",
        "moment-alias",
        "counter-alias",
        "shared-counter",
    )
    for case in fallback_cases:
        model = activation_model()
        policy = StepPolicy()
        flags = dict(exclusive_use=True)
        if case == "disabled":
            policy = None
        if case == "skipped":
            policy = StepPolicy(optimizer_step=False)
        if case == "topology":
            policy = StepPolicy(topology_change=True)
        if case == "observation":
            policy = StepPolicy(observation_before_step=True)
        if case == "accumulation":
            policy = StepPolicy(single_backward=False)
        if case == "scaled":
            policy = StepPolicy(unscaled_gradients=False)
        if case == "exclusive":
            flags["exclusive_use"] = False
        if case == "processing":
            flags["gradient_processing"] = True
        if case == "declared-higher":
            flags["higher_order"] = True
        if case == "optimizer":
            model.optimizer = torch.optim.SGD(model.optimizer.param_groups, lr=0.01)
        if case == "foreach":
            model.optimizer.param_groups[1]["foreach"] = True
        if case == "frozen":
            model._scaling.requires_grad_(False)
        if case == "accumulated":
            model._opacity.grad = torch.ones_like(model._opacity)
        if case == "learning-rate":
            model.optimizer.param_groups[1]["lr"] = torch.tensor(0.005, device="cuda")
        if case == "moment-alias":
            model.optimizer.state[model._scaling]["exp_avg"] = model._rotation.view(-1)[
                :105
            ].view(35, 3)
        if case == "counter-alias":
            with torch.no_grad():
                model._opacity[0].zero_()
            model.optimizer.state[model._opacity]["step"] = model._opacity.view(-1)[:1]
        if case == "shared-counter":
            step = torch.zeros((), device="cuda")
            for state in model.optimizer.state.values():
                state["step"] = step
        if case not in ("moment-alias", "counter-alias", "shared-counter"):
            model.optimizer.state.clear()
        before = [
            p.detach().clone()
            for p in (model._rotation, model._scaling, model._opacity)
        ]
        keys = list(model.optimizer.state)
        old_state = activation_capture(model) if keys else None
        consumer = DenseActivationConsumer(extension)
        assert consumer.bind(model, policy, **flags) is None
        assert consumer.last_reason and consumer.active is None
        assert list(model.optimizer.state) == keys
        if old_state is not None:
            activation_unchanged(model, old_state)
        for raw, previous in zip(
            (model._rotation, model._scaling, model._opacity), before
        ):
            assert torch.equal(raw, previous)
    # Missing state starts at zero. The next frame reads the real counters.
    model = activation_model()
    model.optimizer.state.clear()
    consumer = DenseActivationConsumer(extension)
    outputs = consumer.bind(model, StepPolicy(), exclusive_use=True)
    torch.autograd.backward(outputs, [torch.ones_like(v) for v in outputs])
    assert [
        int(model.optimizer.state[getattr(model, attr)]["step"]) for attr in ATTRS
    ] == [1, 1, 1]
    consumer.end_step()
    assert consumer.active is None
    model.optimizer.zero_grad(set_to_none=True)
    outputs = consumer.bind(model, StepPolicy(), exclusive_use=True)
    torch.autograd.backward(outputs, [torch.ones_like(v) for v in outputs])
    assert [
        int(model.optimizer.state[getattr(model, attr)]["step"]) for attr in ATTRS
    ] == [2, 2, 2]
    consumer.end_step()
    print(
        f"Activation binding: {len(fallback_cases)} native fallbacks, nine unsafe backward rejections, unequal counters and fresh first-use state passed"
    )



def check_sh_binding():
    from adapter import DenseSHConsumer, StepPolicy

    def setup():
        dc = torch.nn.Parameter(torch.zeros((2, 1, 3), device="cuda"))
        rest = torch.nn.Parameter(torch.zeros((2, 15, 3), device="cuda"))
        optimizer = torch.optim.Adam([
            dict(params=[dc], name="f_dc", lr=0.0025),
            dict(params=[rest], name="f_rest", lr=0.000125),
        ], eps=1e-15, fused=True, foreach=False)
        for parameter in (dc, rest):
            optimizer.state[parameter] = dict(
                step=torch.zeros((), device="cuda"), exp_avg=torch.zeros_like(parameter),
                exp_avg_sq=torch.zeros_like(parameter),
            )
        return dc, rest, optimizer, DenseSHConsumer(optimizer, native=None, extension=None)

    def capture(dc, rest, optimizer):
        return [value for p in (dc, rest) for value in
                (p, *[v for v in optimizer.state[p].values() if torch.is_tensor(v)])]

    unsafe = ("parameter", "state", "moment", "moment-write", "counter", "options",
              "group", "gradient", "frozen")
    for case in unsafe:
        dc, rest, optimizer, consumer = setup()
        frame = consumer.bind(dc, rest, StepPolicy())
        if case == "parameter":
            with torch.no_grad():
                dc.add_(1)
        elif case == "state":
            optimizer.state[dc] = dict(optimizer.state[dc])
        elif case == "moment":
            optimizer.state[dc]["exp_avg"] = optimizer.state[dc]["exp_avg"].clone()
        elif case == "moment-write":
            optimizer.state[dc]["exp_avg"].add_(1)
        elif case == "counter":
            optimizer.state[dc]["step"].add_(1)
        elif case == "options":
            optimizer.param_groups[0]["lr"] = 0.1
        elif case == "group":
            optimizer.param_groups[0] = dict(optimizer.param_groups[0])
        elif case == "gradient":
            dc.grad = torch.ones_like(dc)
        elif case == "frozen":
            dc.requires_grad_(False)
        values = capture(dc, rest, optimizer)
        before = [v.detach().clone() for v in values]
        versions = [v._version for v in values]
        try:
            frame.validate()
        except RuntimeError:
            for value, previous, version in zip(values, before, versions):
                assert torch.equal(value, previous) and value._version == version
        else:
            raise AssertionError("SH ownership change accepted: " + case)

    invalid = ("moment-alias", "cross-group", "counter-alias", "shared-counter",
               "counter-device", "counter-dtype", "counter-shape", "counter-missing",
               "counter-nan", "counter-negative", "counter-fraction", "counter-overflow",
               "tensor-rate")
    for case in invalid:
        dc, rest, optimizer, consumer = setup()
        state = optimizer.state[dc]
        if case == "moment-alias":
            state["exp_avg"] = state["exp_avg_sq"]
        elif case == "cross-group":
            state["exp_avg"] = rest.detach().reshape(-1)[:dc.numel()].view_as(dc)
        elif case == "counter-alias":
            state["step"] = dc.detach().reshape(-1)[0]
        elif case == "shared-counter":
            optimizer.state[rest]["step"] = state["step"]
        elif case == "counter-device":
            state["step"] = torch.zeros(())
        elif case == "counter-dtype":
            state["step"] = state["step"].double()
        elif case == "counter-shape":
            state["step"] = torch.zeros(2, device="cuda")
        elif case == "counter-missing":
            del state["step"]
        elif case.startswith("counter-"):
            state["step"].fill_({"counter-nan": math.nan, "counter-negative": -1,
                                 "counter-fraction": 0.5, "counter-overflow": 2**24}[case])
        else:
            optimizer.param_groups[0]["lr"] = torch.tensor(0.0025, device="cuda")
        values = capture(dc, rest, optimizer)
        before = [v.detach().clone() for v in values]
        versions = [v._version for v in values]
        assert consumer.bind(dc, rest, StepPolicy()) is None
        assert consumer.active is None and consumer.last_reason
        for value, previous, version in zip(values, before, versions):
            torch.testing.assert_close(value, previous, rtol=0, atol=0, equal_nan=True)
            assert value._version == version

    dc, rest, optimizer, consumer = setup()
    for state in optimizer.state.values():
        state["step"].fill_(99)
    frame = consumer.bind(dc, rest, StepPolicy())
    frame.validate()
    writable = [dc, rest, *[v for pair in frame.moments.values() for v in pair]]
    versions = [v._version for v in writable]
    assert frame.iteration == 100
    frame.finish()
    assert all(v._version == version + 1 for v, version in zip(writable, versions))
    assert all(int(state["step"]) == 100 for state in optimizer.state.values())
    before = [v.detach().clone() for v in capture(dc, rest, optimizer)]
    optimizer.step()
    for value, previous in zip(capture(dc, rest, optimizer), before):
        assert torch.equal(value, previous)
    try:
        frame.validate()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Repeated SH consumption accepted")
    consumer.end_step()
    assert consumer.bind(dc, rest, StepPolicy()).iteration == 101
    for policy in (StepPolicy(topology_change=True), StepPolicy(optimizer_step=False)):
        dc, rest, optimizer, consumer = setup()
        optimizer.state.clear()
        assert consumer.bind(dc, rest, policy) is None and not optimizer.state
    print(f"SH binding: {len(unsafe)} ownership rejections, {len(invalid)} state fallbacks, "
          "counter/version advancement and native-step skipping passed")


def check_optimizer_setup():
    import copy
    from optimizer_setup import fused_adam

    parameter = torch.nn.Parameter(torch.tensor([1., -2.], device="cuda"))
    original = torch.optim.Adam([dict(params=[parameter], name="probe")],
                                lr=.003, eps=1e-15, foreach=False, fused=False)
    parameter.grad = torch.tensor([.2, -.4], device="cuda")
    original.step()
    saved = copy.deepcopy(original.state_dict())
    before = parameter.detach().clone()
    selected = fused_adam(original)
    assert selected.param_groups[0]["params"][0] is parameter
    assert selected.param_groups[0]["lr"] == .003
    assert selected.param_groups[0]["fused"] is True
    assert selected.param_groups[0]["foreach"] is False
    state = selected.state[parameter]
    prior = saved["state"][0]
    assert state["step"].device == parameter.device
    assert torch.equal(state["step"].cpu(), prior["step"].cpu())
    assert torch.equal(state["exp_avg"], prior["exp_avg"])
    assert torch.equal(state["exp_avg_sq"], prior["exp_avg_sq"])
    assert torch.equal(parameter, before)
    assert original.param_groups[0]["fused"] is False

    other = torch.nn.Parameter(before.clone())
    reference = torch.optim.Adam([dict(params=[other], name="probe")],
                                 lr=.003, eps=1e-15, fused=True, foreach=False)
    for group in saved["param_groups"]:
        group.update(fused=True, foreach=False)
    reference.load_state_dict(saved)
    parameter.grad = torch.tensor([-.3, .1], device="cuda")
    other.grad = parameter.grad.clone()
    selected.step()
    reference.step()
    assert torch.equal(parameter, other)
    for key in ["step", "exp_avg", "exp_avg_sq"]:
        assert torch.equal(selected.state[parameter][key], reference.state[other][key])

    for unsupported in [torch.optim.SGD([parameter], lr=.1),
                        torch.optim.Adam([torch.nn.Parameter(torch.ones(2))])]:
        try:
            fused_adam(unsupported)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported fused launcher optimizer accepted")
    print("Fused optimizer preserves state and matches the native fused update")


def check_routing_summary():
    import json
    import tempfile
    from routing import RoutingSummary

    def context(iteration):
        return dict(iteration=iteration, enabled=True, activation_enabled=True,
                    policy_reason=None, sh_called=True, activations_called=True)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "routing.json"
        sh = SimpleNamespace(active=SimpleNamespace(consumed=True), last_reason=None)
        activation = SimpleNamespace(active=None, last_reason="unsupported optimizer")
        with RoutingSummary(path, dict(fused_adam=True)) as summary:
            row = context(7)
            summary.capture(row, sh, activation)
            # Snapshot the reason before evaluation overwrites it.
            activation.last_reason = "evaluation without gradients"
            summary.record(row)
            skipped = context(8)
            skipped["policy_reason"] = "topology changes before optimizer step"
            sh.active = None
            summary.capture(skipped, sh, activation)
            summary.record(skipped)
            uncompleted = context(9)
            summary.capture(uncompleted, sh, activation)
            assert json.loads(path.read_text())["completed_training_steps"] == 0
        result = json.loads(path.read_text())
        assert result["status"] == "completed"
        assert result["completed_training_steps"] == 2
        assert (result["first_iteration"], result["last_iteration"]) == (7, 8)
        assert result["sh"]["consumed_steps"] == 1
        assert result["activations"]["fallback_reasons"] == {
            "unsupported optimizer": 1, "topology changes before optimizer step": 1}
        saved = path.read_bytes()
        try:
            RoutingSummary(path, {})
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing summary was overwritten")
        assert path.read_bytes() == saved

        failed = Path(directory) / "failed.json"
        try:
            with RoutingSummary(failed, {}) as summary:
                row = context(10)
                row["enabled"] = False
                summary.capture(row, None, None)
                summary.record(row)
                raise RuntimeError("fixture failure")
        except RuntimeError as error:
            assert str(error) == "fixture failure"
        result = json.loads(failed.read_text())
        assert result["status"] == "failed" and result["completed_training_steps"] == 1
        assert result["sh"]["fallback_reasons"] == {"consumption disabled": 1}
        for enabled, called, reason in (
            (False, False, "activation consumption disabled"),
            (True, False, "unsupported renderer path"),
            (True, True, "backward did not consume"),
        ):
            row = context(11)
            row.update(activation_enabled=enabled, activations_called=called)
            summary.capture(row, None, None)
            assert row["activations"]["reason"] == reason
    print("Routing summary: captured reasons, completed steps, resume bounds, failure and overwrite checks passed")


def capture_raster_gradients(extension, output):
    results = {}
    for name, n, height, width, scale, opacity in (
        ("full-warp", 1, 32, 32, 2., .3),
        ("partial-tiles", 17, 17, 31, .2, .5),
        ("occluded", 73, 32, 32, .8, .95),
        ("empty", 0, 17, 31, .2, .5),
    ):
        for antialiasing in (False, True):
            torch.manual_seed(42)
            means = torch.randn(n, 3, device="cuda") * .08
            means[:, 2] = torch.linspace(2., 3., n, device="cuda")
            sh = torch.randn(n, 16, 3, device="cuda") * .02
            opacities = torch.full((n, 1), opacity, device="cuda")
            scales = torch.full((n, 3), scale, device="cuda")
            rotations = torch.zeros(n, 4, device="cuda")
            rotations[:, 0] = 1
            view = torch.eye(4, device="cuda")
            projection = torch.tensor([
                [1., 0, 0, 0], [0, 1., 0, 0],
                [0, 0, 100 / 99.99, -1 / 99.99], [0, 0, 1., 0],
            ], device="cuda").T.contiguous()
            camera = torch.zeros(3, device="cuda")
            background = torch.tensor([.1, .2, .3], device="cuda")
            empty = torch.empty(0, device="cuda")
            count, color, radii, geometry, binning, image, depth = (
                extension.rasterize_gaussians(
                    background, means, empty, opacities, scales, rotations, 1.,
                    empty, view, projection, 1., 1., height, width, sh, 3,
                    camera, False, antialiasing, False,
                )
            )
            if name == "full-warp":
                assert (depth > 0).all(), "Full-warp fixture must cover every pixel"
            gradients = extension.rasterize_gaussians_backward(
                background, means, radii, empty, opacities, scales, rotations, 1.,
                empty, view, projection, 1., 1., torch.randn_like(color) * .01,
                torch.randn_like(depth) * .01, sh, 3, camera, geometry, count,
                binning, image, antialiasing, False,
            )
            results[f"{name}/aa={antialiasing}"] = [
                value.detach().cpu() for value in (color, radii, depth, *gradients)
            ]
    torch.save(results, output)


def check_native_raster(candidate, baseline):
    from reproduction.prepare import verify_build

    candidate_record = verify_build(candidate, direct_inputs=True, warp_accumulation=True)
    native_record = verify_build(baseline, direct_inputs=False, warp_accumulation=False)
    if candidate_record["toolchain"] != native_record["toolchain"]:
        raise ValueError("Native raster comparison requires matching recorded toolchains")
    with tempfile.TemporaryDirectory() as directory:
        captures = []
        # Same extension name, so load each build in its own process.
        for name, build in (("native", baseline), ("candidate", candidate)):
            output = Path(directory) / (name + ".pt")
            subprocess.run([
                sys.executable, "-B", str(Path(__file__).resolve()),
                "--build-dir", str(build), "--raster-output", str(output),
            ], check=True)
            captures.append(torch.load(output, weights_only=True))
        native, generated = captures
        assert native.keys() == generated.keys()
        checks = 0
        for case, reference in native.items():
            for index, (got, expected) in enumerate(zip(generated[case], reference)):
                assert torch.isfinite(got).all() and torch.isfinite(expected).all()
                if index < 3:
                    assert torch.equal(got, expected), (case, "forward", index)
                else:
                    torch.testing.assert_close(got, expected, rtol=5e-5, atol=2e-6,
                                               msg=f"{case}: native raster adjoint {index - 3}")
                    checks += 1
    print(f"Original native/warp raster: {len(native)} fixtures, {checks} gradient tensors passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--baseline-build", type=Path,
                        help="also compare warp pixel gradients with an original-native build")
    parser.add_argument("--raster-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--require-direct-inputs", action="store_true")
    parser.add_argument("--require-activations", action="store_true")
    args = parser.parse_args()
    if args.baseline_build is not None:
        check_native_raster(args.build_dir.resolve(), args.baseline_build.resolve())
    sys.path.insert(0, str(args.build_dir.resolve()))
    import current_native_sh_consumer as extension

    if args.raster_output is not None:
        capture_raster_gradients(extension, args.raster_output)
        return

    if args.require_direct_inputs and not all(
        hasattr(extension, name) for name in (
            "rasterize_gaussians_direct", "rasterize_gaussians_backward_direct",
            "rasterize_gaussians_backward_consume_direct",
        )
    ):
        parser.error("This check requires a build with --direct-inputs")
    normalization = args.build_dir.resolve() / "normalization"
    if args.require_activations and not normalization.is_dir():
        parser.error("This check requires a build with --normalization-object")
    if normalization.is_dir():
        sys.path.insert(0, str(normalization))
        import clad_quaternion

        if args.require_activations and not hasattr(clad_quaternion, "consume_activations"):
            parser.error("This check requires the combined activation consumer")

    check_routing_summary()
    check_optimizer_setup()
    check_sh_binding()
    device = "cuda"
    means = torch.tensor([[0.3, 0.1, 2.0], [0.15, -0.2, 2.5]], device=device)
    camera = torch.tensor([0.2, -0.1, 0.05], device=device)
    coefficients = torch.arange(96, device=device).reshape(2, 16, 3).float() * 0.001
    opacity = torch.tensor([[0.55], [0.4]], device=device)
    scales = torch.full((2, 3), 0.2, device=device)
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device)
    view = torch.eye(4, device=device)
    view[:3, 3] = -camera
    near, far = 0.01, 100.0
    projection = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, far / (far - near), -far * near / (far - near)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )
    full_projection = (projection @ view).T.contiguous()
    view = view.T.contiguous()
    bg = torch.zeros(3, device=device)
    empty = torch.empty(0, device=device)
    seed = torch.tensor([0.3, -0.2, 0.7], device=device).reshape(3, 1, 1)
    depth_seed = torch.zeros((1, 1, 1), device=device)
    failures = []
    checks = 0
    for degree in range(4):
        count, color, radii, geometry, binning, image, depth = (
            extension.rasterize_gaussians(
                bg,
                means,
                empty,
                opacity,
                scales,
                rotation,
                1.0,
                empty,
                view,
                full_projection,
                1.0,
                1.0,
                1,
                1,
                coefficients,
                degree,
                camera,
                False,
                False,
                False,
            )
        )
        assert torch.isfinite(color).all() and (radii > 0).all()

        def run(layout, step=1):
            inputs = dict(
                means=means, radii=radii, coefficients=coefficients, camera=camera
            )
            if layout in ("all", "direct-strided"):
                inputs = {key: strided(value) for key, value in inputs.items()}
            elif layout in ("camera", "means", "radii", "coefficients"):
                inputs[layout] = strided(inputs[layout])
            dc, rest = (
                coefficients[:, :1].contiguous().clone(),
                coefficients[:, 1:].contiguous().clone(),
            )
            moments = [
                torch.full_like(value, fill)
                for value, fill in zip((dc, dc, rest, rest),
                                       (0.002, 0.01, -0.003, 0.02) if step > 1 else (0.,) * 4)
            ]
            if layout == "direct-alias":
                moments[0] = dc
            direct = layout.startswith("direct")
            function = (
                extension.rasterize_gaussians_backward_consume_direct
                if direct
                else extension.rasterize_gaussians_backward_consume
            )
            arguments = (
                bg,
                inputs["means"],
                inputs["radii"],
                empty,
                opacity,
                scales,
                rotation,
                1.0,
                empty,
                view,
                full_projection,
                1.0,
                1.0,
                seed,
                depth_seed,
                empty if direct else inputs["coefficients"],
                degree,
                inputs["camera"],
                geometry,
                count,
                binning,
                image,
                False,
                False,
                dc,
                rest,
                *moments,
                1 - 0.9**step,
                math.sqrt(1 - 0.999**step),
            )
            if layout == "native":
                return list(extension.rasterize_gaussians_backward(*arguments[:24]))
            if layout == "native-direct":
                return list(
                    extension.rasterize_gaussians_backward_direct(
                        *(arguments[:15] + (dc, rest) + arguments[16:24])
                    )
                )
            grads = function(*arguments)
            return [dc, rest, *moments, *grads]

        native_grads = run("native")
        for step in (1, 100):
            layouts = ["packed"]
            if hasattr(extension, "rasterize_gaussians_backward_consume_direct"):
                layouts.append("direct")
            generated = [run(layout, step) for layout in layouts]
            parameters = [torch.nn.Parameter(coefficients[:, :1].contiguous().clone()),
                          torch.nn.Parameter(coefficients[:, 1:].contiguous().clone())]
            optimizer = torch.optim.Adam(
                [dict(params=[parameters[0]], lr=0.0025),
                 dict(params=[parameters[1]], lr=0.000125)],
                eps=1e-15, fused=True, foreach=False,
            )
            for parameter, gradient, first, second in zip(
                parameters, (native_grads[5][:, :1], native_grads[5][:, 1:]),
                (0.002, -0.003), (0.01, 0.02),
            ):
                optimizer.state[parameter] = dict(
                    step=torch.tensor(float(step - 1), device=device),
                    exp_avg=torch.full_like(parameter, first if step > 1 else 0.),
                    exp_avg_sq=torch.full_like(parameter, second if step > 1 else 0.),
                )
                parameter.grad = gradient.contiguous()
            optimizer.step()
            expected = [*parameters, *[
                optimizer.state[parameter][key]
                for parameter in parameters for key in ("exp_avg", "exp_avg_sq")
            ]]
            # Compare consumed Adam to native SH grads plus fused Adam.
            for layout, actual in zip(layouts, generated):
                for index, wanted in enumerate(expected):
                    torch.testing.assert_close(actual[index], wanted, rtol=3e-5, atol=2e-7,
                                               msg=f"{layout}: SH Adam state {index}, degree {degree}, step {step}")
                for index, wanted in enumerate(native_grads):
                    if index != 5:  # Consumed SH grads are absent.
                        torch.testing.assert_close(actual[6 + index], wanted,
                                                   rtol=3e-5, atol=2e-7,
                                                   msg=f"{layout}: adjoint {index}, degree {degree}, step {step}")
        reference = run("packed")
        layouts = ["camera", "means", "radii", "coefficients", "all"]
        if hasattr(extension, "rasterize_gaussians_backward_consume_direct"):
            layouts += ["direct", "direct-strided"]
            for index, (got, expected) in enumerate(
                zip(run("native-direct"), run("native"))
            ):
                checks += 1
                if not torch.equal(got, expected):
                    failures.append(
                        (
                            degree,
                            "native-direct",
                            index,
                            float((got - expected).abs().max()),
                        )
                    )
            try:
                run("direct-alias")
            except RuntimeError as error:
                if "disjoint" not in str(error):
                    raise
            else:
                raise AssertionError(
                    "Direct SH updates accepted aliased moment storage"
                )
        for layout in layouts:
            actual = run(layout)
            for index, (got, expected) in enumerate(zip(actual, reference)):
                checks += 1
                if not torch.equal(got, expected):
                    failures.append(
                        (degree, layout, index, float((got - expected).abs().max()))
                    )
    print("SH/native backward and fused-Adam checks: degrees 0–3, fresh and nonzero state passed")
    print(f"{checks} packed/strided tensor comparisons; {len(failures)} failures")
    if failures:
        print(failures)
        raise SystemExit(2)
    if normalization.is_dir():
        check_normalization(clad_quaternion)
        if hasattr(clad_quaternion, "consume_activations"):
            with torch.cuda.stream(torch.cuda.Stream()):
                check_activation_consumer(clad_quaternion)
            check_activation_binding(clad_quaternion)


if __name__ == "__main__":
    main()
