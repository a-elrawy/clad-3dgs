"""Explicit dense-Adam integration; optional readback instrumentation for tests."""

import json
from pathlib import Path
import torch
import diff_gaussian_rasterization as native
from diff_gaussian_rasterization import _C as extension
from sh_consumer_adapter import DenseSHConsumer, official_step_policy
from clad_activations import ATTRS, DenseActivationConsumer

TRACE = None
DIRECT_INPUTS = False
ACTIVATION_EXTENSION = None


def counters(model, parameters=None):
    values = []
    for p in (
        [model._features_dc, model._features_rest] if parameters is None else parameters
    ):
        state = model.optimizer.state.get(p, {})
        values.append(int(state["step"].item()) if "step" in state else 0)
    return values


def begin_step(model, iteration, options, dataset, testing, saving, enabled):
    policy = official_step_policy(
        iteration,
        options,
        white_background=dataset.white_background,
        testing_iterations=testing,
        saving_iterations=saving,
    )
    activation = getattr(model, "_clad_activations", None)
    if activation is not None and activation.active is not None:
        raise RuntimeError("finish the previous activation step first")
    if TRACE is not None:
        model._clad_trace = dict(
            iteration=iteration,
            enabled=enabled,
            policy_reason=policy.reason(),
            count_before=len(model.get_xyz),
            degree=model.active_sh_degree,
            counters_before=counters(model),
            dc=model._features_dc.detach().clone(),
            rest=model._features_rest.detach().clone(),
        )
        if ACTIVATION_EXTENSION is not None:
            raw = [getattr(model, name) for name in ATTRS]
            model._clad_trace["activation"] = dict(
                counters_before=counters(model, raw),
                before=[p.detach().clone() for p in raw],
            )
    return policy if enabled else None


def activate_consumer(model, policy, *, eligible):
    if ACTIVATION_EXTENSION is None:
        return None
    consumer = getattr(model, "_clad_activations", None)
    if consumer is None:
        consumer = DenseActivationConsumer(ACTIVATION_EXTENSION)
        model._clad_activations = consumer
    if not eligible:
        consumer.last_reason = "unsupported activation renderer path"
        return None
    return consumer.bind(model, policy, exclusive_use=True)


def render_consumer(model, means, screen, opacity, scale, rotation, settings, policy):
    consumer = getattr(model, "_clad_consumer", None)
    if consumer is None or consumer.optimizer is not model.optimizer:
        consumer = DenseSHConsumer(
            model.optimizer,
            native=native,
            extension=extension,
            direct_inputs=DIRECT_INPUTS,
        )
        model._clad_consumer = consumer
    return consumer.render(
        means,
        screen,
        model._features_dc,
        model._features_rest,
        opacity,
        scale,
        rotation,
        settings,
        policy,
    )


def after_backward(model, iteration, loss):
    if TRACE is None:
        return
    row = model._clad_trace
    consumer = getattr(model, "_clad_consumer", None)
    consumed = (
        consumer is not None
        and consumer.active is not None
        and consumer.active.consumed
    )
    row["consumed"] = consumed
    row["loss"] = float(loss.detach())
    assert torch.isfinite(loss)
    row["counters_after_backward"] = counters(model)
    if consumed:
        assert row["counters_after_backward"] == [s + 1 for s in row["counters_before"]]
        assert model._features_dc.grad is None and model._features_rest.grad is None
    else:
        assert row["counters_after_backward"] == row["counters_before"]
        assert torch.equal(model._features_dc, row["dc"]) and torch.equal(
            model._features_rest, row["rest"]
        )
    row["dc_after_backward"] = model._features_dc.detach().clone()
    row["rest_after_backward"] = model._features_rest.detach().clone()
    if "activation" in row:
        activation = getattr(model, "_clad_activations", None)
        frame = activation.active if activation is not None else None
        detail = row["activation"]
        detail["consumed"] = bool(frame is not None and frame.consumed)
        detail["reason"] = (
            activation.last_reason if activation is not None else "disabled"
        )
        raw = [getattr(model, name) for name in ATTRS]
        detail["counters_after_backward"] = counters(model, raw)
        if detail["consumed"]:
            assert detail["counters_after_backward"] == [
                c + 1 for c in detail["counters_before"]
            ]
            assert all(p.grad is None for p in raw)
        else:
            assert detail["counters_after_backward"] == detail["counters_before"]
            assert all(
                torch.equal(p, before) for p, before in zip(raw, detail["before"])
            )
        detail["after_backward"] = [p.detach().clone() for p in raw]


def after_step(model, iteration):
    consumer = getattr(model, "_clad_consumer", None)
    if TRACE is not None:
        row = model._clad_trace
        row["count_after"] = len(model.get_xyz)
        row["counters_after_step"] = counters(model)
        if row["consumed"]:
            assert row["counters_after_step"] == row["counters_after_backward"]
            assert torch.equal(
                model._features_dc, row["dc_after_backward"]
            ) and torch.equal(model._features_rest, row["rest_after_backward"])
        if row["policy_reason"] == "optimizer step skipped":
            assert row["counters_after_step"] == row["counters_before"]
            assert torch.equal(model._features_dc, row["dc"]) and torch.equal(
                model._features_rest, row["rest"]
            )
        row["moment_shapes"] = [
            (
                list(model.optimizer.state[p]["exp_avg"].shape)
                if p in model.optimizer.state
                else None
            )
            for p in [model._features_dc, model._features_rest]
        ]
        if "activation" in row:
            detail = row["activation"]
            raw = [getattr(model, name) for name in ATTRS]
            detail["counters_after_step"] = counters(model, raw)
            if detail["consumed"]:
                assert (
                    detail["counters_after_step"] == detail["counters_after_backward"]
                )
                assert all(
                    torch.equal(p, after)
                    for p, after in zip(raw, detail["after_backward"])
                )
            if row["policy_reason"] == "optimizer step skipped":
                assert detail["counters_after_step"] == detail["counters_before"]
                assert all(
                    torch.equal(p, before) for p, before in zip(raw, detail["before"])
                )
            detail["moment_shapes"] = [
                (
                    list(model.optimizer.state[p]["exp_avg"].shape)
                    if p in model.optimizer.state
                    else None
                )
                for p in raw
            ]
            del detail["before"], detail["after_backward"]
        for key in ["dc", "rest", "dc_after_backward", "rest_after_backward"]:
            del row[key]
        path = Path(TRACE)
        rows = json.loads(path.read_text()) if path.exists() else []
        rows.append(row)
        path.write_text(json.dumps(rows, indent=2))
        del model._clad_trace
    if consumer is not None:
        consumer.end_step()
    activation = getattr(model, "_clad_activations", None)
    if activation is not None:
        activation.end_step()
