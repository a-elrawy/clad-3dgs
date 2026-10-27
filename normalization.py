"""Quaternion normalize, with a Clad backward."""

import torch
import torch.nn.functional as functional
from functools import wraps


class NormalizeQuaternion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw, extension):
        ctx.extension = extension
        norm = raw.norm(2, dim=1, keepdim=True)
        ctx.save_for_backward(raw, norm)
        return raw / norm.clamp_min(1e-12).expand_as(raw)

    @staticmethod
    def backward(ctx, seed):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Quaternion activation supports first-order backward only"
            )
        raw, norm = ctx.saved_tensors
        return ctx.extension.backward(raw, seed, norm), None


def normalize_quaternion(q, extension):
    if not (q.is_cuda and q.dtype == torch.float32 and q.ndim == 2 and q.shape[1] == 4):
        return functional.normalize(q, dim=1, eps=1e-12)
    return NormalizeQuaternion.apply(q, extension)


def enable_quaternion_backward(model_class, extension):
    original = model_class.setup_functions

    @wraps(original)
    def setup(model):
        original(model)
        model.rotation_activation = lambda q: normalize_quaternion(q, extension)

    model_class.setup_functions = setup
