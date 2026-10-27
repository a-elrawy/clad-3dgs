"""Persistent split identities for matched native/Clad training experiments."""

from array import array
import hashlib
import math
import struct
import sys
import types


class Lineage:
    """Encode roots and successive births without a fixed-width integer limit."""

    VERSION = 1

    def __init__(self, roots, max_iteration, seed=42):
        if type(roots) is not int or roots < 0:
            raise ValueError("Invalid initial point count")
        if type(max_iteration) is not int or max_iteration < 1:
            raise ValueError("Invalid training horizon")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("Invalid sampling seed")
        self.roots, self.max_iteration, self.seed = roots, max_iteration, seed
        self.root_bits = max(1, (roots - 1).bit_length())
        self.hop_bits = max_iteration.bit_length() + 2
        self.ids = list(range(1 << self.root_bits, (1 << self.root_bits) + roots))
        self.last_event = 0

    def births(self, parents, iteration, role):
        if type(iteration) is not int or not self.last_event < iteration <= self.max_iteration:
            raise ValueError("Structural iteration is out of order or outside the horizon")
        if role not in (1, 2, 3):
            raise ValueError("Expected clone or one of the two split children")
        return [(parent << self.hop_bits) | (iteration << 2) | role for parent in parents]

    def finish_event(self, iteration):
        if not self.last_event < iteration <= self.max_iteration:
            raise ValueError("Structural event repeated or out of order")
        self.last_event = iteration

    def _validate_id(self, value):
        if type(value) is not int or value <= 0:
            raise ValueError("Invalid point identity")
        width = self.root_bits + 1
        if (value.bit_length() - width) % self.hop_bits:
            raise ValueError("Malformed point lineage")
        previous = self.last_event + 1
        mask = (1 << self.hop_bits) - 1
        while value.bit_length() > width:
            birth = value & mask
            iteration, role = birth >> 2, birth & 3
            if role == 0 or not 0 < iteration < previous:
                raise ValueError("Invalid birth order or operation")
            previous = iteration
            value >>= self.hop_bits
        if not (1 << self.root_bits) <= value < (1 << self.root_bits) + self.roots:
            raise ValueError("Identity does not belong to the initial point set")

    def state_dict(self):
        chunks = [value.to_bytes((value.bit_length() + 7) // 8, "little") for value in self.ids]
        offsets = array("Q", [0])
        for chunk in chunks:
            offsets.append(offsets[-1] + len(chunk))
        if sys.byteorder != "little":
            offsets.byteswap()
        return dict(version=self.VERSION, roots=self.roots, max_iteration=self.max_iteration,
                    seed=self.seed, last_event=self.last_event,
                    offsets=offsets.tobytes(), identities=b"".join(chunks))

    @classmethod
    def from_state_dict(cls, state):
        if state["version"] != cls.VERSION:
            raise ValueError("Unsupported point-identity format")
        result = cls(state["roots"], state["max_iteration"], state["seed"])
        event = state["last_event"]
        if type(event) is not int or not 0 <= event <= result.max_iteration:
            raise ValueError("Invalid last structural iteration")
        result.last_event = event
        raw, data = state["offsets"], state["identities"]
        if not isinstance(raw, bytes) or not isinstance(data, bytes) or len(raw) % 8:
            raise ValueError("Malformed identity checkpoint")
        offsets = array("Q")
        offsets.frombytes(raw)
        if sys.byteorder != "little":
            offsets.byteswap()
        if not offsets or offsets[0] != 0 or offsets[-1] != len(data):
            raise ValueError("Identity checkpoint length mismatch")
        ids = []
        for start, end in zip(offsets, offsets[1:]):
            if not 0 <= start < end <= len(data) or data[end - 1] == 0:
                raise ValueError("Noncanonical point identity")
            value = int.from_bytes(data[start:end], "little")
            result._validate_id(value)
            ids.append(value)
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate point identities")
        result.ids = ids
        return result


def standard_normals(ids, seed):
    """Use a stable per-identity random input; no global RNG state is consumed."""
    key = seed.to_bytes(8, "little")
    result = []
    for value in ids:
        encoded = value.to_bytes((value.bit_length() + 7) // 8, "little")
        digest = hashlib.blake2b(encoded, key=key, person=b"clad-split-v1", digest_size=32).digest()
        words = struct.unpack("<4Q", digest)
        # The open interval keeps Box-Muller finite, including rounded endpoints.
        u = [((word >> 12) + .5) / 2**52 for word in words]
        a, b = math.sqrt(-2 * math.log(u[0])), 2 * math.pi * u[1]
        c, d = math.sqrt(-2 * math.log(u[2])), 2 * math.pi * u[3]
        result.append((a * math.cos(b), a * math.sin(b), c * math.cos(d)))
    return result


class SplitSampler:
    """Wrap one native model's topology methods, leaving global Torch untouched."""

    def __init__(self, model, lineage):
        import torch

        if len(lineage.ids) != len(model.get_xyz):
            raise ValueError("Identity/model point count mismatch")
        self.model, self.lineage = model, lineage
        self.iteration = None
        self.pending = None
        self.children = None
        self.in_event = False
        self.seen = set()
        self.failed = False
        names = ("densify_and_prune", "densify_and_clone", "densify_and_split",
                 "densification_postfix", "prune_points")
        self.original = {name: getattr(model, name) for name in names}
        self.overrides = {name: model.__dict__.get(name) for name in names}
        owner = self

        class TorchProxy:
            def __getattr__(self, name):
                return getattr(torch, name)

            def normal(self, *, mean, std):
                if owner.children is None or tuple(std.shape) != (len(owner.children), 3):
                    raise RuntimeError("Unexpected native split sampling call")
                values = standard_normals(owner.children, lineage.seed)
                z = torch.tensor(values, dtype=std.dtype, device=std.device).reshape(-1, 3)
                return mean + std * z

        # Run the pinned native split body with a local Torch proxy. Other callers
        # and other models continue to see the original torch.normal function.
        original = self.original["densify_and_split"].__func__
        namespace = dict(original.__globals__, torch=TorchProxy())
        local = types.FunctionType(original.__code__, namespace, original.__name__,
                                   original.__defaults__, original.__closure__)
        local.__kwdefaults__ = original.__kwdefaults__
        native_split = types.MethodType(local, model)

        def selected(mask):
            flags = mask.detach().cpu().tolist()
            if len(flags) != len(lineage.ids):
                raise RuntimeError("Selection/identity count mismatch")
            return [value for value, keep in zip(lineage.ids, flags) if keep]

        def densify(_model, *args, **kwargs):
            if self.failed or self.in_event or self.iteration is None:
                raise RuntimeError("Invalid sampler lifecycle")
            if not lineage.last_event < self.iteration <= lineage.max_iteration:
                raise RuntimeError("Repeated or out-of-range structural event")
            self.in_event = True
            self.seen.clear()
            try:
                result = self.original["densify_and_prune"](*args, **kwargs)
                if self.pending is not None or self.seen != {"clone", "split"}:
                    raise RuntimeError("Incomplete native topology event")
                lineage.finish_event(self.iteration)
                return result
            except Exception:
                self.failed = True
                raise
            finally:
                self.in_event = False
                self.children = None

        def begin(operation):
            if not self.in_event or operation in self.seen or self.pending is not None:
                raise RuntimeError("Unexpected or repeated topology operation")
            self.seen.add(operation)

        def clone(_model, grads, threshold, extent):
            begin("clone")
            mask = (torch.norm(grads, dim=-1) >= threshold) & (
                model.get_scaling.max(1).values <= model.percent_dense * extent)
            self.pending = lineage.births(selected(mask), self.iteration, 1)
            return self.original["densify_and_clone"](grads, threshold, extent)

        def split(_model, grads, threshold, extent, N=2):
            begin("split")
            if N != 2 or threshold <= 0:
                raise RuntimeError("Expected the pinned positive-threshold, two-child split")
            padded = torch.zeros(len(model.get_xyz), device=grads.device)
            padded[:len(grads)] = grads.squeeze()
            mask = (padded >= threshold) & (
                model.get_scaling.max(1).values > model.percent_dense * extent)
            parents = selected(mask)
            self.pending = (lineage.births(parents, self.iteration, 2) +
                            lineage.births(parents, self.iteration, 3))
            self.children = self.pending
            try:
                return native_split(grads, threshold, extent, N)
            finally:
                self.children = None

        def postfix(_model, *args, **kwargs):
            if self.pending is None:
                raise RuntimeError("Untracked point append")
            expected = len(lineage.ids) + len(self.pending)
            result = self.original["densification_postfix"](*args, **kwargs)
            if len(model.get_xyz) != expected:
                raise RuntimeError("Point append/identity mismatch")
            lineage.ids.extend(self.pending)
            self.pending = None
            return result

        def prune(_model, mask):
            remaining = selected(~mask)
            result = self.original["prune_points"](mask)
            if len(model.get_xyz) != len(remaining):
                raise RuntimeError("Point prune/identity mismatch")
            lineage.ids = remaining
            return result

        for name, function in zip(names, (densify, clone, split, postfix, prune)):
            setattr(model, name, types.MethodType(function, model))

    def set_iteration(self, iteration):
        if self.in_event or self.failed or type(iteration) is not int:
            raise RuntimeError("Invalid sampler iteration boundary")
        if not self.lineage.last_event <= iteration <= self.lineage.max_iteration:
            raise ValueError("Iteration outside the sampler horizon")
        self.iteration = iteration

    def state_dict(self):
        if self.in_event or self.pending is not None or self.failed:
            raise RuntimeError("Cannot checkpoint a partial or failed topology event")
        if len(self.lineage.ids) != len(self.model.get_xyz):
            raise RuntimeError("Identity/model count mismatch at checkpoint")
        return self.lineage.state_dict()

    def close(self):
        if self.in_event:
            raise RuntimeError("Cannot remove an active sampler")
        for name, value in self.overrides.items():
            if value is None:
                delattr(self.model, name)
            else:
                setattr(self.model, name, value)
