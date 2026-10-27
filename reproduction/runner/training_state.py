"""Checkpoint sidecars for a paired, staged native training experiment."""

from pathlib import Path
import hashlib
import random

from sampling import Lineage


class CameraOrder:
    """Replay the native random-without-replacement camera selection on resume."""

    def __init__(self, names, rng_state=None):
        if not names or len(set(names)) != len(names):
            raise ValueError("Expected a nonempty set of distinct camera names")
        self.names = list(names)
        self.remaining = list(range(len(names)))
        self.draws = 0
        self.rng = random.Random()
        self.rng.setstate(random.getstate() if rng_state is None else rng_state)

    def next(self):
        if not self.remaining:
            self.remaining = list(range(len(self.names)))
        index = self.remaining.pop(self.rng.randint(0, len(self.remaining) - 1))
        self.draws += 1
        return self.names[index], index

    def state_dict(self):
        return dict(version=1, names=self.names.copy(), remaining=self.remaining.copy(),
                    draws=self.draws, rng=self.rng.getstate())

    @classmethod
    def from_state_dict(cls, state):
        if state["version"] != 1:
            raise ValueError("Unsupported camera-order state")
        result = cls(state["names"], state["rng"])
        remaining, draws = state["remaining"], state["draws"]
        if type(draws) is not int or draws < 0:
            raise ValueError("Invalid camera draw count")
        if any(type(i) is not int or not 0 <= i < len(result.names) for i in remaining):
            raise ValueError("Invalid remaining camera index")
        expected = len(result.names) if draws == 0 else (-draws) % len(result.names)
        if len(set(remaining)) != len(remaining) or len(remaining) != expected:
            raise ValueError("Incomplete camera order checkpoint")
        result.remaining, result.draws = list(remaining), draws
        return result


def capture_rng():
    import numpy as np
    import torch

    numpy = np.random.get_state()
    return dict(python=random.getstate(),
                numpy=(numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch

    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("CUDA device count differs from the saved RNG state")
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sidecar_path(checkpoint):
    return Path(str(checkpoint) + ".state.pt")


def save_bundle(checkpoint, native_state, iteration, sampler, cameras, metadata, auxiliary=None):
    """Preserve the native checkpoint tuple and publish its sidecar last."""
    import torch

    checkpoint = Path(checkpoint)
    companion = sidecar_path(checkpoint)
    if checkpoint.exists() or companion.exists():
        raise FileExistsError(checkpoint)
    if type(iteration) is not int or not 0 <= iteration <= sampler["max_iteration"]:
        raise ValueError("Checkpoint iteration outside declared horizon")
    if cameras["draws"] != iteration or sampler["last_event"] > iteration:
        raise ValueError("Checkpoint components are at different iterations")
    # Validate the serialized identity and camera forms before writing anything.
    Lineage.from_state_dict(sampler)
    order = CameraOrder.from_state_dict(cameras)
    state = dict(version=1, iteration=iteration, metadata=metadata,
                 sampler=sampler, cameras=order.state_dict(), rng=capture_rng(),
                 auxiliary={} if auxiliary is None else auxiliary)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("xb") as stream:
        torch.save((native_state, iteration), stream)
    state["checkpoint_sha256"] = file_digest(checkpoint)
    with companion.open("xb") as stream:
        torch.save(state, stream)
    return dict(checkpoint=str(checkpoint), sidecar=str(companion),
                checkpoint_sha256=state["checkpoint_sha256"],
                sidecar_sha256=file_digest(companion))


def load_sidecar(checkpoint, metadata):
    import torch

    checkpoint = Path(checkpoint)
    state = torch.load(sidecar_path(checkpoint), map_location="cpu", weights_only=True)
    if state["version"] != 1 or state["metadata"] != metadata:
        raise ValueError("Checkpoint configuration/source metadata does not match")
    if file_digest(checkpoint) != state["checkpoint_sha256"]:
        raise ValueError("Native checkpoint and sidecar do not match")
    lineage = Lineage.from_state_dict(state["sampler"])
    order = CameraOrder.from_state_dict(state["cameras"])
    iteration = state["iteration"]
    if (type(iteration) is not int or not 0 <= iteration <= lineage.max_iteration or
            order.draws != iteration or lineage.last_event > iteration):
        raise ValueError("Inconsistent checkpoint iteration")
    return state, lineage, order


def load_bundle(checkpoint, metadata, *, map_location="cuda"):
    import torch

    state, lineage, order = load_sidecar(checkpoint, metadata)
    # This is the trainer's unchanged local checkpoint format. In particular,
    # native scalar fields may include NumPy types not accepted by weights_only.
    native, native_iteration = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if native_iteration != state["iteration"]:
        raise ValueError("Native checkpoint iteration differs from its sidecar")
    return native, state, lineage, order
