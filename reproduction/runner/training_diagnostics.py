"""Keep useful state when a training stage fails."""

from pathlib import Path
import json
import math


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "+Inf" if value > 0 else "-Inf"
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def describe_state(model, torch, identities=(), limit=8):
    """Describe nonfinite fields and a bounded set of affected point rows."""
    fields, samples = [], []
    for group in model.optimizer.param_groups:
        name, parameter = group["name"], group["params"][0]
        state = model.optimizer.state.get(parameter, {})
        tensors = {"parameter": parameter}
        tensors.update(
            {key: state[key] for key in ("exp_avg", "exp_avg_sq") if key in state}
        )
        if parameter.grad is not None:
            tensors["gradient"] = parameter.grad
        points = set()
        for key, tensor in tensors.items():
            if bool(torch.isfinite(tensor).all()):
                continue
            count, indices = 0, []
            flat = tensor.detach().reshape(-1)
            # Bound temporary index storage even if an entire field is invalid.
            for offset in range(0, flat.numel(), 65536):
                bad = ~torch.isfinite(flat[offset : offset + 65536])
                count += int(bad.sum())
                if len(indices) < limit:
                    found = torch.nonzero(bad).reshape(-1)[: limit - len(indices)]
                    indices.extend(offset + int(index) for index in found.cpu())
            width = tensor.numel() // len(parameter)
            points.update(index // width for index in indices)
            fields.append(
                dict(
                    group=name,
                    tensor=key,
                    shape=list(tensor.shape),
                    nonfinite_count=count,
                    first_flat_indices=indices,
                )
            )
        for index in sorted(points)[:limit]:
            row = dict(
                group=name,
                point_index=index,
                values={
                    key: tensor.detach()[index].cpu().tolist()
                    for key, tensor in tensors.items()
                },
            )
            if index < len(identities):
                row["identity"] = str(identities[index])
            samples.append(row)
    return _json_value(dict(nonfinite_fields=fields, point_rows=samples))


def write_failure(directory, error, stage):
    """Preserve the original error even if inspecting the device also fails."""
    directory = Path(directory)
    path = directory / "failure.json"
    record = dict(
        error=str(error),
        iteration=getattr(stage, "completed_iteration", None),
        mode=getattr(stage, "mode", None),
        camera=getattr(stage, "camera_name", None),
    )
    path.write_text(json.dumps(record, indent=2, allow_nan=False))
    try:
        model = getattr(stage, "model", None)
        if model is not None:
            identities = getattr(getattr(stage, "lineage", None), "ids", ())
            record["state"] = describe_state(model, stage.torch, identities)
    except Exception as diagnostic_error:
        record["state_diagnostic_error"] = str(diagnostic_error)
    try:
        rows = getattr(stage, "rows", ())
        if rows:
            # CUDA events may be unusable after an error. Keep scalar history.
            history = [
                {
                    key: value
                    for key, value in row.items()
                    if key not in ("start", "end")
                }
                for row in rows
            ]
            history_path = directory / "failure-iterations.json"
            history_path.write_text(json.dumps(_json_value(history), allow_nan=False))
            record["iteration_history"] = str(history_path)
    except Exception as history_error:
        record["history_diagnostic_error"] = str(history_error)
    path.write_text(json.dumps(record, indent=2, allow_nan=False))
