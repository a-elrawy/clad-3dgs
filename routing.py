"""Count routing decisions in Python. No tensor reads, no CUDA sync."""

import json


class RoutingSummary:
    def __init__(self, path, configuration):
        self.path = path
        self.result = dict(
            status="running",
            configuration=configuration,
            completed_training_steps=0,
            first_iteration=None,
            last_iteration=None,
            sh=dict(consumed_steps=0, fallback_steps=0, fallback_reasons={}),
            activations=dict(consumed_steps=0, fallback_steps=0, fallback_reasons={}),
        )
        # Create the file first so a second run cannot overwrite it.
        with path.open("x") as output:
            output.write(json.dumps(self.result, indent=2) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        success = kind is None or (isinstance(error, SystemExit) and error.code in (None, 0))
        self.result["status"] = "completed" if success else "failed"
        if not success:
            self.result["error"] = f"{kind.__name__}: {error}"
        self.path.write_text(json.dumps(self.result, indent=2) + "\n")
        print(f"Routing summary: {self.path} ({self.result['status']}, "
              f"{self.result['completed_training_steps']} completed steps)")
        return False

    def capture(self, context, sh, activations):
        # Record the reason before evaluation overwrites it.
        for name, owner in (("sh", sh), ("activations", activations)):
            frame = owner.active if owner is not None else None
            consumed = bool(frame is not None and frame.consumed)
            if consumed:
                reason = None
            elif not context["enabled"]:
                reason = "consumption disabled"
            elif name == "activations" and not context["activation_enabled"]:
                reason = "activation consumption disabled"
            elif context["policy_reason"]:
                reason = context["policy_reason"]
            elif not context[name + "_called"]:
                reason = "unsupported renderer path"
            else:
                reason = owner.last_reason if owner is not None else None
                reason = reason or "backward did not consume"
            context[name] = dict(consumed=consumed, reason=reason)

    def record(self, context):
        for name in ("sh", "activations"):
            route, counts = context[name], self.result[name]
            if route["consumed"]:
                counts["consumed_steps"] += 1
            else:
                counts["fallback_steps"] += 1
                reasons = counts["fallback_reasons"]
                reason = route["reason"]
                reasons[reason] = reasons.get(reason, 0) + 1
        self.result["completed_training_steps"] += 1
        if self.result["first_iteration"] is None:
            self.result["first_iteration"] = context["iteration"]
        self.result["last_iteration"] = context["iteration"]
