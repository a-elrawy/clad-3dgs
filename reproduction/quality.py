"""Summarize held-out views for the pair runner."""

import math


def quality(modes):
    indexed = {}
    for mode, result in modes.items():
        rows = result["metrics"]
        indexed[mode] = {(r["split"], r["camera"]): r for r in rows}
        if len(indexed[mode]) != len(rows):
            raise ValueError("Duplicate quality view")
        if any(not math.isfinite(r[k]) for r in rows for k in ("objective", "psnr")):
            raise ValueError("Nonfinite quality metric")
    if indexed["native"].keys() != indexed["clad"].keys():
        raise ValueError("Quality views differ")
    splits, comparisons = {}, []
    for key, native in indexed["native"].items():
        clad = indexed["clad"][key]
        if native["objective"] <= 0 or clad["objective"] < 0:
            raise ValueError("Invalid objective")
        comparisons.append(dict(split=key[0], camera=key[1],
                                objective_increase_percent=100 * (clad["objective"] / native["objective"] - 1),
                                psnr_delta_db=clad["psnr"] - native["psnr"]))
    for split in ("train", "heldout"):
        keys = [key for key in indexed["native"] if key[0] == split]
        if not keys or (split == "heldout" and len(keys) != 38):
            raise ValueError("Missing declared held-out or diagnostic views")
        means = {mode: {field: math.fsum(indexed[mode][key][field] for key in keys) / len(keys)
                        for field in ("objective", "psnr")} for mode in indexed}
        rows = [r for r in comparisons if r["split"] == split]
        increase = means["clad"]["objective"] / means["native"]["objective"] - 1
        splits[split] = dict(views=len(keys), means=means,
                            mean_objective_increase_percent=100 * increase,
                            mean_psnr_delta_db=means["clad"]["psnr"] - means["native"]["psnr"],
                            worst_loss_view=max(rows, key=lambda r: r["objective_increase_percent"]),
                            worst_psnr_view=min(rows, key=lambda r: r["psnr_delta_db"]),
                            views_above_two_percent=sum(r["objective_increase_percent"] > 2 for r in rows))
    increase = splits["heldout"]["mean_objective_increase_percent"]
    return dict(passed=increase <= 2, within_one_percent_target=increase <= 1,
                splits=splits, comparisons=comparisons)
