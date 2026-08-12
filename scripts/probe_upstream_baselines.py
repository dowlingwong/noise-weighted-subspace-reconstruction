"""Route A probe: can the published TIDMAD baseline comparison be recovered?

Why
---
P1 found four validation files (0016-0019) with injections above the frozen
800.8 kHz band, so a band-limited arm cannot be compared to the published
20-file scores (FCNet 6.55, Transformer 4.18, PUNet 3.84, none 1.10).

The comparison is recoverable **if** the upstream release ships something that
lets us regenerate those baselines' denoised outputs. Then every method —
baselines and both arms — is scored over the same in-band subset with
``tidmad.score --in-band-only`` and the comparison is like-for-like again.

This script decides that, and prints the next command either way. It reads the
upstream checkout and the data directory; it modifies nothing.

Usage
-----
    python scripts/probe_upstream_baselines.py \\
        --upstream ~/src/TIDMAD \\
        --data-dir /ceph/dwong/paper1_dataset/tidmad/raw \\
        --out results/rps2026/RouteA_baseline_probe.json

Exit status: 0 recoverable, 2 not recoverable, 1 usage/IO error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Tags in the published table. The release's own naming is authoritative; these
# are candidates to look for, not an assertion about what exists.
PUBLISHED = {
    "none": 1.10,
    "savgol": -2.35,
    "moving_average": 0.64,
    "punet": 3.84,
    "transformer": 4.18,
    "fcnet": 6.55,
}
TAG_ALIASES = {
    "moving_average": ("moving_average", "movavg", "moveavg", "ma", "movingaverage"),
    "punet": ("punet", "pu_net"),
    "transformer": ("transformer", "xformer"),
    "fcnet": ("fcnet", "fc_net", "fcn"),
    "savgol": ("savgol", "sg"),
    "none": ("none",),
}

WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".tar", ".safetensors", ".bin")
WEIGHT_HINTS = ("checkpoint", "pretrained", "weights", "model_zoo", "ckpt")


def find_weights(upstream: Path) -> list[dict]:
    """Checkpoint-like files anywhere in the upstream checkout."""
    found = []
    for path in sorted(upstream.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        rel = str(path.relative_to(upstream)).lower()
        looks_like = path.suffix.lower() in WEIGHT_SUFFIXES or any(h in rel for h in WEIGHT_HINTS)
        if not looks_like:
            continue
        if path.suffix.lower() not in WEIGHT_SUFFIXES:
            continue  # a hinted directory alone is not a weight file
        found.append(
            {
                "path": str(path),
                "relative": str(path.relative_to(upstream)),
                "size_bytes": path.stat().st_size,
                "matched_tag": _tag_for(name),
            }
        )
    return found


def _tag_for(text: str) -> str | None:
    for canonical, aliases in TAG_ALIASES.items():
        if any(a in text for a in aliases):
            return canonical
    return None


def inspect_inference_cli(upstream: Path) -> dict:
    """What the release's own inference entry point accepts."""
    info: dict = {"script": None, "arguments": [], "model_choices": [], "notes": []}
    script = upstream / "inference.py"
    if not script.exists():
        candidates = sorted(upstream.rglob("inference.py"))
        script = candidates[0] if candidates else None
    if script is None:
        info["notes"].append("no inference.py found in the upstream checkout")
        return info

    info["script"] = str(script)
    text = script.read_text(errors="replace")
    info["arguments"] = sorted(set(re.findall(r"add_argument\(\s*[\"'](--[A-Za-z0-9_-]+)", text)))
    # choices=[...] on the model argument, if declared
    for m in re.finditer(r"choices\s*=\s*\[([^\]]+)\]", text):
        info["model_choices"] += re.findall(r"[\"']([A-Za-z0-9_-]+)[\"']", m.group(1))
    info["model_choices"] = sorted(set(info["model_choices"]))
    if not any(a in info["arguments"] for a in ("--checkpoint", "--model-path", "--weights", "--ckpt")):
        info["notes"].append(
            "inference.py exposes no checkpoint/weights argument; it may train "
            "before inferring, or load a hard-coded path"
        )
    return info


def existing_denoised(data_dir: Path) -> dict[str, int]:
    """Count denoised outputs already present, per tag."""
    counts: dict[str, int] = {}
    for path in sorted(data_dir.glob("abra_validation_denoised_*.h5")):
        m = re.match(r"abra_validation_denoised_(.+)_(\d{4})\.h5$", path.name)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.upstream.is_dir():
        print(f"upstream not a directory: {args.upstream}", file=sys.stderr)
        return 1
    if not args.data_dir.is_dir():
        print(f"data dir not a directory: {args.data_dir}", file=sys.stderr)
        return 1

    weights = find_weights(args.upstream)
    cli = inspect_inference_cli(args.upstream)
    present = existing_denoised(args.data_dir)

    print("=== checkpoint-like files in the upstream checkout ===")
    if weights:
        for w in weights:
            tag = f"  [tag guess: {w['matched_tag']}]" if w["matched_tag"] else ""
            print(f"  {w['relative']}  ({w['size_bytes'] / 1e6:.1f} MB){tag}")
    else:
        print("  none found")

    print("\n=== release inference entry point ===")
    print(f"  script: {cli['script']}")
    if cli["arguments"]:
        print(f"  arguments: {' '.join(cli['arguments'])}")
    if cli["model_choices"]:
        print(f"  model choices: {', '.join(cli['model_choices'])}")
    for n in cli["notes"]:
        print(f"  note: {n}")

    print("\n=== denoised outputs already on disk ===")
    if present:
        for tag, n in sorted(present.items()):
            published = PUBLISHED.get(_tag_for(tag) or "", None)
            extra = f"  (published 20-file score {published})" if published is not None else ""
            print(f"  {tag}: {n} files{extra}")
    else:
        print("  none")

    # A baseline is recoverable if its outputs already exist, or weights exist
    # for it and inference.py can be pointed at them.
    have_outputs = {(_tag_for(t) or t) for t, n in present.items() if n >= 16}
    have_weights = {w["matched_tag"] for w in weights if w["matched_tag"]}
    can_infer = bool(weights) and any(
        a in cli["arguments"] for a in ("--checkpoint", "--model-path", "--weights", "--ckpt")
    )
    recoverable = sorted((have_outputs | (have_weights if can_infer else set())) - {"none", None})
    strong = sorted(set(recoverable) & {"fcnet", "punet", "transformer"})

    verdict = {
        "recoverable_tags": recoverable,
        "strong_baselines_recoverable": strong,
        "have_outputs_on_disk": sorted(have_outputs),
        "weights_found": sorted(have_weights),
        "inference_accepts_checkpoint": can_infer,
        "weights": weights,
        "inference_cli": cli,
        "denoised_counts": present,
    }

    print("\n=== VERDICT ===")
    if strong:
        print(f"  RECOVERABLE for: {', '.join(strong)}")
        print("  Regenerate their outputs with the release's own inference script, then")
        print("  score every method over the same in-band subset:")
        print()
        print("    python -m tidmad.score --upstream $TIDMAD_UPSTREAM \\")
        print("      --data-dir $TIDMAD_DATA_ROOT --model-tag <tag> \\")
        print("      --in-band-only $RPS_RESULTS/P1_injection_survey.json \\")
        print("      --out $RPS_RESULTS/RouteA_<tag>_inband.json")
        status = 0
    else:
        print("  NOT RECOVERABLE from this checkout: no usable weights or outputs for")
        print("  FCNet / PUNet / Transformer.")
        print("  Fall back to Route C: report the published numbers as regime context,")
        print("  not as a comparison, and rest the claim on the paired T-mse/T-chi2")
        print("  contrast scored in band. Do NOT train the baselines yourself -- that is")
        print("  a separate project and not what this paper claims.")
        status = 2

    if present:
        print(f"\n  Note: {sorted(present)} already on disk; savgol is not a useful")
        print("  comparison point (the contract records it as not bit-reproducible).")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        print(f"\n  probe record written to {args.out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
