"""Protocol files: loading, hashing and the minimal structural checks.

A protocol is a JSON document with ``schema == "mlst-local-package/1"``.  It is
the frozen scientific design of one package.  The runner reads it, never
writes it; a change of design is a new versioned file with a ``supersedes``
pointer, not an edit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "mlst-local-package/1"

REQUIRED_KEYS = (
    "schema", "package", "version", "title", "question", "hypotheses",
    "independent_unit", "estimands", "controls", "seed_namespaces",
    "inference", "stopping_rule", "failure_interpretation", "budget",
    "explicitly_not_claimed",
)

REQUIRED_SEED_NAMESPACES = ("development", "pilot", "confirmation")


class ProtocolError(ValueError):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path | str) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_protocol(path: Path | str) -> dict:
    """Load and structurally validate a protocol; returns the dict with
    ``_path`` and ``_sha256`` attached (both start with an underscore and are
    never part of the canonical hash)."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        P = json.loads(raw)
    except json.JSONDecodeError as e:  # pragma: no cover - message path
        raise ProtocolError(f"{path}: not valid JSON ({e})") from e
    validate_protocol(P, where=str(path))
    P["_path"] = str(path)
    P["_sha256"] = sha256_bytes(raw)
    return P


def validate_protocol(P: dict, *, where: str = "protocol") -> None:
    if not isinstance(P, dict):
        raise ProtocolError(f"{where}: protocol must be a JSON object")
    if P.get("schema") != SCHEMA:
        raise ProtocolError(f"{where}: schema must be {SCHEMA!r}, got {P.get('schema')!r}")
    missing = [k for k in REQUIRED_KEYS if k not in P]
    if missing:
        raise ProtocolError(f"{where}: missing required keys {missing}")
    ns = P["seed_namespaces"]
    for k in REQUIRED_SEED_NAMESPACES:
        if k not in ns:
            raise ProtocolError(f"{where}: seed_namespaces lacks {k!r}")
        rng = ns[k]
        if not (isinstance(rng, list) and len(rng) == 2 and all(isinstance(x, int) for x in rng) and rng[0] <= rng[1]):
            raise ProtocolError(f"{where}: seed_namespaces[{k!r}] must be [lo, hi] ints")
    # the three namespaces must be pairwise disjoint: development seeds can
    # never be confirmation seeds
    names = list(REQUIRED_SEED_NAMESPACES)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = ns[names[i]], ns[names[j]]
            if not (a[1] < b[0] or b[1] < a[0]):
                raise ProtocolError(f"{where}: seed namespaces {names[i]} and {names[j]} overlap: {a} vs {b}")
    budget = P["budget"]
    for k in ("estimator_fits_cap",):
        if k not in budget or not isinstance(budget[k], int) or budget[k] <= 0:
            raise ProtocolError(f"{where}: budget.{k} must be a positive int")
    if P.get("stopping_rule") in (None, "", "budget spent, or acceptance decided, whichever first"):
        raise ProtocolError(f"{where}: stopping_rule must be a concrete fixed allocation "
                            "(an outcome-dependent 'stop when decided' rule is not accepted)")


def protocol_hash(P: dict) -> str:
    """Hash of the canonical protocol content (private keys excluded)."""
    clean = {k: v for k, v in P.items() if not k.startswith("_")}
    return sha256_bytes(canonical_json(clean).encode("utf-8"))


def seed_in_namespace(P: dict, seed: int, namespace: str) -> bool:
    lo, hi = P["seed_namespaces"][namespace]
    return lo <= int(seed) <= hi


def namespace_of_seed(P: dict, seed: int) -> str | None:
    for name, (lo, hi) in P["seed_namespaces"].items():
        if lo <= int(seed) <= hi:
            return name
    return None
