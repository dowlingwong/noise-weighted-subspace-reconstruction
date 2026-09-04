"""Generate ONE LeptonInjector file, to be shared by both geometries.

This is step 1 of the pairing.  It runs Prometheus with ``inject=True`` on the
*reference* geometry of the pair (by convention the larger one), with the
injection cylinder sized by `geo.shared_cylinder` so the file also serves the
other geometry after re-centring.

The output .h5 + .lic pair is the pairing key.  Nothing downstream may
regenerate it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .geo import load_geo, shared_cylinder


def build(pair: str, cfg_dir: Path, geodir: Path, out: Path) -> dict:
    geoms = yaml.safe_load((cfg_dir / "geometries.yaml").read_text())
    inj = yaml.safe_load((cfg_dir / "injection.yaml").read_text())
    spec = geoms["pairs"][pair]
    a = geoms["datasets"][spec["a"]]
    b = geoms["datasets"][spec["b"]]

    ga = load_geo(geodir / a["geofile"])
    gb = load_geo(geodir / b["geofile"])
    ref, other = (a, b) if ga.r_horizontal >= gb.r_horizontal else (b, a)
    gref = ga if ref is a else gb

    cyl = shared_cylinder(
        ga, gb,
        radius_margin=inj["cylinder"]["radius_margin_m"],
        height_margin=inj["cylinder"]["height_margin_m"],
    )

    manifest = {
        "pair": pair,
        "reference_geometry": ref["geofile"],
        "other_geometry": other["geofile"],
        "reference_offset_m": gref.offset.tolist(),
        "cylinder": cyl,
        "n_events": inj["n_events_per_pair"],
        "seed": inj["random_state_seed"],
        "energy_range_gev": [
            max(a["energy_range_gev"][0], b["energy_range_gev"][0]),
            min(a["energy_range_gev"][1], b["energy_range_gev"][1]),
        ],
        "power_law": inj["power_law"],
        "zenith_deg": [inj["min_zenith"], inj["max_zenith"]],
        "is_ranged": inj["is_ranged"],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "injection_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run(manifest: dict, out: Path) -> None:
    """Actually call Prometheus.  Import is local so the manifest step above
    stays runnable without a Prometheus install (that is what CI checks)."""
    from prometheus import Prometheus, config          # noqa: PLC0415

    config.run.nevents = manifest["n_events"]
    config.run.random_state_seed = manifest["seed"]
    config.run.storage_prefix = str(out) + "/"

    config.detector.geo_file = manifest["reference_geometry_path"]

    config.injection.name = "LeptonInjector"
    config.injection.lepton_injector.inject = True
    sim = config.injection.lepton_injector.simulation
    sim.is_ranged = manifest["is_ranged"]
    sim.final_state_1 = "MuMinus"
    sim.final_state_2 = "Hadrons"
    sim.minimal_energy = manifest["energy_range_gev"][0]
    sim.maximal_energy = manifest["energy_range_gev"][1]
    sim.power_law = manifest["power_law"]
    sim.min_zenith, sim.max_zenith = manifest["zenith_deg"]
    sim.cylinder_radius = manifest["cylinder"]["cylinder_radius_m"]
    sim.cylinder_height = manifest["cylinder"]["cylinder_height_m"]

    Prometheus().sim()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", required=True)
    p.add_argument("--config-dir", type=Path, default=Path(__file__).parent / "config")
    p.add_argument("--geodir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--execute", action="store_true", help="actually run Prometheus")
    args = p.parse_args()
    m = build(args.pair, args.config_dir, args.geodir, args.out)
    print(json.dumps(m, indent=2))
    if args.execute:
        m["reference_geometry_path"] = str(args.geodir / m["reference_geometry"])
        run(m, args.out)
