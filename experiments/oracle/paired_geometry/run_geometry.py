"""Step 2: replay the SHARED injection through one geometry.

`inject = False` + `paths.injection_file` is the documented Prometheus route
for re-using an injection (config_types.py:LeptonInjectorConfig.inject).
The re-centring below is NOT done by Prometheus and is the reason this file
exists -- see geo.py's module docstring.

Run once per geometry per pair.  Then run again on the SAME geometry with a
different `--photon-seed` to get the pairing null (check_pairing.py).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml

from .geo import load_geo, recentre_delta


def recentre_injection(src_file: Path, dst_file: Path, delta: np.ndarray) -> None:
    """Translate every vertex by `delta`, mirroring apply_detector_offset.

    Mutates a COPY: the reference injection file is immutable, because it is
    the pairing key.
    """
    shutil.copy(src_file, dst_file)
    with h5py.File(dst_file, "r+") as f:
        for key in f.keys():
            grp = f[key]
            if "Position" in grp:
                grp["Position"][...] = grp["Position"][...] + delta
            if "properties" in grp:
                props = grp["properties"]
                for i, ax in enumerate("xyz"):
                    if ax in props.dtype.names:
                        props[ax] = props[ax] + delta[i]


def override_medium(geofile: Path, medium: str, out: Path) -> Path:
    """NuBench ran all six geometries in water; the shipped headers disagree.

    Rewrites only the `Medium:` line.  Prometheus selects PPC (ice) vs olympus
    (water) from this, so it decides the whole photon-propagation branch.
    """
    lines = geofile.read_text().splitlines()
    out.write_text("\n".join(
        f"Medium:\t{medium}" if ln.strip().lower().startswith("medium") else ln
        for ln in lines
    ) + "\n")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", required=True)
    p.add_argument("--dataset", required=True, help="key in geometries.yaml:datasets")
    p.add_argument("--injection", type=Path, required=True, help="shared .h5")
    p.add_argument("--lic", type=Path, required=True, help="shared .lic")
    p.add_argument("--geodir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--photon-seed", type=int, default=1,
                   help="vary this, with everything else fixed, for the null")
    p.add_argument("--config-dir", type=Path, default=Path(__file__).parent / "config")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    geoms = yaml.safe_load((args.config_dir / "geometries.yaml").read_text())
    spec = geoms["pairs"][args.pair]
    ds = geoms["datasets"][args.dataset]
    ref_key = spec["a"] if geoms["datasets"][spec["a"]]["r_horizontal_m"] >= \
        geoms["datasets"][spec["b"]]["r_horizontal_m"] else spec["b"]

    g_ref = load_geo(args.geodir / geoms["datasets"][ref_key]["geofile"])
    g_dst = load_geo(args.geodir / ds["geofile"])
    delta = recentre_delta(g_ref, g_dst)

    args.out.mkdir(parents=True, exist_ok=True)
    local_inj = args.out / f"injection_{args.dataset}.h5"
    recentre_injection(args.injection, local_inj, delta)
    geo_local = override_medium(
        args.geodir / ds["geofile"], ds["medium_nubench"],
        args.out / f"{args.dataset}.geo",
    )
    print(f"recentred by {delta} -> {local_inj}")
    print(f"medium set to {ds['medium_nubench']} -> {geo_local}")

    if not args.execute:
        return

    from prometheus import Prometheus, config          # noqa: PLC0415
    config.detector.geo_file = str(geo_local)
    config.run.storage_prefix = str(args.out) + "/"
    config.run.random_state_seed = args.photon_seed
    config.injection.name = "LeptonInjector"
    config.injection.lepton_injector.inject = False        # <-- the pairing switch
    config.injection.lepton_injector.paths.injection_file = str(local_inj)
    config.injection.lepton_injector.paths.lic_file = str(args.lic)
    if ds["medium_nubench"] == "ice":
        config.photon_propagator.name = "PPC"
        config.photon_propagator.ppc.paths.force = True
    Prometheus().sim()


if __name__ == "__main__":
    main()
