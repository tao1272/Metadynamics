"""Analytic g-xTB forces for tao1272/Metadynamics/my_md_rmsd.py.

Requires the modified xtb executable from https://github.com/grimme-lab/g-xtb.
Coordinates are supplied in angstrom; forces are returned in Hartree/bohr
and energy in Hartree. The force sign and units match calc_xtb_gradients()
and get_acceleration() in the repository inspected on 2026-11-17.

Example (use at initialization and in BOTH MD loops):
    forces, energy = calc_gxtb_gradients(
        coords, atoms, charge=charge, multi=multi,
        executable="/absolute/path/to/gxtb-package/bin/xtb",
        ncores=XTB_CORES,
        electronic_temperature=6000.0,
        workdir="gxtb_run",
    )
    accs = get_acceleration(forces, masses)

Use one workdir per trajectory; concurrent calls must use different directories.
The directory retains xtbrestart for the next step, while gxtb.out is overwritten.
For a fresh independent trajectory, use a new workdir. This interface launches
one executable per force evaluation, unlike an in-process Calculator object.
"""

from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np


@lru_cache(maxsize=8)
def _gxtb_executable(executable):
    """Resolve before changing cwd; reject a regular xTB without --gxtb."""
    path = shutil.which(os.path.expanduser(os.fspath(executable)))
    if path is None:
        raise FileNotFoundError(f"g-xTB executable not found: {executable}")
    path = str(Path(path).resolve())
    probe = subprocess.run(
        [path, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=30,
    )
    if probe.returncode != 0 or "--gxtb" not in probe.stdout:
        raise RuntimeError(
            f"{path} does not advertise --gxtb. Use the modified executable "
            "from https://github.com/grimme-lab/g-xtb."
        )
    return path


def _read_engrad(path, natoms):
    """Read the documented ORCA-format output; accept Fortran D exponents."""
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 2 + 3 * natoms or int(lines[0]) != natoms:
        raise ValueError("Incorrect atom count or incomplete .engrad output")
    energy = float(lines[1].replace("D", "E").replace("d", "e"))
    gradient = np.array([
        float(line.replace("D", "E").replace("d", "e"))
        for line in lines[2:2 + 3 * natoms]
    ], dtype=np.float64).reshape(natoms, 3)
    if not np.isfinite(energy) or not np.all(np.isfinite(gradient)):
        raise ValueError("Non-finite energy or gradient in .engrad output")
    return -gradient, energy


def calc_gxtb_gradients(
    coords, atoms, charge=0, multi=1, *, executable="xtb", ncores=1,
    electronic_temperature=6000.0, accuracy=1.0, max_iterations=250,
    workdir="gxtb_run", restart=True, timeout=None,
):
    """Return (forces, energy), using g-xTB analytic gradients.

    coords: Cartesian array (natoms, 3), ANGSTROM (no conversion before XYZ).
    atoms: element symbols, in the same order as coords (not atomic numbers).
    charge: integer total charge; multi: positive integer spin multiplicity.
    multi > 1 passes --uhf multi-1, selecting unrestricted g-xTB. For multi=1,
    --uhf is omitted to preserve the binary's default restricted treatment.
    electronic_temperature is in kelvin; 6000 matches the existing MD script.
    Set restart=False to disable reading/writing the native restart file.

    Returns forces = -dE/dR in HARTREE/BOHR, float64, shape (natoms, 3),
    and the energy from the same .engrad calculation in HARTREE. Mass division
    and conversion to acceleration are intentionally left to get_acceleration().
    """
    x = np.asarray(coords, dtype=np.float64)
    symbols = [str(atom) for atom in atoms]
    if not symbols or x.shape != (len(symbols), 3) or not np.all(np.isfinite(x)):
        raise ValueError("coords must be a finite (len(atoms), 3) array")
    if any(re.fullmatch(r"[A-Z][a-z]?", symbol) is None for symbol in symbols):
        raise ValueError("atoms must contain element symbols, e.g. ['O', 'H', 'Li']")
    for name, value, minimum in (
        ("multi", multi, 1), ("ncores", ncores, 1),
        ("max_iterations", max_iterations, 1),
    ):
        if int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if int(charge) != charge:
        raise ValueError("charge must be an integer")
    if not np.isfinite(accuracy) or accuracy <= 0:
        raise ValueError("accuracy must be finite and positive")
    if not np.isfinite(electronic_temperature) or electronic_temperature < 0:
        raise ValueError("electronic_temperature must be finite and nonnegative")

    binary = _gxtb_executable(os.fspath(executable))
    run_dir = Path(workdir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    xyz = run_dir / "gxtb_input.xyz"
    engrad = run_dir / "gxtb_input.engrad"
    log = run_dir / "gxtb.out"
    # Prevent a failed calculation from reusing a previous step's gradient.
    for name in (engrad.name, ".sccnotconverged", "NOT_CONVERGED"):
        (run_dir / name).unlink(missing_ok=True)
    # A .UHF file would override the intended restricted singlet default.
    if multi == 1 and (run_dir / ".UHF").exists():
        raise ValueError("Use a clean workdir: .UHF would force unrestricted g-xTB")
    with xyz.open("w") as handle:
        handle.write(f"{len(symbols)}\ng-xTB MD gradient\n")
        for atom, (px, py, pz) in zip(symbols, x):
            handle.write(f"{atom:2s} {px:.16f} {py:.16f} {pz:.16f}\n")

    command = [
        binary, xyz.name, "--strict", "--gxtb", "--grad", "--silent",
        "--chrg", str(int(charge)), "--acc", str(float(accuracy)),
        "--etemp", str(float(electronic_temperature)),
        "--iterations", str(int(max_iterations)),
    ]
    if multi > 1:
        command.extend(["--uhf", str(int(multi) - 1)])
    if not restart:
        command.append("--norestart")
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(int(ncores)),
        "MKL_NUM_THREADS": str(int(ncores)),
        "OPENBLAS_NUM_THREADS": str(int(ncores)),
        "OMP_MAX_ACTIVE_LEVELS": "1",
    })
    with log.open("w") as output:
        try:
            completed = subprocess.run(
                command, cwd=run_dir, env=env, stdout=output,
                stderr=subprocess.STDOUT, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"g-xTB timed out; see {log}") from exc
    failed_marker = any(
        (run_dir / name).exists() for name in (".sccnotconverged", "NOT_CONVERGED")
    )
    if completed.returncode != 0 or failed_marker or not engrad.is_file():
        tail = log.read_text(errors="replace")[-2500:]
        raise RuntimeError(
            f"g-xTB failed (exit {completed.returncode}); see {log}\n{tail}"
        )
    try:
        return _read_engrad(engrad, len(symbols))
    except (ValueError, OSError, IndexError) as exc:
        raise RuntimeError(f"Invalid g-xTB gradient output; see {log}") from exc
