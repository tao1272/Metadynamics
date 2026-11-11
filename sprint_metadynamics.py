"""SPRINT metadynamics for a small, non-periodic molecular cluster.

Reference: F. Pietrucci and W. Andreoni, Phys. Rev. Lett. 107, 085504
(2011), https://doi.org/10.1103/PhysRevLett.107.085504, Eqs. (2)-(3).

Only NumPy is required. Coordinates and r0 are in angstrom, energies in
hartree. Returned Cartesian FORCES are in hartree/bohr, matching
``-result.get_gradient()`` from the xTB Python API. Do not divide these
forces by masses here; add them before your existing acceleration conversion.

This is ordinary fixed-height, multidimensional Gaussian metadynamics.
The switching function is the paper's Eq. (3) with m = 2*n:
    (1 - x**n)/(1 - x**(2*n)) = 1/(1 + x**n).
This identity removes the apparent 0/0 at r = r0 exactly. No distance
cutoff is used, preserving the weak connections between fragments.

All selected atoms enter ONE contact matrix. Sorting is performed separately
within each element, in alphabetical element order. cv_indices selects
components of this sorted vector; atom_indices selects atoms in the graph.
Sorting is piecewise differentiable: equal components can produce a cusp.
The implementation takes the stable-sort branch at an exact tie. A nearly
degenerate largest eigenvalue raises an error instead of clipping derivatives.

Example integration for an EC/Li atom list (trial parameters):

    # Covalent radii in angstrom, Cordero et al., DOI:10.1039/B801115J.
    # The factor 1.2 is an illustrative heuristic, not a SPRINT-paper setting.
    rcov = {"H": 0.31, "C": 0.76, "O": 0.66, "Li": 1.28}
    r0_pairs = {(a, b): 1.2 * (rcov[a] + rcov[b])
                for a in set(atoms) for b in set(atoms)}

    mtd = SPRINTMetadynamics(
        atoms, r0=r0_pairs, height=1e-4, sigma=0.1, pace=500,
    )

    # Use this inside get_acceleration, at the initial step and after each
    # position update, BEFORE converting force to acceleration:
    ebias, fbias, sprint = mtd(coords, step=0)
    forces = forces_xtb + forces_wall + fbias
    # At subsequent MD steps use the current integer step instead of 0:
    ebias, fbias, sprint = mtd(coords, step=step)
    forces = forces_xtb + forces_wall + fbias

Call at every force evaluation, not just when depositing a hill. The first
hill is placed at step=0 by default; subsequent hills every ``pace`` steps.
Passing step=None evaluates the current bias WITHOUT depositing a hill.
Repeated calls at the same step deposit at most one hill. Energy includes
the newly deposited hill; its force at the deposition point is zero.

The example height, sigma, pace, and default n=6 are illustrative choices,
not calibrated EC/Li parameters or parameters recovered from the supplement.
sigma is dimensionless, unlike the width in RMSD metadynamics. The width
must be calibrated from SPRINT fluctuations; the old RMSD alpha cannot be
transferred. The bias cost grows with the number of stored hills.
"""

import operator

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903


def _integer(value, name, minimum=0):
    try:
        number = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, (bool, np.bool_)) or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return number


def _indices(values, size, name):
    if values is None:
        return np.arange(size)
    result = np.asarray([_integer(v, name) for v in values], dtype=int)
    if result.size == 0 or np.any(result >= size):
        raise ValueError(f"{name} must be nonempty and contain indices < {size}")
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} must not contain duplicate indices")
    return result


def _pair_lengths(r0, elements):
    """Accept a scalar, graph-sized symmetric matrix, or element-pair dict."""
    n = len(elements)
    if isinstance(r0, dict):
        lengths = np.ones((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                pair = (elements[i], elements[j])
                reverse = pair[::-1]
                if pair in r0 and reverse in r0 and r0[pair] != r0[reverse]:
                    raise ValueError(f"Conflicting r0 values for {pair}")
                if pair in r0:
                    value = r0[pair]
                elif reverse in r0:
                    value = r0[reverse]
                else:
                    raise ValueError(f"Missing r0 in angstrom for pair {pair}")
                lengths[i, j] = lengths[j, i] = value
    else:
        lengths = np.asarray(r0, dtype=float)
        if lengths.ndim == 0:
            lengths = np.full((n, n), float(lengths))
        elif lengths.shape != (n, n):
            raise ValueError(f"r0 must be a scalar or an ({n}, {n}) matrix")
        else:
            lengths = lengths.copy()
    if not np.isfinite(lengths).all() or not np.allclose(
        lengths, lengths.T, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("r0 must be finite and symmetric")
    pair_i, pair_j = np.triu_indices(n, 1)
    if np.any(lengths[pair_i, pair_j] <= 0):
        raise ValueError("All off-diagonal r0 values must be positive")
    return lengths


class SPRINTMetadynamics:
    """A callable SPRINT bias: ``ebias, fbias, s = mtd(coords, step)``.

    Parameters
    ----------
    atoms : sequence of str
        Element symbols in the same order as the full MD coordinate array.
    r0 : float, dict, or ndarray
        Switching distances in angstrom. Dict keys are pairs of element
        symbols, e.g. ("C", "O"). Supply every pair present in the graph.
        A matrix must have shape (n_graph_atoms, n_graph_atoms), ordered
        according to atom_indices. One scalar applies to every pair.
    height : float
        Height of each multidimensional Gaussian hill, in hartree (>0).
    sigma : float or array
        Dimensionless Gaussian standard deviation, one per biased SPRINT
        component. One scalar applies to all selected CVs.
    pace : int
        Number of MD steps between hill depositions.
    nn : int
        Numerator exponent in Eq. (3). Denominator exponent is 2*nn.
        Default: 6 and 12, respectively. This code deliberately supports
        only m=2*n, using the exact, nonsingular simplified expression.
    atom_indices : sequence of int, optional
        Zero-based atom indices included in the graph. Excluded atoms retain
        zero DIRECT bias force, but still participate in the physical MD.
    cv_indices : sequence of int, optional
        Zero-based ranks in the element-wise sorted SPRINT vector to bias.
        By default all n_graph_atoms SPRINT coordinates are biased jointly.
    start_step : int
        First hill deposition, default 0. Calls before this step deposit none.

    Notes
    -----
    Return values are bias energy (hartree), force array (hartree/bohr,
    full MD shape), and the FULL sorted SPRINT vector (dimensionless).
    This method does not run the electronic structure calculation, integrate
    positions, apply a thermostat, or include a wall potential. Keep those
    operations in your existing MD loop. Do not add the old RMSD bias as well
    unless you intentionally want both biases.

    No periodic boundary conditions, minimum-image convention, virial,
    Gaussian truncation, or well-tempered height rescaling are implemented.
    Use continuous Cartesian coordinates for your isolated/confined cluster.
    """

    def __init__(self, atoms, r0, height, sigma, pace, *, nn=6,
                 atom_indices=None, cv_indices=None, start_step=0):
        self.atoms = np.asarray(list(atoms), dtype=str)
        if self.atoms.ndim != 1 or not self.atoms.size or np.any(self.atoms == ""):
            raise ValueError("atoms must be a nonempty sequence of element symbols")
        self.atom_indices = _indices(atom_indices, len(self.atoms), "atom_indices")
        self.elements = self.atoms[self.atom_indices]
        self.n = len(self.atom_indices)
        if self.n < 2:
            raise ValueError("The SPRINT graph needs at least two atoms")
        self.r0 = _pair_lengths(r0, self.elements)
        self.nn = _integer(nn, "nn", minimum=2)
        self.pace = _integer(pace, "pace", minimum=1)
        self.start_step = _integer(start_step, "start_step")
        self.cv_indices = _indices(cv_indices, self.n, "cv_indices")
        self.height = float(height)
        if not np.isfinite(self.height) or self.height <= 0:
            raise ValueError("height must be positive and finite")
        self.sigma = np.broadcast_to(
            np.asarray(sigma, dtype=float), (len(self.cv_indices),)
        ).copy()
        if not np.isfinite(self.sigma).all() or np.any(self.sigma <= 0):
            raise ValueError("sigma must contain positive, finite CV widths")
        self._groups = [np.flatnonzero(self.elements == e)
                        for e in sorted(set(self.elements))]
        self._i, self._j = np.triu_indices(self.n, 1)
        self._r0_pairs = self.r0[self._i, self._j]
        self._centers = []
        self._hill_steps = []

    @property
    def centers(self):
        """Copy of Gaussian centers, shape (number_of_hills, number_of_CVs)."""
        return np.asarray(self._centers, dtype=float).reshape(
            -1, len(self.cv_indices)
        ).copy()

    @property
    def hill_steps(self):
        """Copy of deposition steps."""
        return np.asarray(self._hill_steps, dtype=int)

    @property
    def cv_elements(self):
        """Element label of each component of the FULL returned SPRINT vector."""
        return np.concatenate([self.elements[g] for g in self._groups])

    def _sprint(self, coords):
        xyz = np.asarray(coords, dtype=float)
        if xyz.shape != (len(self.atoms), 3) or not np.isfinite(xyz).all():
            raise ValueError(f"coords must be finite with shape ({len(self.atoms)}, 3)")
        graph_xyz = xyz[self.atom_indices]
        displacement = graph_xyz[self._i] - graph_xyz[self._j]
        distance = np.linalg.norm(displacement, axis=1)
        if not np.isfinite(distance).all() or np.any(distance <= 0):
            raise ValueError("Pair distances must be finite and nonzero")

        # s(r)=1/(1+(r/r0)**nn), evaluated without overflow at large r/r0.
        # logaddexp also evaluates r=r0 exactly: contact=1/2.
        log_power = self.nn * (np.log(distance) - np.log(self._r0_pairs))
        contact = np.exp(-np.logaddexp(0.0, log_power))
        dcontact_dr = -self.nn * contact * (1.0 - contact) / distance
        adjacency = np.zeros((self.n, self.n))  # A_ii = 0: no self-contacts.
        adjacency[self._i, self._j] = contact
        adjacency[self._j, self._i] = contact

        eigenvalues, eigenvectors = np.linalg.eigh(adjacency)
        largest = eigenvalues[-1]
        gap = largest - eigenvalues[-2]
        if gap <= 1e-10 * max(1.0, abs(largest)):
            raise FloatingPointError(
                "Largest SPRINT eigenvalue is nearly degenerate. The eigenvector "
                "derivative is ill-conditioned; check fragment separation and r0."
            )
        principal = eigenvectors[:, -1].copy()
        if principal.sum() < 0:
            principal *= -1  # One global sign change, preserving derivatives.
        order = np.concatenate([
            group[np.argsort(principal[group], kind="stable")]
            for group in self._groups
        ])
        sprint = np.sqrt(self.n) * largest * principal[order]
        data = (largest, principal, eigenvalues[:-1], eigenvectors[:, :-1],
                order, displacement / distance[:, None], dcontact_dr)
        return sprint, data

    def coordinates(self, coords):
        """Calculate the full SPRINT vector without modifying bias history."""
        return self._sprint(coords)[0]

    def __call__(self, coords, step=None):
        """Evaluate bias/forces; deposit once if step is a scheduled hill step."""
        if step is not None:
            step = _integer(step, "step")
            if self._hill_steps and step < self._hill_steps[-1]:
                raise ValueError("step precedes the last hill; use step=None to evaluate")
        sprint, data = self._sprint(coords)
        values = sprint[self.cv_indices]

        if (step is not None and step >= self.start_step
                and (step - self.start_step) % self.pace == 0
                and (not self._hill_steps or step != self._hill_steps[-1])):
            self._centers.append(values.copy())
            self._hill_steps.append(step)

        if not self._centers:
            return 0.0, np.zeros((len(self.atoms), 3)), sprint

        delta = values[None, :] - np.asarray(self._centers)
        scaled = delta / self.sigma
        hill_energy = self.height * np.exp(-0.5 * np.sum(scaled**2, axis=1))
        energy = float(hill_energy.sum())
        # dV/dS, including every stored multidimensional Gaussian.
        dV_dS = -np.sum(hill_energy[:, None] * scaled / self.sigma, axis=0)

        largest, principal, other_values, other_vectors, order, unit, da_dr = data
        # Undo sorting/selection so g_i=dV/d(sqrt(N)*lambda*v_i) in atom order.
        g = np.zeros(self.n)
        g[order[self.cv_indices]] = dV_dS

        # Symmetric-matrix perturbation theory, contracted with dV/dS:
        # d(lambda) = v.T dA v
        # dv = sum_k u_k (u_k.T dA v)/(lambda-lambda_k).
        # Only one contracted eigenvector response is needed, avoiding the
        # complete (n_CVs, n_atoms, 3) Jacobian at every MD step.
        response = other_vectors @ (
            (other_vectors.T @ g) / (largest - other_values)
        )
        vi, vj = principal[self._i], principal[self._j]
        dV_da = np.sqrt(self.n) * (
            2.0 * np.dot(g, principal) * vi * vj
            + largest * (response[self._i] * vj + response[self._j] * vi)
        )
        # A single upper-triangle pair changes BOTH A_ij and A_ji. The factor
        # two for d(lambda) is already included above: do not count it again.
        pair_force = -(dV_da * da_dr)[:, None] * unit  # hartree/angstrom
        graph_force = np.zeros((self.n, 3))
        np.add.at(graph_force, self._i, pair_force)
        np.add.at(graph_force, self._j, -pair_force)
        force = np.zeros((len(self.atoms), 3))
        # F[Eh/bohr] = F[Eh/angstrom] * (angstrom/bohr).
        force[self.atom_indices] = graph_force * BOHR_TO_ANGSTROM
        return energy, force, sprint
