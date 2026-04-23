# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Warning: this module is not documented and it does not have an RST file.
# If we ever publicly expose interfaces users can import from this module,
# we should set up its RST file.

"""Forward evolved unequal time commutator bounds."""

from __future__ import annotations

import logging
from functools import partial

import numpy as np
import scipy
from pauli_prop.propagation import (
    RotationGates,
    propagate_through_rotation_gates,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Pauli,
    PauliList,
    QubitSparsePauliList,
    SparseObservable,
    SparsePauliOp,
)
from qiskit.utils import deprecate_arg

from .. import globals as slc_globals
from ..utils import get_extremal_eigenvalue, remove_measure
from ..utils.tracing import get_tracer
from .commutator_bounds import Bounds, CommutatorBounds, compute_bounds
from .light_cone import LightCone

LOGGER = logging.getLogger(__name__)
WARNING_TOL: float = 1e-8


def time_evolved_norm_forward(
    pauli: Pauli,
    gates: RotationGates,
    observable: Pauli,
    *,
    evolution_max_terms: int = np.iinfo(np.uint).max,
    eigval_max_qubits: int = np.iinfo(np.uint).max,
    comm_norm_order: int = 2,
    atol_simplify: float = 1e-8,
    atol_eigenvalue: float = 1e-8,
) -> CommutatorBounds:
    """Compute the bound of an error Pauli term evolved forward to the target observable.

    Given an error Pauli term (``pauli``), the non-Clifford component of a circuit (``gates``), and
    a target observable (``observable``) that is to be measured at the end of the circuit, this
    function computes the unequal-time commutator between the noise term and observable. That means,
    the error Pauli term (which may reside anywhere in the middle of the target circuit) is being
    evolved forwards to the *end* of the circuit where the target observable is being measured. In
    doing so, it must be evolved through the non-Clifford component of the circuit within the
    light-cone (here, given by ``gates``).

    Args:
        pauli: the error Pauli term to be evolved.
        gates: the non-Clifford circuit component encoded as rotation gates (as produced by
            :func:`~pauli_prop.propagation.propagate_through_rotation_gates`).
        observable: the target observable to be measured at the end of the circuit.
        evolution_max_terms: the maximum number of operator terms to keep track of during the
            evolution.
        eigval_max_qubits: the maximum number of qubits of a commutator for which the eigenvalue
            computation will still be attempted. When this value is exceeded, the bound is
            approximated via a simpler and more loose triangle inequality.
        comm_norm_order: the order of the commutator norm to compute.
        atol_simplify: the absolute tolerance used for trimming terms from the commutator. Loosening
            this tolerance will result in a greater truncation of the commutator's terms, rendering
            the computation of its eigenvalue cheaper but less accurate.
        atol_eigenvalue: the absolute tolerance used for detecting convergence of the commutator's
            eigenvalue. Loosening this tolerance will result in a less accurate eigenvalue as
            computed by the iterative Davidson eigensolver.

    Returns:
        The unequal-time commutator bound.
    """
    tracer = get_tracer(__name__)

    # NOTE: we rely on implicit span context management
    with tracer.start_as_current_span(
        "forward_norm_computation",
        attributes={
            "pauli": str(pauli),
            "pauli.num_qubits": pauli.num_qubits,
            "observable": str(observable),
            "observable.num_qubits": observable.num_qubits,
            "gates.count": len(gates.gates),
        },
    ) as span:
        # Convert the single Pauli to a SparsePauliOp which we can then evolve
        orig = pauli
        pauli = SparsePauliOp(pauli)

        span.add_event("pauli_propagation_started")
        pauli, trunc_onenorm = propagate_through_rotation_gates(
            operator=pauli,
            rot_gates=gates,
            max_terms=evolution_max_terms,
            atol=slc_globals.ZERO_ATOL,
            frame="s",
        )
        trunc_bias = float(2 * trunc_onenorm)
        span.add_event("pauli_propagation_completed")
        span.set_attribute("truncation.one_norm", float(trunc_onenorm))

        if trunc_bias >= 2.0:
            result = CommutatorBounds(float("NaN"), trunc_bias, False)
            span.add_event("computation_aborted", {"reason": "truncation_bias_exceeds_bound"})
            span.set_attribute("result.commutator_bound", result.commutator_bound)
            span.set_attribute("result.truncation_bias", result.truncation_bias)
            span.set_attribute("result.fallback_to_tri_ineq", result.fallback_to_tri_ineq)
            return result

        # Handle case of a single-Pauli:
        # Ignore limit on num qubits since don't need to go to computational basis.
        # Efficiently handles Clifford case.
        if len(pauli.paulis) == 1:
            span.add_event("single_pauli_optimization")
            comm_norm = 2 * np.abs(pauli.coeffs[0]) * (pauli.paulis[0].anticommutes(observable))
            result = CommutatorBounds(float(comm_norm), trunc_bias, False)
            span.set_attribute("result.commutator_bound", result.commutator_bound)
            span.set_attribute("result.truncation_bias", result.truncation_bias)
            span.set_attribute("result.fallback_to_tri_ineq", result.fallback_to_tri_ineq)
            return result

        span.add_event("commutator_computation_started")
        # NOTE: we must use .dot for the second operation because we need the implementation of
        # `SparsePauliOp.dot(Pauli)` since `Pauli.compose(SparsePauliOp)` is not implemented
        commutator = pauli.compose(observable) - pauli.dot(observable)
        # NOTE: since pauli and observable are both hermitian, we know that their commutator is
        # anti-hermitian. Therefore, multiplying it by 1j below we can make it hermitian again (without
        # changing its norm). This guarantees the imaginary phase of all coefficients to be zero (but
        # asserting this will work only after a call to simplify(atol=0) to de-duplicate terms and
        # ensure coefficients cancel correctly).
        commutator *= -1j
        # NOTE: even though we do not call simplify (yet) we force all imaginary phases to be exactly 0
        # to avoid numerical noise.
        commutator.coeffs.imag = 0

        # compute 1-norm before simplifying to atol
        commutator = commutator.simplify(atol=0)
        one_norm_before = np.linalg.norm(commutator.coeffs, ord=1)
        # compute 1-norm after simplifying to atol
        commutator = commutator.simplify(atol=atol_simplify)
        one_norm_after = np.linalg.norm(commutator.coeffs, ord=1)
        # compute loss in 1-norm due to simplifying to atol
        one_norm_loss = one_norm_before - one_norm_after
        one_norm_loss = max(one_norm_loss, np.float64(0.0))
        trunc_bias += float(one_norm_loss)
        span.add_event("commutator_computation_completed")

        if trunc_bias >= 2.0:
            span.add_event("computation_aborted", {"reason": "truncation_bias_exceeds_bound"})
            result = CommutatorBounds(float("NaN"), trunc_bias, False)
            span.set_attribute("result.commutator_bound", result.commutator_bound)
            span.set_attribute("result.truncation_bias", result.truncation_bias)
            span.set_attribute("result.fallback_to_tri_ineq", result.fallback_to_tri_ineq)
            return result

        # Handle case where commutator is 0:
        if np.logical_not(np.any((commutator.paulis.x, commutator.paulis.z))) and np.isclose(
            np.sum(commutator.coeffs), 0
        ):
            result = CommutatorBounds(0.0, trunc_bias, False)
            span.add_event("zero_commutator")
            span.set_attribute("result.commutator_bound", result.commutator_bound)
            span.set_attribute("result.truncation_bias", result.truncation_bias)
            span.set_attribute("result.fallback_to_tri_ineq", result.fallback_to_tri_ineq)
            return result

        # If any qubits have only identity Paulis, remove those qubits.
        # Not that important for operator evolution but possibly important for evaluating spectral norm:
        identity_qb_mask = np.logical_not(
            np.any((commutator.paulis.z, commutator.paulis.x), axis=(0, 1))
        )
        if np.any(identity_qb_mask):
            identity_qbs = np.where(identity_qb_mask)[0]
            commutator = SparsePauliOp(
                commutator.paulis.delete(identity_qbs, qubit=True),
                commutator.coeffs.copy(),
                copy=True,
            ).simplify(atol=0)

        # Handle case where a comm_norm_order other than 2 was requested:
        if comm_norm_order != 2:
            span.add_event("non_standard_norm_order", {"order": comm_norm_order})
            comm_norm = np.linalg.norm(commutator, ord=comm_norm_order)
            result = CommutatorBounds(float(comm_norm), trunc_bias, False)
            span.set_attribute("result.commutator_bound", result.commutator_bound)
            span.set_attribute("result.truncation_bias", result.truncation_bias)
            span.set_attribute("result.fallback_to_tri_ineq", result.fallback_to_tri_ineq)
            return result

        def fallback_to_tri_ineq(coeffs, trunc_bias_) -> CommutatorBounds:
            comm_norm_ = 2 * np.abs(coeffs).sum()
            result = CommutatorBounds(float(comm_norm_), trunc_bias_, True)
            span.add_event("fallback_to_triangle_inequality")
            span.set_attribute("result.commutator_bound", result.commutator_bound)
            span.set_attribute("result.truncation_bias", result.truncation_bias)
            span.set_attribute("result.fallback_to_tri_ineq", result.fallback_to_tri_ineq)
            return result

        # When the number of qubits is too large, fall back
        if commutator.num_qubits > eigval_max_qubits:
            span.add_event(
                "qubit_limit_exceeded",
                {"num_qubits": commutator.num_qubits, "max_qubits": eigval_max_qubits},
            )
            return fallback_to_tri_ineq(commutator.coeffs, trunc_bias)

        # When the number of qubits is sufficiently small, compute the smallest eigenvalue directly
        span.add_event("eigenvalue_computation_started")
        if commutator.num_qubits <= 4:
            span.add_event("direct_eigenvalue_computation", {"num_qubits": commutator.num_qubits})
            commutator = commutator.to_matrix()
            comm_norm = np.abs(
                scipy.linalg.eigvalsh(
                    commutator,
                    subset_by_index=(
                        commutator.shape[0] - 1,
                        commutator.shape[0] - 1,
                    ),
                )[0]
            )
            success = True

        else:
            span.add_event("davidson_eigensolver", {"num_qubits": commutator.num_qubits})
            success, comm_norm = get_extremal_eigenvalue(commutator, tol=atol_eigenvalue)
        span.add_event("eigenvalue_computation_completed")

        if success:
            comm_norm = np.abs(comm_norm)
            if comm_norm - 2.0 > WARNING_TOL:
                span.add_event("warning_comm_norm_exceeds_bound", {"comm_norm": float(comm_norm)})
                LOGGER.debug(
                    f"Solver found comm norm {comm_norm:.6f} > {2.0 + WARNING_TOL} for Pauli error "
                    f"{orig!s}."
                )
        else:
            # If this failure is common, could sort SPO by |coeffs|, break into chunks, and call
            # Davidson on each chunk.
            span.add_event("eigensolver_failed")
            LOGGER.debug("Eigensolver failed, reverting to triangle inequality...")
            return fallback_to_tri_ineq(commutator.coeffs, trunc_bias)

        result = CommutatorBounds(float(comm_norm), trunc_bias, False)
        span.set_attribute("result.commutator_bound", result.commutator_bound)
        span.set_attribute("result.truncation_bias", result.truncation_bias)
        span.set_attribute("result.fallback_to_tri_ineq", result.fallback_to_tri_ineq)

        return result


@deprecate_arg(
    name="atol",
    since="0.2.0",
    package_name="qiskit-addon-slc",
    additional_msg="Use `atol_simplify` and `atol_eigenvalue` instead.",
)
def compute_forward_bounds(
    circuit: QuantumCircuit,
    noise_model_paulis: dict[str, QubitSparsePauliList],
    /,
    observable: Pauli | PauliList | SparseObservable | SparsePauliOp,
    *,
    evolution_max_terms: int = 1_000_000,
    eigval_max_qubits: int = 14,
    atol: float = 1e-8,
    atol_simplify: float = 1e-8,
    atol_eigenvalue: float = 1e-8,
    **kwargs,
) -> Bounds:
    r"""Compute the forward-evolved unequal-time commutator bounds.

    Starting at the end of the circuit, compute the forward-evolved unequal-time commutator bounds
    for all Pauli error terms of each noisy layer in the target circuit.

    That is, compute :math:`\| \left[ E_F, A_F \right] \|_2` for all error terms, :math:`E_F`, where
    :math:`A_F` is the target ``observable`` to be measured on ``circuit``.

    The error terms, :math:`E_I`, are dictated by ``noise_model_paulis``. This dictionary maps noise
    model identifiers (:attr:`samplomatic.InjectNoise.ref`) to a list of Pauli error terms. The
    corresponding terms will be used whenever a :class:`~qiskit.circuit.BoxOp` with a matching
    :class:`~samplomatic.InjectNoise` annotation is encountered during the iteration over
    ``circuit``.

    Args:
        circuit: the target circuit.
        noise_model_paulis: the Pauli error terms to consider for each noise model.
        observable: the target observable to be measured at the end of the circuit.
        evolution_max_terms: the maximum number of operator terms to keep track of during the
            evolution.
        eigval_max_qubits: the maximum number of qubits of a commutator for which the eigenvalue
            will still be attempted to be computed. When this value is exceeded, the bound is
            approximated via a simpler and more loose triangle inequality.
        atol: **DEPRECATED** use ``atol_simplify`` and ``atol_eigenvalue`` instead!
        atol_simplify: the absolute tolerance used for trimming terms from the commutator. Loosening
            this tolerance will result in a greater truncation of the commutator's terms, rendering
            the computation of its eigenvalue cheaper but less accurate.
        atol_eigenvalue: the absolute tolerance used for detecting convergence of the commutator's
            eigenvalue. Loosening this tolerance will result in a less accurate eigenvalue as
            computed by the iterative Davidson eigensolver.
        kwargs: any additional keyword arguments will be forward to :func:`.compute_bounds`.

    Returns:
        The unequal-time commutator bound.

    Raises:
        NotImplementedError: when the ``observable`` contains more than a single Pauli term. If you
            run into this, you will need to call this function for each target Pauli separately.
    """
    if not isinstance(observable, Pauli):
        if len(observable) != 1:
            raise NotImplementedError(
                "Cannot compute the bounds for an observable with more than 1 Pauli term! "
                "Please iterate over the Paulis one at a time to compute their bounds."
            )
        if isinstance(observable, PauliList):
            pauli = observable[0]
        elif isinstance(observable, SparsePauliOp):
            pauli = observable.paulis[0]
        elif isinstance(observable, SparseObservable):
            pauli = SparsePauliOp.from_sparse_observable(observable).paulis[0]
    else:
        pauli = observable

    # NOTE: Forward evolution means evolve errors to later times, e.g. towards measurements.
    LOGGER.info("Evolving Pauli error terms forwards through the circuit.")
    LOGGER.info("Modelling errors as though they happen *after* each noise layer.")

    tracer = get_tracer(__name__)

    with tracer.start_as_current_span(
        "compute_forward_bounds",
        attributes={
            "circuit.num_qubits": circuit.num_qubits,
            "observable.num_qubits": pauli.num_qubits,
        },
    ) as span:
        span.add_event("circuit_preparation_started")
        circuit = remove_measure(circuit)
        span.add_event("circuit_preparation_completed")

        if (
            not np.isclose(atol, 1e-8, atol=1e-9)
            and np.isclose(atol_simplify, 1e-8, atol=1e-9)
            and np.isclose(atol_eigenvalue, 1e-8, atol=1e-9)
        ):
            # the user specified the deprecated `atol` argument but neither of the other two new
            # replacement arguments
            atol_simplify = atol  # pragma: no cover
            atol_eigenvalue = atol  # pragma: no cover

        norm_fn = partial(
            time_evolved_norm_forward,
            observable=pauli,
            evolution_max_terms=evolution_max_terms,
            eigval_max_qubits=eigval_max_qubits,
            comm_norm_order=2,
            atol_simplify=atol_simplify,
            atol_eigenvalue=atol_eigenvalue,
        )

        span.add_event("light_cone_initialization")
        lc = LightCone.initialize_from_pauli(circuit, pauli)

        span.add_event("bounds_computation_started")
        comm_norms = compute_bounds(
            circuit,
            noise_model_paulis,
            lc,
            norm_fn,
            backwards=False,
            **kwargs,
        )
        span.add_event("bounds_computation_completed")

        return comm_norms
