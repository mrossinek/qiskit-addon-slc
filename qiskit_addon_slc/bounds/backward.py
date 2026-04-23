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

"""Backward evolved unequal time commutator bounds."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import numpy as np
from pauli_prop.propagation import (
    RotationGates,
    propagate_through_rotation_gates,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import (
    Pauli,
    QubitSparsePauliList,
    SparsePauliOp,
)

from .. import globals as slc_globals
from ..utils import remove_measure
from ..utils.tracing import (
    attach_context,
    detach_context,
    extract_trace_context,
    is_tracing_enabled,
)
from .commutator_bounds import Bounds, CommutatorBounds, compute_bounds
from .light_cone import LightCone

LOGGER = logging.getLogger(__name__)


def _time_evolved_norm_backward(
    pauli: Pauli,
    gates: RotationGates,
    *,
    evolution_max_terms: int = np.iinfo(np.uint).max,
    trace_context: dict[str, str] | None = None,
) -> CommutatorBounds:
    r"""Bound the effect of an error Pauli term on the quantum state by evolving the error backward.

    Evolving backwards means moving the error through the circuit to its start.

    Given an error Pauli term (``pauli``) and the non-Clifford component of a circuit (``gates``),
    this function computes the unequal-time commutator between the Pauli error and the quantum state.
    The error Pauli term (which may reside anywhere in the middle of the
    target circuit) is evolved backwards to the *start* of the circuit. In doing so, it must
    be evolved through the non-Clifford component of the circuit within the light-cone (here, given
    by ``gates``).

    Note, that this function is designed for the context of :func:`.compute_backward_bounds` which
    inverts the target circuit first. Hence, a backward evolution actually amounts to the
    Schrödinger frame evolution of ``pauli`` through ``gates``.

    Args:
        pauli: the error Pauli term to be evolved.
        gates: the non-Clifford circuit component encoded as rotation gates (as produced by
            :func:`~pauli_prop.propagation.propagate_through_rotation_gates`).
        evolution_max_terms: the maximum number of operator terms to keep track of during the
            evolution.
        trace_context: optional trace context for distributed tracing across process boundaries.

    Returns:
        The unequal-time commutator bound :math:`\| \left[E, \rho\right] \|_1` for Pauli error
        :math:`E` and state :math:`\rho`, where the norm is the Schatten 1 norm (nuclear norm).
    """
    # Convert the single Pauli to a SparsePauliOp which we can then evolve
    pauli_op = SparsePauliOp(pauli)

    # Branch based on whether tracing is enabled
    if not is_tracing_enabled():
        return _compute_backward_norm(pauli_op, gates, evolution_max_terms, span=None)

    # Tracing is enabled, wrap in span
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)

    # Extract and attach trace context if provided
    ctx = extract_trace_context(trace_context)
    token = attach_context(ctx)

    try:
        with tracer.start_as_current_span(
            "backward_norm_computation",
            attributes={
                "pauli.num_qubits": len(pauli),
                "gates.count": len(gates.gates),
                "evolution_max_terms": evolution_max_terms,
            },
        ) as span:
            return _compute_backward_norm(pauli_op, gates, evolution_max_terms, span)
    finally:
        detach_context(token)


def _compute_backward_norm(
    pauli: SparsePauliOp,
    gates: RotationGates,
    evolution_max_terms: int,
    span: Any | None,
) -> CommutatorBounds:
    """Internal implementation of backward norm computation with optional span tracking."""
    pauli, trunc_onenorm = propagate_through_rotation_gates(
        operator=pauli,
        rot_gates=gates,
        max_terms=evolution_max_terms,
        atol=slc_globals.ZERO_ATOL,
        frame="s",
    )
    trunc_bias = 2 * trunc_onenorm

    if span is not None:
        span.set_attribute("truncation.one_norm", float(trunc_onenorm))
        span.set_attribute("truncation.bias", float(trunc_bias))

    if trunc_bias >= 2.0:
        if span is not None:
            span.add_event("computation_aborted", {"reason": "truncation_bias_exceeds_bound"})
        result = CommutatorBounds(float("NaN"), trunc_bias, False)
        if span is not None:
            span.set_attribute("result.commutator_bound", "NaN")
            span.set_attribute("result.truncation_bias", result.truncation_bias)
            span.set_attribute("result.fallback_to_tri_ineq", result.fallback_to_tri_ineq)
        return result

    acts_on_zero = np.any(pauli.paulis.x, axis=1)
    x = pauli.paulis.x[acts_on_zero]
    z = pauli.paulis.z[acts_on_zero]
    c = pauli.coeffs[acts_on_zero]

    uniques, which_unique_x = np.unique(x, return_inverse=True, axis=0)
    s = np.zeros(len(uniques), dtype=complex)
    np.add.at(s, which_unique_x, c * ((1j) ** np.sum(z * x, axis=1)))
    sqrt_s = np.linalg.norm(s)
    comm_norm = 2 * sqrt_s

    result = CommutatorBounds(float(comm_norm), trunc_bias, False)

    # Add result attributes to span
    if span is not None:
        span.set_attribute("result.commutator_bound", result.commutator_bound)
        span.set_attribute("result.truncation_bias", result.truncation_bias)
        span.set_attribute("result.fallback_to_tri_ineq", result.fallback_to_tri_ineq)
        span.set_attribute("result.min_bound", result.min())

    return result


def compute_backward_bounds(
    circuit: QuantumCircuit,
    noise_model_paulis: dict[str, QubitSparsePauliList],
    /,
    *,
    evolution_max_terms: int = 1_000_000,
    **kwargs,
) -> Bounds:
    r"""Compute the backward-evolved unequal-time commutator bounds.

    Starting at the beginning of the circuit, compute the backward-evolved unequal-time commutator
    bounds for all Pauli error terms of each noisy layer in the target circuit.

    That is, compute :math:`\| \left[ E_I, \rho_I \right] \|_1` (using the Schatten 1-norm aka
    nuclear norm) for all error terms, :math:`E_I`, where :math:`\rho_I` is assumed to be the
    all-zero state, :math:`\ket{0 \ldots 0}`, on all active qubits in ``circuit``.

    The error terms, :math:`E_I`, are dictated by ``noise_model_paulis``. This dictionary maps noise
    model identifiers (:attr:`samplomatic.InjectNoise.ref`) to a list of Pauli error terms. The
    corresponding terms will be used whenever a :class:`~qiskit.circuit.BoxOp` with a matching
    :class:`~samplomatic.InjectNoise` annotation is encountered during the iteration over
    ``circuit``.

    .. caution::
      Before computing the bounds, this function removes **all** :class:`.Measure` operations from
      ``circuit``. This is required because the circuit is being inverted before being
      processed in reverse order, which allows the backward evolution to be treated like a forward
      evolution (in the inverted circuit).

    Args:
        circuit: the target circuit.
        noise_model_paulis: the Pauli error terms to consider for each noise model.
        evolution_max_terms: the maximum number of operator terms to keep track of during the
            evolution. (If the operator exceeds this size, the smallest terms are truncated).
        kwargs: any additional keyword arguments will be forward to :func:`.compute_bounds`.

    Returns:
        The backward-evolved unequal-time commutator bounds.
    """
    LOGGER.info("Evolving Pauli error terms backwards through the circuit.")
    LOGGER.info("Modelling errors as though they happen *after* each noise layer.")

    # Branch based on whether tracing is enabled
    if not is_tracing_enabled():
        return _compute_backward_bounds_impl(
            circuit,
            noise_model_paulis,
            evolution_max_terms,
            parent_span=None,
            **kwargs,
        )

    # Tracing is enabled, wrap in span
    return _compute_backward_bounds_with_tracing(
        circuit,
        noise_model_paulis,
        evolution_max_terms,
        **kwargs,
    )


def _compute_backward_bounds_with_tracing(
    circuit: QuantumCircuit,
    noise_model_paulis: dict[str, QubitSparsePauliList],
    evolution_max_terms: int,
    **kwargs,
) -> Bounds:
    """Internal function that wraps backward bounds computation with tracing."""
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span(
        "compute_backward_bounds",
        attributes={
            "circuit.num_qubits": circuit.num_qubits,
            "circuit.depth": circuit.depth(),
            "evolution_max_terms": evolution_max_terms,
        },
    ) as parent_span:
        return _compute_backward_bounds_impl(
            circuit,
            noise_model_paulis,
            evolution_max_terms,
            parent_span=parent_span,
            **kwargs,
        )


def _compute_backward_bounds_impl(
    circuit: QuantumCircuit,
    noise_model_paulis: dict[str, QubitSparsePauliList],
    evolution_max_terms: int,
    parent_span: Any | None,
    **kwargs,
) -> Bounds:
    """Internal implementation of backward bounds computation."""
    # Circuit preparation
    if parent_span is not None:
        parent_span.add_event("circuit_preparation_started")
    circuit = remove_measure(circuit).inverse()
    if parent_span is not None:
        parent_span.add_event("circuit_inversion_completed")

    # Create norm function
    if parent_span is not None:
        parent_span.add_event("norm_function_created")
    norm_fn = partial(
        _time_evolved_norm_backward,
        evolution_max_terms=evolution_max_terms,
    )

    # Initialize light cone
    if parent_span is not None:
        parent_span.add_event("light_cone_initialization")
    lc = LightCone.initialize_from_measurements(circuit, measure_active=True)

    # Compute bounds
    if parent_span is not None:
        parent_span.add_event("bounds_computation_started")
    comm_norms = compute_bounds(
        circuit,
        noise_model_paulis,
        lc,
        norm_fn,
        backwards=True,
        parent_span=parent_span,
        **kwargs,
    )
    if parent_span is not None:
        parent_span.add_event("bounds_computation_completed")

    return comm_norms
