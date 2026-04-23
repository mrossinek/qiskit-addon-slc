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

"""Unequal time commutator bounds."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import numpy as np
from pauli_prop.propagation import (
    KNOWN_CLIFFS,
    RotationGates,
)
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, CircuitInstruction
from qiskit.quantum_info import (
    Clifford,
    Pauli,
    PauliLindbladMap,
    QubitSparsePauliList,
    SparsePauliOp,
)

from .. import globals as slc_globals
from ..utils import find_indices, iter_circuit
from ..utils.tracing import (
    get_tracer,
    inject_trace_context,
    is_tracing_enabled,
)
from .light_cone import LightCone

Bounds = dict[str, PauliLindbladMap]

LOGGER = logging.getLogger(__name__)


class CommutatorBounds(NamedTuple):
    """A dataclass to store metadata about the computed commutator bounds."""

    commutator_bound: float
    """The bound on the commutator.

    This bound will be computed in different means depending on the application. For example,
    backward bounds will compute the nuclear norm (Schatten 1-norm) while forward bounds are
    typically computed using the spectral norm (Schatten infinity-norm).

    If the norm computation exceeds specified difficulty limits, it will be abandoned in favor of a
    simpler bound based on the triangle inequality, which is indicated by
    :attr:`fallback_to_tri_ineq` being set to ``True``.

    This value may be ``NaN`` when the computation of the commutor bound was aborted. This can
    happen when the :attr:`truncation_bias` already exceeds the theoretical bound of ``2.0``.
    """

    truncation_bias: float
    """The bias on the bound due to truncation of the commutator."""

    fallback_to_tri_ineq: bool
    """Whether :attr:`commutator_bound` was computed "loosely" using a simple triangle inequality.
    """
    # TODO: document the triangle inequality that is being used here

    def min(self) -> float:
        """Returns the minimum bound encoded by this metadata.

        The minimal bound is the smaller of the sum of :attr:`commutator_bound` and
        :attr:`truncation_bias` or the theoretical bound of ``2.0``.

        The value of ``2.0`` is used because a Pauli observable bounded on the range
        ``[-1, +1]`` cannot be biased by more than ``2.0``.
        """
        return float(np.nanmin((self.commutator_bound + self.truncation_bias, 2.0)))


def compute_bounds(
    circuit: QuantumCircuit,
    noise_model_paulis: dict[str, QubitSparsePauliList],
    light_cone: LightCone,
    norm_fn: Callable[[Pauli, RotationGates], CommutatorBounds],
    *,
    backwards: bool,
    max_num_boxes: int | None = None,
    num_processes: int = 1,
    timeout: float | None = None,
    parent_span: Any | None = None,
) -> Bounds:
    """Computes the unequal time commutator bounds.

    Given a circuit with :class:`.BoxOp` instructions with :class:`.InjectNoise` annotations and a
    mapping of noise model identifiers (:attr:`.InjectNoise.ref`) to list of Pauli error terms
    (``noise_model_paulis``), this function computes the unequal time commutator bounds (the details
    of which are implemented by ``norm_fn``). In doing so, it only considers gates that lie within
    the light-cone of the observable (initialized by ``light_cone``). These computed bounds form the
    basis of the shaded light-cone.

    Since this function performs a long-running computation, it gracefully handles
    ``KeyboardInterrupt`` exceptions, allowing the user to interrupt the computation at an arbitrary
    point in time and still obtain the results that have been computed up to that point.

    Args:
        circuit: the target circuit.
        noise_model_paulis: the Pauli error terms to consider for each noise model.
        light_cone: the initialized and stateful :class:`.LightCone` tracker.
        norm_fn: the function implementing the specific unequal time commutator.
        backwards: whether to iterate over the ``circuit`` in reverse.
        max_num_boxes: the maximum number of boxes for which to compute bounds. Bounds for any
            additional boxes will be given the trivial upper bound value of :math:`2.0`.
        num_processes: the number of parallel processes to use.
        timeout: an optional timeout (in seconds) after which all remaining layers are filled with
            trivial numerical bounds of ``2.0``. Note, that this is not a strict timeout and the
            layer being processed at the time of reaching this timeout will complete normally.
        parent_span: an optional parent span for distributed tracing. If provided, a child span
            will be created under it. If None, a root span will be created.

    Returns:
        The computed unequal time commutator bounds.
    """
    LOGGER.debug(f"Using {num_processes} processes")

    tracer = get_tracer(__name__)

    # Create child span if parent_span is provided, otherwise create root span
    span_attributes = {
        "circuit.num_qubits": circuit.num_qubits,
        "num_processes": num_processes,
        "backwards": backwards,
        "max_num_boxes": max_num_boxes if max_num_boxes is not None else -1,
    }

    # Set context based on whether parent_span is provided
    # Note: We still need this conditional because set_span_in_context is an OpenTelemetry API
    # that requires a real span object (NoOpSpan won't work here)
    if is_tracing_enabled() and parent_span is not None:
        from opentelemetry.trace import set_span_in_context

        ctx = set_span_in_context(parent_span)
    else:
        ctx = None

    with tracer.start_as_current_span(
        "compute_bounds", context=ctx, attributes=span_attributes
    ) as child_span:
        return _compute_bounds_impl(
            circuit,
            noise_model_paulis,
            light_cone,
            norm_fn,
            backwards=backwards,
            max_num_boxes=max_num_boxes,
            num_processes=num_processes,
            timeout=timeout,
            parent_span=child_span,
        )


def _compute_bounds_impl(
    circuit: QuantumCircuit,
    noise_model_paulis: dict[str, QubitSparsePauliList],
    light_cone: LightCone,
    norm_fn: Callable[[Pauli, RotationGates], CommutatorBounds],
    *,
    backwards: bool,
    max_num_boxes: int | None,
    num_processes: int,
    timeout: float | None,
    parent_span: Any,
) -> Bounds:
    """Internal implementation of compute_bounds with tracing support."""
    net_clifford = Clifford.from_label("I" * circuit.num_qubits)
    rot_gates = RotationGates([], [], [])

    def _handle_circuit_instruction(instruction: CircuitInstruction) -> None:
        """Handles a circuit instruction.

        1. If the instruction commutes with the current light-cone, do nothing. Note, that calling
           ``light_cone.commutes`` will append the provided instruction to the stateful
           ``light_cone`` object!
        2. Find ``qargs`` which are the indices of the instruction's qubits in the context of the
           global circuit from which this instruction stems.
        3. If the gate is Clifford, update our global ``net_clifford``, effectively accumulating all
           Clifford gates at the beginning of the circuit.
        4. If the gate is *not* Clifford, append it to our global ``rot_gates`` under which our
           Pauli terms are later going to be evolved. When doing so, we provide ``net_clifford`` to
           ensure the rotation gate gets moved through it, ensuring the ``net_clifford`` remains at
           the beginning of the circuit.
        """
        nonlocal light_cone
        nonlocal circuit
        nonlocal net_clifford
        nonlocal rot_gates
        nonlocal parent_span
        parent_span.add_event("handling_circuit_instruction_started")

        if light_cone.commutes(instruction):
            parent_span.add_event(
                "handling_circuit_instruction_completed", {"reason": "commuted_with_lightcone"}
            )
            return

        qargs = find_indices(circuit, instruction)

        if isinstance(instruction.operation, Barrier):
            LOGGER.debug(f"Ignoring instruction of type '{type(instruction.operation)}'")
        elif instruction.name in KNOWN_CLIFFS:
            parent_span.add_event("composing_net_clifford_started")
            net_clifford = net_clifford.dot(instruction.operation, qargs)
            parent_span.add_event("composing_net_clifford_completed")
        else:
            rot_gates.append_circuit_instruction(
                instruction, qargs, circuit.num_qubits, clifford=net_clifford
            )

    gathered_bounds: dict[str, tuple[np.ndarray, QubitSparsePauliList]] = {}

    def _insert_rate(bound: CommutatorBounds, box_id: str, rate_idx: int) -> None:
        nonlocal gathered_bounds

        gathered_bounds[box_id][0][rate_idx] = bound.min()

    pool = mp.Pool(num_processes)
    tasks = set()

    start = time.time()
    if parent_span is not None:
        parent_span.add_event("task_spawning_started")
    LOGGER.debug("Starting to spawn bound computation tasks")

    # Inject trace context for propagation to worker processes
    trace_context = inject_trace_context() if is_tracing_enabled() else None

    encountered_num_boxes = 0

    for circ_inst, qargs, box_id, noise_id in iter_circuit(circuit, reverse=True):
        if box_id is None:
            _handle_circuit_instruction(circ_inst)
            continue

        # NOTE: we know for a fact that noise_id can only be None when box_id is None
        assert noise_id is not None

        noise_terms: QubitSparsePauliList = noise_model_paulis[noise_id]
        # pre-populate computed bounds with trivial upper bound
        gathered_bounds[box_id] = (np.full(len(noise_terms), 2.0), noise_terms)

        encountered_num_boxes += 1
        if max_num_boxes is not None and encountered_num_boxes > max_num_boxes:
            # skipping actual bound computation and limiting bound estimate to trivial value
            continue

        if backwards:
            # NOTE: we unroll the BoxOp immediately to allow gates contained within the box be
            # pruned by the LightCone pass
            for inst in circ_inst.operation.body[::-1]:
                _handle_circuit_instruction(inst)

        parent_span.add_event("evolving_noise_terms_started")
        # Ensure that the noise model Pauli terms are defined on the entire width of the circuit.
        local_noise_terms = noise_terms.apply_layout(qargs, num_qubits=circuit.num_qubits)
        # NOTE: both FIXMEs below can be resolved by simply implementing QubitSparsePauliList.evolve
        local_noise_terms = QubitSparsePauliList.from_sparse_list(
            [
                tuple(parts)
                # FIXME: we convert temporarily to SparsePauliOp to leverage its to_sparse_list
                for *parts, _ in SparsePauliOp(
                    # FIXME: we convert temporarily to PauliList to leverage its evolve
                    local_noise_terms.to_pauli_list().evolve(net_clifford, frame="s")
                ).to_sparse_list()
            ],
            circuit.num_qubits,
        )
        parent_span.add_event("evolving_noise_terms_completed")

        norm_fn = partial(  # type: ignore[call-arg]
            norm_fn,
            gates=RotationGates(
                rot_gates.gates[::-1], rot_gates.qargs[::-1], rot_gates.thetas[::-1]
            ),
        )

        for pauli_idx, pauli in enumerate(local_noise_terms.to_pauli_list()):
            task = pool.apply_async(
                norm_fn,
                [pauli],
                {"trace_context": trace_context},
                callback=partial(_insert_rate, box_id=box_id, rate_idx=pauli_idx),
            )
            tasks.add(task)

        if not backwards:
            # NOTE: we unroll the BoxOp immediately to allow gates contained within the box be
            # pruned by the LightCone pass
            for inst in circ_inst.operation.body[::-1]:
                _handle_circuit_instruction(inst)

    total_num_tasks = len(tasks)
    LOGGER.debug(f"Total number of spawned tasks: {total_num_tasks}")

    # Add span event for task spawning completion
    parent_span.add_event(
        "tasks_spawned",
        {"total_tasks": total_num_tasks, "encountered_boxes": encountered_num_boxes},
    )

    len_progress_indicator = 50
    per_progress_char = total_num_tasks / len_progress_indicator

    try:
        while tasks:
            next(iter(tasks)).wait(slc_globals.PROGRESS_POLLING_PERIOD)
            tasks = {t for t in tasks if not t.ready()}
            completed = total_num_tasks - len(tasks)
            perc = (completed / total_num_tasks) * 100
            progress = "." * int(completed / per_progress_char)
            LOGGER.info(
                f"Progress: {progress:{len_progress_indicator}} "
                f"[{completed}/{total_num_tasks}] {perc:.1f}%"
            )
            # Add span event for progress tracking
            parent_span.add_event(
                "progress_update",
                {"completed": completed, "total": total_num_tasks, "percentage": perc},
            )
            if timeout is not None and (time.time() - start) > timeout:
                LOGGER.warning(f"Reached user-specified time out of {timeout} seconds!")
                pool.terminate()
                break
        else:
            pool.close()
    except KeyboardInterrupt:
        LOGGER.warning("Caught KeyboardInterrupt! Terminating pending bound computations.")
        pool.terminate()

    tasks = {t for t in tasks if not t.ready()}
    completed = total_num_tasks - len(tasks)
    LOGGER.info(f"Successfully completed [{completed}/{total_num_tasks}] tasks!")

    pool.join()

    # Add span event for computation completion
    parent_span.add_event(
        "computation_completed",
        {"completed_tasks": completed, "total_tasks": total_num_tasks},
    )

    comm_norms: Bounds = {
        box_id: PauliLindbladMap.from_components(bounds[0], bounds[1])
        for box_id, bounds in gathered_bounds.items()
    }
    return comm_norms
