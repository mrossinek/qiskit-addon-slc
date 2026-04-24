# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# The development of this module was assisted by Bob 1.0.1

"""OpenTelemetry tracing utilities for profiling parallel computations.

This module provides utilities for distributed tracing across multiprocessing boundaries
using OpenTelemetry. It handles trace context propagation, tracer initialization, and
configuration management.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import re
import signal
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from ..globals import (
    OTEL_SERVICE_NAME,
    TRACER_FLUSH_TIMEOUT_MS,
    WORKER_INDEX_NOT_INITIALIZED,
)
from ..optionals import HAS_OPENTELEMETRY

LOGGER = logging.getLogger(__name__)

_tracer_provider: Any = None
_is_initialized: bool = False
_init_lock = threading.Lock()

# Process-local storage for worker span and metadata
_worker_span_storage = threading.local()


class NoOpSpan:
    """A no-op span that accepts all operations but does nothing.

    This class provides the same interface as OpenTelemetry spans but performs
    no operations. It's used when tracing is disabled to avoid conditional checks
    throughout the codebase.
    """

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """No-op implementation of add_event.

        Args:
            name: Event name (ignored).
            attributes: Event attributes (ignored).
        """
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op implementation of set_attribute.

        Args:
            key: Attribute key (ignored).
            value: Attribute value (ignored).
        """
        pass

    def set_status(self, status: Any) -> None:
        """No-op implementation of set_status.

        Args:
            status: Status to set (ignored).
        """
        pass

    def record_exception(self, exception: Exception) -> None:
        """No-op implementation of record_exception.

        Args:
            exception: Exception to record (ignored).
        """
        pass

    def __enter__(self) -> NoOpSpan:
        """Context manager entry.

        Returns:
            Self for context manager protocol.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit.

        Args:
            exc_type: Exception type (ignored).
            exc_val: Exception value (ignored).
            exc_tb: Exception traceback (ignored).
        """
        pass


class NoOpTracer:
    """A no-op tracer that returns no-op spans.

    This class provides the same interface as OpenTelemetry tracers but returns
    NoOpSpan instances. It's used when tracing is disabled to avoid conditional
    checks throughout the codebase.
    """

    def start_span(self, _name: str, _context: Any = None, **_kwargs: Any) -> NoOpSpan:
        """Return a no-op span.

        Args:
            _name: Span name (ignored).
            _context: Span context (ignored).
            **_kwargs: Additional arguments (ignored).

        Returns:
            A NoOpSpan instance.
        """
        return NoOpSpan()

    def start_as_current_span(
        self,
        _name: str,
        _context: Any = None,
        _attributes: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> NoOpSpan:
        """Return a no-op span as context manager.

        Args:
            _name: Span name (ignored).
            _context: Span context (ignored).
            _attributes: Span attributes (ignored).
            **_kwargs: Additional arguments (ignored).

        Returns:
            A NoOpSpan instance that can be used as a context manager.
        """
        return NoOpSpan()


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled via environment variable.

    Returns:
        True if tracing is enabled and OpenTelemetry is available, False otherwise.
    """
    if not HAS_OPENTELEMETRY:
        return False

    value = os.getenv("QISKIT_SLC_TRACING_ENABLED", "false").lower()
    return value in ("true", "1", "yes", "on")


def _shutdown_tracing() -> None:
    """Shutdown tracer provider on exit."""
    global _tracer_provider
    if _tracer_provider is not None and HAS_OPENTELEMETRY:
        try:
            _tracer_provider.force_flush(timeout_millis=TRACER_FLUSH_TIMEOUT_MS)
            _tracer_provider.shutdown()
        except Exception as e:
            LOGGER.debug("Error during tracer shutdown: %s", e)


def _initialize_tracing() -> None:
    """Initialize the OpenTelemetry tracer provider.

    This function sets up the tracer provider with appropriate exporters based on
    environment variables. It should only be called once.
    """
    global _tracer_provider, _is_initialized

    with _init_lock:
        if _is_initialized:
            return

        if not HAS_OPENTELEMETRY or not is_tracing_enabled():
            _is_initialized = True
            return

        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import SERVICE_NAME, Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

            # Create resource with service name
            resource = Resource(attributes={SERVICE_NAME: OTEL_SERVICE_NAME})

            # Create tracer provider
            _tracer_provider = TracerProvider(resource=resource)

            # Determine which exporter to use
            exporter_type = os.getenv("OTEL_TRACES_EXPORTER", "console").lower()

            if exporter_type == "otlp":
                # Use OTLP exporter if endpoint is configured
                otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
                if otlp_endpoint:
                    exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
                    LOGGER.info("Initialized OTLP span exporter to %s", otlp_endpoint)
                else:
                    LOGGER.warning(
                        "OTLP exporter requested but OTEL_EXPORTER_OTLP_ENDPOINT not set. "
                        "Falling back to console exporter."
                    )
                    exporter = ConsoleSpanExporter()
            else:
                # Default to console exporter
                exporter = ConsoleSpanExporter()
                LOGGER.info("Initialized console span exporter")

            # Add span processor
            _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

            # Set as global tracer provider
            trace.set_tracer_provider(_tracer_provider)

            _is_initialized = True
            LOGGER.info("OpenTelemetry tracing initialized successfully")

            # Register cleanup handler for main process
            atexit.register(_shutdown_tracing)

        except ImportError as e:
            LOGGER.warning(
                "Failed to initialize OTLP exporter: %s. "
                "Install 'opentelemetry-exporter-otlp' for OTLP support. "
                "Falling back to console exporter.",
                e,
            )
            # Fall back to console exporter
            if _tracer_provider is None:
                from opentelemetry import trace
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

                _tracer_provider = TracerProvider()
                _tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
                trace.set_tracer_provider(_tracer_provider)
            _is_initialized = True
        except Exception as e:
            LOGGER.error("Failed to initialize tracing: %s", e)
            _is_initialized = True


def get_tracer(name: str = "qiskit_addon_slc") -> Any:
    """Get or create a tracer instance.

    Args:
        name: The name of the tracer, typically the module name.

    Returns:
        A Tracer instance if OpenTelemetry is available and tracing is enabled,
        otherwise a NoOpTracer instance.
    """
    if not HAS_OPENTELEMETRY or not is_tracing_enabled():
        return NoOpTracer()

    if not _is_initialized:
        _initialize_tracing()

    from opentelemetry import trace

    return trace.get_tracer(name)


# Unified tracing utilities for cleaner code patterns


def _inject_trace_context() -> dict[str, str] | None:
    """Internal: Inject current trace context into a carrier dictionary."""
    if not HAS_OPENTELEMETRY or not is_tracing_enabled():
        return None

    try:
        from opentelemetry.propagate import inject

        carrier: dict[str, str] = {}
        inject(carrier)
        return carrier
    except Exception as e:
        LOGGER.debug("Failed to inject trace context: %s", e)
        return carrier


def _extract_trace_context(carrier: dict[str, str] | None) -> Any:
    """Internal: Extract trace context from carrier dictionary."""
    if carrier is None or not HAS_OPENTELEMETRY or not is_tracing_enabled():
        return None

    try:
        from opentelemetry.propagate import extract

        return extract(carrier)
    except Exception as e:
        LOGGER.debug("Failed to extract trace context: %s", e)
        return None


def _attach_context(ctx: Any) -> Any:
    """Internal: Attach a trace context to the current execution context."""
    if ctx is None or not HAS_OPENTELEMETRY:
        return None

    try:
        from opentelemetry import context

        return context.attach(ctx)
    except Exception as e:
        LOGGER.debug("Failed to attach context: %s", e)
        return None


def _detach_context(token: Any) -> None:
    """Internal: Detach a previously attached trace context."""
    if token is None or not HAS_OPENTELEMETRY:
        return

    try:
        from opentelemetry import context

        context.detach(token)
    except Exception as e:
        LOGGER.debug("Failed to detach context: %s", e)


@contextmanager
def traced_span(
    name: str,
    *,
    trace_context: dict[str, str] | None = None,
    parent_span: Any | None = None,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Context manager for creating traced spans with automatic context management.

    This context manager handles all the complexity of:
    - Extracting and attaching trace context from cross-process boundaries
    - Creating child spans from parent spans
    - Automatic cleanup and detachment
    - Graceful degradation when tracing is disabled

    Args:
        name: Name of the span.
        trace_context: Optional serialized trace context from another process.
        parent_span: Optional parent span to create a child span under.
        attributes: Optional span attributes.

    Yields:
        The created span (or NoOpSpan if tracing is disabled).

    Example:
        Simple usage::

            with traced_span("my_operation") as span:
                span.add_event("started")
                do_work()

        With parent span::

            with traced_span("child_op", parent_span=parent) as span:
                do_work()

        With cross-process context::

            with traced_span("worker_task", trace_context=ctx_dict) as span:
                do_work()
    """
    if not is_tracing_enabled():
        yield NoOpSpan()
        return

    tracer = get_tracer("qiskit_addon_slc")

    # Handle trace context from cross-process boundary
    ctx = None
    token = None
    if trace_context is not None:
        ctx = _extract_trace_context(trace_context)
        token = _attach_context(ctx)

    # Handle parent span
    elif parent_span is not None and HAS_OPENTELEMETRY:
        try:
            from opentelemetry.trace import set_span_in_context

            ctx = set_span_in_context(parent_span)
        except Exception as e:
            LOGGER.debug("Failed to set span in context: %s", e)
            ctx = None

    try:
        with tracer.start_as_current_span(name, context=ctx, attributes=attributes or {}) as span:
            yield span
    finally:
        if token is not None:
            _detach_context(token)


def prepare_worker_context() -> dict[str, str] | None:
    """Prepare trace context for worker processes.

    This should be called in the main process before spawning workers.
    The returned dict can be passed to worker processes.

    Returns:
        Serialized trace context or None if tracing disabled.

    Example:
        In main process::

            ctx = prepare_worker_context()

            # Pass to worker
            pool.apply_async(
                worker_func,
                args=(...),
                kwds={"trace_context": ctx}
            )
    """
    return _inject_trace_context() if is_tracing_enabled() else None


def _get_current_worker_index() -> int:
    """Infer a stable worker index for the current multiprocessing worker process.

    Returns:
        A zero-based worker index when running in a multiprocessing pool worker,
        otherwise ``WORKER_INDEX_NOT_INITIALIZED`` as a safe fallback.
    """
    current_process = mp.current_process()

    identity = getattr(current_process, "_identity", ())
    if identity:
        return int(identity[0]) - 1

    name_match = re.search(r"(\d+)$", current_process.name)
    if name_match is not None:
        return int(name_match.group(1)) - 1

    return WORKER_INDEX_NOT_INITIALIZED


def _cleanup_worker_span(worker_index: int, pid: int) -> None:
    """Clean up worker span when process terminates.

    Args:
        worker_index: The worker's index for logging.
        pid: The worker's process ID for logging.
    """
    # Detach worker context first
    worker_ctx_token = getattr(_worker_span_storage, "worker_context_token", None)
    if worker_ctx_token is not None and HAS_OPENTELEMETRY:
        try:
            from opentelemetry import context as otel_context

            otel_context.detach(worker_ctx_token)
        except Exception as e:
            LOGGER.debug("Failed to detach worker context: %s", e)

    # End worker span
    worker_span = getattr(_worker_span_storage, "span", None)
    if worker_span is not None:
        LOGGER.debug("Ending worker span for worker %s (PID: %s)", worker_index, pid)
        worker_span.end()

        # Force flush to ensure span is exported before process terminates
        if HAS_OPENTELEMETRY and _tracer_provider is not None:
            try:
                # Force flush all pending spans
                _tracer_provider.force_flush(timeout_millis=TRACER_FLUSH_TIMEOUT_MS)
                LOGGER.debug("Flushed spans for worker %s", worker_index)
            except Exception as e:
                LOGGER.debug("Failed to flush tracer provider: %s", e)

    # Detach parent context if it was attached
    ctx_token = getattr(_worker_span_storage, "context_token", None)
    if ctx_token is not None:
        _detach_context(ctx_token)


def _create_sigterm_handler(worker_index: int, pid: int) -> Any:
    """Create a SIGTERM handler for worker cleanup.

    Args:
        worker_index: The worker's index for logging.
        pid: The worker's process ID for logging.

    Returns:
        A signal handler function.
    """

    def sigterm_handler(_signum: int, _frame: Any) -> None:
        """Handle SIGTERM by cleaning up spans before process terminates."""
        LOGGER.debug("Worker %s (PID: %s) received SIGTERM, cleaning up spans", worker_index, pid)
        try:
            _cleanup_worker_span(worker_index, pid)
        except Exception as e:
            LOGGER.error("Error during cleanup: %s", e)
        finally:
            sys.exit(0)

    return sigterm_handler


def initialize_worker(trace_context: dict[str, str] | None = None) -> None:
    """Initialize worker process with its own parent span.

    This function should be called once per worker process when it starts (typically
    via mp.Pool's initializer parameter). It creates a long-lived parent span for the
    worker that will contain all task spans executed by this worker.

    Args:
        trace_context: Serialized trace context from main process to link worker span
            to the main computation trace.

    Example:
        In main process::

            trace_context = prepare_worker_context()
            pool = mp.Pool(
                num_processes,
                initializer=initialize_worker,
                initargs=(trace_context,)
            )
    """
    if not is_tracing_enabled():
        return

    worker_index = _get_current_worker_index()

    # Get process ID
    pid = os.getpid()

    LOGGER.debug("Initializing worker %s (PID: %s)", worker_index, pid)

    # Extract parent context if provided
    ctx = _extract_trace_context(trace_context) if trace_context else None
    token = _attach_context(ctx) if ctx else None

    # Create tracer for worker
    tracer = get_tracer("qiskit_addon_slc.worker")

    # Create long-lived span for this worker
    try:
        # Start span and make it the current span in this process
        span = tracer.start_span(
            f"worker_{worker_index}",
            context=ctx,
            attributes={
                "worker.index": worker_index,
                "worker.pid": pid,
            },
        )

        # Make this span the current span in the worker process
        # This ensures all child spans are properly linked
        if HAS_OPENTELEMETRY:
            try:
                from opentelemetry import context as otel_context
                from opentelemetry.trace import set_span_in_context

                # Set worker span as current in this process
                worker_ctx = set_span_in_context(span, ctx)
                worker_token = otel_context.attach(worker_ctx)

                # Store the token so we can detach it later
                _worker_span_storage.worker_context_token = worker_token
            except Exception as e:
                LOGGER.debug("Failed to set worker span as current: %s", e)

        # Store in process-local storage
        _worker_span_storage.span = span
        _worker_span_storage.worker_index = worker_index
        _worker_span_storage.pid = pid
        _worker_span_storage.context_token = token

        # Register cleanup function
        atexit.register(lambda: _cleanup_worker_span(worker_index, pid))

        # Register signal handler for SIGTERM
        signal.signal(signal.SIGTERM, _create_sigterm_handler(worker_index, pid))

        LOGGER.debug("Worker %s (PID: %s) initialized successfully", worker_index, pid)

    except Exception as e:
        LOGGER.warning("Failed to initialize worker span: %s", e)
        if token is not None:
            _detach_context(token)


def get_worker_span() -> Any | None:
    """Get the current worker's parent span.

    Returns:
        The worker's parent span if available, None otherwise.

    Example:
        In worker function::

            worker_span = get_worker_span()
            with traced_span("task", parent_span=worker_span) as span:
                do_work()
    """
    return getattr(_worker_span_storage, "span", None)


def get_worker_info() -> dict[str, int]:
    """Get worker metadata (index, PID).

    Returns:
        Dictionary with 'worker_index' and 'pid' keys. Returns -1 for both
        if not in a worker process or worker not initialized.

    Example:
        In worker function::

            info = get_worker_info()
            print(f"Running in worker {info['worker_index']} (PID: {info['pid']})")
    """
    return {
        "worker_index": getattr(_worker_span_storage, "worker_index", -1),
        "pid": getattr(_worker_span_storage, "pid", -1),
    }
