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

import logging
import os
from typing import Any

from ..optionals import HAS_OPENTELEMETRY

LOGGER = logging.getLogger(__name__)

_tracer_provider: Any = None
_is_initialized: bool = False


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

    return os.getenv("QISKIT_SLC_TRACING_ENABLED", "false").lower() == "true"


def _initialize_tracing() -> None:
    """Initialize the OpenTelemetry tracer provider.

    This function sets up the tracer provider with appropriate exporters based on
    environment variables. It should only be called once.
    """
    global _tracer_provider, _is_initialized

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
        service_name = os.getenv("OTEL_SERVICE_NAME", "qiskit-addon-slc")
        resource = Resource(attributes={SERVICE_NAME: service_name})

        # Create tracer provider
        _tracer_provider = TracerProvider(resource=resource)

        # Determine which exporter to use
        exporter_type = os.getenv("OTEL_TRACES_EXPORTER", "console").lower()

        if exporter_type == "otlp":
            # Use OTLP exporter if endpoint is configured
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            if otlp_endpoint:
                exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
                LOGGER.info(f"Initialized OTLP span exporter to {otlp_endpoint}")
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

    except ImportError as e:
        LOGGER.warning(
            f"Failed to initialize OTLP exporter: {e}. "
            "Install 'opentelemetry-exporter-otlp' for OTLP support. "
            "Falling back to console exporter."
        )
        # Fall back to console exporter
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        _tracer_provider = TracerProvider()
        _tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(_tracer_provider)
        _is_initialized = True
    except Exception as e:
        LOGGER.error(f"Failed to initialize tracing: {e}")
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
