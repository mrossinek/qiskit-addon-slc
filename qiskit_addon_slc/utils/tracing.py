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
        otherwise None.
    """
    if not HAS_OPENTELEMETRY or not is_tracing_enabled():
        return None

    if not _is_initialized:
        _initialize_tracing()

    from opentelemetry import trace

    return trace.get_tracer(name)


def inject_trace_context() -> dict[str, str] | None:
    """Inject current trace context into a carrier dictionary.

    This function extracts the current trace context and serializes it into a
    dictionary that can be passed across process boundaries.

    Returns:
        A dictionary containing the serialized trace context, or None if tracing
        is disabled, OpenTelemetry is not available, or no active context exists.
    """
    if not HAS_OPENTELEMETRY or not is_tracing_enabled():
        return None

    try:
        from opentelemetry.propagate import inject

        carrier: dict[str, str] = {}
        inject(carrier)
        return carrier if carrier else None
    except Exception as e:
        LOGGER.debug(f"Failed to inject trace context: {e}")
        return None


def extract_trace_context(carrier: dict[str, str] | None) -> Any:
    """Extract trace context from carrier dictionary.

    This function deserializes trace context from a carrier dictionary that was
    created by inject_trace_context() in another process.

    Args:
        carrier: A dictionary containing serialized trace context, or None.

    Returns:
        The extracted context, or None if carrier is None, OpenTelemetry is not
        available, or extraction fails.
    """
    if carrier is None or not HAS_OPENTELEMETRY or not is_tracing_enabled():
        return None

    try:
        from opentelemetry.propagate import extract

        return extract(carrier)
    except Exception as e:
        LOGGER.debug(f"Failed to extract trace context: {e}")
        return None


def attach_context(ctx: Any) -> Any:
    """Attach a trace context to the current execution context.

    Args:
        ctx: The context to attach, typically obtained from extract_trace_context().

    Returns:
        A token that can be used to detach the context later, or None if ctx is None
        or OpenTelemetry is not available.
    """
    if ctx is None or not HAS_OPENTELEMETRY:
        return None

    try:
        from opentelemetry import context

        return context.attach(ctx)
    except Exception as e:
        LOGGER.debug(f"Failed to attach context: {e}")
        return None


def detach_context(token: Any) -> None:
    """Detach a previously attached trace context.

    Args:
        token: The token returned by attach_context().
    """
    if token is None or not HAS_OPENTELEMETRY:
        return

    try:
        from opentelemetry import context

        context.detach(token)
    except Exception as e:
        LOGGER.debug(f"Failed to detach context: {e}")
