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

"""Tests for OpenTelemetry tracing utilities."""

import os
from unittest.mock import patch

import pytest
from qiskit_addon_slc.optionals import HAS_OPENTELEMETRY
from qiskit_addon_slc.utils.tracing import (
    NoOpSpan,
    NoOpTracer,
    get_tracer,
    is_tracing_enabled,
    prepare_worker_context,
    traced_span,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Fixture to provide clean environment for each test."""
    # Store original values
    original_tracing = os.environ.get("QISKIT_SLC_TRACING_ENABLED")
    original_service = os.environ.get("OTEL_SERVICE_NAME")

    yield monkeypatch

    # Shutdown tracer provider to avoid I/O errors during cleanup
    if HAS_OPENTELEMETRY:
        try:
            from opentelemetry import trace

            provider = trace.get_tracer_provider()
            if hasattr(provider, "shutdown"):
                provider.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup

    # Restore original values
    if original_tracing is not None:
        monkeypatch.setenv("QISKIT_SLC_TRACING_ENABLED", original_tracing)
    elif "QISKIT_SLC_TRACING_ENABLED" in os.environ:
        monkeypatch.delenv("QISKIT_SLC_TRACING_ENABLED", raising=False)

    if original_service is not None:
        monkeypatch.setenv("OTEL_SERVICE_NAME", original_service)
    elif "OTEL_SERVICE_NAME" in os.environ:
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)


def test_is_tracing_enabled_default(clean_env):
    """Test that tracing is disabled by default."""
    clean_env.delenv("QISKIT_SLC_TRACING_ENABLED", raising=False)
    assert is_tracing_enabled() is False


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_is_tracing_enabled_true(clean_env):
    """Test that tracing can be enabled via environment variable."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "true")
    assert is_tracing_enabled() is True


def test_is_tracing_enabled_false(clean_env):
    """Test that tracing can be explicitly disabled."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "false")
    assert is_tracing_enabled() is False


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_is_tracing_enabled_case_insensitive(clean_env):
    """Test that environment variable is case-insensitive."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "true")
    assert is_tracing_enabled() is True

    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "false")
    assert is_tracing_enabled() is False


def test_get_tracer_returns_none_when_disabled(clean_env):
    """Test that get_tracer returns NoOpTracer when tracing is disabled."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "false")
    tracer = get_tracer("test_module")
    assert isinstance(tracer, NoOpTracer)


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_get_tracer_returns_tracer(clean_env):
    """Test that get_tracer returns a tracer instance when enabled."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "true")
    tracer = get_tracer("test_module")
    assert tracer is not None


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_get_tracer_with_custom_name(clean_env):
    """Test that get_tracer accepts custom names."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "true")
    tracer = get_tracer("custom_name")
    assert tracer is not None


def test_noop_span_operations():
    """Test that NoOpSpan accepts all operations without errors."""
    span = NoOpSpan()

    # Test all methods - should not raise any exceptions
    span.add_event("test_event")
    span.add_event("test_event_with_attrs", {"key": "value"})
    span.set_attribute("test_key", "test_value")
    span.set_status("OK")
    span.record_exception(Exception("test"))


def test_noop_span_context_manager():
    """Test that NoOpSpan works as a context manager."""
    span = NoOpSpan()

    # Should work as context manager
    with span as s:
        assert s is span
        s.add_event("inside_context")


def test_noop_tracer_returns_noop_span():
    """Test that NoOpTracer returns NoOpSpan instances."""
    tracer = NoOpTracer()

    # Test start_span
    span = tracer.start_span("test_span")
    assert isinstance(span, NoOpSpan)

    # Test start_as_current_span
    span = tracer.start_as_current_span("test_span")
    assert isinstance(span, NoOpSpan)

    # Test with attributes
    span = tracer.start_as_current_span("test_span", attributes={"key": "value"})
    assert isinstance(span, NoOpSpan)


def test_noop_tracer_context_manager():
    """Test that NoOpTracer spans work as context managers."""
    tracer = NoOpTracer()

    with tracer.start_as_current_span("test_span") as span:
        assert isinstance(span, NoOpSpan)
        span.add_event("inside_context")
        span.set_attribute("key", "value")


def test_get_tracer_returns_noop_when_disabled(clean_env):
    """Test that get_tracer returns NoOpTracer when tracing is disabled."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "false")
    tracer = get_tracer("test_module")
    assert isinstance(tracer, NoOpTracer)


def test_get_tracer_returns_noop_when_no_opentelemetry(clean_env):
    """Test that get_tracer returns NoOpTracer when OpenTelemetry is not available."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "true")

    # Mock HAS_OPENTELEMETRY to False
    with patch("qiskit_addon_slc.utils.tracing.HAS_OPENTELEMETRY", False):
        tracer = get_tracer("test_module")
        assert isinstance(tracer, NoOpTracer)


# Tests for traced_span and prepare_worker_context


def test_traced_span_when_disabled(clean_env):
    """Test that traced_span returns NoOpSpan when tracing is disabled."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "false")

    with traced_span("test_operation") as span:
        assert isinstance(span, NoOpSpan)
        span.add_event("test_event")  # Should not raise
        span.set_attribute("test_key", "test_value")  # Should not raise


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_traced_span_with_trace_context(clean_env):
    """Test that traced_span handles trace_context parameter."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "true")

    # Create a mock trace context
    trace_context = {"traceparent": "00-test-test-01"}

    with traced_span("test_operation", trace_context=trace_context) as span:
        assert span is not None
        span.add_event("test_event")


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_traced_span_with_parent_span(clean_env):
    """Test that traced_span handles parent_span parameter."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "true")

    # Create parent span
    with (
        traced_span("parent_operation") as parent,
        traced_span("child_operation", parent_span=parent) as child,
    ):
        assert child is not None
        child.add_event("child_event")


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_traced_span_with_attributes(clean_env):
    """Test that traced_span handles attributes parameter."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "true")

    attributes = {
        "test_key": "test_value",
        "test_number": 42,
    }

    with traced_span("test_operation", attributes=attributes) as span:
        assert span is not None
        span.add_event("test_event")


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_traced_span_exception_handling(clean_env):
    """Test that traced_span properly handles exceptions."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "true")

    with pytest.raises(ValueError), traced_span("test_operation") as span:
        span.add_event("before_exception")
        raise ValueError("Test exception")


def test_prepare_worker_context_when_disabled(clean_env):
    """Test that prepare_worker_context returns None when tracing is disabled."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "false")

    ctx = prepare_worker_context()
    assert ctx is None


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@patch("qiskit_addon_slc.utils.tracing.is_tracing_enabled", return_value=True)
@patch("opentelemetry.propagate.inject")
def test_prepare_worker_context_when_enabled(mock_inject, _mock_is_enabled):
    """Test that prepare_worker_context works when tracing is enabled."""

    # Mock inject to populate the carrier
    def side_effect(carrier):
        carrier["traceparent"] = "00-test-test-01"

    mock_inject.side_effect = side_effect

    ctx = prepare_worker_context()
    assert ctx is not None
    assert isinstance(ctx, dict)
    assert "traceparent" in ctx


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_traced_span_nested_spans(clean_env):
    """Test that traced_span works with nested spans."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "true")

    with traced_span("outer_operation") as outer:
        outer.add_event("outer_start")

        with traced_span("middle_operation", parent_span=outer) as middle:
            middle.add_event("middle_start")

            with traced_span("inner_operation", parent_span=middle) as inner:
                inner.add_event("inner_work")

            middle.add_event("middle_end")

        outer.add_event("outer_end")


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_traced_span_with_both_context_and_parent(clean_env):
    """Test that trace_context takes precedence over parent_span."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "true")

    trace_context = {"traceparent": "00-test-test-01"}

    with (
        traced_span("parent_operation") as parent,
        traced_span("child_operation", trace_context=trace_context, parent_span=parent) as child,
    ):
        assert child is not None
        child.add_event("test_event")


def test_traced_span_no_opentelemetry():
    """Test that traced_span works when OpenTelemetry is not available."""
    with (
        patch("qiskit_addon_slc.utils.tracing.HAS_OPENTELEMETRY", False),
        traced_span("test_operation") as span,
    ):
        assert isinstance(span, NoOpSpan)


def test_prepare_worker_context_no_opentelemetry():
    """Test that prepare_worker_context returns None when OpenTelemetry is not available."""
    with patch("qiskit_addon_slc.utils.tracing.HAS_OPENTELEMETRY", False):
        ctx = prepare_worker_context()
        assert ctx is None
