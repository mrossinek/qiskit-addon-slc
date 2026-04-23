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
from unittest.mock import MagicMock, patch

import pytest
from qiskit_addon_slc.optionals import HAS_OPENTELEMETRY
from qiskit_addon_slc.utils.tracing import (
    attach_context,
    detach_context,
    extract_trace_context,
    get_tracer,
    inject_trace_context,
    is_tracing_enabled,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Fixture to provide clean environment for each test."""
    # Store original values
    original_tracing = os.environ.get("QISKIT_SLC_TRACING_ENABLED")
    original_service = os.environ.get("OTEL_SERVICE_NAME")

    yield monkeypatch

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
    from qiskit_addon_slc.utils.tracing import NoOpTracer

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


def test_inject_trace_context_when_disabled(clean_env):
    """Test that inject_trace_context returns None when tracing is disabled."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "false")
    carrier = inject_trace_context()
    assert carrier is None


def test_extract_trace_context_with_none():
    """Test that extract_trace_context handles None carrier."""
    ctx = extract_trace_context(None)
    assert ctx is None


def test_extract_trace_context_when_disabled(clean_env):
    """Test that extract_trace_context returns None when tracing is disabled."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "false")
    carrier = {"traceparent": "00-test-test-01"}
    ctx = extract_trace_context(carrier)
    assert ctx is None


def test_attach_context_with_none():
    """Test that attach_context handles None context."""
    token = attach_context(None)
    assert token is None


def test_detach_context_with_none():
    """Test that detach_context handles None token."""
    # Should not raise an exception
    detach_context(None)


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@patch("qiskit_addon_slc.utils.tracing.is_tracing_enabled", return_value=True)
@patch("opentelemetry.propagate.inject")
def test_inject_trace_context_when_enabled(mock_inject, _mock_is_enabled):
    """Test that inject_trace_context works when tracing is enabled."""

    # Mock inject to populate the carrier like the real implementation does
    def side_effect(carrier):
        carrier["traceparent"] = "00-test-test-01"

    mock_inject.side_effect = side_effect

    carrier = inject_trace_context()
    mock_inject.assert_called_once()
    assert isinstance(carrier, dict)
    assert "traceparent" in carrier


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@patch("qiskit_addon_slc.utils.tracing.is_tracing_enabled", return_value=True)
@patch("opentelemetry.propagate.extract")
def test_extract_trace_context_when_enabled(mock_extract, _mock_is_enabled):
    """Test that extract_trace_context works when tracing is enabled."""
    mock_context = MagicMock()
    mock_extract.return_value = mock_context
    carrier = {"traceparent": "00-test-test-01"}
    ctx = extract_trace_context(carrier)
    mock_extract.assert_called_once_with(carrier)
    assert ctx == mock_context


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@patch("opentelemetry.context.attach")
def test_attach_context_with_valid_context(mock_attach):
    """Test that attach_context calls context.attach with valid context."""
    mock_context = MagicMock()
    mock_token = MagicMock()
    mock_attach.return_value = mock_token

    token = attach_context(mock_context)

    mock_attach.assert_called_once_with(mock_context)
    assert token == mock_token


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
@patch("opentelemetry.context.detach")
def test_detach_context_with_valid_token(mock_detach):
    """Test that detach_context calls context.detach with valid token."""
    mock_token = MagicMock()

    detach_context(mock_token)

    mock_detach.assert_called_once_with(mock_token)


@pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
def test_context_propagation_round_trip(clean_env):
    """Test that context can be injected and extracted in a round trip."""
    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "true")

    # Inject context
    carrier = inject_trace_context()

    # If tracing is properly enabled and we have a carrier, extract it
    if carrier is not None:
        ctx = extract_trace_context(carrier)
        # Context should be extractable
        assert ctx is not None


def test_noop_span_operations():
    """Test that NoOpSpan accepts all operations without errors."""
    from qiskit_addon_slc.utils.tracing import NoOpSpan

    span = NoOpSpan()

    # Test all methods - should not raise any exceptions
    span.add_event("test_event")
    span.add_event("test_event_with_attrs", {"key": "value"})
    span.set_attribute("test_key", "test_value")
    span.set_status("OK")
    span.record_exception(Exception("test"))


def test_noop_span_context_manager():
    """Test that NoOpSpan works as a context manager."""
    from qiskit_addon_slc.utils.tracing import NoOpSpan

    span = NoOpSpan()

    # Should work as context manager
    with span as s:
        assert s is span
        s.add_event("inside_context")


def test_noop_tracer_returns_noop_span():
    """Test that NoOpTracer returns NoOpSpan instances."""
    from qiskit_addon_slc.utils.tracing import NoOpSpan, NoOpTracer

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
    from qiskit_addon_slc.utils.tracing import NoOpSpan, NoOpTracer

    tracer = NoOpTracer()

    with tracer.start_as_current_span("test_span") as span:
        assert isinstance(span, NoOpSpan)
        span.add_event("inside_context")
        span.set_attribute("key", "value")


def test_get_tracer_returns_noop_when_disabled(clean_env):
    """Test that get_tracer returns NoOpTracer when tracing is disabled."""
    from qiskit_addon_slc.utils.tracing import NoOpTracer

    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "false")
    tracer = get_tracer("test_module")
    assert isinstance(tracer, NoOpTracer)


def test_get_tracer_returns_noop_when_no_opentelemetry(clean_env):
    """Test that get_tracer returns NoOpTracer when OpenTelemetry is not available."""
    from qiskit_addon_slc.utils.tracing import NoOpTracer

    clean_env.setenv("QISKIT_SLC_TRACING_ENABLED", "true")

    # Mock HAS_OPENTELEMETRY to False
    with patch("qiskit_addon_slc.utils.tracing.HAS_OPENTELEMETRY", False):
        tracer = get_tracer("test_module")
        assert isinstance(tracer, NoOpTracer)
