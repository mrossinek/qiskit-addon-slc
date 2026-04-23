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
import unittest
from unittest.mock import MagicMock, patch

from qiskit_addon_slc.optionals import HAS_OPENTELEMETRY
from qiskit_addon_slc.utils.tracing import (
    attach_context,
    detach_context,
    extract_trace_context,
    get_tracer,
    inject_trace_context,
    is_tracing_enabled,
)


class TestTracingUtilities(unittest.TestCase):
    """Test cases for tracing utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Store original environment variable
        self.original_tracing_enabled = os.environ.get("QISKIT_SLC_TRACING_ENABLED")
        self.original_service_name = os.environ.get("OTEL_SERVICE_NAME")

    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment variables
        if self.original_tracing_enabled is not None:
            os.environ["QISKIT_SLC_TRACING_ENABLED"] = self.original_tracing_enabled
        elif "QISKIT_SLC_TRACING_ENABLED" in os.environ:
            del os.environ["QISKIT_SLC_TRACING_ENABLED"]

        if self.original_service_name is not None:
            os.environ["OTEL_SERVICE_NAME"] = self.original_service_name
        elif "OTEL_SERVICE_NAME" in os.environ:
            del os.environ["OTEL_SERVICE_NAME"]

    def test_is_tracing_enabled_default(self):
        """Test that tracing is disabled by default."""
        if "QISKIT_SLC_TRACING_ENABLED" in os.environ:
            del os.environ["QISKIT_SLC_TRACING_ENABLED"]
        self.assertFalse(is_tracing_enabled())

    @unittest.skipIf(not HAS_OPENTELEMETRY, "OpenTelemetry not installed")
    def test_is_tracing_enabled_true(self):
        """Test that tracing can be enabled via environment variable."""
        os.environ["QISKIT_SLC_TRACING_ENABLED"] = "true"
        self.assertTrue(is_tracing_enabled())

    def test_is_tracing_enabled_false(self):
        """Test that tracing can be explicitly disabled."""
        os.environ["QISKIT_SLC_TRACING_ENABLED"] = "false"
        self.assertFalse(is_tracing_enabled())

    @unittest.skipIf(not HAS_OPENTELEMETRY, "OpenTelemetry not installed")
    def test_is_tracing_enabled_case_insensitive(self):
        """Test that environment variable is case-insensitive."""
        os.environ["QISKIT_SLC_TRACING_ENABLED"] = "TRUE"
        self.assertTrue(is_tracing_enabled())

        os.environ["QISKIT_SLC_TRACING_ENABLED"] = "False"
        self.assertFalse(is_tracing_enabled())

    def test_get_tracer_returns_none_when_disabled(self):
        """Test that get_tracer returns None when tracing is disabled."""
        os.environ["QISKIT_SLC_TRACING_ENABLED"] = "false"
        tracer = get_tracer("test_module")
        if not HAS_OPENTELEMETRY:
            self.assertIsNone(tracer)

    @unittest.skipIf(not HAS_OPENTELEMETRY, "OpenTelemetry not installed")
    def test_get_tracer_returns_tracer(self):
        """Test that get_tracer returns a tracer instance when enabled."""
        os.environ["QISKIT_SLC_TRACING_ENABLED"] = "true"
        tracer = get_tracer("test_module")
        self.assertIsNotNone(tracer)

    @unittest.skipIf(not HAS_OPENTELEMETRY, "OpenTelemetry not installed")
    def test_get_tracer_with_custom_name(self):
        """Test that get_tracer accepts custom names."""
        os.environ["QISKIT_SLC_TRACING_ENABLED"] = "true"
        tracer = get_tracer("custom_name")
        self.assertIsNotNone(tracer)

    def test_inject_trace_context_when_disabled(self):
        """Test that inject_trace_context returns None when tracing is disabled."""
        os.environ["QISKIT_SLC_TRACING_ENABLED"] = "false"
        carrier = inject_trace_context()
        self.assertIsNone(carrier)

    def test_extract_trace_context_with_none(self):
        """Test that extract_trace_context handles None carrier."""
        ctx = extract_trace_context(None)
        self.assertIsNone(ctx)

    def test_extract_trace_context_when_disabled(self):
        """Test that extract_trace_context returns None when tracing is disabled."""
        os.environ["QISKIT_SLC_TRACING_ENABLED"] = "false"
        carrier = {"traceparent": "00-test-test-01"}
        ctx = extract_trace_context(carrier)
        self.assertIsNone(ctx)

    def test_attach_context_with_none(self):
        """Test that attach_context handles None context."""
        token = attach_context(None)
        self.assertIsNone(token)

    def test_detach_context_with_none(self):
        """Test that detach_context handles None token."""
        # Should not raise an exception
        detach_context(None)

    @unittest.skipIf(not HAS_OPENTELEMETRY, "OpenTelemetry not installed")
    @patch("qiskit_addon_slc.utils.tracing.is_tracing_enabled", return_value=True)
    @patch("opentelemetry.propagate.inject")
    def test_inject_trace_context_when_enabled(self, mock_inject, _mock_is_enabled):
        """Test that inject_trace_context works when tracing is enabled."""

        # Mock inject to populate the carrier like the real implementation does
        def side_effect(carrier):
            carrier["traceparent"] = "00-test-test-01"

        mock_inject.side_effect = side_effect

        carrier = inject_trace_context()
        mock_inject.assert_called_once()
        self.assertIsInstance(carrier, dict)
        self.assertIn("traceparent", carrier)

    @unittest.skipIf(not HAS_OPENTELEMETRY, "OpenTelemetry not installed")
    @patch("qiskit_addon_slc.utils.tracing.is_tracing_enabled", return_value=True)
    @patch("opentelemetry.propagate.extract")
    def test_extract_trace_context_when_enabled(self, mock_extract, _mock_is_enabled):
        """Test that extract_trace_context works when tracing is enabled."""
        mock_context = MagicMock()
        mock_extract.return_value = mock_context
        carrier = {"traceparent": "00-test-test-01"}
        ctx = extract_trace_context(carrier)
        mock_extract.assert_called_once_with(carrier)
        self.assertEqual(ctx, mock_context)

    @unittest.skipIf(not HAS_OPENTELEMETRY, "OpenTelemetry not installed")
    @patch("qiskit_addon_slc.utils.tracing.HAS_OPENTELEMETRY", True)
    @patch("opentelemetry.context.attach")
    def test_attach_context_with_valid_context(self, mock_attach):
        """Test that attach_context calls context.attach with valid context."""
        mock_context = MagicMock()
        mock_token = MagicMock()
        mock_attach.return_value = mock_token

        token = attach_context(mock_context)

        mock_attach.assert_called_once_with(mock_context)
        self.assertEqual(token, mock_token)

    @unittest.skipIf(not HAS_OPENTELEMETRY, "OpenTelemetry not installed")
    @patch("qiskit_addon_slc.utils.tracing.HAS_OPENTELEMETRY", True)
    @patch("opentelemetry.context.detach")
    def test_detach_context_with_valid_token(self, mock_detach):
        """Test that detach_context calls context.detach with valid token."""
        mock_token = MagicMock()

        detach_context(mock_token)

        mock_detach.assert_called_once_with(mock_token)

    @unittest.skipIf(not HAS_OPENTELEMETRY, "OpenTelemetry not installed")
    def test_context_propagation_round_trip(self):
        """Test that context can be injected and extracted in a round trip."""
        os.environ["QISKIT_SLC_TRACING_ENABLED"] = "true"

        # Inject context
        carrier = inject_trace_context()

        # If tracing is properly enabled and we have a carrier, extract it
        if carrier is not None:
            ctx = extract_trace_context(carrier)
            # Context should be extractable
            self.assertIsNotNone(ctx)


if __name__ == "__main__":
    unittest.main()
