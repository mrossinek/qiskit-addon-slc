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

# Warning: this module is not documented and it does not have an RST file.
# If we ever publicly expose interfaces users can import from this module,
# we should set up its RST file.

"""Global settings.

.. currentmodule:: qiskit_addon_slc.globals

This module provides a number of globally configurable settings.

.. autoclass:: PROGRESS_POLLING_PERIOD

.. autoclass:: ZERO_ATOL

.. autoclass:: TRACING_ENABLED

.. autoclass:: OTEL_SERVICE_NAME

.. autoclass:: OTEL_TRACES_EXPORTER

.. autoclass:: OTEL_EXPORTER_OTLP_ENDPOINT
"""

import os
import sys

PROGRESS_POLLING_PERIOD = 5
"""The polling period for the progress indicator of the commutator bound task computation.

This number corresponds to the number of seconds to wait between progress indicator updates.
It defaults to ``5``.
"""

ZERO_ATOL = 10 * sys.float_info.epsilon
"""The absolute tolerance value below which terms are considered truly zero and are truncated.

This defaults to the value of ``10 * sys.float_info.epsilon``.
"""

TRACING_ENABLED = os.getenv("QISKIT_SLC_TRACING_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
    "on",
)
"""Whether OpenTelemetry tracing is enabled for profiling parallel computations.

This can be controlled via the ``QISKIT_SLC_TRACING_ENABLED`` environment variable.
Set to ``"true"``, ``"1"``, ``"yes"``, or ``"on"`` to enable tracing. Defaults to ``False``.
"""

OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "qiskit-addon-slc")
"""The service name to use for OpenTelemetry traces.

This can be controlled via the ``OTEL_SERVICE_NAME`` environment variable.
Defaults to ``"qiskit-addon-slc"``.
"""

OTEL_TRACES_EXPORTER = os.getenv("OTEL_TRACES_EXPORTER", "console")
"""The OpenTelemetry traces exporter type to use.

This can be controlled via the ``OTEL_TRACES_EXPORTER`` environment variable.
Supported values are ``"console"`` (default) and ``"otlp"``.
Defaults to ``"console"``.
"""

OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
"""The OTLP exporter endpoint URL.

This can be controlled via the ``OTEL_EXPORTER_OTLP_ENDPOINT`` environment variable.
Required when ``OTEL_TRACES_EXPORTER`` is set to ``"otlp"``.
Defaults to ``None``.
"""

TRACER_FLUSH_TIMEOUT_MS = 5000
"""The timeout in milliseconds for flushing OpenTelemetry spans on shutdown.

This defaults to ``5000`` milliseconds (5 seconds).
"""

WORKER_INDEX_NOT_INITIALIZED = -1
"""Sentinel value indicating a worker process has not been initialized or is not in a worker context.

This defaults to ``-1``.
"""
