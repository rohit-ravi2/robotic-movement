"""Telemetry: trace.Observer (per-step channels via a ctrl_fn observer), descriptors.describe (behaviour descriptors, descriptive
only), summary.summarize (engine.emergence wrapper)."""
from . import trace, descriptors, summary   # noqa
from .trace import Observer, observe   # noqa
from .descriptors import describe, DEFAULT_KEYS   # noqa
from .summary import summarize, one_line   # noqa
