"""Custom exception hierarchy for backend robustness."""
from __future__ import annotations


class BackendError(Exception):
    """Base class for backend related errors."""


class ExternalServiceError(BackendError):
    """Raised when an external dependency fails permanently."""


class RateLimitExceeded(BackendError):
    """Raised when a rate limit threshold is surpassed."""
