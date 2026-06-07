class DataCollectionError(RuntimeError):
    """Raised when market data collection fails."""


class DataValidationError(ValueError):
    """Raised when collected market data is invalid."""
