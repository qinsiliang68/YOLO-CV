class GapValueError(RuntimeError):
    """Base exception."""

class ContractError(GapValueError):
    pass

class ValidationError(GapValueError):
    pass

class ConfigurationError(GapValueError):
    pass

class InfeasibleMatchError(GapValueError):
    pass

class ArtifactExistsError(GapValueError):
    pass

class ExternalCommandError(GapValueError):
    pass

class LockHeldError(ValidationError):
    """A live local/remote owner currently holds a retryable execution lock."""
