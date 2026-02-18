from abc import ABC, abstractmethod
from typing import Optional


class PermanentError(Exception):
    """Error that should not be retried (e.g., URL fetch failure)."""
    pass


class APIProvider(ABC):
    @abstractmethod
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
    ) -> str:
        """Transcribe audio and return the text."""
        ...


_REGISTRY: dict[str, type[APIProvider]] = {}


def register(prefix: str):
    """Decorator to register a provider class under a model prefix."""
    def decorator(cls: type[APIProvider]):
        _REGISTRY[prefix] = cls
        return cls
    return decorator


def get_provider(model_name: str) -> tuple[APIProvider, str]:
    """Look up provider by model_name prefix, return (provider_instance, variant)."""
    for prefix, cls in _REGISTRY.items():
        if model_name.startswith(prefix + "/"):
            variant = model_name[len(prefix) + 1:]
            return cls(), variant
    raise ValueError(
        f"No provider registered for model '{model_name}'. "
        f"Known prefixes: {list(_REGISTRY.keys())}"
    )


# Auto-import all provider modules so they register themselves
from . import speechmatics_provider
from . import assemblyai_provider
from . import openai_provider
from . import elevenlabs_provider
from . import revai_provider
from . import aquavoice_provider
