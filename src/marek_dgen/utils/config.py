import json
from pathlib import Path

from pydantic import BaseModel, Field


class TranslationConfig(BaseModel):
    default_model: str = "qwen3.5"
    default_temperature: float = 0.05
    default_max_tokens: int = 75


class TTSConfig(BaseModel):
    voices_mapping: dict[str, str]
    language: str = "cs"
    output_format: str = "wav"


class OutputConfig(BaseModel):
    directory: str = "output"
    naming_pattern: str = "{speaker}_{index}.wav"


class AppConfig(BaseModel):
    content_description: str = ""
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    tts: TTSConfig
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate the advanced configuration JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AppConfig(**data)
