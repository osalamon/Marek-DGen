import json
from pathlib import Path

from pydantic import BaseModel, Field


class TranscriptionConfig(BaseModel):
    device: str | None = None
    batch_size: int | None = None
    compute_type: str | None = None
    speakers_count: int | None = 2


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


class AssemblyConfig(BaseModel):
    video_suffix: str = "_CZ"
    output_format: str = "mp4"


class AppConfig(BaseModel):
    content_description: str = ""
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    tts: TTSConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    assembly: AssemblyConfig = Field(default_factory=AssemblyConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate the advanced configuration JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AppConfig(**data)
