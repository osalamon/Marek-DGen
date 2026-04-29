from pathlib import Path
from TTS.api import TTS


def synthesize_segment(
    tts_model: TTS,
    text: str,
    speaker_wav: str,
    output_path: str,
    language: str = "cs",
) -> str:
    """Synthesize a single text segment to speech and save to a WAV file.

    Args:
        tts_model: An initialized TTS model instance.
        text: The text to synthesize (Czech).
        speaker_wav: Path to the speaker reference audio file.
        output_path: Destination file path for the generated audio.
        language: Target language code (default: "cs").

    Returns:
        The output file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tts_model.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=output_path,
    )
    return output_path
