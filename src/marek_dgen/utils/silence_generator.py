from pathlib import Path

from marek_dgen.utils.ffmpeg_runner import run_ffmpeg_command


def generate_silence(
    duration: float, sample_rate: int, layout: str, out_path: Path
) -> None:
    """Generate a silent WAV file of the given duration."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl={layout}",
        "-t",
        str(duration),
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        str(out_path),
    ]
    run_ffmpeg_command(cmd)
