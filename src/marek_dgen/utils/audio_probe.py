import json
import subprocess
from pathlib import Path


def probe_audio_format(wav_path: Path) -> tuple[int, str]:
    """Return (sample_rate, channel_layout) for a WAV file."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "json",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    sample_rate = int(stream["sample_rate"])
    channels = int(stream["channels"])
    layout = "stereo" if channels >= 2 else "mono"
    return sample_rate, layout
