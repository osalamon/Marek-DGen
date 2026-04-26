import json
import subprocess
import tempfile
from pathlib import Path

from marek_dgen.utils.config import AssemblyConfig


def _run_ffmpeg_command(cmd: list[str]) -> None:
    """Run an ffmpeg/ffprobe command and raise on failure with stderr."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stderr: {result.stderr}\n"
            f"stdout: {result.stdout}"
        )


def _probe_audio_format(wav_path: Path) -> tuple[int, str]:
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


def _generate_silence(
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
    _run_ffmpeg_command(cmd)


def assemble_dubbed_video(
    manifest_path: Path,
    original_video: Path,
    output_dir: Path,
    assembly_config: AssemblyConfig,
) -> Path:
    """Assemble a gap-preserving dubbed audio track and mux it into the original video."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = [
        e
        for e in manifest
        if not e.get("skipped", False) and e.get("wav_path")
    ]
    if not entries:
        raise ValueError("No audio segments found in manifest to assemble.")

    first_wav = output_dir / entries[0]["wav_path"]
    sample_rate, layout = _probe_audio_format(first_wav)

    with tempfile.TemporaryDirectory(
        dir=output_dir, prefix="tmp_assembly_"
    ) as tmpdir:
        tmpdir_path = Path(tmpdir)
        concat_items: list[str] = []

        # Leading silence before the first segment
        first_start = float(entries[0]["start"])
        if first_start > 0:
            silence_path = tmpdir_path / "silence_start.wav"
            _generate_silence(first_start, sample_rate, layout, silence_path)
            concat_items.append(str(silence_path.resolve()))

        for i, entry in enumerate(entries):
            wav_path = output_dir / entry["wav_path"]
            concat_items.append(str(wav_path.resolve()))

            if i < len(entries) - 1:
                gap = float(entries[i + 1]["start"]) - float(entry["end"])
                if gap > 0:
                    silence_path = tmpdir_path / f"silence_{i}.wav"
                    _generate_silence(gap, sample_rate, layout, silence_path)
                    concat_items.append(str(silence_path.resolve()))

        # Write ffmpeg concat demuxer list
        list_path = tmpdir_path / "concat_list.txt"
        with open(list_path, "w", encoding="utf-8") as f:
            for item in concat_items:
                # Escape single quotes for ffmpeg concat syntax
                escaped = item.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        dubbed_wav = output_dir / "dubbed_track.wav"
        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            str(dubbed_wav),
        ]
        _run_ffmpeg_command(concat_cmd)

        # Mux dubbed audio with original video
        suffix = assembly_config.video_suffix
        out_ext = assembly_config.output_format
        final_name = f"{original_video.stem}{suffix}.{out_ext}"
        final_video_path = output_dir / final_name

        mux_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(original_video),
            "-i",
            str(dubbed_wav),
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            str(final_video_path),
        ]
        _run_ffmpeg_command(mux_cmd)

    return final_video_path
