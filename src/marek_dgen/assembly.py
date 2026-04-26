import json
import tempfile
from pathlib import Path

from marek_dgen.utils.config import AssemblyConfig
from marek_dgen.utils.audio_probe import probe_audio_format
from marek_dgen.utils.ffmpeg_runner import run_ffmpeg_command
from marek_dgen.utils.silence_generator import generate_silence


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
    sample_rate, layout = probe_audio_format(first_wav)

    with tempfile.TemporaryDirectory(
        dir=output_dir, prefix="tmp_assembly_"
    ) as tmpdir:
        tmpdir_path = Path(tmpdir)
        concat_items: list[str] = []

        # Leading silence before the first segment
        first_start = float(entries[0]["start"])
        if first_start > 0:
            silence_path = tmpdir_path / "silence_start.wav"
            generate_silence(first_start, sample_rate, layout, silence_path)
            concat_items.append(str(silence_path.resolve()))

        for i, entry in enumerate(entries):
            wav_path = output_dir / entry["wav_path"]
            concat_items.append(str(wav_path.resolve()))

            if i < len(entries) - 1:
                gap = float(entries[i + 1]["start"]) - float(entry["end"])
                if gap > 0:
                    silence_path = tmpdir_path / f"silence_{i}.wav"
                    generate_silence(gap, sample_rate, layout, silence_path)
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
        run_ffmpeg_command(concat_cmd)

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
        run_ffmpeg_command(mux_cmd)

    return final_video_path
