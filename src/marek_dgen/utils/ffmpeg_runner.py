import subprocess


def run_ffmpeg_command(cmd: list[str]) -> None:
    """Run an ffmpeg/ffprobe command and raise on failure with stderr."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stderr: {result.stderr}\n"
            f"stdout: {result.stdout}"
        )
