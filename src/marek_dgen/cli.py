import click
from marek_dgen.transcriber import transcribe_and_diarize

@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--hf-token', required=True, help='Your Hugging Face token for diarization.')
def main(audio_file, hf_token):
    """
    DABUJ: Automated AI/ML pipeline for dubbing.
    Runs the WhisperX transcription and diarization pipeline.
    """
    click.echo(f"Starting processing for: {audio_file}")

    # Call the logic from your other file
    segments = transcribe_and_diarize(audio_file, hf_token)

    click.echo("\n--- FINAL OUTPUT ---")
    for segment in segments:
        click.echo(f"[{segment.get('start'):.2f} - {segment.get('end'):.2f}] Speaker {segment.get('speaker', 'UNKNOWN')}: {segment.get('text')}")

if __name__ == '__main__':
    main()
