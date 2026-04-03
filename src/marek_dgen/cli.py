import click
from marek_dgen.transcriber import transcribe_and_diarize
from marek_dgen.translation import translation_request

content_desc = "conversation in the episode nr. 311 of the Peter Attia's podcast Drive called Longevity 101 where he concisely summarize results of his more than one decade lasting research into human longevity"

@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--hf-token', required=True, help='Your Hugging Face token for diarization.')
@click.option('--einfra-apikey', required=True, help='Your E-infra AIaaS API key for translation.')
@click.option('--einfra-baseurl', default='https://llm.ai.e-infra.cz/v1', help='Base URL for E-infra AIaaS API.')
@click.option('--translating-model', default='aya-expanse', help='Model to use for translation from ENG to CZ.')
@click.option('--temperature', default=0.1, help='Temperature setting for translation model.')
@click.option('--max-tokens', default=2048, help='Max tokens for translation output.')
def main(audio_file, hf_token, einfra_apikey, einfra_baseurl, translating_model, temperature, max_tokens):
    """
    DABUJ: Automated AI/ML pipeline for dubbing.
    Runs the WhisperX transcription and diarization pipeline.
    """
    click.echo(f"=====================================================================")
    click.echo(f">>> Processing of `{audio_file}` has started! <<<")
    click.echo(f"=====================================================================")

    # Call the logic from your other file
    segments = transcribe_and_diarize(audio_file, hf_token)

    click.echo("\n--- DIARIZATION FINAL OUTPUT ---")
    for segment in segments:
        click.echo(f"[{segment.get('start'):.2f} - {segment.get('end'):.2f}] Speaker {segment.get('speaker', 'UNKNOWN')}: {segment.get('text')}")

if __name__ == '__main__':
    main()
