import click
from marek_dgen.transcriber import transcribe_and_diarize
from marek_dgen.translation import translation_request
from TTS.api import TTS

content_desc = "conversation in the episode nr. 311 of the Peter Attia's podcast Drive called Longevity 101 where he concisely summarize results of his more than one decade lasting research into human longevity"

@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--hf-token', required=True, help='Your Hugging Face token for diarization.')
@click.option('--einfra-baseurl', default='https://llm.ai.e-infra.cz/v1', help='Base URL for E-infra AIaaS API.')
@click.option('--einfra-apikey', required=True, help='Your E-infra AIaaS API key for translation.')
@click.option('--translating-model', default='aya-expanse', help='Model to use for translation from ENG to CZ.')
@click.option('--temperature', default=0.05, help='Temperature setting for translation model.')
@click.option('--max-tokens', default=75, help='Max tokens for translation output.')
@click.option('--voices', default=None, help='Comma-separated list of speaker voices for TTS (e.g. "speaker1,speaker2"). The speakers will be assigned voices exactly in the same order as they appear in the transcript. If not provided, voices will be assigned to speaker randomly.')
def main(audio_file, hf_token, einfra_apikey, einfra_baseurl, translating_model, temperature, max_tokens):
    """
    DABUJ: Automated AI/ML pipeline for dubbing.
    Runs the WhisperX transcription and diarization pipeline.
    """
    click.echo(f"==================================================================================")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    print(tts.list_models())
    click.echo(f"==================================================================================")


    click.echo(f"==================================================================================")
    click.echo(f">>> Processing of `{audio_file}` has started! <<<")
    click.echo(f"==================================================================================")

    click.echo(f"\n>>> Phase 1: Transcription and Diarization")
    click.echo(f"----------------------------------------------------------------------------------")

    segments = transcribe_and_diarize(audio_file, hf_token)

    click.echo("\n>>> Final output of Phase 1 (Diarization):::")
    long_transcript = ""
    for segment in segments:
        long_transcript += f"[{segment.get('start'):.2f} - {segment.get('end'):.2f}] Speaker {segment.get('speaker', 'UNKNOWN')}: {segment.get('text')}\n"
    click.echo_via_pager(long_transcript)

    if not click.confirm("\nDo you want to proceed to Phase 2 - Translation?", default=True, show_default=True):
      click.echo("Aborting the execution. The program exits soon.")
      return

    click.echo(f"\n>>> Phase 2: Translation")
    click.echo(f"----------------------------------------------------------------------------------")

    for segment in segments:
        eng_text = segment["text"]

        cz_text = translation_request(einfra_baseurl, einfra_apikey, translating_model, temperature, max_tokens,eng_text, content_desc)

        segment["text"] = cz_text

    click.echo("\n>>> Final output of Phase 2 (Translation):::")
    long_translation = ""
    for segment in segments:
        long_translation += f"[{segment.get('start'):.2f} - {segment.get('end'):.2f}] Speaker {segment.get('speaker', 'UNKNOWN')}: {segment.get('text')}\n"
    click.echo_via_pager(long_translation)

if __name__ == '__main__':
    main()
