import click
from marek_dgen.transcriber import transcribe_and_diarize
from marek_dgen.translation import translation_request
from TTS.api import TTS

content_desc_default = "conversation in the episode nr. 311 of the Peter Attia's podcast Drive called Longevity 101 where he concisely summarize results of his more than one decade lasting research into human longevity"


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "--hf-token", required=True, help="Your Hugging Face token for diarization."
)
@click.option(
    "--einfra-baseurl",
    default="https://llm.ai.e-infra.cz/v1",
    help="Base URL for E-infra AIaaS API.",
)
@click.option(
    "--einfra-apikey", required=True, help="Your E-infra AIaaS API key for translation."
)
@click.option(
    "--translating-model",
    default="deepseek-v3.2",
    help="Model to use for translation from ENG to CZ.",
)
@click.option(
    "--temperature", default=0.05, help="Temperature setting for translation model."
)
@click.option("--max-tokens", default=75, help="Max tokens for translation output.")
@click.option(
    "--voices",
    default=None,
    help='Comma-separated list of speaker voices for TTS (e.g. "speaker1,speaker2"). The speakers will be assigned voices exactly in the same order as they appear in the transcript. If not provided, voices will be assigned to speaker randomly.',
)
@click.option(
    "--content-description",
    # default=None,
    default=content_desc_default,  # TODO change after testing
    help='Enabling the Context Injection into the prompt which asks the E-Infra large model for translating the original English transcript.  It should improve quality of translation thanks to:\n[A] Hinting LLM with a Domain-Specific Vocabulary. The term "bug" would probably denoted something very different in 1) YouTube education show on Computer Science, 2) BBC documentary about spiders from Amazonia, or even 3) some dystopian movie about junkies in near-future pandemia of a new drug.\b[B] The difference in Czech between e.g. formal, academic, professional, spoken/informal, slang, etc language is very distinct, compared to the English. This may halp also.\n[NOTE:] The overall prompt is written in such way to mitigate a risk of hallucionations, caused by providing this context. If you spot that the model hallucinates even though or if the context do not provide desired results, just do not use it or provide an empty value.',
)
def main(
    audio_file,
    hf_token,
    einfra_apikey,
    einfra_baseurl,
    translating_model,
    temperature,
    max_tokens,
):
    """
    DABUJ: Automated AI/ML pipeline for dubbing.
    Runs the WhisperX transcription and diarization pipeline.
    """
    click.echo(
        f"=================================================================================="
    )
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    print(tts.list_models())
    click.echo(
        f"=================================================================================="
    )

    # PHASE 1 - TRANSCRIPT + DIARIZE
    click.echo(
        f"\n=================================================================================="
    )
    click.echo(f">>> Processing of `{audio_file}` has started! <<<")
    click.echo(
        f"=================================================================================="
    )

    click.echo(f"\n\n>>> Phase 1: Transcription and Diarization")
    click.echo(
        f"----------------------------------------------------------------------------------"
    )

    segments = transcribe_and_diarize(audio_file, hf_token)

    click.echo("\n>>> Final output of Phase 1 (Transcribe & Diarize):::")
    long_transcript = ""
    for segment in segments:
        long_transcript += f"[{segment.get('start'):.2f} - {segment.get('end'):.2f}] Speaker {segment.get('speaker', 'UNKNOWN')}: {segment.get('text')}\n"
    click.echo_via_pager(long_transcript)

    # PHASE 2 - TRANSLATE
    click.echo(f"\n>>> PHASE 1 FINISHED WITH SUCCESS! <<<")
    click.echo(
        f"--- The transcription of speeches in the video is available. The exact timing of the start and the end of each speaker's speech secured. ---"
    )
    click.echo(
        f"--------------------------------------------------------------------------------------------------------------------------------------------"
    )

    if not click.confirm(
        "\nDo you want to proceed to Phase 2 - Translation?",
        default=True,
        show_default=True,
    ):
        click.echo("Aborting the execution. The program exits soon.")
        return

    click.echo(f"\n\n>>> Phase 2: Translation")
    click.echo(
        f"----------------------------------------------------------------------------------"
    )

    for segment in segments:
        eng_text = segment["text"]

        cz_text = translation_request(
            content_desc,
            einfra_baseurl,
            einfra_apikey,
            translating_model,
            temperature,
            max_tokens,
            eng_text,
        )

        segment["text"] = cz_text

    click.echo("\n>>> Final output of Phase 2 (Translation):::")
    long_translation = ""
    for segment in segments:
        long_translation += f"[{segment.get('start'):.2f} - {segment.get('end'):.2f}] Speaker {segment.get('speaker', 'UNKNOWN')}: {segment.get('text')}\n"
    click.echo_via_pager(long_translation)

    # PHASE 3 - TTS
    click.echo(f"\n>>> PHASE 2 FINISHED WITH SUCCESS! <<<")
    click.echo(
        f"--- The transcript was translated from Englisht to Czech by the choosen LLM. ---"
    )
    click.echo(
        f"--------------------------------------------------------------------------------------------------------------------------------------------"
    )

    if not click.confirm(
        "\nDo you want to proceed to Phase 3 - Text To Speech?",
        default=True,
        show_default=True,
    ):
        click.echo("Aborting the execution. The program exits soon.")
        return

    click.echo(f"\n\n>>> Phase 3: Performing Text to Speech")
    click.echo(
        f"----------------------------------------------------------------------------------"
    )


if __name__ == "__main__":
    main()
