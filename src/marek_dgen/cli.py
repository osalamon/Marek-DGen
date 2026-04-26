import click
from pathlib import Path
from marek_dgen.transcriber import transcribe_and_diarize
from marek_dgen.translation import translation_request
from marek_dgen.tts import synthesize_segment
from marek_dgen.utils.config import load_config, AppConfig
from marek_dgen.utils.create_manifest import create_manifest
from marek_dgen.assembly import assemble_dubbed_video
from TTS.api import TTS

content_desc_default = "conversation in the episode nr. 311 of the Peter Attia's podcast Drive called Longevity 101 where he concisely summarize results of his more than one decade lasting research into human longevity"


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "--config",
    type=click.Path(exists=True),
    default="advanced_conf.json",
    help="Path to advanced configuration JSON file.",
)
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
    default=None,
    help="Model to use for translation from ENG to CZ.",
)
@click.option(
    "--temperature", default=None, type=float, help="Temperature setting for translation model."
)
@click.option("--max-tokens", default=None, type=int, help="Max tokens for translation output.")
@click.option(
    "--voices",
    default=None,
    help='Comma-separated list of speaker voices for TTS (e.g. "speaker1,speaker2"). The speakers will be assigned voices exactly in the same order as they appear in the transcript. If not provided, voices will be assigned from config mapping.',
)
@click.option(
    "--content-description",
    default=None,
    help='Enabling the Context Injection into the prompt which asks the E-Infra large model for translating the original English transcript.  It should improve quality of translation thanks to:\n[A] Hinting LLM with a Domain-Specific Vocabulary. The term "bug" would probably denoted something very different in 1) YouTube education show on Computer Science, 2) BBC documentary about spiders from Amazonia, or even 3) some dystopian movie about junkies in near-future pandemia of a new drug.\b[B] The difference in Czech between e.g. formal, academic, professional, spoken/informal, slang, etc language is very distinct, compared to the English. This may halp also.\n[NOTE:] The overall prompt is written in such way to mitigate a risk of hallucionations, caused by providing this context. If you spot that the model hallucinates even though or if the context do not provide desired results, just do not use it or provide an empty value.',
)
def main(
    audio_file,
    config,
    hf_token,
    einfra_apikey,
    einfra_baseurl,
    translating_model,
    temperature,
    max_tokens,
    voices,
    content_description,
):
    """
    DABUJ: Automated AI/ML pipeline for dubbing.
    Runs the WhisperX transcription and diarization pipeline.
    """
    # Load configuration
    app_config: AppConfig = load_config(config)

    # Resolve settings: CLI overrides config defaults
    content_desc = content_description if content_description is not None else app_config.content_description
    if not content_desc:
        content_desc = content_desc_default

    translating_model = translating_model if translating_model is not None else app_config.translation.default_model
    temperature = temperature if temperature is not None else app_config.translation.default_temperature
    max_tokens = max_tokens if max_tokens is not None else app_config.translation.default_max_tokens

    click.echo(
        f"=================================================================================="
    )
    click.echo(f"Configuration loaded from: {config}")
    click.echo(f"Content description: {content_desc[:80]}...")
    click.echo(f"Translation model: {translating_model} | temp: {temperature} | max_tokens: {max_tokens}")
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
        f"--- The transcript was translated from English to Czech by the chosen LLM. ---"
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

    # Prepare output directory
    output_dir = Path(app_config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine unique speakers in order of appearance
    unique_speakers = []
    seen = set()
    for segment in segments:
        spk = segment.get('speaker', 'UNKNOWN')
        if spk not in seen:
            seen.add(spk)
            unique_speakers.append(spk)

    # Map speakers to voice names
    if voices:
        voice_names = [v.strip() for v in voices.split(",")]
    else:
        voice_names = list(app_config.tts.voices_mapping.keys())

    speaker_to_voice_name = {}
    for i, speaker in enumerate(unique_speakers):
        if i < len(voice_names):
            speaker_to_voice_name[speaker] = voice_names[i]
        else:
            speaker_to_voice_name[speaker] = voice_names[i % len(voice_names)]

    # Resolve voice names to WAV file paths
    speaker_to_wav = {}
    for speaker, voice_name in speaker_to_voice_name.items():
        if voice_name in app_config.tts.voices_mapping:
            speaker_to_wav[speaker] = app_config.tts.voices_mapping[voice_name]
        else:
            # Fallback: treat as direct path
            speaker_to_wav[speaker] = voice_name

    click.echo("Speaker mapping:")
    for spk, wav in speaker_to_wav.items():
        voice_name = speaker_to_voice_name[spk]
        click.echo(f"  {spk} -> {voice_name} -> {wav}")

    # Init TTS
    click.echo("\nInitializing TTS model (this may take a while)...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    # Generate audio for each segment and collect manifest entries
    manifest_entries = []
    generated_count = 0
    for i, segment in enumerate(segments):
        speaker = segment.get('speaker', 'UNKNOWN')
        text = segment['text']
        speaker_wav = speaker_to_wav.get(speaker)

        entry = {
            "index": i,
            "start": segment.get("start", 0.0),
            "end": segment.get("end", 0.0),
            "speaker": speaker,
            "text": text,
        }

        if not speaker_wav:
            click.echo(f"Warning: No voice mapping for {speaker}, skipping segment {i}")
            entry["wav_path"] = None
            entry["skipped"] = True
            manifest_entries.append(entry)
            continue

        if not text or not text.strip():
            click.echo(f"Warning: Empty text for segment {i}, skipping")
            entry["wav_path"] = None
            entry["skipped"] = True
            manifest_entries.append(entry)
            continue

        output_filename = app_config.output.naming_pattern.format(
            speaker=speaker.replace(" ", "_"),
            index=i
        )
        output_path = output_dir / output_filename

        click.echo(f"[{i+1}/{len(segments)}] Synthesizing {speaker} -> {output_path.name}")
        synthesize_segment(
            tts_model=tts,
            text=text,
            speaker_wav=speaker_wav,
            output_path=str(output_path),
            language=app_config.tts.language,
        )
        generated_count += 1
        entry["wav_path"] = str(output_path.relative_to(output_dir))
        entry["skipped"] = False
        manifest_entries.append(entry)

    click.echo(f"\n>>> PHASE 3 FINISHED WITH SUCCESS! <<<")
    click.echo(f"Generated {generated_count} audio files in {output_dir.absolute()}")

    # Create manifest and assemble final video
    manifest_path = create_manifest(manifest_entries, output_dir)
    click.echo(f"Manifest written to: {manifest_path}")

    final_video_path = assemble_dubbed_video(
        manifest_path=manifest_path,
        original_video=Path(audio_file),
        output_dir=output_dir,
        assembly_config=app_config.assembly,
    )
    click.echo(f"Final dubbed video written to: {final_video_path}")


if __name__ == "__main__":
    main()
