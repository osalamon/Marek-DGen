# Marek-DGen

Just a silly little ML pipeline intended for translating and dubbing the most important episodes of my favourite podcasts, making them accessible for members of my family not very well versed in English.

## Configuration

The pipeline uses a two-layer configuration system:

1. **Environment variables** (`.env`) — secrets and deployment-specific values.
   Copy and edit the example file:
   ```bash
   cp .env.example .env
   ```

2. **Structured JSON config** (`advanced_conf.json`) — domain-specific settings such as voice mappings, translation defaults, and content description. This file is safe to commit and reuse across runs.

CLI options override values from `advanced_conf.json`.

### Example `advanced_conf.json`

The repository includes a fully working `advanced_conf.json` that maps speaker keys to files in `./voices` and sets translation/TTS defaults.

## Usage via CLI

Minimal example (reads most settings from `advanced_conf.json`):

```bash
dabuj data/whisperx2mintest.mp4 \
    --hf-token="<hf_YOUR_HUGGINGFACE_API_READ_TOKEN>" \
    --einfra-apikey="sk-<YOUR_EINFRA_AI_AS_A_SERVICE_API_TOKEN>" \
    --config="advanced_conf.json"
```

Optional flags (when omitted, values are read from `advanced_conf.json`):

```bash
    --einfra-baseurl="https://llm.ai.e-infra.cz/v1" \
    --translating-model="qwen3.5-122b" \
    --temperature=0.05 \
    --max-tokens=75 \
    --voices="ctvrtnicek,moravec,steindler,zeman,mucha,holubova,kopecka,vasaryova" \
    --content-description="A conversation about Rust and AI"
```

## Status: PoC 
(It works... sometimes somehow. Need voices with much better quality + access to betterr HW.)
