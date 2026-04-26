# Marek-DGen
Just a silly little ML pipeline intended for translating and dubbing the most important epizodes of my favourite podcasts, making them accessible for members of my family not very well versed in English.

## Usage via CLI

```bash
dabuj data/whisperx2mintest.mp4 \														# input English video to be dubbed into Czech
			--hf-token="<hf_YOUR_HUGGINGFACE_API_READ_TOKEN>" \		# free subscription tier is OK, get your key from HF UI
			--einfra-baseurl="https://llm.ai.e-infra.cz/v1" \			# viz
			--einfra-apikey="sk-<YOUR_EINFRA_AI_AS_A_SERVICE_API_TOKEN>" \	# get your key from E-infra AIaaS OpenWebUI > Settings > Account >
			--translating-model="qwen3.5-122b" \
			--voices="ctvrtnicek,moravec,steindler,zeman,mucha,holubova,kopecka,vasaryova" \
			--output-dir="output/czdabing.wav"
Just a silly little ML pipeline intended for translating and dubbing the most important epizodes of my favourite podcasts, making them accessible for member of my family, who is not very well versed in English.

## Status: work in progress, there're only last two steps of pipeline missing
- TTS is almost finished
- the last missing piece: assembling audio/video files back via ffmpeg
