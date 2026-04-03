import whisperx
import gc
import torch
from whisperx.diarize import DiarizationPipeline

def transcribe_and_diarize(audio_file: str, hf_token: str):
	device = "cuda"
	audio_file = "data/whisperx2mintest.mp4"
	batch_size = 8 # reduce if low on GPU mem
	compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

	# 1. Loading standard WhisperX model
	model_dir = "output/"
	model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

	print(f">>>Transcribing audio: {audio_file}...")
	audio = whisperx.load_audio(audio_file)
	result = model.transcribe(audio, batch_size=batch_size)
	print(f">>>Before alignment:::")
	print(result["segments"])

	# free up GPU memory before speakers' alignment
	gc.collect()
	torch.cuda.empty_cache()
	del model

	print(f">>>Aligning WhisperX output...")
	model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
	result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

	print(f">>>After alignment:::")
	print(result["segments"]) # after alignment

	# delete model if low on GPU resources
	gc.collect()
	torch.cuda.empty_cache()
	del model_a

	print(f">>>3. Assigning speaker labels (diarization)...")
	diarize_model = DiarizationPipeline(token=hf_token, device=device)

	diarize_segments = diarize_model(audio)
	# add min/max number of speakers if known
	# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

	result = whisperx.assign_word_speakers(diarize_segments, result)
	print(diarize_segments)
	print(result["segments"]) # segments are now assigned speaker IDs

	return result["segments"]
