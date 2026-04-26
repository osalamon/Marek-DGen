import torch

# PyTorch 2.6+ defaults torch.load to weights_only=True, which breaks
# loading of older whisperx / pyannote checkpoints that contain
# non-standard globals (e.g. omegaconf).  Force the old behavior.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False even if the caller explicitly passes True,
    # because third-party checkpoints (pyannote, whisperx) contain
    # non-standard globals that PyTorch's safe unpickler rejects.
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import whisperx
import gc
from whisperx.diarize import DiarizationPipeline

def transcribe_and_diarize(audio_file: str, hf_token: str):
	audio_file = audio_file.strip()
	device = "cuda"				# or "cpu"
	batch_size = 4 				# 16, reduce if low on GPU mem
	compute_type = "int8" # "float16", change to "int8" if low on GPU mem (may reduce accuracy)
	speakers_count = 2 	# set to None if unknown, or specify number of speakers
	model_name = "large-v2" # Automatic Speech Recognition (ASR) model, e.g. "base", "large-v2"

	print(f">>>1a) Loading standard WhisperX model...")
	# model_dir = "output/" # add download_root argument lo load model to a specific directory
	model = whisperx.load_model(model_name, device, compute_type=compute_type)

	print(f">>>1b) Transcribing audio: {audio_file}...")
	audio = whisperx.load_audio(audio_file)
	result = model.transcribe(audio, batch_size=batch_size)

	print(f">>>Before alignment:::")
	print(result["segments"])

	# free up GPU memory before speakers' alignment
	gc.collect()
	torch.cuda.empty_cache()
	del model

	print(f">>>2. Aligning WhisperX output...")
	model_align, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
	result = whisperx.align(result["segments"], model_align, metadata, audio, device, return_char_alignments=False)

	print(f">>>After alignment:::")
	print(result["segments"]) # after alignment

	# free up GPU memory again before diarization
	gc.collect()
	torch.cuda.empty_cache()
	del model_align

	print(f">>>3. Assigning speaker labels (diarization)...")
	diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
	diarize_segments = diarize_model(audio, min_speakers=speakers_count, max_speakers=speakers_count)
	result = whisperx.assign_word_speakers(diarize_segments, result)

	print(diarize_segments)
	print(f">>>After diarization:::")
	print(result["segments"]) # segments are now assigned speaker IDs

	return result["segments"]
