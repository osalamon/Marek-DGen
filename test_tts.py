import torch
from TTS.api import TTS

# --- THE PYTORCH 2.6 FIX ---
# This forces PyTorch to allow older model formats like Coqui's to load properly.
_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_load
# ---------------------------

device = "cuda"

# 2. Initialize the model (Warning: This will download ~2GB of data on the very first run!)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 3. Run the local voice clone test
print(
    "Synthesizing audio... (If running on CPU, grab a coffee, this will take a minute!)"
)
tts.tts_to_file(
    text="Zkouším, zda se některý z ukradených hlasů aspoň nahrubo trefí. Zatím lokálně na laptopu, než to nechám chroustat monstrum v mém kanclu, poděděném po Dejvovi.",
    speaker_wav="voices/z04vas.wav",
    language="cs",
    file_path="output/testvasaryova.wav",
)

print("\n[Success] Audio successfully generated! Check `output/testvasaryova.wav`")
