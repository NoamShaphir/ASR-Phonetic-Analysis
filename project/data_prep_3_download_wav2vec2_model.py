import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# --- Wav2Vec2 Base Model Downloader Utility ---
# Downloads the "facebook/wav2vec2-base-960h" processor and CTC model weights
# from Hugging Face and saves them locally inside project/wav2vec2-base-local/
# to allow complete offline execution of the ASR pipeline.

model_name = "facebook/wav2vec2-base-960h"
script_dir = os.path.dirname(os.path.abspath(__file__))
local_directory = os.path.join(script_dir, "wav2vec2-base-local")

print(f"Downloading model and processor from Hugging Face: {model_name}...")
print(f"Local storage destination: {local_directory}")

try:
    # Fetch from Hugging Face Hub (one-time download)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Save weights and config locally
    processor.save_pretrained(local_directory)
    model.save_pretrained(local_directory)

    print(f"\n[+] Model and Processor successfully saved locally to: {local_directory}")
except Exception as e:
    print(f"\n[X] Failed to download or save model: {e}")
