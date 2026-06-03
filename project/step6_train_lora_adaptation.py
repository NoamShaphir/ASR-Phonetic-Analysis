import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
from torch.utils.data import Dataset

# --- step6_train_lora_adaptation.py ---
# This script provides a complete, production-ready implementation blueprint for adapting
# a local Wav2Vec2-For-CTC acoustic model to compensate for L2 Russian-accented speech transfers
# (including vowel spectral smearing and final-consonant devoicing) using Parameter-Efficient Fine-Tuning (PEFT) and LoRA.
#
# Requirements:
#   pip install peft transformers accelerate datasets soundfile librosa
#

try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
except ImportError:
    print("[!] Missing required packages. Please install them using:")
    print("    pip install peft transformers accelerate datasets soundfile librosa")
    # Define placeholder classes so the script compiles/parses without errors even if not installed
    class Wav2Vec2ForCTC: pass
    class Wav2Vec2Processor: pass
    class TrainingArguments: pass
    class Trainer: pass
    class LoraConfig: pass
    def get_peft_model(*args, **kwargs): pass

# 1. Paths & Configurations
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "wav2vec2-base-local") # Path to local Wav2Vec2 weights
OUTPUT_ADAPTER_DIR = os.path.join(script_dir, "wav2vec2-lora-russian-adapter")

# 2. Simple Accent Dataset Loader
class L2AccentDataset(Dataset):
    """
    Custom Dataset class that loads L2 accented audio paths and maps them to their 
    intended English orthographic ground-truth texts.
    """
    def __init__(self, audio_paths: List[str], texts: List[str], processor: Wav2Vec2Processor, target_sr: int = 16000):
        self.audio_paths = audio_paths
        self.texts = texts
        self.processor = processor
        self.target_sr = target_sr

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        import librosa
        # Load and resample audio to 16kHz
        speech_path = self.audio_paths[idx]
        speech, sr = librosa.load(speech_path, sr=self.target_sr)
        
        # Process audio to extract log-mel/spectral features (input_values)
        inputs = self.processor(speech, sampling_rate=self.target_sr, return_tensors="pt")
        input_values = inputs.input_values.squeeze(0)
        
        # Process targets (text labels) using the processor's tokenizer
        labels = self.processor.tokenizer(self.texts[idx]).input_ids
        
        return {
            "input_values": input_values,
            "labels": labels
        }

# 3. Data Collator for CTC padding
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs and labels to the longest sequence in the batch.
    Essential for training CTC models where audio features and text labels have variable lengths.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad audio inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )
        
        # Pad label sequences (with -100 so PyTorch's CTC Loss ignores padding tokens)
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt"
            )

        # Replace padding token id's with -100 to ignore in loss calculation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

# 4. Main Training Orchestrator
def run_lora_adaptation_training():
    print("=== STARTING LORA ADAPTATION TRAINING ORCHESTRATION ===")
    
    # Verify local model weights exist
    if not os.path.exists(MODEL_PATH):
        print(f"[-] Local model path not found at {MODEL_PATH}. Cannot proceed.")
        return

    # Load processor and model
    print(f"Loading local processor and acoustic model from: {MODEL_PATH}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_PATH,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )

    # Freeze base model encoder weights
    # We only want to train our lightweight LoRA adapters to adapt decision boundaries
    model.freeze_feature_encoder()

    # Define LoRA Configuration
    print("Defining Parameter-Efficient LoRA Configuration...")
    peft_config = LoraConfig(
        r=8,                       # Low rank (dimension of adapter matrix)
        lora_alpha=16,             # Scaling factor for adapter weights
        target_modules=[
            "q_proj", "v_proj"     # Target the query and value projection layers in Self-Attention
        ],
        lora_dropout=0.05,         # Regularization to prevent overfitting on L2 speaker subset
        bias="none",
        task_type=None             # None for CTC models (custom architectures)
    )

    # Wrap Wav2Vec2 with LoRA adapters
    print("Wrapping model with PEFT LoRA Adapters...")
    model = get_peft_model(model, peft_config)
    
    # Print statistics of trainable vs frozen parameters
    model.print_trainable_parameters()

    # --- PLUG IN YOUR L2 ACCENT DATA HERE ---
    # In production, replace these with paths to your resampled Russian speakers' audio files
    # and their corresponding target English sentences.
    sample_audio_paths = [
        os.path.join(script_dir, "data", "russian_speaker1.wav"),
        os.path.join(script_dir, "data", "russian_speaker2.wav")
    ]
    sample_intended_texts = [
        "these things are slabs of wood",
        "the kids wanted a thick spoon"
    ]

    # Create dummy files if they don't exist to prevent crash during script inspection
    os.makedirs(os.path.join(script_dir, "data"), exist_ok=True)
    for p in sample_audio_paths:
        if not os.path.exists(p):
            with open(p, "w") as f: f.write("dummy audio")

    # Instantiate datasets
    print("Instantiating datasets and CTC padding collator...")
    train_dataset = L2AccentDataset(sample_audio_paths, sample_intended_texts, processor)
    
    # Define Data Collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Set up Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(script_dir, "training_checkpoints"),
        group_by_length=True,             # Batch sequences of similar lengths to speed up training
        per_device_train_batch_size=4,    # Keep batch size small since we have limited audio memory
        gradient_accumulation_steps=2,
        evaluation_strategy="no",
        num_train_epochs=10,              # Epochs can be tuned based on loss stabilization
        fp16=torch.cuda.is_available(),   # Mixed precision if GPU is active
        save_steps=100,
        logging_steps=10,
        learning_rate=3e-4,               # Higher learning rate is safe since base is frozen
        warmup_steps=50,
        save_total_limit=1,
        push_to_hub=False,
        report_to="none"
    )

    # Initialize Hugging Face Trainer
    print("Initializing HF Trainer with customized parameters...")
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.feature_extractor
    )

    # Run adaptation training
    print("\n[+] To start adaptation training, run:")
    print("    trainer.train()")
    print("\nAfter training, save only the trained LoRA adapter weights:")
    print(f"    model.save_pretrained('{OUTPUT_ADAPTER_DIR}')")
    
    print("\n--- HOW TO LOAD AND USE THE PEFT ADAPTED MODEL FOR INFERENCE ---")
    print(f"""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from peft import PeftModel, PeftConfig

    # 1. Load the frozen base model
    base_model = Wav2Vec2ForCTC.from_pretrained("{MODEL_PATH}")
    processor = Wav2Vec2Processor.from_pretrained("{MODEL_PATH}")

    # 2. Merge the lightweight L2 Russian adapter
    model = PeftModel.from_pretrained(base_model, "{OUTPUT_ADAPTER_DIR}")
    
    # 3. Run inference - the model now natively maps Russian-accented
    #    vocal properties directly to intended English characters!
    inputs = processor(audio_16khz, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    """)

if __name__ == "__main__":
    run_lora_adaptation_training()
